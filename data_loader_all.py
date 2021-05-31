#! -*- coding:utf-8 -*-
import numpy as np
import re, os, json
from random import choice

import torch
from transformers import BertModel

BERT_MAX_LEN = 512
RANDOM_SEED = 2019

def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

def to_tuple(sent):
    triple_list = []
    for triple in sent['triple_list']:
        triple_list.append(tuple(triple))
    sent['triple_list'] = triple_list

def seq_padding(batch, padding=0):  #每一条文本分词序列采用补齐0策略padding。与每批中最长序列长度一致
    length_batch = [len(seq) for seq in batch]
    max_length = max(length_batch)
    return torch.from_numpy(np.array([  #(batch，max-length)
        np.concatenate([seq, [padding] * (max_length - len(seq))]) if len(seq) < max_length else seq for seq in batch
    ]))
    
def seq_padding1(batch, padding):  #每一条文本分词序列采用补齐0策略padding。与每批中最长序列长度一致
    triple_sum = [len(seq) for seq in batch]
    seq_length = [len(seq[0]) for seq in batch]
    triple_max_sum = max(triple_sum)
    s_max_length = max(seq_length)
    batch1 = []
    
    for seq in batch:
        sentence = []
        for triple in seq:
            if len(triple) < s_max_length:
                triple = np.concatenate([triple, [padding] * (s_max_length - len(triple))])
            sentence.append(triple)
        if len(sentence) < triple_max_sum:
            pad = np.zeros((s_max_length,len(padding)))
            sentence = np.concatenate([sentence,[pad] * (triple_max_sum - len(sentence))])
        batch1.append(sentence)
    return torch.tensor(batch1)
    
def seq_padding2(batch):  #每一条文本分词序列采用补齐0策略padding。与每批中最长序列长度一致
    triple_sum = [len(seq) for seq in batch]
    triple_max_sum = max(triple_sum)
    batch1 = []
    
    for sentence in batch:
        if len(sentence) < triple_max_sum:
            sentence = np.concatenate([sentence,[0] * (triple_max_sum - len(sentence))])
        batch1.append(sentence)
    return torch.tensor(batch1)    
    
def load_data(train_path, dev_path, test_path, rel_dict_path): #加载全部数据、关系序列和反序列，并对训练集进行洗牌
    train_data = json.load(open(train_path))
    dev_data = json.load(open(dev_path))
    test_data = json.load(open(test_path))
    id2rel, rel2id = json.load(open(rel_dict_path))

    id2rel = {int(i): j for i, j in id2rel.items()}
    num_rels = len(id2rel)

    random_order = list(range(len(train_data)))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(random_order)
    train_data = [train_data[i] for i in random_order]  #洗牌训练集

    for sent in train_data: 
        to_tuple(sent) #把每一个关系三元组总list中每个元素（，，）变为元组
    for sent in dev_data:  
        to_tuple(sent)
    for sent in test_data: 
        to_tuple(sent)

    print("train_data len:", len(train_data))
    print("dev_data len:", len(dev_data))
    print("test_data len:", len(test_data))

    return train_data, dev_data, test_data, id2rel, rel2id, num_rels


class data_generator:  #数据生成器-->迭代器
    def __init__(self, data, tokenizer, rel2id, num_rels, maxlen, batch_size=32):
        self.data = data
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.num_rels = num_rels
        self.maxlen = maxlen
        self.batch_size = batch_size
       
    def __iter__(self): #返回所有批的输入id矩阵,标注矩阵，迭代器
        #while True:
        #idxs = list(range(len(self.data)))
        idxs = list(range(12000)) #只加载12000条数据
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idxs)
        tokens_batch, segments_batch, sub_heads_batch, sub_tails_batch, sub_head_batch, sub_tail_batch, obj_heads_batch, obj_tails_batch = [], [], [], [], [], [], [], []
        embeddings_batch = []
        # 使用预训练模型拿词向量
        #model = BertModel.from_pretrained('bert-base-cased')
        model = BertModel.from_pretrained('./bert-base-cased')
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        model.eval()
        for idx in idxs:
            line = self.data[idx]
            #print(line['text'])
            sentence="[CLS] "+line['text']+" [SEP]"
            text = ' '.join(sentence.split()[:self.maxlen+1])  #去掉空格制表符等符号。一句话的单词+标点至多maxlen
            text1 = ' '.join(line['text'].split()[:self.maxlen])
            #tokens = self.tokenizer.tokenize(sentence)
            tokens = self.tokenizer.tokenize(text)  #分词
            if len(tokens) > BERT_MAX_LEN: #分词后文本序列长度至多512 BERT模型的输入。 文本序列长度没对齐？？？？？
                tokens = tokens[:BERT_MAX_LEN]
            text_len = len(tokens)

            s2ro_map = {} #dict{(头实体头索引,头实体末索引):[(尾实体头索引,尾实体末索引,关系id数字型)……] , ……}
            for triple in line['triple_list']:
                triple = (self.tokenizer.tokenize(triple[0]), triple[1], self.tokenizer.tokenize(triple[2]))  #tokenize无[cls][sep]
                sub_head_idx = find_head_idx(tokens, triple[0])  #头实体在分词后的句子中的索引
                obj_head_idx = find_head_idx(tokens, triple[2])  #尾实体----
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)  #头实体的（头索引，末索引）
                    if sub not in s2ro_map: #对每个头实体找和它有关系的尾实体
                        s2ro_map[sub] = []
                    s2ro_map[sub].append((obj_head_idx,
                                       obj_head_idx + len(triple[2]) - 1,
                                       self.rel2id[triple[1]]))

            if s2ro_map:
                #token_ids, segment_ids = self.tokenizer.encode(first=text) #keras-bert中的用法。不是问答，只设置第一句话，第二句为None
                embeds = self.tokenizer.encode_plus(text1)   #pytorch中transformers的bert模型调用
                token_ids = embeds['input_ids']#[1:-1]
                segment_ids = embeds['token_type_ids']#[1:-1]
                if len(token_ids) > text_len:
                    token_ids = token_ids[:text_len] #去掉最后标点符号和[sep]或单词对应的token，可能丢失尾实体，保留[cls]保证find_head_idx正确
                    segment_ids = segment_ids[:text_len]
                tokens_batch.append(token_ids)
                segments_batch.append(segment_ids)
                sub_heads, sub_tails = np.zeros(text_len, dtype=np.float32), np.zeros(text_len, dtype=np.float32) ###一个句子的subject标注
                obj_heads=[]
                obj_tails=[]
                sub_head=[]
                sub_tail=[]
                for s in s2ro_map:
                    sub_heads[s[0]] = 1 #头实体的头索引位置为1，标注所有头实体的头索引    
                    sub_tails[s[1]] = 1 #头实体的末索引位置为1，标注所有头实体的末索引
                    sub_head.append(s[0])
                    sub_tail.append(s[1])
                #sub_head, sub_tail = choice(list(s2ro_map.keys()))  #随机返回一个头实体的索引号
                    obj_heads1, obj_tails1 = np.zeros((text_len, self.num_rels), dtype=np.float32), np.zeros((text_len, self.num_rels), dtype=np.float32)
                    for ro in s2ro_map.get((s[0], s[1]), []): 
                        obj_heads1[ro[0]][ro[2]] = 1
                        obj_tails1[ro[1]][ro[2]] = 1
                    obj_heads.append(obj_heads1.tolist())
                    obj_tails.append(obj_tails1.tolist())
                sub_heads_batch.append(sub_heads)
                sub_tails_batch.append(sub_tails)
                sub_head_batch.append(sub_head)
                sub_tail_batch.append(sub_tail)
                obj_heads_batch.append(obj_heads)
                obj_tails_batch.append(obj_tails)
                if len(tokens_batch) == self.batch_size or idx == idxs[-1]:
                    tokens_batch = seq_padding(tokens_batch) #[batch_size,text_len(seq_len一个句子的序列长度)]
                    #print (tokens_batch.dtype) #int32
                    segments_batch = seq_padding(segments_batch)   #[batch_size,text_len(seq_len一个句子的序列长度)]
                    #print (segments_batch.dtype) #int32
                    atten_mask = (tokens_batch.gt(0)).int()
                    #print (atten_mask)

                    with torch.no_grad():
                        encoded_layers = model(tokens_batch, attention_mask=atten_mask, token_type_ids=segments_batch,output_hidden_states=True).hidden_states

                    for batch_i in range(tokens_batch.shape[0]):
                        token_embeddings = []
                        for token_i in range(tokens_batch.shape[1]):  # 得到每个分词的所有隐状态序列
                            hidden_layers = []
                            # For each of the 12 layers...
                            for layer_i in range(len(encoded_layers)):
                                # Lookup the vector for `token_i` in `layer_i`
                                vec = encoded_layers[layer_i][batch_i][token_i]  # tensor [768]
                                hidden_layers.append(vec.numpy().tolist())
                            token_embeddings.append(hidden_layers)

                        embeddings_b = [(np.mean(token_i_layer[-4:], 0)).tolist() for token_i_layer in token_embeddings]
                        embeddings_batch.append(embeddings_b)
                    
                    embeddings_batch = torch.tensor(embeddings_batch)
                    #print (embeddings_batch.shape) #(batch_size,seq_len,768)
                    #print (embeddings_batch.dtype)#float32
                    #print (type(sub_heads_batch)) #list 每个元素为float32数组

                    sub_heads_batch = seq_padding(sub_heads_batch) #[batch_size,text_len(seq_len一个句子的序列长度)]
                    sub_tails_batch = seq_padding(sub_tails_batch) #[batch_size,text_len(seq_len一个句子的序列长度)]
                    
                    obj_heads_batch = seq_padding1(obj_heads_batch, np.zeros(self.num_rels))#[batch_size,text_len(seq_len一个句子的序列长度),num_rels]
                    #print (obj_heads_batch.shape)
                    obj_tails_batch = seq_padding1(obj_tails_batch, np.zeros(self.num_rels))#[batch_size,text_len(seq_len一个句子的序列长度),num_rels]
                    sub_head_batch, sub_tail_batch = seq_padding2(sub_head_batch), seq_padding2(sub_tail_batch)  #[batch_size,1]
                    #print (sub_head_batch)
                    #均为array--->tensor
                    yield [tokens_batch, segments_batch, embeddings_batch, sub_heads_batch, sub_tails_batch, sub_head_batch, sub_tail_batch, obj_heads_batch, obj_tails_batch], None
                    tokens_batch, segments_batch, embeddings_batch, sub_heads_batch, sub_tails_batch, sub_head_batch, sub_tail_batch, obj_heads_batch, obj_tails_batch, = [], [], [], [], [], [], [], [], []


