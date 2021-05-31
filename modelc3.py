import json

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel
from tqdm import tqdm
import torch.nn.functional as F

class subject_model(nn.Module):
    def __init__(self,hidden_size,lstmnum_layers):
        super(subject_model,self).__init__()
        self.input_size = 768
        self.hidden_size = hidden_size
        self.num_layers = lstmnum_layers
        self.output_size = 2
        self.bilstm1 = nn.LSTM(self.input_size, self.hidden_size,self.num_layers,batch_first=True,bidirectional=True).cuda()
        self.bilstm2 = nn.LSTM(self.hidden_size*2, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True).cuda()   

        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.q = nn.Parameter(torch.Tensor(self.hidden_size * 2, self.hidden_size * 2).cuda()) #Q权值矩阵
        self.k = nn.Parameter(torch.Tensor(self.hidden_size * 2, self.hidden_size * 2).cuda())    #K权值矩阵        
        nn.init.uniform_(self.q, -0.1, 0.1).cuda() #初始化为均匀分布
        nn.init.uniform_(self.k, -0.1, 0.1).cuda()
        
        self.hidden2tag = nn.Linear(self.hidden_size*2, self.output_size).cuda()
        self.posb = nn.Sigmoid().cuda() #转换为概率       
        
    def forward(self, inputs):
        x1,(hn1, cn1) = self.bilstm1(inputs)
        
        # Attention过程
        query = torch.matmul(x1, self.q)
        # query形状是(batch_size, seq_len, 2 * num_hiddens)
        key = torch.tanh(torch.matmul(x1.data, self.k))
        att = torch.matmul(query, key.permute(0,2,1))
        # att形状是(batch_size, seq_len, seq_len)        
        att_score = F.softmax(att, dim=2)
        # att_score形状为(batch_size, seq_len, seq_len)
        scored_x = torch.bmm(att_score,x1.data)
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束        
        x2,(hn2, cn2) = self.bilstm2(scored_x)
        x3 = self.hidden2tag(x2)
        out1 = self.posb(x3)
        return out1
        


def subj_feature(x):  #1修改
    seq, idxs = x #均为tensor
    idxs = idxs.int()
    batch_idxs = torch.arange(0,seq.size(0)).int().cuda()
    batch_idxs = batch_idxs.unsqueeze(dim = 1)
    idxs = torch.cat([batch_idxs, idxs], dim=1)
    
    random_subject_features = []
    for i in range(idxs.size(0)):
        vec = seq[idxs[i][0],idxs[i][1],:]  
        #print (vec)
        random_subject_features.append(torch.unsqueeze(vec,0))
    
    random_subject_features = torch.cat(random_subject_features) #默认dim=0
    return random_subject_features

def feature_sum(x):
    """seq是[None, seq_len, embeds_size]的格式， None为未知的batch_size
    vec是[None, embeds_size]的格式，将vec重复seq_len次，加到seq上，
    得到[None, seq_len, embeds_size]的向量。
    """
    seq, feature = x
    for i in range(feature.size(0)):
        #seq[i,:,:] += feature[i,:]
        seq[i,:,:] = seq[i,:,:]+feature[i,:]
        #print(seq[i,:,:] )
    return seq

class object_model(nn.Module): #多标签分类
    def __init__(self,num_rels,hidden_size,lstmnum_layers):
        super(object_model,self).__init__()
        self.input_size = 768
        self.hidden_size = hidden_size
        self.num_layers = lstmnum_layers
        self.output_size = num_rels *2
        self.bilstm1 = nn.LSTM(self.input_size, self.hidden_size,self.num_layers,batch_first=True,bidirectional=True).cuda()
        self.bilstm2 = nn.LSTM(self.hidden_size*2, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True).cuda()
         # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.q = nn.Parameter(torch.Tensor(self.hidden_size * 2, self.hidden_size * 2).cuda()) #Q权值矩阵
        self.k = nn.Parameter(torch.Tensor(self.hidden_size * 2, self.hidden_size * 2).cuda())    #K权值矩阵        
        nn.init.uniform_(self.q, -0.1, 0.1).cuda() #初始化为均匀分布
        nn.init.uniform_(self.k, -0.1, 0.1).cuda()
        
        self.hidden2tag = nn.Linear(self.hidden_size*2, self.output_size).cuda()
        self.posb = nn.Sigmoid().cuda() #转换为概率         

    def forward(self, inputs,sub_head_in,sub_tail_in):
        sub_head_feature = subj_feature([inputs, sub_head_in])
        sub_tail_feature = subj_feature([inputs, sub_tail_in])
        sub_feature = (sub_head_feature+sub_tail_feature)/2 #得到头实体的向量和
        embeds = feature_sum([inputs,sub_feature])
        x1, (h_n1,c_n1) = self.bilstm1(embeds)
        # Attention过程
        query = torch.matmul(x1, self.q)
        # query形状是(batch_size, seq_len, 2 * num_hiddens)
        key = torch.tanh(torch.matmul(x1.data, self.k))
        att = torch.matmul(query, key.permute(0,2,1))
        # att形状是(batch_size, seq_len, seq_len)        
        att_score = F.softmax(att, dim=2)
        # att_score形状为(batch_size, seq_len, seq_len)
        scored_x = torch.bmm(att_score,x1.data)
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束        
        x2,(hn2, cn2) = self.bilstm2(scored_x)
        x3 = self.hidden2tag(x2)
        out2 = self.posb(x3)      
        return out2


def extract_triples(sub_model, obj_model, tokenizer, bert_model, text_in, id2rel, h_bar=0.5, t_bar=0.5):  #提取模型预测的关系三元组
    text_in1="[CLS] "+text_in+" [SEP]"
    tokens = tokenizer.tokenize(text_in1)
    embeds = tokenizer.encode_plus(text_in)   #pytorch中transformers的bert模型调用
    token_ids = torch.tensor([embeds['input_ids']])
    segment_ids = torch.tensor([embeds['token_type_ids']])
    atten_mask = torch.tensor([embeds['attention_mask']])
    with torch.no_grad():
        encoded_layers = bert_model(token_ids, atten_mask, segment_ids,output_hidden_states=True).hidden_states

        token_embeddings = []
        for token_i in range(token_ids.shape[1]):  # 得到每个分词的所有隐状态序列
            hidden_layers = []
            # For each of the 12 layers...
            for layer_i in range(len(encoded_layers)):
                # Lookup the vector for `token_i` in `layer_i`
                vec = encoded_layers[layer_i][0][token_i]  # tensor [768]
                hidden_layers.append(vec.numpy().tolist())
            token_embeddings.append(hidden_layers)
        embeddings_ids = [(np.mean(token_i_layer[-4:], 0)) for token_i_layer in token_embeddings]
    with torch.no_grad():    
        sub_heads_logits, sub_tails_logits = (sub_model(torch.tensor([embeddings_ids]).float().cuda())).chunk(2, 2)
    sub_heads_logits = (sub_heads_logits[0].squeeze(dim=1).cpu()).numpy() #1维
    #print (sub_heads_logits)
    sub_tails_logits = (sub_tails_logits[0].squeeze(dim=1).cpu()).numpy()
    sub_heads, sub_tails = np.where(sub_heads_logits > h_bar)[0], np.where(sub_tails_logits > t_bar)[0] #求在文本分词中的索引号
    sub_heads = sub_heads.tolist()
    if 0 in sub_heads:
        sub_heads.remove(0) #删除[CLS]
    if len(token_ids)-1 in sub_heads:
        sub_heads.remove(len(token_ids)-1) #删除[SEP]
    sub_heads = np.array(sub_heads)
    sub_tails = sub_tails.tolist()
    if 0 in sub_tails:
        sub_tails.remove(0)
    if len(token_ids)-1 in sub_tails:
        sub_tails.remove(len(token_ids)-1)
    sub_tails = np.array(sub_tails)
    subjects = []
    for sub_head in sub_heads:
        sub_tail = sub_tails[sub_tails >= sub_head] #求每个实体的末索引
        if len(sub_tail) > 0:
            sub_tail = sub_tail[0] #最近法匹配末索引
            subject = tokens[sub_head: sub_tail+1]  #######+1?
            subjects.append((subject, sub_head, sub_tail)) #头实体内容，头索引值，末索引值
            
    triple_set = set()
    #print(subjects)
    if subjects:
        triple_list = []
        embeddings_ids1 = []
        for x in range(len(subjects)):
            embeddings_ids1.append(embeddings_ids)
        sub_heads, sub_tails = np.array([sub[1:] for sub in subjects]).T.reshape((2, -1, 1))
        sub_heads = torch.tensor(sub_heads).cuda()
        #print (sub_heads.dtype) #int64
        sub_tails = torch.tensor(sub_tails).cuda()
        with torch.no_grad():
            obj_heads_logits, obj_tails_logits = (obj_model(torch.tensor(embeddings_ids1).float().cuda(), sub_heads, sub_tails)).chunk(2, 2)
        obj_heads_logits = obj_heads_logits[:,:-1].cpu().numpy() #删除[SEP]
        obj_tails_logits = obj_tails_logits[:,:-1].cpu().numpy()
        
        for i, subject in enumerate(subjects):
            sub = subject[0]
            sub = " ".join(sub).replace(" ##", "").strip() #来自bert源码tokenizer.convert_tokens_to_string("tokens")函数
            obj_heads, obj_tails = np.where(obj_heads_logits[i] > h_bar), np.where(obj_tails_logits[i] > t_bar)
            for obj_head, rel_head in zip(*obj_heads):
                for obj_tail, rel_tail in zip(*obj_tails):
                    if obj_head <= obj_tail and rel_head == rel_tail:
                        rel = id2rel[rel_head]
                        obj = tokens[obj_head: obj_tail+1]
                        obj = " ".join(obj).replace(" ##", "").strip()
                        triple_list.append((sub, rel, obj))
                        break
        triple_set = set()
        for s, r, o in triple_list:
            triple_set.add((s, r, o))
        #print(triple_set)
        return list(triple_set)
    else:
        return []
    



def partial_match(pred_set, gold_set):
    pred = {(i[0].split(' ')[-1] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(' ')[-1] if len(i[2].split(' ')) > 0 else i[2]) for i in pred_set}
    gold = {(i[0].split(' ')[-1] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(' ')[-1] if len(i[2].split(' ')) > 0 else i[2]) for i in gold_set}
    return pred, gold
    
def evaluate(sub_model, obj_model, tokenizer, eval_data, id2rel, output_path=None, exact_match=False, h_bar=0.5, t_bar=0.5):
    F = None
    if output_path:
        F = open(output_path, 'w') #以只写方式打开文件.如果文件不存在，创建该文件;如果文件已存在，先清空，再打开文件
    orders = ['subject', 'relation', 'object']
    #TP, predict_Num, gold_Num = 1e-10, 1e-10, 1e-10
    TP, predict_Num, gold_Num = 0,0,0

    bert_model = BertModel.from_pretrained('../bert-base-cased')
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    bert_model.eval()
    for line in tqdm(iter(eval_data)):
    #for idx in range(3):
        #line = eval_data[idx]
        Pred_triples = set(extract_triples(sub_model, obj_model, tokenizer,bert_model, line['text'], id2rel,h_bar, t_bar))
        #print (Pred_triples)
        Gold_triples = set(line['triple_list'])
        
        predict, gold = partial_match(Pred_triples, Gold_triples) if not exact_match else (Pred_triples, Gold_triples)        
        #print(predict)
        #print (gold)
        TP += len(predict & gold)
        #print (TP)
        predict_Num += len(predict)
        gold_Num += len(gold)
        
        if output_path:
            result = json.dumps({
                'text': line['text'],
                'triple_list_gold': [
                    dict(zip(orders, triple)) for triple in Gold_triples
                ],
                'triple_list_pred': [
                    dict(zip(orders, triple)) for triple in Pred_triples
                ],
                'new': [
                    dict(zip(orders, triple)) for triple in Pred_triples - Gold_triples
                ],
                'lack': [
                    dict(zip(orders, triple)) for triple in Gold_triples - Pred_triples
                ]
            }, ensure_ascii=False, indent=4)
            F.write(result + '\n')
    if output_path:
        F.close()

    if predict_Num !=0 and gold_Num !=0:
        return TP / predict_Num, TP / gold_Num, 2 * TP / (predict_Num + gold_Num)
    else:
        return 0, 0, 0