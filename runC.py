#! -*- coding:utf-8 -*-
from data_loader1 import data_generator, load_data
from transformers import BertTokenizer, BertModel
from modelC import subject_model,object_model,evaluate
import os, argparse,torch
import torch.nn as nn
import numpy as np 


device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    
    dataset = 'NYT'
    train_path = 'data/'+dataset+'/train_triples.json'
    dev_path = './data/'+dataset+'/dev_triples.json'
    test_path = './data/'+dataset+'/test_triples.json' # overall test
    rel_dict_path = './data/'+dataset+'/rel2id.json'    
    output_path = './data/'+dataset+'/output.json'
    LR = 1e-4
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    train_data, dev_data, test_data, id2rel, rel2id, num_rels = load_data(train_path, dev_path, test_path, rel_dict_path)
    
    
    BATCH_SIZE = 6
    EPOCH = 20
    MAX_LEN = 100
    data_manager = data_generator(train_data, tokenizer, rel2id, num_rels, MAX_LEN, BATCH_SIZE)
    Input_size = 60
    num_layers = 1
    
    #CHECKPOINT = torch.load('data/'+dataset+'/all/paramsg_31.pkl',map_location=device)
    sub_model = subject_model(Input_size,num_layers)
    #sub_model.load_state_dict(CHECKPOINT['sub_state_dict'])
    sub_model.to(device)
    obj_model = object_model(num_rels,Input_size,num_layers)
    #obj_model.load_state_dict(CHECKPOINT['obj_state_dict'])
    obj_model.to(device)
    params = list(sub_model.parameters()) + list(obj_model.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)
    state = {'sub_state_dict': sub_model.state_dict(),  # 模型训练好的参数/store the parameters of 2 models
             'obj_state_dict': obj_model.state_dict()
             }
    loss_func = nn.BCELoss()
    
    file="outc.txt" #保存每个epoch下模型的性能/store the performance of the whole model in every epoch
    
    for epoch in range(EPOCH): 
        data_loader1 = data_manager.__iter__()
        loss = 0.0
        #enumerate(sequence, [start=0])，i序号，data是数据
        for i, data in enumerate(data_loader1):
            optimizer.zero_grad() #将参数的grad值清为0
            # get the inputs
            bert_inputs = data[0][2]
            sub_heads_label = data[0][3]
            sub_tails_label = data[0][4]
            sub_head_input = data[0][5]
            sub_tail_input = data[0][6]
            obj_heads_label = data[0][7]
            obj_tails_label = data[0][8]
            
            s = sub_model(bert_inputs.data)
            o = obj_model(bert_inputs.data,sub_head_input,sub_tail_input)

            sub_label = torch.cat([sub_heads_label.unsqueeze(dim=2),sub_tails_label.unsqueeze(dim=2)],2)
            sub_loss=loss_func(s,sub_label.float())
            
            obj_label = torch.cat([obj_heads_label,obj_tails_label],2)
            obj_loss=loss_func(o,obj_label.float())
            
            loss = sub_loss+obj_loss
            
            loss.backward() #反向传播
            
            optimizer.step() #更新参数

        print("epoch:",epoch,"loss:",loss.data)
        torch.save(state, 'data/'+dataset+'/all/paramsc_' + str(epoch) + '.pkl')
        f = open(file, 'a+')
        f.write('epoch: %d, loss: %.6f \n' % (epoch,loss.data))
   
        precision, recall, f1= evaluate(sub_model,obj_model,tokenizer,dev_data,id2rel)
        f.write('Precision: %.4f, Recall: %.4f, F1: %.4f \n' % (precision, recall, f1))
        print('Precision: %.4f, Recall: %.4f, F1: %.4f' 
            % (precision, recall, f1))        
        f.close()
