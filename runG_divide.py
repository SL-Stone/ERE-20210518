#! -*- coding:utf-8 -*-
from data_loader_divide import data_generator, load_data
from transformers import BertTokenizer, BertModel
from modelG import subject_model,object_model,evaluate
import os, argparse,torch
import torch.nn as nn
import numpy as np 
from multiprocessing import Pool

device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    
    dataset = 'NYT'
    train_path1 = 'data/'+dataset+'/train/train_triples_1.json'
    train_path2 = 'data/'+dataset+'/train/train_triples_2.json'
    train_path3 = 'data/'+dataset+'/train/train_triples_3.json'
    train_path4 = 'data/'+dataset+'/train/train_triples_4.json'
    train_path5 = 'data/'+dataset+'/train/train_triples_5.json'
    dev_path = './data/'+dataset+'/dev_triples1.json'
    test_path = './data/'+dataset+'/test_triples1.json' # overall test
    rel_dict_path = './data/'+dataset+'/rel2id1.json'    
    output_path = './data/'+dataset+'/output1.json'
    LR = 1e-3
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    train_data1, train_data2, train_data3, id2rel, rel2id, num_rels = load_data(train_path1, train_path2, train_path3, rel_dict_path)
    train_data4, train_data5, dev_data, id2rel, rel2id, num_rels = load_data(train_path4, train_path5, dev_path, rel_dict_path)
    
    BATCH_SIZE = 6
    EPOCH = 10
    MAX_LEN = 100
    data_manager1 = data_generator(train_data1[:7872], tokenizer, rel2id, num_rels, MAX_LEN, BATCH_SIZE)
    data_manager2 = data_generator(train_data2[:2575], tokenizer, rel2id, num_rels, MAX_LEN, BATCH_SIZE)
    data_manager3 = data_generator(train_data3[:782], tokenizer, rel2id, num_rels, MAX_LEN, BATCH_SIZE)
    data_manager4 = data_generator(train_data4[:559], tokenizer, rel2id, num_rels, MAX_LEN, BATCH_SIZE)
    data_manager5 = data_generator(train_data5[:212], tokenizer, rel2id, num_rels, MAX_LEN, BATCH_SIZE)
    Input_size = 60
    num_layers = 1
    #CHECKPOINT = torch.load('./data/'+dataset+'/all/paramsd_9.pkl',map_location=device)
    sub_model = subject_model(Input_size,num_layers).cuda()
    #sub_model.load_state_dict(CHECKPOINT['sub_state_dict'])
    sub_model.to(device)
    obj_model = object_model(num_rels,Input_size,num_layers).cuda()
    #obj_model.load_state_dict(CHECKPOINT['obj_state_dict'])
    obj_model.to(device)
    params = list(sub_model.parameters()) + list(obj_model.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)
    state = {'sub_state_dict': sub_model.state_dict(),  # 模型训练好的参数
             'obj_state_dict': obj_model.state_dict()
             }
    loss_func = nn.BCELoss().cuda()
    
    file="outg_d.txt"

       

    for epoch in range(EPOCH):
        data_loader=[]
        data_loader.append(data_manager1.__iter__())
        data_loader.append(data_manager2.__iter__())
        data_loader.append(data_manager3.__iter__())
        data_loader.append(data_manager4.__iter__())
        data_loader.append(data_manager5.__iter__())

        for j in [0,3,1,4,2]:
             for i, data in enumerate(data_loader[j]):
                optimizer.zero_grad() #将参数的grad值初始化为0
                # get the inputs
                bert_inputs = data[0][2].cuda()
                sub_heads_label = data[0][3]
                sub_tails_label = data[0][4]
                sub_head_input = data[0][5].cuda()
                sub_tail_input = data[0][6].cuda()
                obj_heads_label = data[0][7]
                obj_tails_label = data[0][8]

                s = sub_model(bert_inputs.data)
                sub_label = torch.cat([sub_heads_label.unsqueeze(dim=2),sub_tails_label.unsqueeze(dim=2)],2)
                sub_loss=loss_func(s,sub_label.float().cuda())
                loss = sub_loss
                triples_sum = sub_head_input.shape[1]
                sub_head_l = sub_head_input.chunk(triples_sum,1)
                sub_tail_l = sub_tail_input.chunk(triples_sum,1)
                obj_heads_l = obj_heads_label.chunk(triples_sum,1)
                obj_tails_l = obj_tails_label.chunk(triples_sum,1)

                for i in range(triples_sum):
                    o = obj_model(bert_inputs.data,sub_head_l[i],sub_tail_l[i])
                    obj_label = torch.cat([obj_heads_l[i].squeeze(1),obj_tails_l[i].squeeze(1)],2).cuda().float()
                    #o=o*obj_label                
                    obj_loss=loss_func(o,obj_label)
                    loss = loss+obj_loss                            

                loss.backward() #反向传播
                optimizer.step() #更新参数
            

        print("epoch:",epoch,"loss:",loss)
        torch.save(state, 'data/'+dataset+'/all/paramsd_' + str(epoch) + '.pkl')
        f = open(file, 'a+')
        f.write('epoch: %d, loss: %.6f \n' % (epoch,loss))
        #if epoch !=2:        
        precision, recall, f1= evaluate(sub_model,obj_model,tokenizer,dev_data,id2rel)
        f.write('Precision: %.4f, Recall: %.4f, F1: %.4f \n' % (precision, recall, f1))
        print('Precision: %.4f, Recall: %.4f, F1: %.4f' 
            % (precision, recall, f1))        
        f.close()