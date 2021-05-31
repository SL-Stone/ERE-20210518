#! -*- coding:utf-8 -*-
from data_loaderC2 import data_generator, load_data
from transformers import BertTokenizer, BertModel
from modelc2 import subject_model,object_model,evaluate_num
import os, argparse,torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt

device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    
    dataset = 'NYT'
    
    test_overall_path='C:/Users/Administrator/Desktop/毕设/Codes/data/' + dataset + '/test_triples.json' # overall test
    
    rel_dict_path = 'C:/Users/Administrator/Desktop/毕设/Codes/data/'+dataset+'/rel2id.json'    
    p1 = 'C:/Users/Administrator/Desktop/毕设/Codes/记录/程序/'
    p2 = 'C:/Users/Administrator/Desktop/毕设/Codes/记录/'
    output_eva_file =p2+'/output_eva_dataset.txt'
    
    test_num_path = []
    for i in range(5):
        p = 'C:/Users/Administrator/Desktop/毕设/Codes/data/' + dataset + '/test_split_by_num/test_triples_'+str(i+1)+'.json' #['1','2','3','4','5']
        test_num_path.append(p)
    test_num=[0,0]   
    test_overall,test_num[0],test_num[1],id2rel, rel2id, num_rels = load_data(test_overall_path,test_num_path[0],test_num_path[1], rel_dict_path)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    Input_size = 60
    num_layers = 1
    
    
    CHECKPOINT0 = torch.load(p1+'paramsAtt1_13.pkl',map_location=device)
    sub_model0 = subject_model(Input_size,num_layers)
    sub_model0.load_state_dict(CHECKPOINT0['sub_state_dict'])
    obj_model0 = object_model(num_rels,Input_size,num_layers)
    obj_model0.load_state_dict(CHECKPOINT0['obj_state_dict'])
    
    CHECKPOINT2 = torch.load(p1+'paramsAtt_16.pkl',map_location=device)
    sub_model2 = subject_model(Input_size,num_layers)
    sub_model2.load_state_dict(CHECKPOINT2['sub_state_dict'])
    obj_model2 = object_model(num_rels,Input_size,num_layers)
    obj_model2.load_state_dict(CHECKPOINT2['obj_state_dict'])
    
    #sub_model3 = subject_model(Input_size,num_layers)
    #sub_model3.load_state_dict(CHECKPOINT3['sub_state_dict'])
    #obj_model3 = object_model(num_rels,Input_size,num_layers)
    #obj_model3.load_state_dict(CHECKPOINT3['obj_state_dict'])
    
             
    f=open(output_eva_file,'a+')
    
    #对总test集算pr曲线/10个点
    
    p_all0=[0,0,0,0,0,0,0,0,0,0]
    r_all0=[0,0,0,0,0,0,0,0,0,0]
    f1_all0=[0,0,0,0,0,0,0,0,0,0]
    for i in range(10):
        x=(i+1)*50
        p_all0[i], r_all0[i], f1_all0[i]= evaluate_num(x,sub_model0,obj_model0,tokenizer,test_overall,id2rel)        
        print (i)
        print('Precision: %.4f, Recall: %.4f, F1: %.4f' % (p_all0[i], r_all0[i], f1_all0[i]))              
        f.write('Precision: %.4f, Recall: %.4f, F1: %.4f \n' % (p_all0[i], r_all0[i], f1_all0[i]))
    
    p_all2=[0,0,0,0,0,0,0,0,0,0]
    r_all2=[0,0,0,0,0,0,0,0,0,0]
    f1_all2=[0,0,0,0,0,0,0,0,0,0]
    for i in range(10):
        x=(i+1)*50
        p_all2[i], r_all2[i], f1_all2[i]= evaluate_num(x,sub_model2,obj_model2,tokenizer,test_overall,id2rel)        
        print (i)
        print('Precision: %.4f, Recall: %.4f, F1: %.4f' % (p_all2[i], r_all2[i], f1_all2[i]))              
        f.write('Precision: %.4f, Recall: %.4f, F1: %.4f \n' % (p_all2[i], r_all2[i], f1_all2[i]))   
    f.close()

plt.figure()
plt.plot(r_all0,p_all0, color='r', linewidth=1.0,label="方法 1")
plt.plot(r_all2,p_all2, color='b', linewidth=1.0,label="方法 2")
plt.legend()
plt.xlabel("Recall") 
plt.ylabel("Precision") 
plt.show()        