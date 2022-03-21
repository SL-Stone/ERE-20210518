# 实体关系联合抽取模型

#### My project on joint exraction of entities and relations



#### Model Structure

![image](https://github.com/SL-Stone/ERE-20210518/blob/c478b4623c9cbd327031a73fd1827e1a458dd2a3/model_image/%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%840.png)

1. use the bert-base-cased model for generating word embeddings. [BERT: Pre-training of Deep Bidirectional Transformers for
   Language Understanding ](https://www.aclweb.org/anthology/N19-1423/)

2. This model is composed of two parts: a subject recognition module and a corresponding object and relation recognition module. The structure of both: BiLSTM + Self-Attenion Mechanism + BiLSTM.
3. Apply the CasRel Sequence Labeling strategy proposed by Wei, et al. [A Novel Cascade Binary Tagging Framework](https://arxiv.org/abs/1909.03227) Thanks them a lot for releasing code based on tensorflow .



#### Dataset

[NYT](https://github.com/weizhepei/CasRel/tree/master/data/NYT)



#### Requirements

Python: `python 3.8.5` 

Pytorch: `torch 1.8.0` 

Transformers: `transformers 4.4.2`



#### Documentation

Program execution entry: `run***.py`, containing model training and evaluation.

Data Preprocessing: `data_loader***.py`.

Model Structure: `model***.py`.

1. `**runG.py**`: GPU version. When training the relation-specific object tagging module, we only choose one subject in a sentence as an input instance. With data_loader1.py, modelG.py.

2. 关系及其尾实体识别模块训练时，每个句子只选择一个头实体输入.

3. `runG_all.py`: GPU version. When training the relation-specific object tagging module, we choose all subjects in a sentence as an input instance. With data_loader_all.py, modelG.py.

   关系及其尾实体识别模块训练时，每个句子的所有头实体都会输入.

4. `runG_divide.py`: GPU version.When training the relation-specific object tagging module, we first categorize all sentences by the number of triplets they contain, then select one certain length of sentences pair with all subjects at a time as an input instance. With data_loader_all.py, modelG.py.

   关系及其尾实体识别模块训练时，对训练集按照每个句子所含的三元组个数对句子进行分类，分类输入句子的所有头实体.

5. `runG_ls.py`: Remove the Self-Atention mechanism, for ablation analysis. With data_loader1.py, modelG_ls.py。

6. `runG_att.py`: Remove the second Bi-LSTM layer, for ablation analysis. With data_loader1.py, modelG_att.py.

7. `**runC.py**`: CPU version. With data_loader1.py, modelc.py.



#### Performance

F1: 0.758.

This model can extract overlapping relation triplets, including SEO(SingleEntityOverlap), EPO(EntityPairOverlap).
