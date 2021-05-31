# ERE-20210518
实体关系联合抽取模型 / My project on joint exraction of entities and relations

我的毕设项目的主模型结构如下：An overview of the framework of my models. 

![image](https://github.com/SL-Stone/ERE-20210518/blob/a0cb36948b415b8842645845af0502e9482cce9a/model_image/%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%841.png)

1.采用[BERT预训练模型](https://www.aclweb.org/anthology/N19-1423/)bert-base-cased生成词向量

2.模型分为头实体识别模块和关系及其尾实体识别模块，每个模块结构structure：Bi-LSTM + Self-Attenion Mechanism + Bi-LSTM

3.采用Wei等人提出的CASREL序列标注策略，[A Novel Cascade Binary Tagging Framework](https://arxiv.org/abs/1909.03227)


数据集 / Dataset: [NYT](https://github.com/weizhepei/CasRel/tree/master/data/NYT)


文件说明：
以“run”开头的.py文件是程序运行入口，包含模型训练和评估 / The .py programs whose names begin with "run" is the execution entry point, containing model training and evaluation.

以“data_loader”开头的.py文件完成数据预处理

以“model”开头的.py文件定义模型和评估函数

1.runG.py:主模型的GPU版本。关系及其尾实体识别模块训练时，每个句子只选择一个头实体输入。data_loader1.py 和 modelG.py。

2.runC.py:主模型的CPU版本，对应data_loader1.py 和 modelc.py。

3.runG_all.py:主模型的GPU版本。关系及其尾实体识别模块训练时，每个句子的所有头实体都会输入。data_loader_all.py 和 modelG.py。

4.runG_divide.py:主模型的GPU版本。关系及其尾实体识别模块训练时，对训练集按照每个句子所含的三元组个数对句子进行分类，分类输入句子的所有头实体。data_loader_all.py 和 modelG.py。

5.runG_ls.py:删除模型的Self-Atention机制部分，进行消融分析。data_loader1.py 和 modelG_ls.py。

6.runG_att.py:删除模型的第二个Bi-LSTM层，进行消融分析。data_loader1.py 和 modelG_att.py。

本文实体关系联合抽取模型的F1值可达0.758，可以抽取重叠的关系三元组，SEO（SingleEntityOverlap）、EPO（EntityPairOverlap）
