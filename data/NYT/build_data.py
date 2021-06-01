#! -*- coding:utf-8 -*-


import json
from tqdm import tqdm
import codecs

rel_set = set()


train_data = []
dev_data = []
test_data = []

with open('new_train.json') as f:
    for l in tqdm(f):
        a = json.loads(l)
        if not a['relationMentions']:
            continue
        line = {
                'text': a['sentText'].lstrip('\"').strip('\r\n').rstrip('\"'),
                'triple_list': [(i['em1Text'], i['label'], i['em2Text']) for i in a['relationMentions'] if i['label'] != 'None']
               }
        if not line['triple_list']:
            continue
        train_data.append(line)
        for rm in a['relationMentions']:
            if rm['label'] != 'None':
                rel_set.add(rm['label'])


with open('new_valid.json') as f:  #原文档为dev.json，根据含义此处为验证集
    for l in tqdm(f):
        a = json.loads(l)
        if not a['relationMentions']:
            continue
        line = {
                'text': a['sentText'].lstrip('\"').strip('\r\n').rstrip('\"'),
                'triple_list': [(i['em1Text'], i['label'], i['em2Text']) for i in a['relationMentions'] if i['label'] != 'None']
               }
        if not line['triple_list']:
            continue
        dev_data.append(line)
        for rm in a['relationMentions']:
            if rm['label'] != 'None':
                rel_set.add(rm['label'])

cnt = 0
with open('test_split_by_type/new_test.json') as f:
    for l in tqdm(f):
        a = json.loads(l)
        if not a['relationMentions']:
            continue
        line = {
                'text': a['sentText'].lstrip('\"').strip('\r\n').rstrip('\"'),
                'triple_list': [(i['em1Text'], i['label'], i['em2Text']) for i in a['relationMentions'] if i['label'] != 'None']
               }
        if not line['triple_list']:
            continue
        cnt += len(line['triple_list'])
        test_data.append(line)

print(f'test triples:{cnt}')


id2predicate = {i:j for i,j in enumerate(sorted(rel_set))}
predicate2id = {j:i for i,j in id2predicate.items()}


with codecs.open('rel2id.json', 'w', encoding='utf-8') as f:
    json.dump([id2predicate, predicate2id], f, indent=4, ensure_ascii=False)


with codecs.open('train_triples.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)


with codecs.open('dev_triples.json', 'w', encoding='utf-8') as f:
    json.dump(dev_data, f, indent=4, ensure_ascii=False)


with codecs.open('test_triples.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)
