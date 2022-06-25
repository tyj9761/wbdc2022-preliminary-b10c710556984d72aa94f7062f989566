import logging
import os
import time
import torch
import json
import gc
import pandas as pd
from category_id_map import category_id_to_lv2id
from sklearn.model_selection import StratifiedKFold


def main():
    X, y = [i for i in range(100000)], []
    with open('data/annotations/labeled.json', 'r', encoding='utf8') as f:
        anns = json.load(f)
    for item in anns:
        y.append(category_id_to_lv2id(item['category_id']))
    train_index, val_index = None, None

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2022)
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        a=[]
        b=[]
        for i in train_index:
            a.append(anns[i])
        # traindata, valdata = anns[train_index], anns[val_index]
        filename='train_'+str(fold)+'.json'
        with open(filename,'w') as f:
            json.dump(a,f)
        for j in val_index:
            b.append(anns[j])
        # traindata, valdata = anns[train_index], anns[val_index]
        filename1='val_'+str(fold)+'.json'
        with open(filename1,'w') as f1:
            json.dump(b,f1)


if __name__ == '__main__':
    main()