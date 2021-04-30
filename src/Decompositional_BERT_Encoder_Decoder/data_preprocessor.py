# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:45:05 2021

@author: Debela
"""
import re
import unicodedata
import torch
import numpy as np

from transformers import BertTokenizer





def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

#Load proposition pairs

import pandas as pd


def load_files(data_dir, MAX_LENGTH=60):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    df=pd.read_csv("aspect3.csv")
    #category,c1,oc1,a1,oa1,c2,oc2,a2,oa2,rel
    cat=df.category.values
    encoded_labels=dumy_cat(cat)
    
    c1=df.c1.values
    c2=df.c2.values
    c1_c2=c1+" [SEP] "+c2
    
    a1=df.a1.values
    a2=df.a2.values    
    a1_a2=a1+" [SEP] "+a2
    
    oc1=df.oc1.values
    oc2=df.oc2.values
    oc1_oc2=oc1+" [SEP] "+oc2
    
    oa1=df.oa1.values
    oa2=df.oa2.values
    oa1_oa2=oa1+" [SEP] "+oa2


    c1_c2_inputs = tokenizer.batch_encode_plus(c1_c2, return_tensors="pt",add_special_tokens=True, max_length=MAX_LENGTH,
                      pad_to_max_length=True)
    a1_a2_inputs = tokenizer.batch_encode_plus(a1_a2, return_tensors="pt",add_special_tokens=True, max_length=MAX_LENGTH,
                      pad_to_max_length=True)
    oc1_oc2_inputs = tokenizer.batch_encode_plus(oc1_oc2, return_tensors="pt",add_special_tokens=True, max_length=MAX_LENGTH,
                      pad_to_max_length=True)
    oa1_oa2_inputs = tokenizer.batch_encode_plus(oa1_oa2, return_tensors="pt",add_special_tokens=True, max_length=MAX_LENGTH,
                      pad_to_max_length=True)
    return c1_c2_inputs,a1_a2_inputs,oc1_oc2_inputs,oa1_oa2_inputs,encoded_labels

def dumy_cat(labels):
    unique_lebels=set(labels)
    index_to_label=dict({label_id:label for label_id,label in enumerate(unique_lebels)})
    
    index_to_cat = dict(enumerate(sorted(set(labels))))
    cat_to_index = {index_to_cat[i]: i for i in index_to_cat.keys()}    
    encoded_labels = np.array([cat_to_index[x] for x in labels])
    encoded_labels = torch.tensor(encoded_labels).type(torch.LongTensor)

    
    return encoded_labels
#create dataloader from a batch size and the two language lists
def load_batches(c1_c2, a1_a2,oc1_oc2,oa1_oa2,encoded_labels,batch_size, device):
    
    #print(c1_c2)
    data_loader = []
    encoded_labels_tensor = torch.LongTensor(encoded_labels).to(device)  
    data_loader.append([c1_c2, a1_a2,oc1_oc2, oa1_oa2,encoded_labels_tensor])
    return data_loader
    
