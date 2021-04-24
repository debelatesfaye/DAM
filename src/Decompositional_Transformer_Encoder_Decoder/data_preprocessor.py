# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 13:46:49 2020

@author: Debela Gemechu
"""
import numpy as np
import re
import unicodedata
import torch
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

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

    df=pd.read_csv("aspect3.csv")
    #category,c1,oc1,a1,oa1,c2,oc2,a2,oa2,rel
    cat=df.category.values
    encoded_labels=dumy_cat(cat)
    
    c1=df.c1.values
    c2=df.c2.values
    
    a1=df.a1.values
    a2=df.a2.values
    
    oc1=df.oc1.values
    oc2=df.oc2.values
    
    oa1=df.oa1.values
    oa2=df.oa2.values


    #preprocess c1 and c2
    c1_normalized = list(map(normalizeString, c1))
    c2_normalized = list(map(normalizeString, c2))
    c1_max_trim = []
    c2_max_trim = []
    
    
    #preprocess a1 and a2
    a1_normalized = list(map(normalizeString, a1))
    a2_normalized = list(map(normalizeString, a2))
    a1_max_trim = []
    a2_max_trim = []
    
    #preprocess oc1 and oc2
    oc1_normalized = list(map(normalizeString, oc1))
    oc2_normalized = list(map(normalizeString, oc2))
    oc1_max_trim = []
    oc2_max_trim = []
    
    #preprocess oa1 and oa2
    oa1_normalized = list(map(normalizeString, oa1))
    oa2_normalized = list(map(normalizeString, oa2))
    oa1_max_trim = []
    oa2_max_trim = []
    

    for i in range(len(c1_normalized)):
        tokens1 = c1_normalized[i].split(' ')
        tokens2 = c2_normalized[i].split(' ')
        tokens3 = a1_normalized[i].split(' ')
        tokens4 = a2_normalized[i].split(' ')
        tokens5 = oc1_normalized[i].split(' ')
        tokens6 = oc2_normalized[i].split(' ')
        tokens7 = oa1_normalized[i].split(' ')
        tokens8 = oa2_normalized[i].split(' ')
        if (len(tokens1) <= MAX_LENGTH and len(tokens2) <= MAX_LENGTH  and
           len(tokens3) <= MAX_LENGTH and len(tokens4) <= MAX_LENGTH and 
           len(tokens5) <= MAX_LENGTH and len(tokens6) <= MAX_LENGTH and 
           len(tokens7) <= MAX_LENGTH and len(tokens8) <= MAX_LENGTH):
            c1_max_trim.append(c1_normalized[i])
            c2_max_trim.append(c2_normalized[i])
            a1_max_trim.append(a1_normalized[i])
            a2_max_trim.append(a2_normalized[i])
            oc1_max_trim.append(oc1_normalized[i])
            oc2_max_trim.append(oc2_normalized[i])
            oa1_max_trim.append(oa1_normalized[i])
            oa2_max_trim.append(oa2_normalized[i])
            
    dic = Dictionary()

    return dic, c1_max_trim, c2_max_trim,a1_max_trim, a2_max_trim,oc1_max_trim, oc2_max_trim,oa1_max_trim, oa2_max_trim,encoded_labels

#takes in a sentence and dictionary, and tokenizes based on dictionary
def tokenize(sentence, dictionary, MAX_LENGTH=60):
    split_sentence = [word for word in sentence.split(' ')]
    token = [SOS_TOKEN]
    token += [dictionary.word2index[word] for word in sentence.split(' ')]
    token.append(EOS_TOKEN)
    token += [PAD_TOKEN]*(MAX_LENGTH - len(split_sentence))
    return token
def dumy_cat(labels):
    unique_lebels=set(labels)
    index_to_label=dict({label_id:label for label_id,label in enumerate(unique_lebels)})
    
    index_to_cat = dict(enumerate(sorted(set(labels))))
    cat_to_index = {index_to_cat[i]: i for i in index_to_cat.keys()}
    
   
    encoded_labels = np.array([cat_to_index[x] for x in labels])
    encoded_labels = torch.tensor(encoded_labels).type(torch.LongTensor)
    
    print(encoded_labels)
    
    return encoded_labels
#create dataloader from a batch size and the two language lists
def load_batches(c1, c2, a1,a2,oc1,oc2,oa1,oa2,encoded_labels,batch_size, device):
    
    print(encoded_labels)
    data_loader = []
    for i in range(0, len(c1), batch_size):
        seq_length = min(len(c1) - batch_size, batch_size)
        
        c1_batch = c1[i:i+seq_length][:]
        c2_batch = c2[i:i+seq_length][:]
        
        a1_batch = a1[i:i+seq_length][:]
        a2_batch = a2[i:i+seq_length][:]
                
        oc1_batch = oc1[i:i+seq_length][:]
        oc2_batch = oc2[i:i+seq_length][:]
                
        oa1_batch = oa1[i:i+seq_length][:]
        oa2_batch = oa2[i:i+seq_length][:]
        encoded_labels_batch=encoded_labels[i:i+seq_length][:]
        #convrt to tenors
        
        c1_tensor = torch.LongTensor(c1_batch).to(device)
        c2_tensor = torch.LongTensor(c2_batch).to(device)
        a1_tensor = torch.LongTensor(a1_batch).to(device)
        a2_tensor = torch.LongTensor(a2_batch).to(device)   
        oc1_tensor = torch.LongTensor(oc1_batch).to(device)
        oc2_tensor = torch.LongTensor(oc2_batch).to(device)
        oa1_tensor = torch.LongTensor(oa1_batch).to(device)
        oa2_tensor = torch.LongTensor(oa2_batch).to(device)
        encoded_labels_tensor = torch.LongTensor(encoded_labels_batch).to(device)
        data_loader.append([c1_tensor, c2_tensor,a1_tensor, a2_tensor,oc1_tensor, oc2_tensor,oa1_tensor, oa2_tensor,encoded_labels_tensor])
    return data_loader


PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

#from pytorch's documentation website on NLP
class Dictionary:
    def __init__(self):
        #self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
        self.n_count = 3

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_count
            self.word2count[word] = 1
            self.index2word[self.n_count] = word 
            self.n_count += 1
        else:
            self.word2count[word] += 1