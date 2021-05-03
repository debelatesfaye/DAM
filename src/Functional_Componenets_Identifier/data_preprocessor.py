# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:45:05 2021

@author: Debela Gemechu
"""
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

torch.__version__

MAX_LEN = 75
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
def data_procesor(data_path):
    # return the training, and validation dataloder objects using BERT input format
    

    sentences,labels,tag2idx=file_loader(data_path)
    tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(sentences, labels)
    ]

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", value=0.0,
                              truncating="post", padding="post")

    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                         maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                         dtype="long", truncating="post")

    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                                random_state=2018, test_size=0.1)
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=2018, test_size=0.1)

    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)

    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)
    
    return train_dataloader,valid_dataloader,tag2idx
    

def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels
	
	
def file_loader(data_path):
    #return sentences, labels and label_to_index dictionary
    # it takes conll file with (token, label) tuple format with double new lines 
    # at the end of a proposition
    s_id=0
    sentences=[]
    labels=[]
    sent=[]
    leb=[]
    with open(data_path, 'r') as conll:
        for toke_lebl in (conll.readlines()):

            if toke_lebl != '\n':        

                sent.append(toke_lebl.split(" ")[0])
                leb.append(toke_lebl.split(" ")[1])
            else:

                s_id+=1
                sentences.append(sent)
                labels.append(leb)
                sent=[]
                leb=[]
    leb_list=[]
    for leb in labels:
        for ll in leb:
            leb_list.append(ll)

    tag_values = list(set(leb_list))
    tag_values.append("PAD")
    tag2idx = {t: i for i, t in enumerate(tag_values)}
    return sentences,labels,tag2idx
    