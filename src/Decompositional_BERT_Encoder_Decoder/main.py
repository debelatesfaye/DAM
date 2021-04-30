# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 13:41:49 2020

@author: Debela
"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import time
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
from random import shuffle
import pickle



import torch 
import torch.nn as nn

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import time
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
from random import shuffle
import pickle



from model import *
from data_preprocessor import *

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
class Train():

    def initialize_weights(self, model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.xavier_uniform_(model.weight.data)



    def __init__(self, data_directory, MAX_LENGTH, OUTPUT_SIZE,MAX_FILE_SIZE, batch_size, modeltype='bert-base-uncased',
                 lr=0.0005, hidden_size=256, encoder_dropout=0.1, 
                 device='cuda'):
        
        self.MAX_LENGTH = MAX_LENGTH
        self.OUTPUT_SIZE=OUTPUT_SIZE
        self.MAX_FILE_SIZE = MAX_FILE_SIZE
        self.device = device

        self.c1_c2,self.a1_a2,self.oc1_oc2,self.oa1_oa2,self.encoded_labels = load_files(data_directory, self.MAX_LENGTH)

        
     
        self.batch_size = batch_size

        self.data_loader = load_batches(self.c1_c2, 
                                        self.a1_a2,
                                        self.oc1_oc2,
                                        self.oa1_oa2,
                                        self.encoded_labels,
                                        self.batch_size, self.device)


      
        self.multi_dam_bert=Decompositional_Multi_Bert(self.MAX_LENGTH ,self.OUTPUT_SIZE, hidden_size, encoder_dropout, 
                                                       modeltype,self.device)
        
        self.multi_dam_bert.to(self.device)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        self.optimizer = optim.Adam(self.multi_dam_bert.parameters(), lr=lr)


    def train(self, epochs, saved_model_directory):
        start_time = time.time()

        for epoch in range(epochs):
            #shuffle batches to prevent overfitting
            #shuffle(self.data_loader)
            print("hereeeeeeeeee",self.c1_c2)

            start_time = time.time()
            train_loss = 0
            print(self.c1_c2['input_ids'])
            for i in range(0, len(self.c1_c2['input_ids']), self.batch_size):
                inp_c1_inp_c2={}
                inp_a1_inp_a2={}
                inp_oc1_inp_oc2={}
                inp_oa1_inp_oa2={}
                seq_length = min(len(self.c1_c2['input_ids']) - self.batch_size, self.batch_size)
                inp_c1_inp_c2['input_ids'] = self.c1_c2['input_ids'][i:i+seq_length][:]
                inp_c1_inp_c2['token_type_ids'] = self.c1_c2['token_type_ids'][i:i+seq_length][:]
                inp_c1_inp_c2['attention_mask'] = self.c1_c2['attention_mask'][i:i+seq_length][:]
                
                inp_a1_inp_a2['input_ids'] = self.c1_c2['input_ids'][i:i+seq_length][:]
                inp_a1_inp_a2['token_type_ids'] = self.a1_a2['token_type_ids'][i:i+seq_length][:]
                inp_a1_inp_a2['attention_mask'] = self.a1_a2['attention_mask'][i:i+seq_length][:]
                
                inp_oc1_inp_oc2['input_ids'] = self.oc1_oc2['input_ids'][i:i+seq_length][:]
                inp_oc1_inp_oc2['token_type_ids'] = self.oc1_oc2['token_type_ids'][i:i+seq_length][:]
                inp_oc1_inp_oc2['attention_mask'] = self.oc1_oc2['attention_mask'][i:i+seq_length][:]
                
                
                inp_oa1_inp_oa2['input_ids'] = self.oa1_oa2['input_ids'][i:i+seq_length][:]
                inp_oa1_inp_oa2['token_type_ids'] = self.oa1_oa2['token_type_ids'][i:i+seq_length][:]
                inp_oa1_inp_oa2['attention_mask'] = self.oa1_oa2['attention_mask'][i:i+seq_length][:]
                
                batch_labels = self.encoded_labels[i:i+seq_length][:]
                

                
                print("inp_c1_inp_c2........................",inp_c1_inp_c2)
                #zero gradient
                self.optimizer.zero_grad()

                #pass through transformer

                arg_cat= self.multi_dam_bert(inp_c1_inp_c2,inp_a1_inp_a2,inp_oc1_inp_oc2,inp_oa1_inp_oa2)
                


                #loss

                #print(arg_cat, batch_labels)
                loss = self.loss_func(arg_cat, batch_labels)
                
                
                
                print(loss)

                #backprop
                loss.backward()
                nn.utils.clip_grad_norm_(self.multi_dam_bert.parameters(), 1)
                self.optimizer.step()

                train_loss += loss.item()
                
            train_loss /= len(self.data_loader)

            end_time = int(time.time() - start_time)
            torch.save(self.multi_dam_bert.state_dict(), saved_model_directory +
            '2' + '/transformer_model_{}.pt'.format(epoch))

            print('Epoch: {},   Time: {}s,  Estimated {} seconds remaining.'.format(epoch, end_time, (epochs-epoch)*end_time))
            print('\tTraining Loss: {:.4f}\n'.format(train_loss))
        print('Training finished!')

def main():
  
    data_directory = "data"
    reverse = 1
    MAX_LENGTH = 60
    OUTPUT_SIZE=3
    MAX_FILE_SIZE = 100000
    batch_size = 2
    lr = 0.0005
    hidden_size = 256
    encoder_layers = 3
    decoder_layers = 3
    encoder_heads = 8
    decoder_heads = 8
    encoder_ff_size = 512
    decoder_ff_size = 512
    encoder_dropout = 0.1
    decoder_dropout = 0.1
    modeltype='bert-base-uncased'
    #device = 'cuda'
    epochs = 2
    saved_model_directory = 'saved_models/'
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    transformer = Train(data_directory, MAX_LENGTH,OUTPUT_SIZE, MAX_FILE_SIZE, batch_size, lr, hidden_size, encoder_dropout, modeltype,device)
    transformer.train(epochs, saved_model_directory)


if __name__ == "__main__":
    main()