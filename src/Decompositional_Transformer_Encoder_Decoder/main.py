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

#from models import *
#from utilities import *

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
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



    def __init__(self, data_directory, reverse, MAX_LENGTH, MAX_FILE_SIZE, batch_size, lr=0.0005, hidden_size=256, encoder_layers=3, decoder_layers=3,
                 encoder_heads=8, decoder_heads=8, encoder_ff_size=512, decoder_ff_size=512, encoder_dropout=0.1, decoder_dropout=0.1, device='cuda'):
        
        self.MAX_LENGTH = MAX_LENGTH
        self.MAX_FILE_SIZE = MAX_FILE_SIZE
        self.device = device

        self.input_lang_dic, self.c1, self.c2,self.a1, self.a2,self.oc1, self.oc2,self.oa1, self.oa2,encoded_labels = load_files(data_directory, self.MAX_LENGTH)
        
        #dic, c1_max_trim, c2_max_trim,a1_max_trim, a2_max_trim,oc1_max_trim, oc2_max_trim,oa1_max_trim, oa2_max_trim
        
        for index,sentence in enumerate(self.c1):
            self.input_lang_dic.add_sentence(sentence)
            self.input_lang_dic.add_sentence(self.c2[index])      
            self.input_lang_dic.add_sentence(self.a1[index])            
            self.input_lang_dic.add_sentence(self.a2[index])            
            self.input_lang_dic.add_sentence(self.oc1[index])
            self.input_lang_dic.add_sentence(self.oc2[index])
            self.input_lang_dic.add_sentence(self.oa1[index])
            self.input_lang_dic.add_sentence(self.oa2[index])            




        self.tokenized_c1 = [tokenize(sentence, self.input_lang_dic, self.MAX_LENGTH) for sentence in self.c1]
        self.tokenized_c2 = [tokenize(sentence, self.input_lang_dic, self.MAX_LENGTH) for sentence in self.c2]
        self.tokenized_a1 = [tokenize(sentence, self.input_lang_dic, self.MAX_LENGTH) for sentence in self.a1]
        self.tokenized_a2 = [tokenize(sentence, self.input_lang_dic, self.MAX_LENGTH) for sentence in self.a2]
        self.tokenized_oc1 = [tokenize(sentence, self.input_lang_dic, self.MAX_LENGTH) for sentence in self.oc1]
        self.tokenized_oc2 = [tokenize(sentence, self.input_lang_dic, self.MAX_LENGTH) for sentence in self.oc2]
        self.tokenized_oa1 = [tokenize(sentence, self.input_lang_dic, self.MAX_LENGTH) for sentence in self.oa1]
        self.tokenized_oa2 = [tokenize(sentence, self.input_lang_dic, self.MAX_LENGTH) for sentence in self.oa2]
        

        self.batch_size = batch_size

        self.data_loader = load_batches(self.tokenized_c1, self.tokenized_c2, 
                                        self.tokenized_a1, self.tokenized_a2,
                                        self.tokenized_oc1, self.tokenized_oc2,
                                        self.tokenized_oa1, self.tokenized_oa2,
                                        encoded_labels,
                                        self.batch_size, self.device)

        input_size = self.input_lang_dic.n_count
        output_size = self.input_lang_dic.n_count
         
        self.multi_dam_transformer=Decompositional_Transformer(input_size,output_size, hidden_size, decoder_layers,encoder_layers, encoder_heads,decoder_heads,
                 encoder_ff_size, decoder_ff_size,encoder_dropout, self.device)
        
        self.multi_dam_transformer.to(self.device)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        self.optimizer = optim.Adam(self.multi_dam_transformer.parameters(), lr=lr)


    def train(self, epochs, saved_model_directory):
        start_time = time.time()

        for epoch in range(epochs):



            start_time = time.time()
            train_loss = 0

            for inp_c1, inp_c2,inp_a1, inp_a2,inp_oc1, inp_oc2,inp_oa1, inp_oa2,encoded_labels in self.data_loader:
                #zero gradient
                self.optimizer.zero_grad()

                output,arg_cat= self.multi_dam_transformer(inp_c1, inp_c2,inp_a1, inp_a2,inp_oc1, inp_oc2,inp_oa1, inp_oa2)
                
                output_dim = output.shape[-1]

                #flatten and omit SOS from target
                output = output.contiguous().view(-1, output_dim)
                c_target = inp_c2[:,1:].contiguous().view(-1)

                #loss
                #loss = self.loss_func(output, c_target)
                print(arg_cat, encoded_labels)
                loss = self.loss_func(arg_cat, encoded_labels)
                
                
                
                print(loss)

                #backprop
                loss.backward()
                nn.utils.clip_grad_norm_(self.multi_dam_transformer.parameters(), 1)
                self.optimizer.step()

                train_loss += loss.item()
                
            train_loss /= len(self.data_loader)

            end_time = int(time.time() - start_time)
            torch.save(self.multi_dam_transformer.state_dict(), saved_model_directory +
            '2' + '/transformer_model_{}.pt'.format(epoch))

            print('Epoch: {},   Time: {}s,  Estimated {} seconds remaining.'.format(epoch, end_time, (epochs-epoch)*end_time))
            print('\tTraining Loss: {:.4f}\n'.format(train_loss))
        print('Training finished!')

def main():


    data_directory = "data"
    reverse = 1
    MAX_LENGTH = 60
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
    #device = 'cuda'
    epochs = 2
    saved_model_directory = 'saved_models/'
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer = Train(data_directory, reverse, MAX_LENGTH, MAX_FILE_SIZE, batch_size, lr, hidden_size, encoder_layers, decoder_layers, 
                            encoder_heads, decoder_heads, encoder_ff_size, decoder_ff_size, encoder_dropout, decoder_dropout, device)
    transformer.train(epochs, saved_model_directory)


if __name__ == "__main__":
    main()