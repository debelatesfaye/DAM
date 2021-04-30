# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 13:37:21 2020

@author: Debela Gemechu
"""
import torch 
import torch.nn as nn





from transformers import BertForSequenceClassification


class Decompositional_Multi_Bert(nn.Module):
           
    def __init__(self, input_size,output_size, hidden_size, dropout, modeltype,device):
        super().__init__()
        self.device=device
        self.modeltype="bert-base-uncased"     

         
            
        self.bert_1=BertForSequenceClassification.from_pretrained(self.modeltype,num_labels=output_size).to(self.device)
        self.bert_2=BertForSequenceClassification.from_pretrained(self.modeltype,num_labels=output_size).to(self.device)
        self.bert_3=BertForSequenceClassification.from_pretrained(self.modeltype,num_labels=output_size).to(self.device)
        self.bert_4=BertForSequenceClassification.from_pretrained(self.modeltype,num_labels=output_size).to(self.device)
        
        self.output=torch.nn.Linear(output_size*4,output_size)
        self.softmax=torch.nn.Softmax(dim=1)
        
    def forward(self, c1_c2,a1_a2,oc1_oc2,oa1_oa2):      

        output1 = self.bert_1(**c1_c2)[0]
        output2 = self.bert_2(**a1_a2)[0]
        output3 = self.bert_3(**oc1_oc2)[0]
        output4 = self.bert_4(**oa1_oa2)[0]
      
        concatenated_outputs = torch.cat((output1, output2, output3,output4), dim=1)        
        concatenated_outputs=concatenated_outputs.to(self.device)
        out=self.output(concatenated_outputs)        


        return out
   
