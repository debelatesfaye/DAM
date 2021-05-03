# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 13:37:21 2020

@author: Debela Gemechu
"""

   
import transformers
from transformers import BertForTokenClassification



import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformers.__version__

def BERT_NER(model_type,tag2idx):

    model = BertForTokenClassification.from_pretrained(
        model_type,
        num_labels=len(tag2idx),
        output_attentions = False,
        output_hidden_states = False
    )

    model.to(device)
    return model
