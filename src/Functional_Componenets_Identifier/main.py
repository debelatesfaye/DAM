# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 13:41:49 2020

@author: Debela Gemechu
"""
import transformers
from transformers import get_linear_schedule_with_warmup,AdamW

from seqeval.metrics import accuracy_score
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model import *
from data_preprocessor import *


transformers.__version__
def main():
    epochs = 10
    max_grad_norm = 1.0    
    lr=3e-5
    eps=1e-8
    
    model_type="bert-base-uncased"
    data_path="unique_propositions8.conll"
    train_dataloader,valid_dataloader,tag2idx=data_procesor(data_path)
    model=BERT_NER(model_type,tag2idx)
    
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=eps
    )



    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    ## Store the average loss after each epoch so we can plot them.
    loss_values, validation_loss_values = [], []




    for _ in range(epochs):
        model.train()
        total_loss = 0

        # Training loop
        for step, batch in enumerate(train_dataloader):

            input_batch = tuple(data_tuple.to(device) for data_tuple in batch)
            batch_input_ids, batch_input_mask, batch_labels = input_batch
            model.zero_grad()
            outputs = model(batch_input_ids, token_type_ids=None,
                            attention_mask=batch_input_mask, labels=batch_labels)

            loss = outputs[0]
            loss.backward()
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        loss_values.append(avg_train_loss)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for batch in valid_dataloader:
            input_batch = tuple(data_tuple.to(device) for data_tuple in batch)
            batch_input_ids, batch_input_mask, batch_labels = input_batch

            with torch.no_grad():
                outputs = model(batch_input_ids, token_type_ids=None,
                                attention_mask=batch_input_mask, labels=batch_labels)
            logits = outputs[1].detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()

            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(valid_dataloader)
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))
        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                     for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
        valid_tags = [tag_values[l_i] for l in true_labels
                                      for l_i in l if tag_values[l_i] != "PAD"]
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))

if __name__ == "__main__":
    main()