# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 23:28:10 2019

@author: monst
"""

import os
import sys
import argparse
import time
import random
import utils
import pdb
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim


from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from Loader4Text import Loader4Text
from Loader4Text import PaddedTensorDataset

from model_LSTM import LSTMClassifier


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='data/',
											help='data_directory')
	parser.add_argument('--hidden_dim', type=int, default=300,
											help='LSTM hidden dimensions')
	parser.add_argument('--batch_size', type=int, default=20,
											help='size for each minibatch')
	parser.add_argument('--num_epochs', type=int, default=30,
											help='maximum number of epochs')
	parser.add_argument('--embded_dim', type=int, default=300,
											help=' embedding dimensions')
	parser.add_argument('--learning_rate', type=float, default=0.001,
											help='initial learning rate')
	parser.add_argument('--weight_decay', type=float, default=0,
											help='weight_decay rate')
	parser.add_argument('--seed', type=int, default=123,
											help='seed for random initialisation')
	args = parser.parse_args()
	train(args)
    
def apply(model, criterion, batch, targets, lengths):
    pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
    loss = criterion(pred, torch.autograd.Variable(targets))
    return pred, loss

def train_model(model, optimizer, train, dev, x_to_ix, y_to_ix, batch_size, max_epochs,args):
    criterion = nn.CrossEntropyLoss(size_average=None)
    loss_list=[]
    acc_list=[]
    val_loss_list=[]
    val_acc_list=[]
    model.train()
    for epoch in range(max_epochs):
        print('Epoch:', epoch)
        y_true = list()
        y_pred = list()
        total_loss = 0
        for batch, targets, lengths, raw_data in utils.create_dataset(train, x_to_ix, y_to_ix, batch_size=batch_size):
            batch, targets, lengths = utils.sort_batch(batch, targets, lengths)
            model.zero_grad()
            pred, loss = apply(model, criterion, batch, targets, lengths)
            loss.backward()
            optimizer.step()
            
            pred_idx = torch.max(pred, 1)[1]
            y_true += list(targets.int())
            y_pred += list(pred_idx.data.int())
            total_loss += loss
        acc = accuracy_score(y_true, y_pred)
        loss_list.append(total_loss.data.float()/len(train))
        acc_list.append(100*acc)
        val_loss, val_acc = evaluate_validation_set(model, dev, x_to_ix, y_to_ix, criterion)
        print("Train loss: {} - acc: {} \nValidation loss: {} - acc: {}".format(total_loss.data.float()/len(train), acc,
                                                                                val_loss, val_acc))
        val_loss_list.append(val_loss)
        val_acc_list.append(100*val_acc)
        fig, axs = plt.subplots(2, 2, figsize=(20,10))
        fig.suptitle("Hidden_dim: {} -embded_dim: {} - Batch Size: {} - Num Epochs: {} - learning_rate : {} - weight_decay : {}".format(args.hidden_dim,args.embded_dim,args.batch_size,args.num_epochs,args.learning_rate,args.weight_decay), fontsize=16)
        axs[0][0].plot(loss_list)
        axs[0][0].set_title('Training Loss')


        axs[0][1].plot(acc_list)
        axs[0][1].set_title('Training Accuracy')

        axs[1][0].plot(val_loss_list)
        axs[1][0].set_title('Validation Loss')

        axs[1][1].plot(val_acc_list)
        axs[1][1].set_title('Validation Accuracy')
        plt.show()
 
    axs[0][0].annotate(str('%.3f'%(float(loss_list[-1].data))),xy=(len(loss_list)/2,loss_list[-1]))
    axs[0][1].annotate(str('%.3f'%(float(acc_list[-1] ))),xy=(len(acc_list)/2,acc_list[-1]))
    axs[1][0].annotate(str('%.3f'%(float(val_loss_list[-1].data))),xy=(len(val_loss_list)/2,val_loss_list[-1]))
    axs[1][1].annotate(str('%.3f'%(float(val_acc_list[-1]))),xy=(len(val_acc_list)/2,val_acc_list[-1]))
    plt.show()
    fig.savefig('result' + str(args.hidden_dim) + str(args.embded_dim) + str(args.batch_size) + str(args.num_epochs) + str(args.learning_rate) + str(args.weight_decay) + '.png')
        
        
    return model
    
def evaluate_validation_set(model, devset, x_to_ix, y_to_ix, criterion):
    y_true = list()
    y_pred = list()
    total_loss = 0
    model.eval()
    for batch, targets, lengths, raw_data in utils.create_dataset(devset, x_to_ix, y_to_ix, batch_size=1):
        batch, targets, lengths = utils.sort_batch(batch, targets, lengths)
        pred, loss = apply(model, criterion, batch, targets, lengths)
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())
        total_loss += loss
    acc = accuracy_score(y_true, y_pred)
    return total_loss.data.float()/len(devset), acc




def train(args):
    random.seed(args.seed)
    data_loader = Loader4Text(args.data_dir)
    train_data = data_loader.train_data
    dev_data = data_loader.dev_data
    word_vocab=data_loader.word2id
    classes_vocab=data_loader.class2id
    word_vocab_size=len(data_loader.vocab)
    
    print('Training samples:', len(train_data))
    print('Validation samples:', len(dev_data))
    
    
    model = LSTMClassifier(word_vocab_size, args.embded_dim, args.hidden_dim, len(classes_vocab))
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    model = train_model(model, optimizer, train_data, dev_data, word_vocab, classes_vocab, args.batch_size, args.num_epochs,args)
    
if __name__ == '__main__':
	main()   