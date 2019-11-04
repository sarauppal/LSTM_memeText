# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 23:10:27 2019

@author: monst
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class LSTMClassifier(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
        super(LSTMClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size+1 , embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=3,bidirectional=True)
        self.hidden2out = nn.Linear(hidden_dim, output_size)
#		self.softmax = nn.LogSoftmax()

        self.dropout_layer = nn.Dropout(p=0)
        
    def init_hidden(self, batch_size):
        return(autograd.Variable(torch.randn(6, batch_size, self.hidden_dim)),
						autograd.Variable(torch.randn(6, batch_size, self.hidden_dim)))
        
    def forward(self, batch, lengths):
		
        self.hidden = self.init_hidden(batch.size(-1))

        embeds = self.embedding(batch)
        packed_input = pack_padded_sequence(embeds, lengths)
        outputs, (ht, ct) = self.lstm(packed_input, self.hidden)
#        Output[-1] same as ht[-1] can use either to get the last output, 
#        we dont need intermediate output for a classifier
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
		

        return output