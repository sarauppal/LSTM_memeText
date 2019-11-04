# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 23:05:41 2019

@author: monst
"""

import os
import sys
import math
import random
import argparse
import operator
import pdb
import numpy as np
import pandas as pd
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
set(stopwords.words('english'))
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from collections import defaultdict
from collections import Counter
from torch.autograd import Variable


# data dir C:\CPSC532P\LSTM_memeText\data

class Loader4Text:
    def __init__(self,datadir,clean_data=False):
        self.clean=clean_data
        self.texts_col=3
        self.labels_col=8
        self.cleaned_data=self.load_data(datadir,self.clean)
        self.vocab,self.classes=self.build_vocab(self.cleaned_data)
        self.word2id=self.get_index(self.vocab,'UNK')
        self.class2id=self.get_index(self.classes)
        self.train_data,self.dev_data=self.split_data(self.cleaned_data)
       
        
        
    def load_data(self, datadir,clean=False):
        if (clean):
            texts = []
            labels = []
            clean_texts = []
            clean_labels = []
            filenames=os.listdir(datadir)
           
            for f in filenames:
                if not f.endswith('csv'):
                    continue
                df=pd.read_csv(datadir+f)
                texts[len(texts):]=np.array(df.iloc[:,self.texts_col])
                labels[len(labels):]=np.array(df.iloc[:,self.labels_col])
                texts_n_labels=list(zip(texts,labels))
                
                for text,label in texts_n_labels:
                    
                    if not (isinstance(text, str)):
                        continue
                    if (len(text)==0):
#                        print("found him")
                        print (len(text))
                    text=self.preprocess(text)
                    label=self.preprocess(label)
                    if (len(text)==0):
                        continue
                    clean_texts.append(text)
                    clean_labels.append(label)
        
                clean_texts_n_labels=np.array(list(zip(clean_texts,clean_labels)))
                pd.DataFrame(clean_texts_n_labels).to_csv(datadir+'clean/'+f, index=False,header=False)
                clean_texts_n_labels=np.array(pd.read_csv(datadir+'clean/'+f))
                
        else:
            cleandir=datadir+'clean/'
            filenames=os.listdir(cleandir)
            for f in filenames:
                clean_texts_n_labels=np.array(pd.read_csv(cleandir+f))
             
        
        return clean_texts_n_labels
    
    def build_vocab(self,data):
        words_set=set()
        classes_set=set()
        for line,label in data:
#            print(line)
            for word in eval(line):
                words_set.add(word)
            classes_set.add(label)    
                
                
        
        return words_set,classes_set
    
    
    def preprocess(self,text):
   
        #remove everything after time stamp
        text = re.sub(r'((1[0-2]|0?[1-9]):([0-5][0-9]) ?([AaPp][Mm]) ?(-) ([\s\S]+))', '', text)
        #remove @mentions with user ID
        text = re.sub(r'@[A-Za-z0-9]+','',text)
        # Check characters to see if they are in punctuation
        text = re.sub('\d+','', text)
       #remove punctuation
        nopunc = [char for char in text if char not in string.punctuation]
        # Join the characters again to form the string.
        combined = ''.join(nopunc)
        
        # convert text to lower-case
        combined = combined.lower()
        # remove URLs
        combined = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', combined)
    
        
        # remove # but keep the hastag as word
        combined = re.sub(r'#([^\s]+)', r'\1', combined)
        # remove repeated characters
        combined = word_tokenize(combined)
        # remove stopwords from final word list
        return [word for word in combined if word not in stopwords.words('english')]
    
    def get_index(self,item_set, unk = None):
        item2id = defaultdict(int)
        if unk is not None:
            item2id[unk] = 0
        for item in item_set:
            item2id[item] = len(item2id)
            
        return item2id    
            
                
    def split_data(self,data):
        train_split = []
        valid_split = []
        
        print("Data Statistics")
        
        fdist_class = FreqDist(data[:,1])
        
        for key in fdist_class.keys():
            print(key + " : {} Distribution: {:.2f} %" .format( fdist_class[key],100*fdist_class[key]/sum(fdist_class.values()) ))
        
        np.random.shuffle(data)
        train_ratio=int(len(data)*0.9)
        
        train_split=data[:train_ratio]
        valid_split=data[train_ratio:]
        
        return train_split,valid_split
    
    
"""Dataset interface provided with pytorch"""    

class PaddedTensorDataset(Dataset):
    """Dataset wrapping data, target and length tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
        length (Tensor): contains sample lengths.
        raw_data (Any): The data that has been transformed into tensor, useful for debugging
    """

    def __init__(self, data_tensor, target_tensor, length_tensor, raw_data):
        assert data_tensor.size(0) == target_tensor.size(0) == length_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.length_tensor = length_tensor
        self.raw_data = raw_data

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index], self.length_tensor[index], self.raw_data[index]

    def __len__(self):
        return self.data_tensor.size(0)
        
        