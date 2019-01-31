# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 19:31:52 2019

@author: Admin
"""

from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize


__authors__ = ['Sophie Hu','Marion Favre dEchallens']
__emails__  = ['jiahui.hu@student-cs.fr','mariondechallens@gmail.com']

import os
PATH_TO_DATA = "C:/Users/Admin/Documents/Centrale Paris/3A/OMA/Deep Learning/nlp_project/nlp_project/data/"
sent = os.path.join(PATH_TO_DATA, 'sentences.txt')
#words = os.path.join(PATH_TO_DATA, 'crawl-300d-200k.vec')

def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append( l.lower().split() )
    return sentences

def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs

## generating vocabulary from the data
sentences = text2sentences(sent)
vocabulary = []

for sentence in sentences:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)

# counting words occurences
word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

print(word2idx['cat'])
print(idx2word[2276])

# generating words vector for words as words (center) and words as context
window_size = 2 #context windows - exploration around the center word
idx_pairs = []
# for each sentence

for sentence in sentences:
    indices = [word2idx[word] for word in sentence]
    # for each word as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        # we explore the words next to the center_word 
        # within the context window
        for w in range(-window_size, window_size + 1):
            # indice of a contex_word in the context window of the center_word
            context_word_pos = center_word_pos + w 
            # make sure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs) #pairs of (center word, context word)

# computing P(context|center) according to the formula of skip gram
# nominator : exp(u_c'v_w) is the similarity score of a pair (c,w)
# denominator : sum(cont in context) of exp(u_cont'v_w)

sentence = sentences[489] 
idx_pairs = []
indices = [word2idx[word] for word in sentence]
    # for each word as center word
for center_word_pos in range(len(indices)):
    for w in range(-window_size, window_size + 1):
        context_word_pos = center_word_pos + w 
        if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
            continue
        context_word_idx = indices[context_word_pos]
        idx_pairs.append((indices[center_word_pos], context_word_idx))
idx_pairs = np.array(idx_pairs)
#  building center word and context word vectors by one hot encoding
n = len(sentence)
center = np.zeros((n,n))
for i in range(n):
    vec = np.zeros(len(sentence))
    vec[i] = 1
    center[i] = vec

lw = 2*window_size
context = np.zeros((n,lw,n))
for i in range(len(idx_pairs)):
    context[indices.index(idx_pairs[i,0]),i%4] = center[indices.index(idx_pairs[i,1])]
    
for i in range(n):
    print(i, "\n center word =", center[i], "\n context words =\n",context[i])
# skip gram = center word in input and context words in output
    
u_c = context[0]

#input center word
v_w = center[0]

W = np.random.normal(size = (n,2)) #input -> hidden matrix
W2 = np.random.normal(size = (2,n)) #hidden -> output matrix

#hidden layer
h = np.matmul(W.transpose(),v_w)
#output context word
u = np.matmul(W2.transpose(),h)
#soft max
y = softmax(u)

#back progagation and training to improve W and W2

def softmax(x):

    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)

def backprop(W,W2, e, h, x, eta = 0.025):
    dl_dw2 = np.outer(h, e)  
    dl_dw = np.outer(x, np.matmul(W2, e.transpose()))

    # UPDATE WEIGHTS
    W = W - (eta * dl_dw)
    W2 = W2 - (eta * dl_dw2)
    return (W,W2)

def train(center,context,epochs,n,m=2):
    # INITIALIZE WEIGHT MATRICES
    W = np.random.uniform(-0.8, 0.8, (n, m))   # context matrix
    W2 = np.random.uniform(-0.8, 0.8, (m, n))     # embedding matrix

    for i in range(epochs):
        loss = 0
        for j in range(n): 
            
            v_w = center[j]
            
            #taking a context of v_w with P(D=1)
            u_c = context[j]
            u_c = u_c[~np.all(u_c == 0, axis=1)] #removing zero lines
            #adding a context of v_w with P(D=0)
            k = np.random.randint(n)
            word = center[k]
            u_c = np.concatenate((u_c,word.reshape(1,n)), axis = 0)
            
                #hidden layer
            h = np.matmul(W.transpose(),v_w)
                #output context word
            u = np.matmul(W2.transpose(),h)
           
            y = softmax(u) 
            

            # ERROR
            EI = np.sum([np.subtract(y, word) for word in u_c], axis=0)

            # BACKPROPAGATION
            W,W2 = backprop(W = W,W2 = W2,e = EI, h = h, x = v_w)

            # CALCULATE LOSS
            loss += -np.sum([u[np.argmax(word)] for word in u_c]) + \
                    len(u_c) * np.log(np.sum(np.exp(u))) 

        if i%1000== 0:
            print('EPOCH:',i, 'LOSS:', loss)
            
    return(W,W2)

W, W2 = train(center,context,5000,n)
#test sur training data: y == u_c ?
u_c = context[9]
v_w = center[9]
#hidden layer
h = np.matmul(W.transpose(),v_w)
#output context word
u = np.matmul(W2.transpose(),h)
#soft max
y = softmax(u)

#test sur testing data: y == u_c ?
sentence = sentences[99] + ['.']*( len(sentences[489]) - len(sentences[99]) )
print(sentence)
idx_pairs = []
indices = [word2idx[word] for word in sentence]
    # for each word as center word
for center_word_pos in range(len(indices)):
    for w in range(-window_size, window_size + 1):
        context_word_pos = center_word_pos + w 
        if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
            continue
        context_word_idx = indices[context_word_pos]
        idx_pairs.append((indices[center_word_pos], context_word_idx))
idx_pairs = np.array(idx_pairs)
#  building center word and context word vectors by one hot encoding
n = len(sentence)
center = np.zeros((n,n))
for i in range(n):
    vec = np.zeros(len(sentence))
    vec[i] = 1
    center[i] = vec

lw = 2*window_size
context = np.zeros((n,lw,n))
for i in range(len(idx_pairs)):
    context[indices.index(idx_pairs[i,0]),i%4] = center[indices.index(idx_pairs[i,1])]

u_c = context[0]
v_w = center[0]
#hidden layer
h = np.matmul(W.transpose(),v_w)
#output context word
u = np.matmul(W2.transpose(),h)
#soft max
y = softmax(u)

class SkipGram:
    def __init__(self,sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        self.load(sentences)
        self.word2id = dict.fromkeys(self.word2vec.keys())
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.embeddings = np.array(self.word2vec.values())
        #raise NotImplementedError('implement it!')


    def train(self,stepsize, epochs):
        raise NotImplementedError('implement it!')

    def save(self,path):
        raise NotImplementedError('implement it!')

    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        
        raise NotImplementedError('implement it!')

    @staticmethod
    def load(path):
        raise NotImplementedError('implement it!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train(...)
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = mSkipGram.load(opts.model)
        for a,b,_ in pairs:
            print sg.similarity(a,b)
