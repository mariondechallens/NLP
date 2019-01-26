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
__emails__  = ['fatherchristmoas@northpole.dk','mariondechallens@gmail.com']

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


sentences = text2sentences(sent)
vocabulary = []

for sentence in sentences:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

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
