# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:28:38 2019

@author: Sophie HU
"""

# References
'''
•	https://nathanrooy.github.io/posts/2018-03-22/word2vec-from-scratch-with-python-and-numpy/ : code python et théorie
•	http://mediamining.univ-lyon2.fr/people/guille/word_embedding/skip_gram_with_negative_sampling.html : théorie en francais 
•	https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb : code
•	http://www.claudiobellei.com/2018/01/07/backprop-word2vec-python/ : code et théorie
•	https://github.com/deborausujono/word2vecpy: code
•	https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c:code
•	http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/ :theorie
'''


from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize

# ours 
import re
from collections import Counter


__authors__ = ['Sophie Hu','Marion Favre dEchallens']
__emails__  = ['jiahui.hu@student-cs.fr','mariondechallens@gmail.com']

#Data Path
import os
PATH_TO_DATA = "C:/Users/Admin/Documents/Centrale Paris/3A/OMA/Deep Learning/nlp_project/nlp_project/data/"
#PATH_TO_DATA = "C:/Users/Sophie HU/Desktop/CentraleSupelec/DL/NLP/nlp_project/nlp_project/data"
filename = os.path.join(PATH_TO_DATA, 'sentences.txt')




########################
#Code donnes par le prof

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

sentences = text2sentences(filename)
########################

def vocab_ids(sentences,n_most=13000):
    # Creat Tokens. We keep only tokens that is alphabic words
    # join all sentences in to one
    sentences_joint = [inner for outer in sentences for inner in outer]
    # count the occurences of each word: output tuple
    sentences_count = Counter(sentences_joint)
    # get most frequent words
    sentences_most_frqt = sentences_count.most_common(n_most)
    vocabulary = [x[0] for x in sentences_most_frqt if x[0].isalpha()==True]
    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
    return vocabulary,word2idx,idx2word

vocabulary,word2idx,idx2word = vocab_ids(sentences,13000)


def sigmoid(x):  
    return 1 / (1 + np.exp(-x))

class SkipGram:
    def __init__(self,sentences,vocabulary,word2idx,idx2word, dim = 2, nEmbed=100, negativeRate=5, winSize = 2, minCount = 5):#winSize = 5
        # raise NotImplementedError('implement it!')
        # self.sentences_ = sentences
        # self.nEmbed = nEmbed
        # self.negativeRate = negativeRate
        self.winSize = winSize
        # self.minCount = minCount
        self.vocabulary = vocabulary
        self.word2idx = word2idx
        self.idx2word = idx2word 
        self.idx_pairs = []
        self.sentences = sentences 
        self.dim = dim
        self.voc_size = len(vocabulary)
    ########################################################################################################
    #1. Preprcessing Steps#
    ########################################################################################################
    ## generating vocabulary and ids from observed data(most frequent words)
    def create_pairs_pos_neg(self,k=5):
        ''' This function builds center word and context word vectors by one hot encoding '''
        # generating words vector for words as words (center) and words as context
        window_size = self.winSize  #context windows - exploration around the center word
        # word2idx = self.word2idx
        for sentence in sentences:
            indices = [word2idx[word] for word in sentence if word in vocabulary]
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
                    self.idx_pairs.append((indices[center_word_pos], context_word_idx,1))
                    #negative words
                    for i in range(k):
                        indice_neg = np.random.randint(len(vocabulary))
                        self.idx_pairs.append((indices[center_word_pos], indice_neg,-1))
                    
                        
        self.idx_pairs = np.array(self.idx_pairs) #pairs of (center word, context word)
        return self.idx_pairs
    
    ## generating negatif samples
    
    ########################################################################################################
    # 2. Skip-Gram#
    ########################################################################################################
    #  building center word and context word vectors by one hot encoding

    def train(self,stepsize, epochs):
        w0 = np.random.uniform(-0.8/self.dim,0.8/self.dim,(self.voc_size,self.dim))
        w1 = np.zeros((self.voc_size,self.dim))
        for i in range(epochs): 
            #loss = 0
            j= 0
            while j < len(self.idx_pairs):
                context_word= classif[0,1]
                w = np.zeros(self.dim)
                classif = idx_pairs[j:j+k+1]
                for ind in range(k+1):
                    neg_word = classif[ind,1]
                    u = np.dot(w0[context_word], w1[neg])
                    p = sigmoid(u)
                    e = 0.025 * (classif[ind,2] - p)
                    w += e * w1[neg_word]
                    w1[neg_word] += e * w0[context_word] 
                
                w0[context_word] += w
                j += k+1 #passer à la prochaine paire positive



    def save(self,path):
        

    def similarity(self,word1,word2):
       

    def load(path):
        '''

# Test Code
#test = SkipGram(sentences).vocab_ids()
vocabulary,word2idx,idx2word = vocab_ids(sentences[0:5],13000)
test = SkipGram(sentences[0:10],vocabulary,word2idx,idx2word)   
vocabulary = test.vocabulary
word2idx = test.word2idx
idx2word = test.idx2word
test_id_pairs = test.create_pairs_pos_neg()


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
            print(sg.similarity(a,b))
