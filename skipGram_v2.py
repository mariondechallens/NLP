# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:28:38 2019

@author: Sophie HU and Marion Favre d'Echallens
"""


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

def cleaning(sentences) :
    sent_clean = []
    for sentence in sentences :
        sentence2 = [word for word in sentence if word.isalpha() == True]
        sent_clean.append(sentence2)
    return sent_clean

sentences = cleaning(text2sentences(filename))
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

    # probabilité de tirage des paires négatives
def probability(sentences):
    sentences_joint = [inner for outer in sentences for inner in outer]
    sentences_count = Counter(sentences_joint)
    sentences_most_frqt = sentences_count.most_common(13000)
    sentences_clean = [x for x in sentences_most_frqt if x[0].isalpha()==True]
    select_word = np.random.randint(len(sentences_clean),size=32)
    select_word2 = [sentences_clean[i] for i in select_word]
    total = sum(x[1] for x in select_word2)
    prob = np.array([x[1]/total for x in select_word2 if x[0].isalpha()==True]) #occurence proba
    prob_neg = [p**(3/4)/ sum(prob**(3/4)) for p in prob]
    return  prob_neg, select_word

def sigmoid(x):  
    return 1 / (1 + np.exp(-x))

def log_Likelyhood(id_pairs,U,V):
    n_obs = id_pairs.shape[0]   
    ll = 0
    for id_obs in range(n_obs):
        i,j,d = id_pairs[id_obs,:]
           
        u = U[i,:]
        v = V[j,:]

        x = np.dot(u,v)

        ll += np.log(sigmoid(d*x))

    return ll

def adam(dim,d,u,v,x,iter_ = 1, alpha = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
    #theta_0 = 0
    m_t = np.zeros((2,dim))
    v_t = np.zeros((2,dim)) 
    t = 0
    
    while (t<iter_):
        t+=1
        g_t_i = sigmoid(-d * x) * v * d
        g_t_j = sigmoid(-d * x) * u * d
        g_t = np.array([g_t_i,g_t_j])
        #g_t = grad_func(theta_0)		#computes the gradient
        m_t = beta_1*m_t + (1-beta_1)*g_t	#updates the moving averages of the gradient
        v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)	#updates the moving averages of the squared gradient
        m_cap = m_t/(1-(beta_1**t))		#calculates the bias-corrected estimates of the gradient
        v_cap = v_t/(1-(beta_2**t))		#calculates the bias-corrected estimates of the squared gradient
								
        #theta_0 = theta_0 - (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)	#updates the parameters
        u = u - (alpha*m_cap[0])/(np.sqrt(v_cap[0])+epsilon)
        v = v - (alpha*m_cap[1])/(np.sqrt(v_cap[1])+epsilon) 
        
    return u,v



class SkipGram:
    def __init__(self,sentences,vocabulary,word2idx,idx2word,nEmbed=100, negativeRate=5, winSize = 2, minCount = 5):#winSize = 5

        self.nEmbed = nEmbed
        self.negativeRate = negativeRate
        self.winSize = winSize
        # self.minCount = minCount
        self.vocabulary = vocabulary
        self.word2idx = word2idx
        self.idx2word = idx2word 
        self.idx_pairs = []
        self.sentences = sentences 
        self.voc_size = len(vocabulary)
    ########################################################################################################
    #1. Preprocessing Steps#
    ########################################################################################################
    ## generating vocabulary and ids from observed data(most frequent words)

    def create_pairs_pos_neg(self):
        ''' This function builds center word and context word vectors by one hot encoding '''
        # generating words vector for words as words (center) and words as context
        window_size = self.winSize  #context windows - exploration around the center word
        # word2idx = self.word2idx
        
        for sentence in self.sentences:
            
            indices = [self.word2idx[word] for word in sentence if word in self.vocabulary]
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
                    
                    for i in range(self.negativeRate):
                        #indice_neg = np.random.randint(len(vocabulary))
                        prob_neg, neg_word = probability(self.sentences)
                        indice_neg = np.random.choice(neg_word,p=prob_neg)
                        self.idx_pairs.append((indices[center_word_pos], indice_neg,-1))
                    
                        
        self.idx_pairs = np.array(self.idx_pairs) #pairs of (center word, context word)
        return self.idx_pairs
    

    ## generating negatif samples
    
    ########################################################################################################
    # 2. Skip-Gram#
    ########################################################################################################
    #  building center word and context word vectors by one hot encoding
    def train(self, n_iter = 20,lr = 0.002):

    # We initiate our embedding by using normal multivariate distribution
        U = np.random.randn(self.voc_size,self.nEmbed)
        V = np.random.randn(self.voc_size,self.nEmbed)
    
        ll = log_Likelyhood(self.idx_pairs,U,V)
        print("initial likelyhood",ll)
    
        #id_pairs = np.array(self.idx_pairs)
        n_obs = self.idx_pairs.shape[0]

        for iteration in range(n_iter):

        #Randomize observations : stochastic
            np.random.shuffle(self.idx_pairs)
        # to do : mini-batch gradient descent
           ''' indices = np.random.permutation(len(self.idx_pairs))
            U_b = U[indices,:]
            V_b = V[indices,:]  #revient à faire shuffle
            for id_obs in range(0,len(self.idx_pairs,batch_size)):
                u = U_b[id_obs:id_obs+batch_size]
                v = V_b[id_obs:id_obs+batch_size]
                x = np.matmul(u,v.transpose())'''
            for id_obs in range(n_obs):

                i,j,d = self.idx_pairs[id_obs,:]
               
                u = U[i,:]
                v = V[j,:]
            
                x = np.dot(u,v)
                #U[i,:], V[j,:] = adam(self.nEmbed,d,u,v,x)

                grad_u_i = sigmoid(-d * x) * v * d
                U[i,:] = U[i,:] + lr * grad_u_i
            
                grad_v_j = sigmoid(-d * x) * u * d       
                V[j,:] = V[j,:] + lr * grad_v_j
            
        # We compute the likelyhood at the end of the current iteration
            ll = log_Likelyhood(self.idx_pairs,U,V)
            if iteration%100 == 0:
                print("likelyhood at step ",int(iteration + 1)," : ",ll)
        
        return U,V,ll
    
    def similarity(self,word1,word2,U): # similar if we can replace the word1 by the word2, they appear in the same context
        if word1 not in word2idx.keys():
            id1 = 0
        if word2 not in word2idx.keys():
            id2 = 0
        else:
            id1 = self.word2idx[word1]
            id2 = self.word2idx[word2]
        u = U[id1,:]
        v = U[id2,:]
        s = round(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)),3) #cosine 
        print('Similarity : ', s)

    

'''    def save(self,path):
        

    
       

    def load(path):
        '''

# Test Code
#test = SkipGram(sentences).vocab_ids()
vocabulary,word2idx,idx2word = vocab_ids(sentences[0:200],13000)
test = SkipGram(sentences[0:200],vocabulary,word2idx,idx2word,nEmbed = 100,negativeRate = 5)   
vocabulary = test.vocabulary
word2idx = test.word2idx
idx2word = test.idx2word
test_id_pairs = test.create_pairs_pos_neg()


U,V,ll = test.train(n_iter = 300)

#test paire pos
i,j,d = test_id_pairs[0,:]
u = U[i,:]
v = V[j,:]
x = np.dot(u,v)
p = sigmoid(d*x)

#test paire neg
i,j,d = test_id_pairs[3,:]
u = U[i,:]
v = V[j,:]
x = np.dot(u,v)
p = sigmoid(d*x)

test.similarity('blue','white',U) ## mauvais 



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
        sg = SkipGram.load(opts.model)
        for a,b,_ in pairs:
            print(sg.similarity(a,b))
