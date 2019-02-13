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
#from scipy.special import expit
#from sklearn.preprocessing import normalize

# ours 
#import re
from collections import Counter


__authors__ = ['Sophie Hu','Marion Favre dEchallens']
__emails__  = ['jiahui.hu@student-cs.fr','mariondechallens@gmail.com']

#Data Path
import os
PATH_TO_DATA = "C:/Users/Admin/Documents/Centrale Paris/3A/OMA/Deep Learning/nlp_project/nlp_project/data/"
#PATH_TO_DATA = "C:/Users/Sophie HU/Desktop/CentraleSupelec/DL/NLP/nlp_project/nlp_project/data"
PATH_TO_NLP = "C:/Users/Admin/Documents/Centrale Paris/3A/OMA/NLP/Exo 1/"
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

# removing characters we don't want (. # numbers)
def cleaning(sentences) :
    sent_clean = []
    for sentence in sentences :
        sentence2 = [word for word in sentence if word.isalpha() == True]
        sent_clean.append(sentence2)
    return sent_clean

sentences = cleaning(text2sentences(filename))

###################Amelioration######################################
#from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
# instanciation du stemmer
stemmer = SnowballStemmer('english')
#Create a function Stem
def stem(text):
    return ' '.join([stemmer.stem(word) for word in text])
stem_sent = [stem(s) for s in sentences]
########################
def stem2(text) :
    stem = []
    for sentence in text:
        stem.append([stemmer.stem(word) for word in sentence])
    return stem
#stem_sent2 = stem2(sentences)   

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

'''
def sim_matrix(n,voc,path):
#similarity matrix
    sim = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            sim[i][j] = test.similarity(voc[i],voc[j],U)
            
    sim_df = pd.DataFrame(sim)
    sim_df.index = voc
    sim_df.columns = voc
    sim_df.to_csv(path +'similarity.csv',index=True)
    return sim_df '''

def sigmoid(x):  
    return 1 / (1 + np.exp(-x))

def log_Likelihood(id_pairs,U,V):
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
        u = u + (alpha*m_cap[0])/(np.sqrt(v_cap[0])+epsilon) #attention -
        v = v + (alpha*m_cap[1])/(np.sqrt(v_cap[1])+epsilon) 
 
        
    return u,v

def adam_batch(dim,indice_d,u,v,batch_size,iter_ = 1, alpha = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
    #theta_0 = 0
    m_t = np.zeros((2,dim))
    v_t = np.zeros((2,dim)) 
    t = 0
    
    while (t<iter_):
        t+=1
        g_t_i = np.array([sigmoid(-indice_d[l] * np.dot(u[l],v[l])) * v[l] * indice_d[l] for l in range(batch_size)])
        g_t_j = np.array([sigmoid(-indice_d[l] * np.dot(u[l],v[l])) * u[l] * indice_d[l] for l in range(batch_size)])
        g_t = np.array([np.mean(g_t_i,axis = 0),np.mean(g_t_j,axis=0)])
        m_t = beta_1*m_t + (1-beta_1)*g_t	#updates the moving averages of the gradient
        v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)	#updates the moving averages of the squared gradient
        m_cap = m_t/(1-(beta_1**t))		#calculates the bias-corrected estimates of the gradient
        v_cap = v_t/(1-(beta_2**t))		#calculates the bias-corrected estimates of the squared gradient
								
        #theta_0 = theta_0 - (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)	#updates the parameters
        u = u + (alpha*m_cap[0])/(np.sqrt(v_cap[0])+epsilon) #attention -
        v = v + (alpha*m_cap[1])/(np.sqrt(v_cap[1])+epsilon) 
 
        
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

    def create_pairs_pos_neg(self,proba = True,stem = False): #cooccurence matrix
        # generating words vector for words as words (center) and words as context
        window_size = self.winSize  #context windows - exploration around the center word
        # word2idx = self.word2idx
        if stem == True :
            self.sentences = stem2(self.sentences)
        for sentence in self.sentences:
            if proba == True :
                prob_neg, neg_word = probability(self.sentences)
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
                        if proba == False :
                            indice_neg = np.random.randint(len(self.vocabulary))
                        if proba == True :
                            indice_neg = np.random.choice(neg_word,p=prob_neg)
                        self.idx_pairs.append((indices[center_word_pos], indice_neg,-1))
                    
                        
        self.idx_pairs = np.array(self.idx_pairs) #pairs of (center word, context word)
        return self.idx_pairs
    

    ## generating negatif samples
    
    ########################################################################################################
    # 2. Skip-Gram
    ########################################################################################################
    #  building center word and context word vectors by one hot encoding
    def train(self, n_iter = 20,lr = 0.002,batch_size = 20,adam_opt = True, batch = False):

    # We initiate our embedding by using normal multivariate distribution
        U = np.random.randn(self.voc_size,self.nEmbed)
        V = np.random.randn(self.voc_size,self.nEmbed)
    
        ll = log_Likelihood(self.idx_pairs,U,V)
        print("initial likelihood",ll)
        n_obs = self.idx_pairs.shape[0]
        
        if adam_opt == False and batch == True :
            print('Performing Gradient Descent with mini-batches')
            for iteration in range(n_iter):
                np.random.shuffle(self.idx_pairs)
                q = n_obs//batch_size
                for id_obs in range(0,q*batch_size,batch_size):
                
                    batch_ = self.idx_pairs[id_obs:id_obs+batch_size,:]
                    indice_i  = [batch_[i,0] for i in range(batch_size)]
                    indice_j  = [batch_[i,1] for i in range(batch_size)]
                    indice_d  = [batch_[i,2] for i in range(batch_size)]
                    
                    u = U[indice_i,:]
                    v = V[indice_j,:]  
                    
                    grad_u_m = np.mean(np.array([sigmoid(-indice_d[l] * np.dot(u[l],v[l])) * v[l] * indice_d[l] for l in range(batch_size)]),axis = 0)
                    U[indice_i,:] = U[indice_i,:] + lr * grad_u_m
                    
                    grad_v_m = np.mean(np.array([sigmoid(-indice_d[l] * np.dot(u[l],v[l])) * u[l] * indice_d[l] for l in range(batch_size)]),axis = 0)
                    V[indice_j,:] = V[indice_j,:] + lr * grad_v_m
                
                ll = log_Likelihood(self.idx_pairs,U,V)
                if iteration%1 == 0:
                    print("likelihood at step ",int(iteration + 1)," : ",ll)
        

        if adam_opt == False and batch == False : 
            print('Performing Gradient Descent without mini-batches')
            for iteration in range(n_iter):
                np.random.shuffle(self.idx_pairs)
                for id_obs in range(n_obs):

                    i,j,d = self.idx_pairs[id_obs,:]
               
                    u = U[i,:]
                    v = V[j,:]
            
                    x = np.dot(u,v)

                    grad_u_i = sigmoid(-d * x) * v * d
                    U[i,:] = U[i,:] + lr * grad_u_i
            
                    grad_v_j = sigmoid(-d * x) * u * d       
                    V[j,:] = V[j,:] + lr * grad_v_j
                    
                ll = log_Likelihood(self.idx_pairs,U,V)
                if iteration%1 == 0:
                    print("likelihood at step ",int(iteration + 1)," : ",ll)
        
                    
        if adam_opt == True and batch == False :
            print('Performing Adam optimization without mini-batches')
            for iteration in range(n_iter):
                np.random.shuffle(self.idx_pairs)
                for id_obs in range(n_obs):

                    i,j,d = self.idx_pairs[id_obs,:]
               
                    u = U[i,:]
                    v = V[j,:]
            
                    x = np.dot(u,v)
                    U[i,:], V[j,:] = adam(self.nEmbed,d,u,v,x)
                
                ll = log_Likelihood(self.idx_pairs,U,V)
                if iteration%1 == 0:
                    print("likelihood at step ",int(iteration + 1)," : ",ll)
        
            
        if adam_opt == True and batch == True :
            print('Performing Adam optimization with mini-batches')
            for iteration in range(n_iter):
                np.random.shuffle(self.idx_pairs)
                q = n_obs//batch_size
                for id_obs in range(0,q*batch_size,batch_size):
                    
                    batch_ = self.idx_pairs[id_obs:id_obs+batch_size,:]
                    indice_i  = [batch_[i,0] for i in range(batch_size)]
                    indice_j  = [batch_[i,1] for i in range(batch_size)]
                    indice_d  = [batch_[i,2] for i in range(batch_size)]
                    
                    u = U[indice_i,:]
                    v = V[indice_j,:]
                    
                    U[indice_i,:],V[indice_j,:] = adam_batch(self.nEmbed,indice_d,u,v,batch_size)
                    
                    

                    
                ll = log_Likelihood(self.idx_pairs,U,V)
                if iteration%1 == 0:
                    print("likelihood at step ",int(iteration + 1)," : ",ll)
        
        return U,V,ll
    
    def similarity(self,word1,word2,word_emb): # similar if we can replace the word1 by the word2, they appear in the same context
        if word1 not in self.word2idx.keys() or word2 not in self.word2idx.keys():
            #print('At least one of the words is not in the vocabulary')
            s = -10
        else:
            u = np.array(word_emb.loc[word1])
            v = np.array(word_emb.loc[word2])
            s = round(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)),3) #cosine 
        return s

    def similarity_file(self,test_file,path,word_emb):
        df_test = pd.read_csv(path + test_file)
        sim_skip = np.zeros(len(df_test))
        for i in range(len(df_test)):
            sim_skip[i] = self.similarity(df_test['word1'][i],df_test['word2'][i],word_emb)
        df_test['skip_sim'] = sim_skip
        df_test.to_csv(path + 'test_skip_sim.csv')
        
        

    def save(self,path,U,V):
        voc = pd.DataFrame(self.vocabulary,columns=['vocab'])
        # center words matrix
        emb_w = pd.DataFrame(U)
        emb_w = pd.concat([voc, emb_w], axis=1, sort=False)
        emb_w = emb_w.set_index('vocab')
        emb_w.to_csv(path +'emb_word.csv',index=True)
        
        #context words matrix
        emb_c = pd.DataFrame(V)
        emb_c = pd.concat([voc, emb_c], axis=1, sort=False)
        emb_c = emb_c.set_index('vocab')
        emb_c.to_csv(path +'emb_context.csv',index=True)
        
        
        
    
    @staticmethod
    def load(path):
        emb_word = pd.read_csv(path + 'emb_word.csv').set_index('vocab')
        emb_cont = pd.read_csv(path + 'emb_context.csv').set_index('vocab')
        return emb_word, emb_cont
        



# Test Code
# probleme avec stem_sent, divergence de la likelihood
vocabulary,word2idx,idx2word = vocab_ids(sentences[0:5000],13000)
test = SkipGram(sentences[0:5000],vocabulary,word2idx,idx2word,nEmbed = 100,negativeRate = 5)   

#creation of the coocurrence matrix before the training !!
test_id_pairs = test.create_pairs_pos_neg()


U,V,ll = test.train(n_iter = 20, adam_opt= True,batch = True)
#U,V,ll1 = test.train(n_iter = 20, adam_opt= False,batch = False)
#U,V,ll2 = test.train(n_iter = 20, adam_opt= True,batch = False)
#U,V,ll3 = test.train(n_iter = 20, adam_opt= True,batch = True)

test.save(PATH_TO_NLP,U,V)
word, context = test.load(PATH_TO_NLP)


#train file
import gzip
sent_train = []
with gzip.open('C:/Users/Admin/Downloads/training-monolingual-news-commentary.gz','r') as fin:        
    for line in fin:        
       sent_train.append( line.lower().split() )

sent_train_clean = cleaning(sent_train)

#test file
test.similarity_file('test_file.csv',PATH_TO_NLP,word)





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
