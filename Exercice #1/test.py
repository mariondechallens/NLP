# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:28:38 2019

@author: Sophie HU and Marion Favre d'Echallens
"""


from __future__ import division
import argparse
import pandas as pd
import pickle # for saving and loading models
import numpy as np

from collections import Counter


__authors__ = ['Sophie Hu','Marion Favre dEchallens']
__emails__  = ['jiahui.hu@student-cs.fr','mariondechallens@gmail.com']

#Data Path
# import os
PATH_TO_DATA = "C:/Users/Admin/Documents/Centrale Paris/3A/OMA/Deep Learning/nlp_project/nlp_project/data/"
#PATH_TO_NLP = "C:/Users/Sophie HU/Desktop/CentraleSupelec/NLP/NLP-master"
PATH_TO_NLP = "C:/Users/Admin/Documents/Centrale Paris/3A/OMA/NLP/Exo 1/"
#filename = os.path.join(PATH_TO_DATA, 'sentences.txt')


#######################train file####################################
import gzip
sent_train = []
#with gzip.open('C:/Users/Sophie HU/Desktop/CentraleSupelec/NLP/NLP-master/training-monolingual-news-commentary.tgz','r') as fin:        
with gzip.open('C:/Users/Admin/Downloads/training-monolingual-news-commentary.gz','r') as fin:        
  for line in fin:        
       sent_train.append(line.lower().split() )
#####################################################################


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
        sentence2 = [word.decode("utf-8") for word in sentence if word.isalpha() == True]
        sent_clean.append(sentence2)
    return sent_clean


###################Stem######################################
#from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
# instanciation du stemmer
stemmer = SnowballStemmer('english')
#Create a function Stem
def stem(text):
    return ' '.join([stemmer.stem(word) for word in text])

def stem2(text) :
    stem = []
    for sentence in text:
        stem.append([stemmer.stem(word) for word in sentence])
    return stem
########################

def vocab_ids(sentences,n_most=13000):
    ''' Creat Tokens. We keep only tokens that is alphabic words'''
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

    
def probability(sentences):
    ''' probabilité de tirage des paires négatives'''
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
    #theta_0 = 0.1
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

def adam_batch(dim,indice_d,u,v,batch_size,iter_ = 1, alpha = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
    #alpha = 0.1
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


def create_pairs_pos_neg(sentences,word2idx,vocabulary, window_size = 2, negativeRate=5, proba = True,stem = False): #cooccurence matrix
    # generating words vector for words as words (center) and words as context
    #window_size : context windows - exploration around the center word
    # word2idx = self.word2idx
    idx_pairs = []
    if stem == True :
        sentences = stem2(sentences)
        
    for sentence in sentences:
        if proba == True :
            prob_neg, neg_word = probability(sentences)
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
                idx_pairs.append((indices[center_word_pos], context_word_idx,1))
                #negative words
                
                for i in range(negativeRate):
                    if proba == False :
                        indice_neg = np.random.randint(len(vocabulary))
                    if proba == True :
                        indice_neg = np.random.choice(neg_word,p=prob_neg)
                    idx_pairs.append((indices[center_word_pos], indice_neg,-1))
                
    idx_pairs = np.array(idx_pairs) #pairs of (center word, context word)
    return idx_pairs






class SkipGram:
    def __init__(self,sentences,nEmbed=100, minCount = 5):
        vocabulary,word2idx,idx2word = vocab_ids(sentences,13000)
        id_pairs = create_pairs_pos_neg(sentences,word2idx,vocabulary)
        self.nEmbed = nEmbed
        # self.minCount = minCount
        self.vocabulary = vocabulary
        self.word2idx = word2idx
        self.idx2word = idx2word 
        self.idx_pairs = id_pairs
        self.sentences = sentences 
        self.voc_size = len(vocabulary)
            # We initiate our embedding by using normal multivariate distribution
        self.U = np.random.randn(self.voc_size,self.nEmbed) #embeddings for center words
        self.V = np.random.randn(self.voc_size,self.nEmbed) #embeddings for context words 
        self.ll = 0

        #Setting as research paper: proba = true, stem = false
        # arguments of create_pairs_pos_neg(sentences,word2idx,vocabulary, window_size = 2, negativeRate=5, proba = True,stem = False)
        

    ########################################################################################################
    # Skip-Gram Negative Samples
    ########################################################################################################
    #  building center word and context word vectors by one hot encoding
    def train(self, n_iter = 20,lr = 0.002,batch_size = 20,adam_opt = False,batch=False, print_lik = False):

    
        self.ll = log_Likelihood(self.idx_pairs,self.U,self.V)
        print("initial likelihood",self.ll)
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
                    
                    u = self.U[indice_i,:]
                    v = self.V[indice_j,:]  
                    
                    grad_u_m = np.mean(np.array([sigmoid(-indice_d[l] * np.dot(u[l],v[l])) * v[l] * indice_d[l] for l in range(batch_size)]),axis = 0)
                    self.U[indice_i,:] = self.U[indice_i,:] + lr * grad_u_m
                    
                    grad_v_m = np.mean(np.array([sigmoid(-indice_d[l] * np.dot(u[l],v[l])) * u[l] * indice_d[l] for l in range(batch_size)]),axis = 0)
                    self.V[indice_j,:] = self.V[indice_j,:] + lr * grad_v_m
                
                self.ll = log_Likelihood(self.idx_pairs,self.U,self.V)
                if print_lik == True:
                    if iteration%1 == 0:
                        print("likelihood at step ",int(iteration + 1)," : ",self.ll)
        

        if adam_opt == False and batch == False : 
            print('Performing Gradient Descent without mini-batches')
            for iteration in range(n_iter):
                np.random.shuffle(self.idx_pairs)
                for id_obs in range(n_obs):
    
                    i,j,d = self.idx_pairs[id_obs,:]
               
                    u = self.U[i,:]
                    v = self.V[j,:]
            
                    x = np.dot(u,v)
    
                    grad_u_i = sigmoid(-d * x) * v * d
                    self.U[i,:] = self.U[i,:] + lr * grad_u_i
            
                    grad_v_j = sigmoid(-d * x) * u * d       
                    self.V[j,:] = self.V[j,:] + lr * grad_v_j
                    
                self.ll = log_Likelihood(self.idx_pairs,self.U,self.V)
                if print_lik == True:
                    if iteration%1 == 0:
                        print("likelihood at step ",int(iteration + 1)," : ",self.ll)
         
                    
        if adam_opt == True and batch == False :
            print('Performing Adam optimization without mini-batches')
            for iteration in range(n_iter):
                np.random.shuffle(self.idx_pairs)
                for id_obs in range(n_obs):

                    i,j,d = self.idx_pairs[id_obs,:]
               
                    u = self.U[i,:]
                    v = self.V[j,:]
            
                    x = np.dot(u,v)
                    self.U[i,:],self.V[j,:] = adam(self.nEmbed,d,u,v,x)
                
                self.ll = log_Likelihood(self.idx_pairs,self.U,self.V)
                if print_lik == True:
                    if iteration%1 == 0:
                        print("likelihood at step ",int(iteration + 1)," : ",self.ll)
         
            
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
                    
                    u = self.U[indice_i,:]
                    v = self.V[indice_j,:]
                    
                    self.U[indice_i,:],self.V[indice_j,:] = adam_batch(self.nEmbed,indice_d,u,v,batch_size)
                    
                    

                    
                self.ll = log_Likelihood(self.idx_pairs,self.U,self.V)
                if print_lik == True:
                    if iteration%1 == 0:
                        print("likelihood at step ",int(iteration + 1)," : ",self.ll)
         
    
    def similarity(self,word1,word2): # similar if we can replace the word1 by the word2, they appear in the same context
        if word1 not in self.word2idx.keys() or word2 not in self.word2idx.keys():
            #print('At least one of the words is not in the vocabulary')
            s = -10
        else:
            u = np.array(self.U[self.word2idx[word1]])
            v = np.array(self.U[self.word2idx[word2]])
            s = round(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)),3) #cosine 
        return s

    def similarity_file(self,test_file,path):
        df_test = pd.read_csv(path + test_file)
        sim_skip = np.zeros(len(df_test))
        for i in range(len(df_test)):
            sim_skip[i] = self.similarity(df_test['word1'][i],df_test['word2'][i])
        df_test['skip_sim'] = sim_skip
        df_test.to_csv(path + 'test_skip_sim.csv')
        
        

    def save(self,path):
        with open(path, 'wb') as file:  
            pickle.dump(self, file)
   
    @staticmethod
    def load(path):
        with open(path, 'rb') as file:  
            pickle_model = pickle.load(file)
        return pickle_model
        















####Execution
sentences = sent_train[1000:3000]
sentences = cleaning(sentences)


model = SkipGram(sentences)   
model.train(print_lik= True) #n_iter = 20, adam = false and batch = false by default

model.save(PATH_TO_NLP + 'sg.pkl')
sg = model.load(PATH_TO_NLP + 'sg.pkl')

#test file
model.similarity_file('test_file.csv',PATH_TO_NLP)
print(model.similarity('boy','the'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    
    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)
        sg = SkipGram.load(opts.model)
        for a,b,_ in pairs:
            print(sg.similarity(a,b))
