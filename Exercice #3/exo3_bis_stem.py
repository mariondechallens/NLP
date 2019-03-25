# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 09:47:04 2019
@author: Admin
"""
import pandas as pd
import numpy as np

data = 'C:/Users/Admin/Documents/Centrale Paris/3A/OMA/NLP/Exo 3/convai2_fix_723.tar'
rep = 'C:/Users/Admin/'
#data = 'C:/Users/Sophie HU/Desktop/CentraleSupelec/NLP/HW3/convai2_fix_723.tgz'
#rep = 'C:/Users/Sophie HU/Desktop/CentraleSupelec/NLP/HW3/'
#source :http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/

def text2sentences2(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append( l.lower() )
    return sentences

train = text2sentences2(rep+'train_both_original.txt')
#séparer les dialogues 

def sep_dial(train):
    indexes = []
    for i in range(len(train)) :
        if (train[i][0:14] == "1 your persona"):
            indexes.append(i)
    indexes.append(len(train))
    ll = []
    for i in range(len(indexes)-1):
        ll.append(train[indexes[i]:indexes[i+1]])
    return ll 

list_dial = sep_dial(train)

def find_cont(dial):
    indexes = []
    for i in range(len(dial)):
        if (dial[i][2:19] == "partner's persona" or dial[i][3:20] == "partner's persona" ):
            indexes.append(i)
    cont = dial[0:(indexes[-1]+1)]
    return cont
        
    
    
def cleaning(cont):
    cont2 = []
    for sent in cont :
        sent = sent.strip('\n')
        sent = ' '.join([w for w in sent.split()[3:]])
        cont2.append(sent)
        
    cont2 = ' '.join(cont2)    
    return cont2


def add_cont(cont,sent):
    cont2 = []
    cont2.append(cont)
    cont2.append(sent)
    cont2 = ' '.join(cont2) 
    return cont2

def remove_digit(utt):
    c = [w for w in utt if w.isdigit()==False]
    c =''.join(c)
    return c
#df train

def add_rows_train(row, n,train_d):
    for i in range(n):
        dial = train_d[i]
        n_d = len(dial)
        
        cont = find_cont(dial)
        n_s = len(cont)
        
        cont = cleaning(cont)
              
        for j in range(n_s,n_d):
            distr = dial[j].split("\t")
            cor = distr[1]
            utt = distr[0]         
            cont = add_cont(cont,remove_digit(utt))
            row.append({'context': cont, 'utt': cor, 'xlabel': 1})
            answers = distr[3].split("|") 
        
            for i in range(len(answers)-1):
                d = {'context': cont, 'utt': answers[i], 'xlabel': 0}
                row.append(d)  
                
            cont = add_cont(cont,cor)
        
    return row

#df test

def add_rows_test(row2,n,train_d):
    for i in range(n):
        dial = train_d[i]
        n_d = len(dial)
        
        cont = find_cont(dial)
        n_s = len(cont)
        
        cont = cleaning(cont)
       
        for j in range(n_s,n_d):
            distr = dial[j].split("\t")
            cor = distr[1]
            utt = distr[0]
            answers = distr[3].split("|")     
            cont = add_cont(cont,remove_digit(utt))
            row2.append({'context': cont, 'correct': cor, 'dis1': answers[0],'dis2': answers[1],'dis3': answers[2],'dis4': answers[3],'dis5': answers[4],'dis6': answers[5], \
                 'dis7': answers[6],'dis8': answers[7],'dis9': answers[8],'dis10': answers[9],'dis11': answers[10],'dis12': answers[11],'dis13': answers[12], \
                 'dis14': answers[13],'dis15': answers[14],'dis16': answers[15],'dis17': answers[16],'dis18': answers[17], 'dis19': answers[18]})
            cont = add_cont(cont,cor)
    return row2
 

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def dataprocessing(orig):
    # Remove puntuation
    table = str.maketrans('', '', string.punctuation)
    new_train = [w.translate(table) for w in orig] #without puntuations
    
    # split into words
    
    new_train_bis=[]
    for i in range(len(orig)):
        new_train_bis.append(word_tokenize(new_train[i]))
    
    #Stop Words
    stop_words = stopwords.words('english')
    train_=[]
    for i in range(len(new_train_bis)):
        train_.append(' '.join([w for w in new_train_bis[i] if not w in stop_words]))
        
    #Setm
    porter = PorterStemmer()
    stemmed=[]
    for i in range(len(train_)):
        stemmed.append(','.join([porter.stem(word) for word in word_tokenize(train_[i])]))
        
    return stemmed
###
    


N = 20

#building train data set
row = []
row = add_rows_train(row,N,list_dial)
df_train_old = pd.DataFrame(data = row)

#### Data Processing on df_train[context] and df_train[utt]
df_train = pd.DataFrame()


df_train['xlabel']=df_train_old['xlabel']
df_train['context']= dataprocessing(df_train_old['context'])
df_train['utt']= dataprocessing(df_train_old['utt'])


#building test data set
row2 = []
row2 = add_rows_test(row2,N,list_dial)    
df_test_old = pd.DataFrame(data = row2)

df_test = df_test_old
for i in range(df_test_old.shape[1]):
    df_test.iloc[:,i]= dataprocessing(df_test_old.iloc[:,i])

#Recall@k means that we let the model pick the k best responses out of the 20 possible responses (1 true and 19 distractors)
def evaluate_recall(y, y_test, k=1):
    num_examples = float(len(y))
    num_correct = 0
    for predictions, label in zip(y, y_test):
        if label in predictions[:k]:
            num_correct += 1
    return num_correct/num_examples

# Random Predictor
def predict_random(context, utterances):
    return np.random.choice(len(utterances), 20, replace=False)

# Evaluate Random predictor
y_random = [predict_random(df_test.context[x], df_test.iloc[x,1:].values) for x in range(len(df_test))]
y_test = np.zeros(len(y_random))
for n in [1, 2, 5, 10, 15, 20]:
    print('Recall at ',n)
    print(evaluate_recall(y_random, y_test, n))

from sklearn.feature_extraction.text import TfidfVectorizer
    
class TFIDFPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
 
    def train(self, data):
        self.vectorizer.fit(np.append(data.context.values,data.utt.values))
 
    def predict(self, context, utterances):
        # Convert context and utterances into tfidf vector
        vector_context = self.vectorizer.transform([context])
        vector_doc = self.vectorizer.transform(utterances)
        # The dot product measures the similarity of the resulting vectors
        result = np.dot(vector_doc, vector_context.T).todense()
        result = np.asarray(result).flatten()
        # Sort by top results and return the indices in descending order
        return np.argsort(result, axis=0)[::-1]    
    
pred = TFIDFPredictor()
pred.train(df_train)
y = [pred.predict(df_test.context[x], df_test.iloc[x,1:].values) for x in range(len(df_test))]
for n in [1, 2, 5, 10, 15, 20]:
    print('Recall at ',n)
    print(evaluate_recall(y, y_test, n))
    
# 0.49 the best we can get with this method  pour N = 200 et 2000  

    