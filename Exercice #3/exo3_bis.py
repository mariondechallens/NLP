# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 09:47:04 2019

@author: Admin
"""
import pandas as pd
import numpy as np

data = 'C:/Users/Admin/Documents/Centrale Paris/3A/OMA/NLP/Exo 3/convai2_fix_723.tar'
rep = 'C:/Users/Admin/'
#source :http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/

def text2sentences2(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append( l.lower() )
    return sentences

train = text2sentences2(rep+'train_both_original.txt')
dial = train[0:15]
dial2 = train[15:30]

cont = dial[0:8]
distr = dial[8].split("\t")
utt = distr[0]
cor = distr[1]
answers = distr[3].split("|") #correct one at the end
dis1 = answers[0]

def cleaning(cont):
    cont2 = []
    for sent in cont :
        sent = sent.strip('\n')
        sent = ' '.join([w for w in sent.split()[3:]])
        cont2.append(sent)
        
    cont2 = ' '.join(cont2)    
    return cont2

#cont2 = cleaning(cont)

def add_cont(cont,sent):
    cont2 = []
    cont2.append(cont)
    cont2.append(sent)
    cont2 = ' '.join(cont2) 
    return cont2

#cont3 = add_cont(cont2,utt[1:])
def remove_digit(utt):
    c = [w for w in utt if w.isdigit()==False]
    c =''.join(c)
    return c
#df train
row = []
dial = train[0:15]
n_d = len(dial)
cont = dial[0:8]
cont = cleaning(cont)
for j in range(8,n_d):
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

df_train = pd.DataFrame(data = row)

#df test
row2 = []
for i in range(2):
    dial = train[(15*i):((i+1)*15)]
    n_d = len(dial)
    cont = dial[0:8]
    cont = cleaning(cont)
    
    for j in range(8,n_d):
        distr = dial[j].split("\t")
        cor = distr[1]
        utt = distr[0]
        answers = distr[3].split("|")     
        cont = add_cont(cont,remove_digit(utt))
        row2.append({'context': cont, 'correct': cor, 'dis1': answers[0],'dis2': answers[1],'dis3': answers[2],'dis4': answers[3],'dis5': answers[4],'dis6': answers[5], \
                 'dis7': answers[6],'dis8': answers[7],'dis9': answers[8],'dis10': answers[9],'dis11': answers[10],'dis12': answers[11],'dis13': answers[12], \
                 'dis14': answers[13],'dis15': answers[14],'dis16': answers[15],'dis17': answers[16],'dis18': answers[17], 'dis19': answers[18]})
        cont = add_cont(cont,cor)
            
df_test = pd.DataFrame(data = row2)

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
    

#Dual Encoder LSTM : hard
    