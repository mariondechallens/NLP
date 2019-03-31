# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 19:14:01 2019

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

train = text2sentences2(rep+'valid_both_original.txt')
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
    


N = 100

#building train data set
row = []
row = add_rows_train(row,N,list_dial)
df_train_old = pd.DataFrame(data = row)

#### Data Processing on df_train[context] and df_train[utt]
df_train = pd.DataFrame()


df_train['xlabel']=df_train_old['xlabel']
df_train['context']= dataprocessing(df_train_old['context'])
df_train['utt']= dataprocessing(df_train_old['utt'])


# source : https://hub.packtpub.com/build-generative-chatbot-using-recurrent-neural-networks-lstm-rnns/
# Creating Vocabulary

import nltk
import collections
counter = collections.Counter()
for i in range(len(df_train)):
    for word in nltk.word_tokenize(df_train['utt'][i]): 
    #for word in nltk.word_tokenize(' '.join([df_train['context'][i],df_train['utt'][i]])): 
        if word.isalpha() == True :
            counter[word]+=1
            word2idx = {w:(i+1) for i,(w,_) in enumerate(counter.most_common())}
            idx2word = {v:k for k,v in word2idx.items()}
    
idx2word[0] = "PAD"
word2idx['PAD'] = 0
vocab_size = len(word2idx)+1
print(vocab_size)



# encoding and decoding functions
def encode(sentence, maxlen,vocab_size):
    indices = np.zeros((maxlen, vocab_size))

    for i, w in enumerate(nltk.word_tokenize(sentence)):
        if i == maxlen: break
        if w in word2idx.keys():
            indices[i, word2idx[w]] = 1
        if w not in word2idx.keys() :
            indices[i, word2idx['PAD']] = 1
    return indices

def decode(indices, calc_argmax=True):
    if calc_argmax:
        indices = np.argmax(indices, axis=-1)

    return ' '.join(idx2word[x] for x in indices)


question_maxlen = 20

answer_maxlen = 20

def create_questions(question_maxlen,vocab_size,Questions):
    question_idx = np.zeros(shape=(len(Questions),question_maxlen, vocab_size))
    for q in range(len(Questions)):
        question = encode(Questions[q],question_maxlen,vocab_size)
        question_idx[q] = question
    return question_idx



def create_answers(answer_maxlen,vocab_size,Answers):
    answer_idx = np.zeros(shape=(len(Answers),answer_maxlen, vocab_size))
    for q in range(len(Answers)):
        answer = encode(Answers[q],answer_maxlen,vocab_size)
        answer_idx[q] = answer
    return answer_idx

def create_lists_qa(N_t,df_train,nb=10):
    Questions = []
    Answers = []
    for i in range(N_t):
        dial = df_train['context'][i*20].split(',')
        q = dial[(len(dial)-nb):len(dial)]
        q = ' '.join([w for w in q if w.isalpha()==True])
    
        a = df_train['utt'][i*20].split(',')
        a = ' '.join([w for w in a if w.isalpha()==True])
        Questions.append(q)
        Answers.append(a)
    return Questions, Answers

Questions, Answers = create_lists_qa(500,df_train)


quesns_train = create_questions(question_maxlen=question_maxlen, vocab_size=vocab_size,Questions = Questions)
answs_train = create_answers(answer_maxlen=answer_maxlen,vocab_size= vocab_size,Answers = Answers)

from keras.layers import Input,Dense,Activation
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import RepeatVector, TimeDistributed, ActivityRegularization

n_hidden = 128
question_layer = Input(shape=(question_maxlen,vocab_size))
encoder_rnn = LSTM(n_hidden,dropout=0.2,recurrent_dropout=0.2) (question_layer)
repeat_encode = RepeatVector(answer_maxlen)(encoder_rnn)
dense_layer = TimeDistributed(Dense(vocab_size))(repeat_encode)
regularized_layer = ActivityRegularization(l2=1)(dense_layer)
softmax_layer = Activation('softmax')(regularized_layer)
model = Model([question_layer],[softmax_layer])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print (model.summary())

# Model Training
quesns_train_2 = quesns_train.astype('float32')
answs_train_2 = answs_train.astype('float32')
model.fit(quesns_train_2, answs_train_2,batch_size=32,epochs=30, validation_split=0.05)
# accuracy 0 : does not work :'(


# Model prediction : bad
ans_pred = model.predict(quesns_train_2[0:10])
Questions[0:10]
Answers[0:10]
for i in range(10):
    print (decode(ans_pred[i]))  ## bad
