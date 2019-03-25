# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:20:55 2019

@author: Admin
"""
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

train = text2sentences2(rep+'train_none_original.txt')

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

t = train[0].split('\t')
tt = t[3].split('|')
l = []
l.append(t[0])
l.append(t[1])
l.append(t[3].split('|'))
tt.append(t[0])
tt.append(t[1])
c = dataprocessing(tt)
cc = ' '.join(c)

def cleaning(data):
    l = []
    for i in range(len(data)):
        dial = train[i].split('\t')
        ans = dial[3].split('|')
        ans.append(dial[0])
        ans.append(dial[1])
        ans = dataprocessing(ans)
        l.append(' '.join(ans))
    return l
N = 100
train2 = cleaning(train[0:N])
# source : https://hub.packtpub.com/build-generative-chatbot-using-recurrent-neural-networks-lstm-rnns/
# Creating Vocabulary
import numpy as np
import nltk
import collections
counter = collections.Counter()
for i in range(len(train2)):
    for word in nltk.word_tokenize(train2[i]): # to do : cleaning of train[i] before !
        if word.isalpha() == True :
            counter[word]+=1
            word2idx = {w:(i+1) for i,(w,_) in enumerate(counter.most_common())}
            idx2word = {v:k for k,v in word2idx.items()}
    
idx2word[0] = "PAD"
vocab_size = len(word2idx)+1
print(vocab_size)



# encoding and decoding functions
def encode(sentence, maxlen,vocab_size):
    indices = np.zeros((maxlen, vocab_size))

    for i, w in enumerate(nltk.word_tokenize(sentence)):
        if i == maxlen: break
        indices[i, word2idx[w]] = 1
    return indices

def decode(indices, calc_argmax=True):
    if calc_argmax:
        indices = np.argmax(indices, axis=-1)

    return ' '.join(idx2word[x] for x in indices)


question_maxlen = 10

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

Questions = []
Answers = []
for i in range(100):
    dial = train[i].split("\t")
    qa = dataprocessing(dial[0:2])
    q = qa[0].replace(',', ' ')
    a = qa[1].replace(',', ' ')
    Questions.append(' '.join([w for w in q.split() if w.isalpha()==True]))
    Answers.append(' '.join([w for w in a.split() if w.isalpha()==True]))


quesns_train = create_questions(question_maxlen=question_maxlen, vocab_size=vocab_size,Questions = Questions)
answs_train = create_answers(answer_maxlen=answer_maxlen,vocab_size= vocab_size,Answers = Answers)

from keras.layers import Input,Dense,Dropout,Activation
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
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


# Model prediction
ans_pred = model.predict(quesns_train_2[0:3])
print (decode(ans_pred[0]))
print (decode(ans_pred[1]))
