# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 09:27:11 2019

@author: Admin
"""
import numpy as np
import nltk
import string
import random
data = 'C:/Users/Admin/Documents/Centrale Paris/3A/OMA/NLP/Exo 3/convai2_fix_723.tar'

'''
import tarfile
tar = tarfile.open(data)
tar.extractall()
'''
rep = 'C:/Users/Admin/'
def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append( l.lower().split() )
    return sentences

#train = text2sentences(rep+'train_none_original.txt')
#true = text2sentences(rep+'train_none_original_no_cands.txt')


def text2sentences2(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append( l.lower() )
    return sentences

train2 = text2sentences2(rep+'train_none_original.txt')
s2 = train2[5]
s3 = s2.split("\t")

utt = s3[0]
cor = s3[1]
answers = s3[3].split("|") #correct one at the end

lemmer = nltk.stem.WordNetLemmatizer()

#nltk.download('wordnet')
#nltk.download('punkt')

#sent_tokens = nltk.sent_tokenize(utt)# converts to list of sentences 
#word_tokens = nltk.word_tokenize(utt)# converts to list of words

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

utt_t  = []
for ans in answers :
    utt_t.append(ans)
n = len(utt_t[-1])
utt_t[-1] = utt_t[-1][0:(n-2)] #remove /n
utt_t.append(utt[1:len(utt)]) #utterance of user at last
user_response = utt_t 

corpus = ['This is the first document.','This document is the second document.','this is the third document.']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
X.shape
vals = cosine_similarity(X[-1], X)
idx=vals.argsort()[0][-2] #highest similarity except one (the last one)
flat = vals.flatten()
flat.sort()
req_tfidf = flat[-2]

def response(user_response):
    robo_response=''
    #sent_tokens.append(user_response)
    #TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize)
    tfidf = TfidfVec.fit_transform(user_response)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return 55
    else:
        robo_response = robo_response+utt_t[idx]
        return idx

# la phrase correcte et est l'avant dernière phrase de utt_t
# i.e. idx = len(utt_t) -2 : 19
        
##besoin du contexte pour la réponse
def user_response(data):
    all_resp = []
    for i_dial in range(len(data)):
        dial = data[i_dial]
        dial = dial.split("\t")
        utt = dial[0]
        answers = dial[3].split("|") 
        resp  = []
        
        for ans in answers :
            resp.append(ans)
            
        n = len(resp[-1])
        resp[-1] = resp[-1][0:(n-2)] #remove /n
        resp.append(utt[1:len(utt)]) #utterance of user at last
        all_resp.append(resp)
    return all_resp

d = user_response(train2[0:100])   
res = []    
for sent in d:
    idx = response(sent)
    if (idx == 19): #correct answer
        res.append(True)
    else:
        res.append(False)
 
sum(res)/100 #bad
res[5]

flag=True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
            else:
                print("ROBO: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("ROBO: Bye! take care..")
        
     

# source : https://medium.com/analytics-vidhya/building-a-simple-chatbot-in-python-using-nltk-7c8c8215ac6e