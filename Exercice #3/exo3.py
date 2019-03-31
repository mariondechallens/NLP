# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 09:27:11 2019

@author: Admin
"""
import pandas as pd
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



def text2sentences2(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append( l.lower() )
    return sentences

train = text2sentences2(rep+'train_none_original.txt')
s2 = train[5]
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
        

def prep_response(data):
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

d = prep_response(train2[0:100])   
res = []    
for sent in d:
    idx = response(sent)
    if (idx == 19): #correct answer
        res.append(True)
    else:
        res.append(False)
 
sum(res)/len(d) #bad : besoin du contexte pour la réponse
## try with context
train_c = text2sentences2(rep+'train_both_original.txt')

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


all_rep = []
for i in range(2):
    dial = train_c[(15*i):((i+1)*15)]
    n_d = len(dial)
    cont = dial[0:8]
    cont = cleaning(cont)
    
    for j in range(8,n_d):
        distr = dial[j].split("\t")
        cor = distr[1]
        utt = distr[0]
        answers = distr[3].split("|")     
        cont = add_cont(cont,remove_digit(utt))
        resp  = []
        for ans in answers :
            resp.append(ans)
        n = len(resp[-1])
        resp[-1] = resp[-1][0:(n-2)]
        resp.append(cont)
        cont = add_cont(cont,cor)
        all_rep.append(resp)
            
res = []    
for sent in all_rep:
    idx = response(sent)
    if (idx == 19): #correct answer
        res.append(True)
    else:
        res.append(False)
 
sum(res)/len(d) #encore pire avec le contexte ??


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
        
     
# choosing between utterances
# source : https://medium.com/analytics-vidhya/building-a-simple-chatbot-in-python-using-nltk-7c8c8215ac6e
        
# generating utterances 
# https://lizadaly.com/brobot/    
        
# Sentences we'll respond with if the user greeted us

import logging        
from textblob import TextBlob
from config import FILTER_WORDS

logging.basicConfig()

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)        
        
        
GREETING_KEYWORDS = ("hello", "hi", "greetings", "sup", "what's up",)

GREETING_RESPONSES = ["'sup bro", "hey", "*nods*", "hey you get my snap?"]

def check_for_greeting(sentence):
    """If any of the words in the user's input was a greeting, return a greeting response"""
    for word in sentence.words:
        if word.lower() in GREETING_KEYWORDS:
            return random.choice(GREETING_RESPONSES)        
        




def find_verb(sent):

    """Pick a candidate verb for the sentence."""

    verb = None

    pos = None

    for word, part_of_speech in sent.pos_tags:

        if part_of_speech.startswith('VB'):  # This is a verb

            verb = word

            pos = part_of_speech

            break

    return verb, pos


def find_noun(sent):

    """Given a sentence, find the best candidate noun."""

    noun = None



    if not noun:

        for w, p in sent.pos_tags:

            if p == 'NN':  # This is a noun

                noun = w

                break

    if noun:

        logger.info("Found noun: %s", noun)



    return noun



def find_adjective(sent):

    """Given a sentence, find the best candidate adjective."""

    adj = None

    for w, p in sent.pos_tags:

        if p == 'JJ':  # This is an adjective

            adj = w

            break

    return adj

def find_pronoun(sent):
    """Given a sentence, find a preferred pronoun to respond with. Returns None if no candidate
    pronoun is found in the input"""
    pronoun = None

    for word, part_of_speech in sent.pos_tags:
        # Disambiguate pronouns
        if part_of_speech == 'PRP' and word.lower() == 'you':
            pronoun = 'I'
        elif part_of_speech == 'PRP' and word == 'I':
            # If the user mentioned themselves, then they will definitely be the pronoun
            pronoun = 'You'
    return pronoun

def check_for_comment_about_bot(pronoun, noun, adjective):
    """Check if the user's input was about the bot itself, in which case try to fashion a response
    that feels right based on their input. Returns the new best sentence, or None."""
    resp = None
    if pronoun == 'I' and (noun or adjective):
        if noun:
            if random.choice((True, False)):
                resp = random.choice(SELF_VERBS_WITH_NOUN_CAPS_PLURAL).format(**{'noun': noun.pluralize().capitalize()})
            else:
                resp = random.choice(SELF_VERBS_WITH_NOUN_LOWER).format(**{'noun': noun})
        else:
            resp = random.choice(SELF_VERBS_WITH_ADJECTIVE).format(**{'adjective': adjective})
    return resp

# Template for responses that include a direct noun which is indefinite/uncountable
SELF_VERBS_WITH_NOUN_CAPS_PLURAL = [
    "My last startup totally crushed the {noun} vertical",
    "Were you aware I was a serial entrepreneur in the {noun} sector?",
    "My startup is Uber for {noun}",
    "I really consider myself an expert on {noun}",
]

SELF_VERBS_WITH_NOUN_LOWER = [
    "Yeah but I know a lot about {noun}",
    "My bros always ask me about {noun}",
]

SELF_VERBS_WITH_ADJECTIVE = [
    "I'm personally building the {adjective} Economy",
    "I consider myself to be a {adjective}preneur",
]
    
NONE_RESPONSES = [

    "uh whatever",

    "meet me at the foosball table, bro?",

    "code hard bro",

    "want to bro down and crush code?",

    "I'd like to add you to my professional network on LinkedIn",

    "Have you closed your seed round, dog?",

]

# end



# start:example-self.py

# If the user tries to tell us something about ourselves, use one of these responses

COMMENTS_ABOUT_SELF = [

    "You're just jealous",

    "I worked really hard on that",

    "My Klout score is {}".format(random.randint(100, 500)),

]

class UnacceptableUtteranceException(Exception):

    """Raise this (uncaught) exception if the response was going to trigger our blacklist"""

    pass



def starts_with_vowel(word):

    """Check for pronoun compability -- 'a' vs. 'an'"""

    return True if word[0] in 'aeiou' else False

def broback(sentence):

    """Main program loop: select a response for the input sentence and return it"""

    logger.info("Broback: respond to %s", sentence)

    resp = respond(sentence)

    return resp

def preprocess_text(sentence):

    """Handle some weird edge cases in parsing, like 'i' needing to be capitalized

    to be correctly identified as a pronoun"""

    cleaned = []

    words = sentence.split(' ')

    for w in words:

        if w == 'i':

            w = 'I'

        if w == "i'm":

            w = "I'm"

        cleaned.append(w)



    return ' '.join(cleaned)

 
 
def respond(sentence):
    """Parse the user's inbound sentence and find candidate terms that make up a best-fit response"""
    cleaned = preprocess_text(sentence)
    parsed = TextBlob(cleaned)

    # Loop through all the sentences, if more than one. This will help extract the most relevant
    # response text even across multiple sentences (for example if there was no obvious direct noun
    # in one sentence
    pronoun, noun, adjective, verb = find_candidate_parts_of_speech(parsed)

    # If we said something about the bot and used some kind of direct noun, construct the
    # sentence around that, discarding the other candidates
    resp = check_for_comment_about_bot(pronoun, noun, adjective)

    # If we just greeted the bot, we'll use a return greeting
    if not resp:
        resp = check_for_greeting(parsed)

    if not resp:
        # If we didn't override the final sentence, try to construct a new one:
        if not pronoun:
            resp = random.choice(NONE_RESPONSES)
        elif pronoun == 'I' and not verb:
            resp = random.choice(COMMENTS_ABOUT_SELF)
        else:
            resp = construct_response(pronoun, noun, verb)

    # If we got through all that with nothing, use a random response
    if not resp:
        resp = random.choice(NONE_RESPONSES)

    logger.info("Returning phrase '%s'", resp)
    # Check that we're not going to say anything obviously offensive
    filter_response(resp)

    return resp
        
def construct_response(pronoun, noun, verb):
    """No special cases matched, so we're going to try to construct a full sentence that uses as much
    of the user's input as possible"""
    resp = []

    if pronoun:
        resp.append(pronoun)

    # We always respond in the present tense, and the pronoun will always either be a passthrough
    # from the user, or 'you' or 'I', in which case we might need to change the tense for some
    # irregular verbs.
    if verb:
        verb_word = verb[0]
        if verb_word in ('be', 'am', 'is', "'m"):  # This would be an excellent place to use lemmas!
            if pronoun.lower() == 'you':
                # The bot will always tell the person they aren't whatever they said they were
                resp.append("aren't really")
            else:
                resp.append(verb_word)
    if noun:
        pronoun = "an" if starts_with_vowel(noun) else "a"
        resp.append(pronoun + " " + noun)

    resp.append(random.choice(("tho", "bro", "lol", "bruh", "smh", "")))

    return " ".join(resp)

def filter_response(resp):
    """Don't allow any words to match our filter list"""
    tokenized = resp.split(' ')
    for word in tokenized:
        if '@' in word or '#' in word or '!' in word:
            raise UnacceptableUtteranceException()
        for s in FILTER_WORDS:
            if word.lower().startswith(s):
                raise UnacceptableUtteranceException()
       
        
        