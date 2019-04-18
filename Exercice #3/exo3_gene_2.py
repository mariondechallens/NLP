# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 09:49:00 2019

@author: Admin
"""

# generating utterances 
# https://lizadaly.com/brobot/    
        
# Sentences we'll respond with if the user greeted us
import random
import logging        
from textblob import TextBlob

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)        
        
        
GREETING_KEYWORDS = ("hello", "hi", "greetings", "how are you doing", "what's up",)

GREETING_RESPONSES = ["Hi, i am good thanks, what's for you?", "hey","hello ! I am fine thank you and you ?", "hello, happy to see you, I am going well and you ?","hello how are are you ?"]

FILTER_WORDS = set(["skank","wetback","bitch","cunt","dick","douchebag",
"dyke","fag","nigger","tranny","trannies", "paki","pussy","retard","slut",
"titt","tits","wop","whore","chink","fatass","shemale","nigga","daygo",
"dego","dago","gook","kike","kraut","spic","twat","lesbo","homo",
"fatso", "lardass", "jap", "biatch", "tard", "gimp", "gyp", "chinaman",
 "chinamen", "golliwog", "crip","raghead","negro","hooker"])


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
    "let's have a drink soon ?",
    "see you later !",
    "let's talk about that later, I have to go",
    "I'd like to add you to my professional network on LinkedIn"
]


COMMENTS_ABOUT_SELF = [
    "You're just jealous",
    "I worked really hard on that"
]


def check_for_greeting(sentence):
    for word in sentence.words:
        if word.lower() in GREETING_KEYWORDS:
            return random.choice(GREETING_RESPONSES)        
        

def find_verb(sent):
    verb = None
    pos = None
    for word, part_of_speech in sent.pos_tags:
        if part_of_speech.startswith('VB'):  
            verb = word
            pos = part_of_speech
            break
    return verb, pos


def find_noun(sent):
    noun = None
    if not noun:
        for w, p in sent.pos_tags:
            if p == 'NN': 
                noun = w
                break
    if noun:
        logger.info("Found noun: %s", noun)
    return noun



def find_adjective(sent):
    adj = None
    for w, p in sent.pos_tags:
        if p == 'JJ':
            adj = w
            break
    return adj

def find_pronoun(sent):
    pronoun = None
    for word, part_of_speech in sent.pos_tags:
        if part_of_speech == 'PRP' and word.lower() == 'you':
            pronoun = 'I'
        elif part_of_speech == 'PRP' and word == 'I':
            pronoun = 'You'
    return pronoun

def find_candidate_parts_of_speech(parsed):
    pronoun = None
    noun = None
    adjective = None
    verb = None

    for sent in parsed.sentences:
        pronoun = find_pronoun(sent)
        noun = find_noun(sent)
        adjective = find_adjective(sent)
        verb = find_verb(sent)

    logger.info("Pronoun=%s, noun=%s, adjective=%s, verb=%s", pronoun, noun, adjective, verb)
    return pronoun, noun, adjective, verb




def check_for_comment_about_bot(pronoun, noun, adjective):
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

class UnacceptableUtteranceException(Exception):
    pass



def starts_with_vowel(word):

    return True if word[0] in 'aeiou' else False



def preprocess_text(sentence):
    cleaned = []
    words = sentence.split(' ')
    for w in words:
        if w == 'i':
            w = 'I'
        if w == "i'm":
            w = "I'm"
        cleaned.append(w)
    return ' '.join(cleaned)

def filter_response(resp):
    tokenized = resp.split(' ')
    for word in tokenized:
        if '@' in word or '#' in word or '!' in word:
            raise UnacceptableUtteranceException()
        for s in FILTER_WORDS:
            if word.lower().startswith(s):
                raise UnacceptableUtteranceException() 
 
def respond(sentence):
    cleaned = preprocess_text(sentence)
    parsed = TextBlob(cleaned)

    pronoun, noun, adjective, verb = find_candidate_parts_of_speech(parsed)
    resp = check_for_comment_about_bot(pronoun, noun, adjective)

    if not resp:
        resp = check_for_greeting(parsed)
    if not resp:
        if not pronoun:
            resp = random.choice(NONE_RESPONSES)
        elif pronoun == 'I' and not verb:
            resp = random.choice(COMMENTS_ABOUT_SELF)
        else:
            resp = construct_response(pronoun, noun, verb)

    if not resp:
        resp = random.choice(NONE_RESPONSES)

    logger.info("Returning phrase '%s'", resp)
    #filter_response(resp)
    return resp
        
def construct_response(pronoun, noun, verb):
    resp = []
    if pronoun:
        resp.append(pronoun)
    if verb:
        verb_word = verb[0]
        if verb_word in ('be', 'am', 'is', "'m"): 
            if pronoun.lower() == 'you':
                resp.append("aren't really")
            else:
                resp.append(verb_word)
    if noun:
        pronoun = "an" if starts_with_vowel(noun) else "a"
        resp.append(pronoun + " " + noun)
    return " ".join(resp)

def broback(sentence):
    logger.info("Broback: respond to %s", sentence)
    resp = respond(sentence)
    return resp
         

####### Apply to our data
import pandas as pd

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

def add_rows_train_cont(row, n,train_d):
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
            row.append({'context': cont, 'corr_utt': cor,})   
                
            cont = add_cont(cont,cor)
        
    return row

def add_rows_train_utt(row, n,train_d):
    for i in range(n):
        dial = train_d[i]
        n_d = len(dial)
        
        cont = find_cont(dial)
        n_s = len(cont)
        
                    
        for j in range(n_s,n_d):
            distr = dial[j].split("\t")
            cor = distr[1]
            utt = remove_digit(distr[0])   
            row.append({'user_utt': utt, 'corr_ans': cor})
        
    return row   


N = 100

#building train data set
row = []
row = add_rows_train_cont(row,N,list_dial)
df_train_cont = pd.DataFrame(data = row)


N=100
row2 = []
row2 = add_rows_train_utt(row2,N,list_dial)
df_train_utt = pd.DataFrame(data = row2)

l = []
for i in range(len(df_train_utt)):
    sent = df_train_utt['user_utt'][i]
    l.append(broback(sent))

df_train_utt['gene_ans'] = l  # nul

for i in range(10):
    print('correct answer')
    print(df_train_utt['corr_ans'][i])
    print('gene answer')
    print(df_train_utt['gene_ans'][i])
    