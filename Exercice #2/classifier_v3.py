
import pandas as pd
import spacy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Flatten, Convolution1D

from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

#Attention : cette ligne doit pouvoir être lancée de l'ordi de la prof !
nlp = spacy.load('C:/Users/Admin/Anaconda3/envs/py35/lib/site-packages/en_core_web_sm/en_core_web_sm-2.0.0')

PATH_TO_DATA = "C:/Users/Admin/Documents/Centrale Paris/3A/OMA/NLP/Exo 2/exercise2/exercise2/data/"

#dev = pd.read_csv(PATH_TO_DATA + 'devdata.csv',sep='\t',header=None)
#train = pd.read_csv(PATH_TO_DATA + 'traindata.csv',sep='\t',header = None)

#preprocessing of data : extracting the sentiment, the category, the review, the aspect_term
# from the review, extracting the sentiment terms with the library spacy
def clean_data(data):
    data = data.loc[:, [0, 2, 4]]
    data = data.rename(index=str, columns={ 0: "opinion", 2:"aspect_term",4: "sentence"})
    data = context_words_sentences(data)
    opinion_words = []
    for sent in nlp.pipe(data['sentence']):
        if sent.is_parsed:
            opinion_words.append(' '.join([token.lemma_ for token in sent if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
        else:
            opinion_words.append('')  
    data['opinion_words'] = opinion_words

    return data


#dev = clean_data(dev)
#train = clean_data(train)

def context_words(index,window_size,data) :   
    s = data['sentence'][index]
    ss = s.lower().split(' ')
    try :
        i = ss.index(data['aspect_term'][index])
    except ValueError:
        return s
    l = []
    for j in range(-window_size, window_size + 1):
        context_word_pos = i + j
                # make sure not jump out sentence
        if context_word_pos < 0 or context_word_pos >= len(ss) :
            continue
        l.append(ss[context_word_pos])
        
    l = ' '.join(l)
    return(l)
    
def context_words_sentences(data,window_size = 5):
    for i in range(len(data['sentence'])):
        data['sentence'][i] = context_words(i,window_size,data)
    return data
    
                   
            
class Classifier:
    """The Classifier"""
    def __init__(self,vocab_size = 6000,epoch = 5):  
        
        self.vocab_size=vocab_size
        '''
        #With convolutional layers
        model = Sequential()
        model.add(Embedding(self.vocab_size, self.vocab_size, input_length=self.vocab_size))
        model.add(Convolution1D(64, 3, padding='same'))
        model.add(Convolution1D(32, 3, padding='same'))
        model.add(Convolution1D(16, 3, padding='same'))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(180,activation='sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(3,activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        '''
        '''
        #With LSTM
        model = Sequential()
        model.add(Embedding(self.vocab_size, 128))
        model.add(LSTM(128,dropout=0.2, recurrent_dropout=0.2)) 
        model.add(Dense(3))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        ''' 
        
        # Simply with Dense
        model = Sequential()
        model.add(Dense(512, input_shape=(self.vocab_size,), activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.model = model
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        self.label_encoder = LabelEncoder()
        self.epoch = epoch
        
    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        #preprocessing data
        train_data = clean_data(pd.read_csv(trainfile,sep='\t',header = None))
        
        #tokenization from keras according to Bag of Words embeddings techniques
        self.tokenizer.fit_on_texts(train_data.sentence)
        opinion_embedding = pd.DataFrame(self.tokenizer.texts_to_matrix(train_data.opinion_words))
        
        #creating labels with keras
        opinion_label =self.label_encoder.fit_transform(train_data.opinion)
        opinion_cate = to_categorical(opinion_label)


        self.model.fit(opinion_embedding, opinion_cate, epochs=self.epoch, verbose=1)



    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        dev_data = clean_data(pd.read_csv(datafile,sep='\t',header = None))
        
        opinion_embedding = pd.DataFrame(self.tokenizer.texts_to_matrix(dev_data.opinion_words))
        predicted_opinion = self.label_encoder.inverse_transform(self.model.predict_classes(opinion_embedding)) 
        
        return predicted_opinion


classif = Classifier(vocab_size=6000)
classif.train(PATH_TO_DATA + 'traindata.csv')
pred = classif.predict(PATH_TO_DATA + 'devdata.csv')

dev = clean_data(pd.read_csv(PATH_TO_DATA + 'devdata.csv',sep='\t',header=None))

s = sum(pred==dev['opinion'])/len(pred)

# Améliorations : 
# Dense : accu de 0.77, très rapide avec 6000
#LSTM : long avec 300 voc, accu de 0.70, augmenter la taille des batchs ? sans batchs?
# 6000 bcp trop long
#Conv 1D : 600, rapide, 0.70
#6000 : pb de mémoire, trop gros vecteurs
#1000, assez rapide, 0.70
#3000 : sans batchs
# source : https://remicnrd.github.io/Aspect-based-sentiment-analysis/
# https://medium.com/@thoszymkowiak/how-to-implement-sentiment-analysis-using-word-embedding-and-convolutional-neural-networks-on-keras-163197aef623
# ils disent 20 minutes pour 86% sur le site