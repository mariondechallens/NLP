
import pandas as pd
import spacy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Flatten, Convolution1D

from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

nlp = spacy.load('C:/Users/Admin/Anaconda3/envs/py35/lib/site-packages/en_core_web_sm/en_core_web_sm-2.0.0')

PATH_TO_DATA = "C:/Users/Admin/Documents/Centrale Paris/3A/OMA/NLP/Exo 2/exercise2/exercise2/data/"

#dev = pd.read_csv(PATH_TO_DATA + 'devdata.csv',sep='\t',header=None)
#train = pd.read_csv(PATH_TO_DATA + 'traindata.csv',sep='\t',header = None)

#preprocessing of data : extracting the sentiment, the category, the review, the aspect_term
# from the review, extracting the sentiment terms with the library spacy
def clean_data(data):
    data = data.loc[:, [0, 4]]
    data = data.rename(index=str, columns={ 0: "sentiment", 4: "review"})
    sentiment_terms = []
    for review in nlp.pipe(data['review']):
        if review.is_parsed:
            sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
        else:
            sentiment_terms.append('')  
    data['sentiment_terms'] = sentiment_terms

    return data

#dev = clean_data(dev)
#train = clean_data(train)
   
    
class Classifier:
    """The Classifier"""
    def __init__(self,vocab_size = 6000,epoch = 5):  
        
        self.vocab_size=vocab_size
        #With convolutional layers
        sentiment_model = Sequential()
        sentiment_model.add(Embedding(self.vocab_size, self.vocab_size, input_length=self.vocab_size))
        sentiment_model.add(Convolution1D(64, 3, padding='same'))
        sentiment_model.add(Convolution1D(32, 3, padding='same'))
        sentiment_model.add(Convolution1D(16, 3, padding='same'))
        sentiment_model.add(Flatten())
        sentiment_model.add(Dropout(0.2))
        sentiment_model.add(Dense(180,activation='sigmoid'))
        sentiment_model.add(Dropout(0.2))
        sentiment_model.add(Dense(3,activation='sigmoid'))
        sentiment_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        '''
        #With LSTM
        sentiment_model = Sequential()
        sentiment_model.add(Embedding(self.vocab_size, 128))
        sentiment_model.add(LSTM(128,dropout=0.2, recurrent_dropout=0.2)) 
        sentiment_model.add(Dense(3))
        sentiment_model.add(Activation('sigmoid'))
        sentiment_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        ''' 
        '''
        # Simply with Dense
        sentiment_model = Sequential()
        sentiment_model.add(Dense(512, input_shape=(self.vocab_size,), activation='relu'))
        sentiment_model.add(Dense(3, activation='softmax'))
        sentiment_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        '''
        self.model = sentiment_model
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        self.label_encoder = LabelEncoder()
        self.epoch = epoch
        
    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        #preprocessing data
        train_data = clean_data(pd.read_csv(trainfile,sep='\t',header = None))
        
        #tokenization from keras according to Bag of Words embeddings techniques
        self.tokenizer.fit_on_texts(train_data.review)
        sentiment_tokenized = pd.DataFrame(self.tokenizer.texts_to_matrix(train_data.sentiment_terms))
        
        #creating labels with keras
        integer_sentiment =self.label_encoder.fit_transform(train_data.sentiment)
        cat_sentiment = to_categorical(integer_sentiment)


        #self.model.fit(sentiment_tokenized, cat_sentiment, epochs=self.epoch, verbose=1)
        self.model.fit(sentiment_tokenized, cat_sentiment, batch_size=50, epochs=self.epoch)


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        dev_data = clean_data(pd.read_csv(datafile,sep='\t',header = None))
        
        sentiment_tokenized = pd.DataFrame(self.tokenizer.texts_to_matrix(dev_data.sentiment_terms))
        predicted_sentiment = self.label_encoder.inverse_transform(self.model.predict_classes(sentiment_tokenized)) 
        
        return predicted_sentiment


classif = Classifier(vocab_size=1000)
classif.train(PATH_TO_DATA + 'traindata.csv')
pred = classif.predict(PATH_TO_DATA + 'devdata.csv')

dev = clean_data(pd.read_csv(PATH_TO_DATA + 'devdata.csv',sep='\t',header=None))

sum(pred==dev['sentiment'])/len(pred)

# Améliorations : 
# Dense : accu de 0.77, très rapide avec 6000
#LSTM : long avec 300 voc, accu de 0.70, augmenter la taille des batchs ? sans batchs?
# 6000 bcp trop long
#Conv 1D : 600, rapide, 0.70
#6000 : pb de mémoire
#1000, assez rapide, 0.70
# source : https://remicnrd.github.io/Aspect-based-sentiment-analysis/
# https://medium.com/@thoszymkowiak/how-to-implement-sentiment-analysis-using-word-embedding-and-convolutional-neural-networks-on-keras-163197aef623
# ils disent 20 minutes pour 86% sur le site