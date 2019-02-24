# Data Path
import os
#PATH_TO_DATA = "C:/Users/Sophie HU/Desktop/CentraleSupelec/NLP/HW2/exercise2/data"
PATH_TO_DATA ="C:\\Users\\Sophie HU\\Desktop\\CentraleSupelec\\NLP\\HW2\\exercise2\\data"
#PATH_TO_DATA = "C:/Users/Admin/Documents/Centrale Paris/3A/OMA/NLP/Exo 2/exercise2/exercise2/data/"

dev_data = os.path.join(PATH_TO_DATA, 'devdata.csv')
train_data = os.path.join(PATH_TO_DATA, 'traindata.csv')

import pandas as pd

import spacy
#conda install -c conda-forge spacy
# python -m spacy download en
nlp = spacy.load('en')
#nlp = spacy.load('C:/Users/Admin/Anaconda3/envs/py35/lib/site-packages/en_core_web_sm/en_core_web_sm-2.0.0')


from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical



df = pd.read_csv(train_data, sep='delimiter', header=None,names = "r")
df = pd.DataFrame(df.r.str.split('\t',expand=True))
df.columns = ['sentiment','aspect_category','aspect_terms','location_asp_terms','review']
# unique_words_reviews = len(set(df.review.str.cat(sep=' ').split(' ')))
unique_words_aspect_terms = set(df.aspect_terms.str.cat(sep=' ').split(' '))
n_layer_aspect_terms= len(unique_words_aspect_terms)

df_dev = pd.read_csv(dev_data, sep='delimiter', header=None,names = "r")
df_dev = pd.DataFrame(df_dev.r.str.split('\t',expand=True))
df_dev.columns = ['sentiment','aspect_category','aspect_terms','location_asp_terms','review']

#unique_words = set(sentences.split(' '))
'''
https://remicnrd.github.io/Aspect-based-sentiment-analysis/

'''

def train(trainfile,max_vocab_size=6000,sentiment_terms=False):
    """Trains the classifier model on the training set stored in file trainfile"""
    # read trainfile ='patch'+'traindata.csv'
    #logistic regression, svm, deep learning method
    
    #0. load data and lower case on reviews
    df = pd.read_csv(train_data, sep='delimiter', header=None,names = "r")
    df = pd.DataFrame(df.r.str.split('\t',expand=True))
    df.columns = ['sentiment','aspect_category','aspect_terms','location_asp_terms','review']
    df.review = df.review.str.lower()
    
    #count words
    # unique_words_reviews = len(set(df.review.str.cat(sep=' ').split(' ')))
    unique_words_aspect_terms = set(df.aspect_terms.str.cat(sep=' ').split(' '))
    n_layer_aspect_terms= len(unique_words_aspect_terms)
    
    
    # get sentiment_terms in review can improve model accuracy
    if sentiment_terms==False :
        #1. Build the Aspect Categories Model
        aspect_categories_model = Sequential()
        # 512 is the shape of word size according to the website, we need to count words ourself
        aspect_categories_model.add(Dense(n_layer_aspect_terms, input_shape=(6000,), activation='relu'))
        #we have 12 aspects, so we want 12 as size of output
        aspect_categories_model.add(Dropout(0.5))
        aspect_categories_model.add(Dense(12, activation='softmax'))
        aspect_categories_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        #2. Transform words into vectors with word embedding in order to feed FC model
        # Bog of Words Word Embedding
        tokenizer = Tokenizer(num_words=max_vocab_size)
        tokenizer.fit_on_texts(df.review)
        aspect_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(df.aspect_terms))
        
        #3. encode the aspect_category to dummy (binary) variables using Sklearn and Keras
        label_encoder = LabelEncoder()
        integer_category = label_encoder.fit_transform(df.aspect_category)
        dummy_category = to_categorical(integer_category)
        
        aspect_categories_model.fit(aspect_tokenized, dummy_category, epochs=5, verbose=1)
        
        return aspect_categories_model
        
    else:
        sentiment_terms = []
        for review in nlp.pipe(df['review']):
            if review.is_parsed:
                sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
            else:
                sentiment_terms.append('') 
        df['sentiment_terms'] = sentiment_terms
        
        #. Build Sentiment Model
        sentiment_model = Sequential()
        sentiment_model.add(Dense(n_layer_aspect_terms, input_shape=(6000,), activation='relu'))
        sentiment_model.add(Dense(3, activation='softmax'))
        sentiment_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #3. encode sentiment
        tokenizer = Tokenizer(num_words=max_vocab_size)
        tokenizer.fit_on_texts(df.sentiment_terms)
        sentiment_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(df.sentiment_terms))
        label_encoder_2 = LabelEncoder()
        integer_sentiment = label_encoder_2.fit_transform(df.sentiment)
        dummy_sentiment = to_categorical(integer_sentiment)

        sentiment_model.fit(sentiment_tokenized, dummy_sentiment, epochs=5, verbose=1)
        
        return sentiment_model
        
        '''test, to delete'''
#        dev_sentiment_terms = []
#        for review in nlp.pipe(df_dev['review']):
#            if review.is_parsed:
#                dev_sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
#            else:
#                dev_sentiment_terms.append('') 
#        df_dev['sentiment_terms']=dev_sentiment_terms
#        
#        tokenizer = Tokenizer(num_words=6000)
#        tokenizer.fit_on_texts(df_dev.sentiment_terms)
#        dev_sentiment_terms = pd.DataFrame(tokenizer.texts_to_matrix(df_dev.sentiment_terms))
#        
#        label_encoder_2 = LabelEncoder()
#        prediction = sentiment_model.predict_classes(dev_sentiment_terms)
#        dev_sentiment = label_encoder_2.inverse_transform(prediction)
#        return dev_sentiment
        #dev_sentiment = label_encoder_2.inverse_transform(sentiment_model.predict_classes(dev_sentiment_terms))
        '''test, to delete'''
        
    
model_aspect = train(train_data,6000,False)
model_sentiment = train(train_data,6000,True)





#'''to delet'''
#dev_sentiment_terms = []
#for review in nlp.pipe(df_dev['review']):
#    if review.is_parsed:
#        dev_sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
#    else:
#        dev_sentiment_terms.append('') 
#df_dev['sentiment_terms']=dev_sentiment_terms
#
#tokenizer = Tokenizer(num_words=6000)
#tokenizer.fit_on_texts(df_dev.sentiment_terms)
#dev_sentiment_terms = pd.DataFrame(tokenizer.texts_to_matrix(df_dev.sentiment_terms))
#
#label_encoder_2 = LabelEncoder()
#dev_sentiment = label_encoder_2.inverse_transform(model_sentiment.predict_classes(dev_sentiment_terms))
#'''to delet'''

def predict(dev_file,model,max_vocab_size=6000,sentiment_terms=False):
    df_dev = pd.read_csv(dev_data, sep='delimiter', header=None,names = "r")
    df_dev = pd.DataFrame(df_dev.r.str.split('\t',expand=True))
    df_dev.columns = ['sentiment','aspect_category','aspect_terms','location_asp_terms','review']

    if sentiment_terms==False:
        print("sentiment false")
        tokenizer = Tokenizer(num_words=max_vocab_size)
        tokenizer.fit_on_texts(df_dev.review)
        dev_aspect_terms = pd.DataFrame(tokenizer.texts_to_matrix(df_dev.aspect_terms))
        label_encoder = LabelEncoder()
        integer_sentiment = label_encoder.fit_transform(df_dev.aspect_terms)
        print("Model Prediction")
        dev_aspect_categories = label_encoder.inverse_transform(model.predict_classes(dev_aspect_terms))
        
        
    else:
        # sentiment
        dev_sentiment_terms = []
        for review in nlp.pipe(df_dev['review']):
            if review.is_parsed:
                dev_sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
            else:
                dev_sentiment_terms.append('') 
        df_dev['sentiment_terms']=dev_sentiment_terms
        
        tokenizer = Tokenizer(num_words=max_vocab_size)
        tokenizer.fit_on_texts(df_dev.sentiment_terms)
        dev_sentiment_terms = pd.DataFrame(tokenizer.texts_to_matrix(df_dev.sentiment_terms))
        
        label_encoder_2 = LabelEncoder()
        integer_sentiment = label_encoder_2.fit_transform(df_dev.sentiment)
        print("Model Prediction")
        dev_sentiment = label_encoder_2.inverse_transform(model.predict_classes(dev_sentiment_terms))
        
predict(dev_data,model_aspect,6000,False)

predict(dev_data,model_sentiment,6000,True)


#
class Classifier:
    """The Classifier"""
    def __init__(self,max_vocab_size = 60000):    
        self.max_vocab_size=max_vocab_size# randomly set a maximum size for the vocabulary

    #############################################
    def train(trainfile,max_vocab_size=6000,sentiment_terms=False):
    """Trains the classifier model on the training set stored in file trainfile"""
    # read trainfile ='patch'+'traindata.csv'
    #logistic regression, svm, deep learning method
    
    #0. load data and lower case on reviews
    df = pd.read_csv(train_data, sep='delimiter', header=None,names = "r")
    df = pd.DataFrame(df.r.str.split('\t',expand=True))
    df.columns = ['sentiment','aspect_category','aspect_terms','location_asp_terms','review']
    df.review = df.review.str.lower()
    
    #count words
    # unique_words_reviews = len(set(df.review.str.cat(sep=' ').split(' ')))
    unique_words_aspect_terms = set(df.aspect_terms.str.cat(sep=' ').split(' '))
    n_layer_aspect_terms= len(unique_words_aspect_terms)
    
    
    # get sentiment_terms in review can improve model accuracy
    if sentiment_terms==False :
        #1. Build the Aspect Categories Model
        aspect_categories_model = Sequential()
        # 512 is the shape of word size according to the website, we need to count words ourself
        aspect_categories_model.add(Dense(n_layer_aspect_terms, input_shape=(6000,), activation='relu'))
        #we have 12 aspects, so we want 12 as size of output
        aspect_categories_model.add(Dropout(0.5))
        aspect_categories_model.add(Dense(12, activation='softmax'))
        aspect_categories_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        #2. Transform words into vectors with word embedding in order to feed FC model
        # Bog of Words Word Embedding
        tokenizer = Tokenizer(num_words=max_vocab_size)
        tokenizer.fit_on_texts(df.review)
        aspect_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(df.aspect_terms))
        
        #3. encode the aspect_category to dummy (binary) variables using Sklearn and Keras
        label_encoder = LabelEncoder()
        integer_category = label_encoder.fit_transform(df.aspect_category)
        dummy_category = to_categorical(integer_category)
        
        aspect_categories_model.fit(aspect_tokenized, dummy_category, epochs=5, verbose=1)
        
    else:
        sentiment_terms = []
        for review in nlp.pipe(df['review']):
            if review.is_parsed:
                sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
            else:
                sentiment_terms.append('') 
        df['sentiment_terms'] = sentiment_terms
        
        #. Build Sentiment Model
        sentiment_model = Sequential()
        sentiment_model.add(Dense(n_layer_aspect_terms, input_shape=(6000,), activation='relu'))
        sentiment_model.add(Dense(3, activation='softmax'))
        sentiment_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #3. encode sentiment
        tokenizer = Tokenizer(num_words=max_vocab_size)
        tokenizer.fit_on_texts(df.sentiment_terms)
        sentiment_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(df.sentiment_terms))
        label_encoder_2 = LabelEncoder()
        integer_sentiment = label_encoder_2.fit_transform(df.sentiment)
        dummy_sentiment = to_categorical(integer_sentiment)


        integer_category = label_encoder_2.fit_transform(df.sentiment)
        sentiment_model.fit(sentiment_tokenized, dummy_sentiment, epochs=5, verbose=1)
        
        return sentiment_model

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        




