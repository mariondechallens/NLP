
import sklearn as skl
import keras
import pandas as pd
import nltk 
import spacy 

# 1 - Load train/dev sets 
PATH_TO_DATA = "C:/Users/Admin/Documents/Centrale Paris/3A/OMA/NLP/Exo 2/exercise2/exercise2/data/"

dev = pd.read_csv(PATH_TO_DATA + 'devdata.csv',sep='\t',header=None)
train = pd.read_csv(PATH_TO_DATA + 'traindata.csv',sep='\t',header = None)

train = train.loc[:, [0, 1, 4,2]]
train = train.rename(index=str, columns={ 0: "sentiment", 1: "aspect_category", 4: "review", 2: "aspect_term"})
#dataset = dataset.rename(index=str, columns={ 0: "sentiment", 1: "aspect_category", 2: "review"})
train.head(5)

dev = dev.loc[:, [0, 1, 4,2]]
dev = dev.rename(index=str, columns={ 0: "sentiment", 1: "aspect_category", 4: "review", 2: "aspect_term"})
#dataset = dataset.rename(index=str, columns={ 0: "sentiment", 1: "aspect_category", 2: "review"})
dev.head(5)

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation

aspect_categories_model = Sequential()
aspect_categories_model.add(Dense(512, input_shape=(6000,), activation='relu'))
aspect_categories_model.add(Dense(12, activation='softmax'))
aspect_categories_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.preprocessing.text import Tokenizer

vocab_size = 6000 # We set a maximum size for the vocabulary
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train.review)
aspect_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(train.aspect_term))
aspect_tokenized_dev = pd.DataFrame(tokenizer.texts_to_matrix(dev.aspect_term))

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

label_encoder = LabelEncoder()
integer_category = label_encoder.fit_transform(train.aspect_category)
dummy_category = to_categorical(integer_category)

aspect_categories_model.fit(aspect_tokenized, dummy_category, epochs=5, verbose=1)

new_review_category = label_encoder.inverse_transform(aspect_categories_model.predict_classes(aspect_tokenized_dev))

sentiment_terms = []

nlp = spacy.load('C:/Users/Admin/Anaconda3/envs/py35/lib/site-packages/en_core_web_sm/en_core_web_sm-2.0.0')
for review in nlp.pipe(train['review']):
        if review.is_parsed:
            sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
        else:
            sentiment_terms.append('')  
train['sentiment_terms'] = sentiment_terms
train.head(10)

sentiment_model = Sequential()
sentiment_model.add(Dense(512, input_shape=(6000,), activation='relu'))
sentiment_model.add(Dense(3, activation='softmax'))
sentiment_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

sentiment_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(train.sentiment_terms))

label_encoder_2 = LabelEncoder()
integer_sentiment = label_encoder_2.fit_transform(train.sentiment)
dummy_sentiment = to_categorical(integer_sentiment)

sentiment_model.fit(sentiment_tokenized, dummy_sentiment, epochs=5, verbose=1)

new_review_sentiment = label_encoder_2.inverse_transform(sentiment_model.predict_classes(aspect_tokenized_dev)) 
    
    
class Classifier:
    """The Classifier"""


    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """





