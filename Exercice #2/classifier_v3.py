
import pandas as pd
import spacy
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

nlp = spacy.load('C:/Users/Admin/Anaconda3/envs/py35/lib/site-packages/en_core_web_sm/en_core_web_sm-2.0.0')

PATH_TO_DATA = "C:/Users/Admin/Documents/Centrale Paris/3A/OMA/NLP/Exo 2/exercise2/exercise2/data/"

#dev = pd.read_csv(PATH_TO_DATA + 'devdata.csv',sep='\t',header=None)
#train = pd.read_csv(PATH_TO_DATA + 'traindata.csv',sep='\t',header = None)

def clean_data(data):
    data = data.loc[:, [0, 1, 4,2]]
    data = data.rename(index=str, columns={ 0: "sentiment", 1: "aspect_category", 4: "review", 2: "aspect_term"})
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
    def __init__(self,vocab_size = 6000,epoch = 5):  #60000?  
        self.vocab_size=vocab_size# randomly set a maximum size for the vocabulary
        sentiment_model = Sequential()
        sentiment_model.add(Dense(512, input_shape=(self.vocab_size,), activation='relu'))
        sentiment_model.add(Dense(3, activation='softmax'))
        sentiment_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = sentiment_model
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        self.label_encoder = LabelEncoder()
        self.epoch = epoch
    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        #preprocessing data
        train_data = clean_data(pd.read_csv(trainfile,sep='\t',header = None))
        
        #tokenization
        self.tokenizer.fit_on_texts(train_data.review)
        sentiment_tokenized = pd.DataFrame(self.tokenizer.texts_to_matrix(train_data.sentiment_terms))
        
        #labels
        integer_sentiment =self.label_encoder.fit_transform(train_data.sentiment)
        cat_sentiment = to_categorical(integer_sentiment)


        self.model.fit(sentiment_tokenized, cat_sentiment, epochs=self.epoch, verbose=1)


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        dev_data = clean_data(pd.read_csv(datafile,sep='\t',header = None))
        
        sentiment_tokenized = pd.DataFrame(self.tokenizer.texts_to_matrix(dev_data.sentiment_terms))
        predicted_sentiment = self.label_encoder.inverse_transform(self.model.predict_classes(sentiment_tokenized)) 
        
        return predicted_sentiment

"""
classif = Classifier()
classif.train(PATH_TO_DATA + 'traindata.csv')
pred = classif.predict(PATH_TO_DATA + 'devdata.csv')

dev = clean_data(pd.read_csv(PATH_TO_DATA + 'devdata.csv',sep='\t',header=None))

sum(pred==dev['sentiment'])/len(pred) """

#améliorations : ajouter des couches de convolution2D?
# source : https://remicnrd.github.io/Aspect-based-sentiment-analysis/