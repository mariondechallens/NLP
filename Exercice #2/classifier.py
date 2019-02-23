
import sklearn as skl
import keras
import pandas as pd
import nltk 
import spacy 

# 1 - Load train/dev sets 
PATH_TO_DATA = "C:/Users/Admin/Documents/Centrale Paris/3A/OMA/NLP/Exo 2/exercise2/exercise2/data/"
with open(PATH_TO_DATA + 'traindata.csv') as fd:
    train = fd.read().splitlines()

    
with open(PATH_TO_DATA + 'devdata.csv') as fd:
    dev = fd.read().splitlines()

#one hot encoding :  absence/presence of a word in the vocabulary

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)

cv.fit(train)

X = cv.transform(train)

X_dev = cv.transform(dev)
    
    
    
class Classifier:
    """The Classifier"""


    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """





