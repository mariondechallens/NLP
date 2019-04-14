__authors__ = ['Sophie Hu','Marion Favre dEchallens']
__emails__  = ['jiahui.hu@student-cs.fr','mariondechallens@gmail.com']


import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl
import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def text2sentences2(path):
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append( l.lower() )
    return sentences



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



def add_rows_train(row, n,train_d):

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

            row.append({'context': cont, 'utt': cor, 'xlabel': 1})

            answers = distr[3].split("|") 

        

            for i in range(len(answers)-1):

                d = {'context': cont, 'utt': answers[i], 'xlabel': 0}

                row.append(d)  

                

            cont = add_cont(cont,cor)

        

    return row



def add_rows_test2(row2,n,train_d): #we don't know the right utterances in test mode

    for i in range(n):

        dial = train_d[i]

        n_d = len(dial)

        

        cont = find_cont(dial)

        n_s = len(cont)

        

        cont = cleaning(cont)

       

        for j in range(n_s,n_d):

            distr = dial[j].split("\t")

            utt = distr[0]

            answers = distr[3].split("|")    

            n_a = len(answers)

            cont = add_cont(cont,remove_digit(utt))

            

            #create dictionary for row2

            list_of_key = [x for x in range(1,n_a+1)]

            list_of_key.append('context')

            

            list_of_values = [answers[x] for x in range(n_a)]

            list_of_values.append(cont)

            

            dic = dict( zip(list_of_key,list_of_values ))

            row2.append(dic)

    return row2

    



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



def stemming_test(df):

    for i in range(df.shape[1]):

        df.iloc[:,i]= dataprocessing(df.iloc[:,i])

    return df



class DialogueManager:

    def __init__(self):

        #self.vect = TfidfVectorizer(analyzer='word',ngram_range=(1,1))

        self.vect = TfidfVectorizer()



    def load(self,path):

        with open(path,'rb') as f:

            self.vect = pkl.load(f)





    def save(self,path):

        with open(path,'wb') as fout:

            pkl.dump(self.vect,fout)





    def train(self,data):

        #self.vect.fit(data)

        self.vect.fit(np.append(data.context.values,data.utt.values))



    def findBest2(self, context, utterances):

        # Convert context and utterances into tfidf vector

        vector_context = self.vect.transform([context])

        vector_doc = self.vect.transform(utterances)

        # The dot product measures the similarity of the resulting vectors

        result = np.dot(vector_doc, vector_context.T).todense()

        result = np.asarray(result).flatten()

        # Sort by top results and return the indices in descending order

        return np.argsort(result, axis=0)[::-1]

    

    def findBest(self,utterance,options):

        """

            finds the best utterance out of all those given in options

        :param utterance: a single string

        :param options: a sequence of strings

        :return: returns one of the strings of options

        """

        Xtext = [utterance] + options

        X = self.vect.transform(Xtext)

        X = normalize(X,axis=1,norm='l2')

        idx = np.argmax(X[0] * X[1:,:].T)



        return options[idx]



def loadDatatrain(path,N,list_dial):

    
    row = []

    row = add_rows_train(row,N,list_dial)

    df_train_old = pd.DataFrame(data = row)



#### Data Processing on df_train[context] and df_train[utt]

    df_train = pd.DataFrame()

    df_train['xlabel']=df_train_old['xlabel']

    df_train['context']= dataprocessing(df_train_old['context'])

    df_train['utt']= dataprocessing(df_train_old['utt'])

    

    return df_train_old, df_train



def loadDatatest(path,N,list_dial):
    row = []

    row = add_rows_test2(row,N,list_dial)

    df_test_old = pd.DataFrame(data = row)

    df_test = stemming_test(pd.DataFrame(data = row))

    

    return df_test_old, df_test



def retrieve_sentence2(y_pred,df_test): 

    l = []

    for i in range(len(y_pred)) :

        l.append([y_pred[i][0]+1,df_test.iloc[i,:df_test.shape[1]-1][y_pred[i][0]+1]])

    return l 

    

def loadData(path):

    """

        :param path: containing dialogue data of ConvAI (eg:  train_both_original.txt, valid_both_original.txt)

        :return: for each dialogue, yields (description_of_you, description_of_partner, dialogue) where a dialogue

            is a sequence of (utterance, answer, options)

    """

    with open(path) as f:

        descYou, descPartner = [], []

        dialogue = []

        for l in f:

            l=l.strip()

            lxx = l.split()

            idx = int(lxx[0])

            if idx == 1:

                if len(dialogue) != 0:

                    yield descYou,  descPartner, dialogue

                # reinit data structures

                descYou, descPartner = [], []

                dialogue = []



            if lxx[2] == 'persona:':

                # description of people involved

                if lxx[1] == 'your':

                    description = descYou

                elif lxx[1] == "partner's":

                    description = descPartner

                else:

                    assert 'Error, cannot recognize that persona ({}): {}'.format(lxx[1],l)

                description.append(lxx[3:])



            else:

                # the dialogue

                lxx = l.split('\t')

                utterance = ' '.join(lxx[0].split()[1:])

                answer = lxx[1]

                options = [o for o in lxx[-1].split('|')]

                dialogue.append( (idx, utterance, answer, options))


# test avec les données
data = 'C:/Users/Admin/Documents/Centrale Paris/3A/OMA/NLP/Exo 3/convai2_fix_723.tar'
rep = 'C:/Users/Admin/'   

dm = DialogueManager()
path_train = rep +'train_both_original.txt'
list_dial_train = sep_dial(text2sentences2(path_train))
N_train = len(list_dial_train)
df_train_old, df_train = loadDatatrain(path_train,200,list_dial_train)
dm.train(df_train)
dm.save('C:/Users/Admin/Documents/Centrale Paris/3A/OMA/NLP/Exo 3/dm.pkl')

path_test = rep + 'valid_both_original.txt'

    dm.load('C:/Users/Admin/Documents/Centrale Paris/3A/OMA/NLP/Exo 3/dm.pkl')

    df_test_old, df_test = loadDatatest(path_test,200)

    pred = [dm.findBest2(df_test.context[x], df_test.iloc[x,:df_test.shape[1]-1].values) for x in range(len(df_test))]

    l = retrieve_sentence2(pred,df_test_old)

    for i in range(len(l)):

        print(l[i])            


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help='path to model file (for saving/loading)', required=True)

    parser.add_argument('--text', help='path to text file (for training/testing)', required=True)

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--train', action='store_true')

    group.add_argument('--test', action='store_true')

    parser.add_argument('--gen', help='enters generative mode')



    opts = parser.parse_args()




    

    # checking recall on validation set 

    def evaluate_recall(y, y_test, k=1):

        num_examples = float(len(y))

        num_correct = 0

        for predictions, label in zip(y, y_test):

            if label in predictions[:k]:

                num_correct += 1

        return num_correct/num_examples

    

    y_test = np.zeros(len(pred)) + 19

    for n in [1, 2, 5, 10, 15, 20]:

        print('Recall at ',n)

        print(evaluate_recall(pred, y_test, n))









    

    if opts.train:

        df_train_old, df_train = loadDatatrain(opts.text,200)

        dm.train(df_train)

        dm.save(opts.model)

    else:

        assert opts.test,opts.test

        dm.load(opts.model)

        df_test_old, df_test = loadDatatest(opts.text,200)

        pred = [dm.findBest2(df_test.context[x], df_test.iloc[x,:df_test.shape[1]-1].values) for x in range(len(df_test))]

        l = retrieve_sentence2(pred,df_test_old)

        for i in range(len(l)):

            print(l[i])

"""  

    if opts.train:

        text = []

        for _,_, dialogue in loadData(opts.text):

            for idx, _, _,options in dialogue:

                text.extend(options)

        dm.train(text)

        dm.save(opts.model)

    else:

        assert opts.test,opts.test

        dm.load(opts.model)

        for _,_, dialogue in loadData(opts.text):

            for idx, utterance, answer, options in dialogue:

                print(idx,dm.findBest(utterance,options))



"""                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    