import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.autograd as autograd
from torch.nn import init
import torch.nn.utils.rnn 
import datetime
import operator
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

np.random.seed(0)
rep = 'C:/Users/Sophie HU/Desktop/CentraleSupelec/NLP/HW3/'


nb_epochs=10
N=200

###############################################################################
# Prepare input for LSTM
###############################################################################
def text2sentences2(path):
    # feel free to make a better tokenization/pre-processing
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

def remove_empty(df):
    '''remove rows whose utt length equals to 0'''
    n_length = df.shape[0]
    df_res =pd.DataFrame()
    for i in range(n_length):
        if len(df['utt'].iloc[i])<1:
            df['Bool'].iloc[i]=0
        else:
            df['Bool'].iloc[i]=1
    df_res = df.loc[df['Bool'] == 1]
    return df_res

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

def add_rows_test(row2,n,train_d):
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
            answers = distr[3].split("|")     
            cont = add_cont(cont,remove_digit(utt))
            row2.append({'context': cont, 'correct': cor, 'dis1': answers[0],'dis2': answers[1],'dis3': answers[2],'dis4': answers[3],'dis5': answers[4],'dis6': answers[5], \
                 'dis7': answers[6],'dis8': answers[7],'dis9': answers[8],'dis10': answers[9],'dis11': answers[10],'dis12': answers[11],'dis13': answers[12], \
                 'dis14': answers[13],'dis15': answers[14],'dis16': answers[15],'dis17': answers[16],'dis18': answers[17], 'dis19': answers[18]})
            cont = add_cont(cont,cor)
    return row2




def stemming_test(df):
    for i in range(df.shape[1]):
        df.iloc[:,i]= dataprocessing(df.iloc[:,i])
    return df



train = text2sentences2(rep+'train_both_original.txt')
train_test = text2sentences2(rep+'valid_both_original.txt')
list_dial = sep_dial(train)
list_dial_test = sep_dial(train_test)

#building train data set
row = []
row = add_rows_train(row,N,list_dial)
df_train_old = pd.DataFrame(data = row)
df_train_old['Bool'] = np.zeros(len(df_train_old))


df_train = pd.DataFrame()
df_train['xlabel']=df_train_old['xlabel']
df_train['context']= dataprocessing(df_train_old['context'])
df_train['utt']= dataprocessing(df_train_old['utt'])


#building test data set
row2 = []
row2 = add_rows_test(row2,N,list_dial_test)  
df_test_old = pd.DataFrame(data = row2)



df_test = stemming_test(pd.DataFrame(data = row2))
  
del(df_train_old)  
del(df_test_old)
###############################################################################

    
###############################################################################
# Functions for LSTM   
###############################################################################    
    
def create_vocab(dataframe):
    vocab = []
    word_freq = {}
    
    for index, row in dataframe.iterrows():
        
        context_cell = row["context"]
        response_cell = row["utt"]
        
        train_words = str(context_cell).split(',') + str(response_cell).split(',')
        
        for word in train_words:
          
            if word.lower() not in vocab:
                vocab.append(word.lower())         
                       
            if word.lower() not in word_freq:
                word_freq[word.lower()] = 1
            else:
                word_freq[word] += 1
    
    word_freq_sorted = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    vocab = [pair[0] for pair in word_freq_sorted]
    
    return vocab


def create_word_to_id(vocab):             
    word_to_id = {word: id for id, word in enumerate(vocab)}
    
    return word_to_id


def create_id_to_vec(word_to_id, glovefile): 
    lines = open(glovefile,encoding="utf-8").readlines()
    id_to_vec = {}
    vector = None
    
    for line in lines:
        word = line.split()[0]
        vector = np.array(line.split()[1:], dtype='float32') #32
        
        if word in word_to_id:
            id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(vector))
            
    for word, id in word_to_id.items(): 
        if word_to_id[word] not in id_to_vec:
            v = np.zeros(*vector.shape, dtype='float32')
            v[:] = np.random.randn(*v.shape)*0.01
            id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(v))
            
    embedding_dim = id_to_vec[0].shape[0]
    
    return id_to_vec, embedding_dim


def load_ids_and_labels(row, word_to_id):
    context_ids = []
    response_ids = []

    context_cell = row['context']
    response_cell = row['utt']
    label_cell = row['xlabel']

    max_context_len = 160
    
    context_words = context_cell.split(',')
    if len(context_words) > max_context_len:
        context_words = context_words[:max_context_len]
    for word in context_words:
        if word in word_to_id:
            context_ids.append(word_to_id[word])
        else: 
            context_ids.append(0) #UNK
    
    response_words = response_cell.split(',')
    
    for word in response_words:
        if word in word_to_id:
            response_ids.append(word_to_id[word])
        else: 
            response_ids.append(0)
            
    
    label = np.array(label_cell).astype(np.float32)

    return context_ids, response_ids, label

vocab_train = create_vocab(df_train)
vocab_to_id = create_word_to_id(vocab_train)
id_to_vec,emb_dim= create_id_to_vec(vocab_to_id,'glove.6B.100d.txt')
validation_dataframe = df_test
###############################################################################
    



###############################################################################
# Encoder-Decoder Model
###############################################################################
class Encoder(nn.Module):
    def __init__(self, 
            emb_size, 
            hidden_size, 
            vocab_size, 
            p_dropout): 
    
            super(Encoder, self).__init__()
             
            self.emb_size = emb_size
            self.hidden_size = hidden_size
            self.vocab_size = vocab_size
            self.p_dropout = p_dropout
       
            self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
            self.lstm = nn.LSTM(self.emb_size, self.hidden_size)
            self.dropout_layer = nn.Dropout(self.p_dropout) 

            self.init_weights()
             
    def init_weights(self):
        init.uniform(self.lstm.weight_ih_l0, a = -0.01, b = 0.01)
        init.orthogonal(self.lstm.weight_hh_l0)
        self.lstm.weight_ih_l0.requires_grad = True
        self.lstm.weight_hh_l0.requires_grad = True
        
        embedding_weights = torch.FloatTensor(self.vocab_size, self.emb_size)
            
        for id, vec in id_to_vec.items():
            embedding_weights[id] = vec
        
        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad = True)
            
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        _, (last_hidden, _) = self.lstm(embeddings) 
        last_hidden = self.dropout_layer(last_hidden[-1])

        return last_hidden
    
    
class DualEncoder(nn.Module):
    def __init__(self, encoder):
        super(DualEncoder, self).__init__()
        self.encoder = encoder
        self.hidden_size = self.encoder.hidden_size
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)     
        init.xavier_normal(M)
        self.M = nn.Parameter(M, requires_grad = True)

    def forward(self, context_tensor, response_tensor):
        
        context_last_hidden = self.encoder(context_tensor) 
        response_last_hidden = self.encoder(response_tensor) 
        
        context = context_last_hidden.mm(self.M) 
        context = context.view(-1, 1, self.hidden_size) 
        
        response = response_last_hidden.view(-1, self.hidden_size, 1) 
        
        score = torch.bmm(context, response).view(-1, 1) 
        return score
    
    
def shuffle_df(df):
    df.reindex(np.random.permutation(df.index))
    
def create_md(vocab,hidden_size,p_dropout):

    encoder = Encoder(
            emb_size = emb_dim,
            hidden_size = hidden_size,
            vocab_size = len(vocab),
            p_dropout = p_dropout)

    dual_encoder = DualEncoder(encoder)
    
    return encoder, dual_encoder

def increase_count(correct_count, score, label):
    if ((score.data[0][0] >= 0.5) and (label.data[0][0] == 1.0)):
       correct_count +=1  
   
    return correct_count

def get_accuracy(correct_count, dataframe):
    accuracy = correct_count/(len(dataframe))
        
    return accuracy


def model_train(training_dataframe,validation_dataframe,learning_rate, l2_penalty, epochs): 
    
    optimizer = torch.optim.Adam(dual_encoder.parameters(), lr = learning_rate, weight_decay = l2_penalty)
       
    loss_func = torch.nn.BCEWithLogitsLoss()
     
    for epoch in range(epochs): 
                     
            shuffle_df(training_dataframe)
                        
            sum_loss_training = 0.0
            
            training_correct_count = 0
            
            dual_encoder.train()

            for index, row in training_dataframe.iterrows():            
            
                context_ids, response_ids, label = load_ids_and_labels(row, vocab_to_id)
                
                context = autograd.Variable(torch.LongTensor(context_ids).view(-1,1), requires_grad = False) #.cuda()
                
                response = autograd.Variable(torch.LongTensor(response_ids).view(-1, 1), requires_grad = False) #.cuda()
                                
                label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label).reshape(1,1))), requires_grad = False) #.cuda()
                
                score = dual_encoder(context, response)
        
                loss = loss_func(score, label)
                
                sum_loss_training += loss.data#[0]
                
                loss.backward()
        
                optimizer.step()
               
                optimizer.zero_grad()
                
                training_correct_count = increase_count(training_correct_count, score, label)
                                                    
            training_accuracy = get_accuracy(training_correct_count, training_dataframe)
            
                
            shuffle_df(validation_dataframe)
            
            validation_correct_count = 0

            sum_loss_validation = 0.0

            dual_encoder.eval()
            

            
            for index, row in validation_dataframe.iterrows():

                context_ids, response_ids, label = load_ids_and_labels(row, vocab_to_id)
                
                context = autograd.Variable(torch.LongTensor(context_ids).view(-1,1)) 
                
                response = autograd.Variable(torch.LongTensor(response_ids).view(-1, 1))
                
                                
                label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label).reshape(1,1)))) 
                
                score = dual_encoder(context, response)
                
                loss = loss_func(score, label)
                
                sum_loss_validation += loss.data
                
                validation_correct_count = increase_count(validation_correct_count, score, label)
                    
            validation_accuracy = get_accuracy(validation_correct_count, validation_dataframe)
            
            print(str(datetime.datetime.now()).split('.')[0], 
                  "Epoch: %d/%d" %(epoch,epochs),  
                  "TrainLoss: %.3f" %(sum_loss_training/len(training_dataframe)), 
                  "TrainAccuracy: %.3f" %(training_accuracy), 
                  "ValLoss: %.3f" %(sum_loss_validation/len(validation_dataframe)), 
                  "ValAccuracy: %.3f" %(validation_accuracy))

            torch.save(dual_encoder.state_dict(), 'saved_model_%d_examples.pt' %(len(training_dataframe)))
                
    print(str(datetime.datetime.now()).split('.')[0], "Training and validation epochs finished.")
    
    return training_accuracy,validation_accuracy
###############################################################################
    
    
#Choosing hidden size and dropout probability, creating model
encoder, dual_encoder = create_md(vocab_train, hidden_size = 50,p_dropout = 0.85)

'''
for name, param in dual_encoder.named_parameters():
    if param.requires_grad:
        print(name)
# Create Valisation set:10 pct of train, and train will become first 10 pct of train       
validation_dataframe=model_train[int(df_train.shape[0]/10)*9:df_train.shape[0]]  
#df_train = df_train[0:(int(df_train.shape[0]/10)*9-1)]  
df_train2 = df_train[0:int(df_train.shape[0]/10)*9] 

training_accuracy,validation_accuracy=model_train(df_train2,validation_dataframe,learning_rate = 0.01, l2_penalty = 0.01,epochs = nb_epochs)#0.0001,0.0001,100

'''
# load model for testing  
#dual_encoder.load_state_dict(torch.load('C:/Users/Admin/Documents/GitHub/NLP/Exercice #3/saved_model_13230_examples.pt'))      
dual_encoder.load_state_dict(torch.load('saved_model_26460_examples.pt'))
dual_encoder.eval()



###############################################################################
# Chance of getting correct answers
###############################################################################
def load_ids(test_dataframe_different_structure, word_to_id):
    
    print(str(datetime.datetime.now()).split('.')[0], "Loading test IDs...")

    max_context_len = 160
    
    ids_per_example_and_candidate = {}
    
    for i, example in test_dataframe_different_structure.iterrows():
        
        ids_per_candidate = {}
      
        for column_name, cell in  example.iteritems():
            
                id_list = []
            
                words = str(cell).split(',')
                if len(words) > max_context_len:
                    words = words[:max_context_len]
    
                for word in words:
                    if word in word_to_id:
                        id_list.append(word_to_id[word])
                    else: 
                        id_list.append(0) #UNK  
                    
                ids_per_candidate[column_name] = id_list
    
        ids_per_example_and_candidate[i] = ids_per_candidate
    
    print(str(datetime.datetime.now()).split('.')[0], "Test IDs loaded.")
    
    return ids_per_example_and_candidate

ids_per_example_and_candidate = load_ids(df_test, vocab_to_id)


###############################################################################
# Valuation
###############################################################################

def load_scores(ids_per_example_and_candidate): 
    '''Dictionary :  to store each example's scores per utterance 
    Outer dictionary "scores_per_example_and_candidate": keys = examples, values = inner dictionaries

    Inner dictionaries "scores_per_candidate": keys = candidate names, values = score
    '''
    print(str(datetime.datetime.now()).split('.')[0], "Computing test scores...")
    
    scores_per_example_and_candidate = {}
                 
    for example, utterance_ids_dict in sorted(ids_per_example_and_candidate.items()): 
        score_per_candidate = {}

        for utterance_name, ids_list in sorted(utterance_ids_dict.items()):
        
            context = autograd.Variable(torch.LongTensor(utterance_ids_dict['context']).view(-1,1))#.cuda()
            
            if utterance_name != 'context':

                candidate_response = autograd.Variable(torch.LongTensor(utterance_ids_dict[utterance_name]).view(-1, 1))#.cuda()
                
                score = torch.sigmoid(dual_encoder(context, candidate_response))

                score_per_candidate["Score with " + utterance_name] = score.data[0][0]
    
        scores_per_example_and_candidate[example] = score_per_candidate

    print(str(datetime.datetime.now()).split('.')[0], "Test scores computed.")
    
    return scores_per_example_and_candidate


scores_per_example_and_candidate = load_scores(ids_per_example_and_candidate)

def recall_k(k,scores_per_example_and_candidate):
    count_true_hits = 0
    
    for example, score_per_candidate_dict in sorted(scores_per_example_and_candidate.items()): 

        top_k = dict(sorted(score_per_candidate_dict.items(), key=operator.itemgetter(1), reverse=True)[:k])
        if 'Score with correct' in top_k:
            count_true_hits += 1
    
    number_of_examples = len(scores_per_example_and_candidate)

    recall_at_k = count_true_hits/number_of_examples
    
    return recall_at_k

print("recall_at_1 =",recall_k(1,scores_per_example_and_candidate))
print("recall_at_2 =",recall_k(2,scores_per_example_and_candidate)) 
print("recall_at_5 =",recall_k(5,scores_per_example_and_candidate))  
print("recall_at_10 =",recall_k(10,scores_per_example_and_candidate)) 
print("recall_at_15 =",recall_k(15,scores_per_example_and_candidate)) 
print("recall_at_20 =",recall_k(20,scores_per_example_and_candidate)) 


###############################################################################
##Validation
###############################################################################
df_val = df_train[int(df_train.shape[0]/10)*9-1:int(df_train.shape[0])] 

def testing_same_structure(test_dataframe_same_structure):
    
    test_correct_count = 0

    for index, row in test_dataframe_same_structure.iterrows():

        context_ids, response_ids, label = load_ids_and_labels(row, vocab_to_id)

        context = autograd.Variable(torch.LongTensor(context_ids).view(-1,1)) #.cuda()

        response = autograd.Variable(torch.LongTensor(response_ids).view(-1, 1)) #.cuda()

        label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label).reshape(1,1)))) #.cuda()

        score = dual_encoder(context, response)

        test_correct_count = increase_count(test_correct_count, score, label)

    test_accuracy = get_accuracy(test_correct_count, test_dataframe_same_structure)
    
    return test_accuracy

test_accuracy = testing_same_structure(df_val)
print("Test accuracy for %d training examples and %d test examples: %.2f" %(len(df_train),len(df_val),test_accuracy))



