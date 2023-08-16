import os
import random
import time
import pickle
import re,string,unicodedata
import pandas as pd
import math
import time
import pprint
pp = pprint.PrettyPrinter()
import numpy as np
#from sklearn.decomposition import PCA
import torch
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.tokenize.toktok import ToktokTokenizer
from torch.utils.data import TensorDataset, DataLoader
import gensim.downloader as api


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Constants
path_data_train="./snli_data/snli_1.0_train.csv"
path_data_test="./snli_data/snli_1.0_test.csv"

max_word_length2 = 25
max_word_length = 25
embed_dim = 300
positional_embed = 0
special_token = [np.zeros(embed_dim+positional_embed)]
train_dataset_size = 540000 #550152
val_dataset_size = 10000
test_dataset_size = 10000
#Tokenization of text
tokenizer=ToktokTokenizer()
stopword_list=nltk.corpus.stopwords.words('english')

#Removing the html strips
def strip_html(text):
    if type(text) != str:
        return ' '
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

def remove_special_characters(text, remove_digits=True):
    pattern = r'[.]'
    text = re.sub(pattern, ' . ', text)
    pattern = r'[?]'
    text = re.sub(pattern, ' ? ', text)
    pattern = r'[!]'
    text = re.sub(pattern, ' ! ', text)
    pattern=r'[^a-zA-z0-9.?!\s]'
    text=re.sub(pattern,'',text)
    return text

#set stopwords to english
stop=set(stopwords.words('english'))

def remove_stopwords(text, is_lower_case=True):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token.lower() for token in tokens]
    else:
        filtered_tokens = [token for token in tokens]
    return filtered_tokens

def get_embeddings(glove_model, training_word2idx):
    keys = {}
    ind = 0
    c = 0
    for w,i in training_word2idx:
        try:
            keys[ind] = glove_model.get_vector(w, None)
            #keys[ind] = np.append(keys[ind], [math.sin(float(ind + 1) / 1000), math.cos(float(ind + 1) / 1000)])
            ind = ind + 1
        except:
            c = c + 1
            pass
    embeddings = np.zeros((len(keys), embed_dim+positional_embed))
    for x in range(len(keys)):
        embeddings[x] = keys[x]
    return embeddings

def process_snli_train_data(seqlen = 25, batch_size=200):
    loaded_glove_model = api.load("word2vec-google-news-300")  # glove-twitter-200") for 200 dim embed

    max_word_length = max_word_length2 = seqlen
    # Apply function on review column
    snli_data_train = pd.read_csv(path_data_train)
    snli_data_train['sentence1'] = snli_data_train['sentence1'].apply(denoise_text)
    snli_data_train['sentence2'] = snli_data_train['sentence2'].apply(denoise_text)
    snli_data_train['sentence1'] = snli_data_train['sentence1'].apply(remove_special_characters)
    snli_data_train['sentence2'] = snli_data_train['sentence2'].apply(remove_special_characters)
    snli_data_train['sentence1'] = snli_data_train['sentence1'].apply(remove_stopwords)
    snli_data_train['sentence2'] = snli_data_train['sentence2'].apply(remove_stopwords)

    train_data_text = []
    train_data_label = []
    for i in range(train_dataset_size):
        words = snli_data_train['sentence1'][i]
        training_word2idx = list(zip(words, range(len(words))))
        GloveEmbeddings1 = get_embeddings(loaded_glove_model, training_word2idx)
        if GloveEmbeddings1.shape[0] < max_word_length:
            app = special_token*(max_word_length - GloveEmbeddings1.shape[0])
            GloveEmbeddings1 = np.append(GloveEmbeddings1, app, axis= 0)
        else:
            GloveEmbeddings1 = GloveEmbeddings1[:max_word_length]
        words = snli_data_train['sentence2'][i]
        training_word2idx = list(zip(words, range(len(words))))
        GloveEmbeddings2 = get_embeddings(loaded_glove_model, training_word2idx)

        if GloveEmbeddings2.shape[0] < max_word_length2:
            app = special_token*(max_word_length2 - GloveEmbeddings2.shape[0])
            GloveEmbeddings2 = np.append(GloveEmbeddings2, app, axis= 0)
        else:
            GloveEmbeddings2 = GloveEmbeddings2[:max_word_length2]
        lab = 0
        if snli_data_train['gold_label'][i] == 'neutral':
            lab = 0
        elif snli_data_train['gold_label'][i] == 'contradiction':
            lab = 1
        elif snli_data_train['gold_label'][i] == 'entailment':
            lab = 2
        else:
            continue
        train_data_text.append(np.concatenate([GloveEmbeddings1, GloveEmbeddings2]))
        train_data_label.append(lab)

    train_data_text = torch.tensor(train_data_text, dtype=torch.float32)
    train_data_label = torch.tensor(train_data_label, dtype=torch.float32)
    train_dataset = TensorDataset(train_data_text, train_data_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader

def process_snli_test_data(seqlen = 25, batch_size=200):
    loaded_glove_model = api.load("word2vec-google-news-300")  # glove-twitter-200") for 200 dim embed

    max_word_length = max_word_length2 = seqlen
    snli_data_test = pd.read_csv(path_data_test)
    snli_data_test['sentence1'] = snli_data_test['sentence1'].apply(denoise_text)
    snli_data_test['sentence2'] = snli_data_test['sentence2'].apply(denoise_text)
    snli_data_test['sentence1'] = snli_data_test['sentence1'].apply(remove_special_characters)
    snli_data_test['sentence2'] = snli_data_test['sentence2'].apply(remove_special_characters)
    snli_data_test['sentence1'] = snli_data_test['sentence1'].apply(remove_stopwords)
    snli_data_test['sentence2'] = snli_data_test['sentence2'].apply(remove_stopwords)
    test_data_text = []
    test_data_label = []
    for i in range(test_dataset_size):
        words = snli_data_test['sentence1'][i]
        test_word2idx = list(zip(words, range(len(words))))
        GloveEmbeddings1 = get_embeddings(loaded_glove_model, test_word2idx)
        if GloveEmbeddings1.shape[0] < max_word_length:
            app = special_token*(max_word_length - GloveEmbeddings1.shape[0])
            GloveEmbeddings1 = np.append(GloveEmbeddings1, app, axis= 0)
        else:
            GloveEmbeddings1 = GloveEmbeddings1[:max_word_length]
        words = snli_data_test['sentence2'][i]
        test_word2idx = list(zip(words, range(len(words))))
        GloveEmbeddings2 = get_embeddings(loaded_glove_model, test_word2idx)
        if GloveEmbeddings2.shape[0] < max_word_length2:
            app = special_token*(max_word_length2 - GloveEmbeddings2.shape[0])
            GloveEmbeddings2 = np.append(GloveEmbeddings2, app, axis= 0)
        else:
            GloveEmbeddings2 = GloveEmbeddings2[:max_word_length2]
        lab = 0
        if snli_data_test['gold_label'][i] == 'neutral':
            lab = 0
        elif snli_data_test['gold_label'][i] == 'contradiction':
            lab = 1
        elif snli_data_test['gold_label'][i] == 'entailment':
            lab = 2
        else:
            continue
        test_data_text.append(np.concatenate([GloveEmbeddings1, GloveEmbeddings2]))
        test_data_label.append(lab)
    test_data_text = torch.tensor(test_data_text, dtype=torch.float32)
    test_data_label = torch.tensor(test_data_label, dtype=torch.float32)
    test_dataset = TensorDataset(test_data_text, test_data_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
