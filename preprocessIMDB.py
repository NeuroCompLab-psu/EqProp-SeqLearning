from keras.datasets import imdb
import re
from bs4 import BeautifulSoup
import numpy as np
import gensim.downloader as api
from torch.utils.data import TensorDataset, DataLoader
import torch
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer=ToktokTokenizer()

loaded_glove_model = None
embed_dim = 300
positional_embed = 0

train_dataset_size = 25000
test_dataset_size = 25000

special_token = [np.zeros(embed_dim+positional_embed)]

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data()
index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()])

def get_embeddings(glove_model, training_word2idx):
    keys = {}
    ind = 0
    c = 0
    for w,i in training_word2idx:
        try:
            keys[ind] = glove_model.get_vector(w, None)
            #keys[ind] = np.append(keys[ind], [math.sin(float(ind + 1) / 5000.), math.cos(float(ind + 1) / 5000.)])
            ind = ind + 1
        except:
            c = c + 1
            pass
    #print(c)   #Characters skipped
    embeddings = np.zeros((len(keys), embed_dim+positional_embed))
    for x in range(len(keys)):
        embeddings[x] = keys[x]
    return embeddings

def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9.!?,\s]'
    text=re.sub(pattern,'',text)
    return text

#Removing the html strips
def strip_html(text):
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

def remove_stopwords(text, is_lower_case=True):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token.lower() for token in tokens]
    else:
        filtered_tokens = [token for token in tokens]
    return filtered_tokens

def process_train_data(seqlen=150, batch_size=100):
    max_word_length = seqlen
    loaded_glove_model = api.load("word2vec-google-news-300")  #("glove-twitter-50") For embed dim 50
    train_data_text = []
    train_data_label = []
    for i in range(train_dataset_size):
        decoded = " ".join([reverse_index.get(j-3, "#") for j in training_data[i]])
        words = remove_stopwords(remove_special_characters(denoise_text(decoded)))
        training_word2idx = list(zip(words, range(len(words))))
        GloveEmbeddings = get_embeddings(loaded_glove_model, training_word2idx)
        train_data_text.append(GloveEmbeddings)
        train_data_label.append(training_targets[i])

        if train_data_text[i].shape[0] < max_word_length:
            app = special_token*(max_word_length - train_data_text[i].shape[0])
            train_data_text[i] = np.append(train_data_text[i], app, axis= 0)
        else:
            train_data_text[i] = train_data_text[i][:max_word_length]


    train_data_text = torch.tensor(train_data_text, dtype=torch.float32)
    train_data_label = torch.tensor(train_data_label, dtype=torch.float32)
    train_dataset = TensorDataset(train_data_text, train_data_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader

def process_test_data(seqlen=150, batch_size=100):
    max_word_length = seqlen
    loaded_glove_model = api.load("word2vec-google-news-300")
    test_data_text = []
    test_data_label = []
    for i in range(test_dataset_size):
        decoded = " ".join([reverse_index.get(j-3, "#") for j in testing_data[i]])
        words = remove_stopwords(remove_special_characters(denoise_text(decoded)))
        test_word2idx = list(zip(words, range(len(words))))
        GloveEmbeddings = get_embeddings(loaded_glove_model, test_word2idx)
        test_data_text.append(GloveEmbeddings)
        test_data_label.append(testing_targets[i])

        if test_data_text[i].shape[0] < max_word_length:
            app = special_token*(max_word_length - test_data_text[i].shape[0])
            test_data_text[i] = np.append(test_data_text[i], app, axis= 0)
        else:
            test_data_text[i] = test_data_text[i][:max_word_length]

    test_data_text = torch.tensor(test_data_text, dtype=torch.float32)
    test_data_label = torch.tensor(test_data_label, dtype=torch.float32)
    test_dataset = TensorDataset(test_data_text, test_data_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
