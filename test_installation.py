""" Test LLM"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import brown
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import string

nltk.download('brown')
real_data = brown.sents()
# print(real_data[:5])

def preprocess(data):
    "Preprocess data"
    stop_words = set(stopwords.words('english'))
    preprocessed_data = []
    for sentence in data:
        sentence = [word.lower() for word in sentence if word.isalpha()]
        sentence = [word for word in sentence if word not in stop_words]
        preprocessed_data.append(sentence)
    return preprocessed_data

real_preprocessed_data = preprocess(real_data)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(real_preprocessed_data)
sequences = tokenizer.texts_to_sequence(real_preprocessed_data)

#Defining the Model
vocab_size = len(tokenizer.word_index) + 1 #To account for padding token
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_seq_length -1))
model.add(LSTM(128))
model.add(Dense(vocab_size,activation='softmax'))
model.summary()


