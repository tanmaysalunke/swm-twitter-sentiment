import os
import tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential

# from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer


import pandas as pd
from termcolor import colored

# Get the tweets data
print(colored("Getting data for training purpose and testing purpose", "orange"))
training_tweets_data = pd.read_csv('data/clean_train.csv')
testing_tweets_data = pd.read_csv('data/clean_test.csv')
print(colored("Data received", "orange"))

# Process the data with tokens and padding
print(colored("Getting tokens and applying padding to the tweets data", "orange"))
tweets_tokenizing = Tokenizer(num_words = 2000, split = ' ')
tweets_tokenizing.fit_on_texts(training_tweets_data['Clean_tweet'].astype(str).values)
training_tweet_sequence = tweets_tokenizing.texts_to_sequences(training_tweets_data['Clean_tweet'].astype(str).values)
maximum_training_tweets_sequence = max([len(i) for i in training_tweet_sequence])
training_tweet_sequence = pad_sequences(training_tweet_sequence, maxlen = maximum_training_tweets_sequence)
testing_tweets_sequence = tweets_tokenizing.texts_to_sequences(testing_tweets_data['Clean_tweet'].astype(str).values)
testing_tweets_sequence = pad_sequences(testing_tweets_sequence, maxlen = maximum_training_tweets_sequence)
print(colored("Process complete", "orange"))

# Prepare the model for training
print(colored("Building the model: Long Short Term Memory", "orange"))
model = Sequential()
model.add(Embedding(2000, 128, input_length = training_tweet_sequence.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(256, dropout = 0.2))
model.add(Dense(2, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

# Starting to train the LSTM
print(colored("Process Long Short Term Memory model for train", "green"))
history = model.fit(training_tweet_sequence, pd.get_dummies(training_tweets_data['Sentiment']).values, epochs = 10, batch_size = 128, validation_split = 0.2)
print(colored(history, "green"))

# Starting to test the LSTM
print(colored("Process the Long Short Term Memory for test", "green"))
score, accuracy = model.evaluate(testing_tweets_sequence, pd.get_dummies(testing_tweets_data['Sentiment']).values, batch_size = 128)
print("Test accuracy: {}".format(accuracy))