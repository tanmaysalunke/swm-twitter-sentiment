import numpy as np
import pandas as pd
from termcolor import colored
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading the data
print(colored("Loading train and test data", "yellow"))
training_tweets_data = pd.read_csv('data/clean_train.csv')
testing_tweets_data = pd.read_csv('data/clean_test.csv')
print(colored("Data loaded", "yellow"))

# Tf-IDF
print(colored("Applying TF-IDF transformation", "yellow"))
tfidfVectorizer = TfidfVectorizer(min_df = 5, max_features = 1000)
tfidfVectorizer.fit(training_tweets_data['Clean_tweet'].apply(lambda x: np.str_(x)))
train_tweet_vector = tfidfVectorizer.transform(training_tweets_data['Clean_tweet'].apply(lambda x: np.str_(x)))
test_tweet_vector = tfidfVectorizer.transform(testing_tweets_data['Clean_tweet'].apply(lambda x: np.str_(x)))

# Training on data
print(colored("Training Random Forest Classifier", "yellow"))
randomForestClassifier = RandomForestClassifier()
randomForestClassifier.fit(train_tweet_vector, training_tweets_data['Sentiment'])

# Prediction
print(colored("Predicting on train data", "yellow"))
prediction = randomForestClassifier.predict(train_tweet_vector)
print(colored("Training accuracy: {}%".format(accuracy_score(training_tweets_data['Sentiment'], prediction)*100), "green"))

print(colored("Predicting on test data", "yellow"))
prediction = randomForestClassifier.predict(test_tweet_vector)
print(colored("Testing accuracy: {}%".format(accuracy_score(testing_tweets_data['Sentiment'], prediction)*100), "green"))