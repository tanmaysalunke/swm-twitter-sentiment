import re
import nltk
import numpy as np
import pandas as pd
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from termcolor import colored
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Import datasets
print("Loading data")
training_tweets_data = pd.read_csv('data/train.csv')
testing_tweets_data = pd.read_csv('data/test.csv')

# Setting stopwords
RetrievedDelimitingWords = set(stopwords.words('english'))
RetrievedDelimitingWords.remove("not")

# Function to expand tweet
def elaborateTweetData(tweet):
	elaboratedTweetsData = []
	for t in tweet:
		if re.search("n't", t):
			elaboratedTweetsData.append(t.split("n't")[0])
			elaboratedTweetsData.append("not")
		else:
			elaboratedTweetsData.append(t)
	return elaboratedTweetsData

# Function to process tweets
def process_data(un_processed_tweets, lemmatizingMethod, stemmingMethod):
	un_processed_tweets['Clean_tweet'] = un_processed_tweets['Tweet']
	print(colored("Cleaning data containing user data with @  symbol", "orange"))
	un_processed_tweets['Clean_tweet'] = un_processed_tweets['Clean_tweet'].str.replace("@[\w]*","")
	print(colored("Cleaning and discarding speacial chars to reduce non-meaningful data", "orange"))
	un_processed_tweets['Clean_tweet'] = un_processed_tweets['Clean_tweet'].str.replace("[^a-zA-Z' ]","")
	print(colored("Discarding web links: not meaningful data", "orange"))
	un_processed_tweets['Clean_tweet'] = un_processed_tweets['Clean_tweet'].replace(re.compile(r"((www\.[^\s]+)|(https?://[^\s]+))"), "")
	print(colored("Discarding one letter words: not meaning ful", "orange"))
	un_processed_tweets['Clean_tweet'] = un_processed_tweets['Clean_tweet'].replace(re.compile(r"(^| ).( |$)"), " ")
	print(colored("Generating tokens from data: Splitting technique", "orange"))
	un_processed_tweets['Clean_tweet'] = un_processed_tweets['Clean_tweet'].str.split()
	print(colored("Discarding delimitting words: stopping words", "orange"))
	un_processed_tweets['Clean_tweet'] = un_processed_tweets['Clean_tweet'].apply(lambda tweet: [word for word in tweet if word not in RetrievedDelimitingWords])
	print(colored("Data elaboration", "orange"))
	un_processed_tweets['Clean_tweet'] = un_processed_tweets['Clean_tweet'].apply(lambda tweet: elaborateTweetData(tweet))
	print(colored("Performing Data lematization", "orange"))
	un_processed_tweets['Clean_tweet'] = un_processed_tweets['Clean_tweet'].apply(lambda tweet: [lemmatizingMethod.lemmatize(word) for word in tweet])
	print(colored("Getting root words: reducing set of reference data", "orange"))
	un_processed_tweets['Clean_tweet'] = un_processed_tweets['Clean_tweet'].apply(lambda tweet: [stemmingMethod.stem(word) for word in tweet])
	print(colored("Merging the cleaned data to make it ready for learning", "orange"))
	un_processed_tweets['Clean_tweet'] = un_processed_tweets['Clean_tweet'].apply(lambda tweet: ' '.join(tweet))
	return un_processed_tweets

# Define processing methods
lemmatizingMethod = WordNetLemmatizer()
stemmingMethod = PorterStemmer()

# Pre-processing the tweets
print(colored("Cleaning the raw data", "green"))
training_tweets_data = process_data(training_tweets_data, lemmatizingMethod, stemmingMethod)
training_tweets_data.to_csv('data/clean_train.csv', index = False)
print(colored("Tweets processed and saved to csv data", "green"))
print(colored("Applying cleaning process to test data", "green"))
testing_tweets_data = process_data(testing_tweets_data, lemmatizingMethod, stemmingMethod)
testing_tweets_data.to_csv('data/clean_test.csv', index = False)
print(colored("Test tweets processed and saved to csv file", "green"))
