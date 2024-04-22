import joblib
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import tkinter as tk
from tkinter import messagebox


cur_dir = input("Please input your current path till Sentiment_Analysis_Logistic_Regression\nEx. '/Desktop/..../Sentiment_Analysis_Logistic_Regression'")

"""**Testing pre-trained model**"""

load_model = joblib.load(cur_dir + '/CODE/model.pkl')

# Stemming object
ps = PorterStemmer()

stop_words = set(stopwords.words('english'))


# Function to stem all words in the tweet
def stemming(content):

    # Removes every letter in the content that is not a to z letters
    stemmed = re.sub('[^a-zA-Z]', ' ', content)
    stemmed = stemmed.lower()
    stemmed = stemmed.split()

    stemmed = [ps.stem(word) for word in stemmed if not word in stop_words]

    stemmed = ' '.join(stemmed)

    return stemmed

# Sentences to test it out with
# I love rainy days.            ##Positive
# I hate rainy days.            ##Negative
# I like rainy days.            #Supposed to be positive but is negative
# I love to do all the chores but don't like to do the dishes.          ##Positive since the person likes most of the chores but one


def analyze_sentiment():
    # sample_sentence = "I love rainy days."

    sample_sentence = sentence_input.get()

    # Stemming the sentence words
    preprocess_sentence = stemming(sample_sentence)

    vectorizer = joblib.load(cur_dir + '/CODE/vectorizer.pkl')

    # Tokenizing
    tokenized_sentence = vectorizer.transform([preprocess_sentence])

    # Prediction
    sentence_prediction = load_model.predict(tokenized_sentence)

    sen = ""
    if sentence_prediction == 0:
        sen = "The sentiment of this sentence is negative :/"
    else:
        sen = "The sentiment of this sentence is positive!"

    # Show the prediction in a message box
    messagebox.showinfo("Sentiment Analysis Result" + "The sentiment of this sentence is,", sen)


# Window
root = tk.Tk()
root.title("Sentiment analysis Demo")

# User input
tk.Label(root, text="Enter your sentence:").pack()
sentence_input = tk.Entry(root, width=200)
sentence_input.pack()

# Button
button = tk.Button(root, text="Calculate sentiment", command=analyze_sentiment)
button.pack()

# Output
result = tk.Label(root, text="Result!")

# Run
root.mainloop()

