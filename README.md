# Twitter Sentiment Analysis Using Neural Networks

This project contains code for processing text, creating features, and conducting sentiment analysis with Neural Networks. We specifically use LSTM (Long Short-Term Memory networks) to train our models, achieving a test accuracy of 79%.

## Setup Instructions

## Requirements

1. Python installation
2. Necessary library installation from the requirements.txt

### Setting Up the Code

2. **Change to the project directory:**

```
cd Twitter-Sentiment-Analysis-using-Neural-Networks
```

3. **Install the required Python depndencies:**

```
pip install -r requirements.txt
```

### Preparing the Dataset

1. **Download the dataset from Kaggle:**
   The dataset is available at [this link](https://www.kaggle.com/kazanova/sentiment140).
2. **Unzip the downloaded file and rename the resulting CSV file to `dataset.csv`.**
3. **Create a `data` folder inside the project directory and move `dataset.csv` into this folder.**

## Using the Code

### Performing Twitter Sentiment Analysis

1. **Split the dataset into training and testing sets:**

```
python twitter_dataset_split.py
```

2. **Process the tweets to prepare for analysis. This includes removing mentions, links, non-letters, and more:**

```
python twitter_preprocessing.py
```

3. **Train the LSTM model on the processed data and test its accuracy:**

```
python twitter_lstm_model_training.py
```

This setup and process will allow you to analyze sentiments from Twitter data using advanced neural network techniques.

# CSE 573: Sentiment Analysis Utilizing Logistic Regression

Please download this directory! <br>
This project contains code to train a Logistic Regression model to classify between negative and positive sentiments. Additionally, there is a testModel.py file to test out the model with new sentences. It takes about 6 minutes to train the model. The training accuracy is about 81% and the testing accuracy is about 77.9%

## Please make sure to have Python installed

## Prerequisite Python libraries:

Please install these libraries to run the program

- pandas
- re
- nltk
- sklearn
- joblib
- seaborn
- matplotlib
- tkinter

## Data:

- Please go to this link:
  https://www.kaggle.com/datasets/kazanova/sentiment140 <br>
- Download this dataset, and put it in the DATA folder
- Rename the file name to dataset.csv

## To run the code:

`cd CODE` <br>
Train the model: <br>
`python logistic_regression_model.py` <br>
Test the model: <br>
`python test_model.py` <br>
