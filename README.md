# Twitter Sentiment Analysis Using Neural Networks

This project contains code for processing text, creating features, and conducting sentiment analysis with Neural Networks. We specifically use LSTM (Long Short-Term Memory networks) to train our models, achieving a test accuracy of 79%.

## Setup Instructions

### Installing Python

1. **Install pyenv (a tool to manage different Python versions):**

```
brew install pyenv
```

2. **Use pyenv to install Python version 3.7.2:**

```
CFLAGS="-I$(xcrun --show-sdk-path)/usr/include" pyenv install 3.7.2
```

### Setting Up the Code

1. **Download the project repository:**

```
git clone https://github.com/kb22/Twitter-Sentiment-Analysis-using-Neural-Networks.git
```

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

### Analyzing the Dataset

1. **Start the Jupyter Notebook to explore the dataset:**

```
jupyter notebook
```

2. **Open the `Dataset analysis.ipynb` notebook from Jupyter to view the dataset analysis.**

### Performing Twitter Sentiment Analysis

1. **Split the dataset into training and testing sets:**

```
python train-test-split.py
```

2. **Process the tweets to prepare for analysis. This includes removing mentions, links, non-letters, and more:**

```
python preprocessing.py
```

3. **Train the LSTM model on the processed data and test its accuracy:**

```
python lstm.py
```

This setup and process will allow you to analyze sentiments from Twitter data using advanced neural network techniques.
