# Depressive-and-Non-Depressive-Tweets-Classification-using-a-sequential-deep-learning-model
This project aims to perform sentiment analysis on depressive and random tweets using a combination of traditional machine learning techniques and deep learning models. The repository contains code for data cleaning, exploratory data analysis, and model building and evaluation.

Introduction
This project leverages natural language processing (NLP) techniques and deep learning models to classify tweets into depressive or random categories based on their sentiment. The main goal is to identify depressive tweets and analyze their sentiment distribution.

Datasets
The project uses two datasets:

Depressive Tweets: Contains tweets labeled as depressive.

Random Tweets: Contains a random sample of tweets.

Files

depressive_tweets_processed.csv: CSV file containing processed depressive tweets.

Sentiment Analysis Dataset 2.csv: CSV file containing random tweets.

GoogleNews-vectors-negative300.bin.gz: Pre-trained Word2Vec embeddings from Google News.

Preprocessing

The preprocessing steps include:

Loading and reading datasets.
Cleaning tweets by removing URLs, mentions, emojis, and special characters.
Expanding contractions and removing stopwords.
Stemming the words.
Exploratory Data Analysis
Word Clouds: Visualize the most common words in depressive and random tweets using word clouds.
Sentiment Distribution: Analyze the distribution of sentiments (positive, negative, neutral) in both datasets using TextBlob.
Sentiment Analysis
Sentiment analysis is performed using:

TextBlob: A simple library for processing textual data and performing sentiment analysis.
Model Training and Evaluation
Deep learning models are built and trained to classify tweets:

CNN-LSTM Model: Combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks.
Training and Validation: The dataset is split into training, validation, and test sets. Early stopping and model checkpointing are used to optimize training.
Evaluation Metrics
Accuracy
Confusion Matrix
Classification Report: Precision, recall, F1-score for each class.
Requirements
The project requires the following Python libraries:

ftfy
matplotlib
nltk
numpy
pandas
scikit-learn
gensim
keras
wordcloud
textblob


Results
The trained model achieves competitive performance in classifying depressive and random tweets. The evaluation metrics and visualizations provide insights into the model's effectiveness and areas for improvement.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.
