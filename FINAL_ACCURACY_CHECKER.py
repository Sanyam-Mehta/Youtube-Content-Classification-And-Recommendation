from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from time import time
import pandas as pd
from textblob import TextBlob
import codecs
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import collections
from time import time
from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

csv='test_set.csv'
tf = pd.read_csv(csv,encoding='utf-8')

csv='validation_set.csv'
vf = pd.read_csv(csv,encoding='utf-8')

from joblib import dump, load
sentiment_fit = load('Saved_Model.joblib')


#START TESTING HERE
x_test=vf.comment_text
y_test=vf.polarity
y_pred = sentiment_fit.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(y_pred)
print(accuracy)
