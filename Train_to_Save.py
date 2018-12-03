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
import pickle


csv='test_set.csv'
tf = pd.read_csv(csv,encoding='utf-8')

csv='validation_set.csv'
vf = pd.read_csv(csv,encoding='utf-8')



def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == -1]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == -1]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == -1]) / (len(x_test)*1.))
    t0 = time()
    sentiment_fit = pipeline.fit(x_train, y_train)
    
    return sentiment_fit
    
   



cvec = CountVectorizer()
lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')


def nfeature_accuracy_checker(vectorizer=cvec, stop_words=None, ngram_range=(1, 1), classifier=lr):
    result = []
    print(classifier)
    print("\n")
    #for n in n_features:
    n=40000
    vectorizer.set_params(stop_words=stop_words, max_features=40000, ngram_range=ngram_range)
    checker_pipeline = Pipeline([
       ('vectorizer', vectorizer),
       ('classifier', classifier)
    ])
    print("Validation result for {} features".format(n))
    nfeature_accuracy = accuracy_summary(checker_pipeline, tf.comment_text, tf.polarity, vf.comment_text, vf.polarity)
    
    return nfeature_accuracy


#prints the most frequent words occuring
csv = 'term_frequency.csv'
term_freq_df = pd.read_csv(csv,index_col=0)
#print(term_freq_df.sort_values(by='total', ascending=False).iloc[:10])

#Check if it is sklearn's stop words list
#a = frozenset(list(term_freq_df.sort_values(by='total', ascending=False).iloc[:10].index))
#b = text.ENGLISH_STOP_WORDS
#print(set(a).issubset(set(b)))


#contains the first 10 stop words
my_stop_words = frozenset(list(term_freq_df.sort_values(by='total', ascending=False).iloc[:9].index))



feature_result_tgwocsw = nfeature_accuracy_checker(stop_words=my_stop_words,ngram_range=(1, 3))

s = pickle.dumps(feature_result_tgwocsw)

from joblib import dump, load
dump(feature_result_tgwocsw, 'Saved_Model.joblib')
