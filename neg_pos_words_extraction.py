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


csv='scored_comments.csv'
my_df = pd.read_csv(csv,encoding='utf-8')

cvec = CountVectorizer()
cvec.fit(my_df.comment_text)

print(len(cvec.get_feature_names()))



neg_doc_matrix = cvec.transform(my_df[my_df.polarity == -1].comment_text)
neu_doc_matrix= cvec.transform(my_df[my_df.polarity == 0].comment_text)
pos_doc_matrix = cvec.transform(my_df[my_df.polarity == 1].comment_text)
neg_tf = np.sum(neg_doc_matrix,axis=0)
neu_tf= np.sum(neu_doc_matrix,axis=0)
pos_tf = np.sum(pos_doc_matrix,axis=0)
neg = np.squeeze(np.asarray(neg_tf))
neu = np.squeeze(np.asarray(neu_tf))
pos = np.squeeze(np.asarray(pos_tf))
term_freq_df = pd.DataFrame([neg,neu,pos],columns=cvec.get_feature_names()).transpose()


term_freq_df.columns = ['negative', 'neutral','positive']
term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['neutral'] + term_freq_df['positive']

term_freq_df.to_csv('term_frequency.csv', encoding='utf-8')
