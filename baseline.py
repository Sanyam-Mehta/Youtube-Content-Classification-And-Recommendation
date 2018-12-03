import pandas as pd
from textblob import TextBlob
import codecs
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import collections

csv='train_set.csv'
uf = pd.read_csv(csv,encoding='utf-8')

csv='test_set.csv'
df = pd.read_csv(csv,encoding='utf-8')

csv='validation_set.csv'
mf = pd.read_csv(csv,encoding='utf-8')


print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(uf.comment_text),(len(uf.comment_text[uf.polarity == 0]) / (len(uf.comment_text)*1.))*100,(len(uf.comment_text[uf.polarity == 1]) / (len(uf.comment_text)*1.))*100))


print("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(mf.comment_text),(len(mf.comment_text[mf.polarity == 0]) / (len(mf.comment_text)*1.))*100,(len(mf.comment_text[mf.polarity == 1]) / (len(mf.comment_text)*1.))*100))


print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(df.comment_text),(len(df.comment_text[df.polarity == 0]) / (len(df.comment_text)*1.))*100, (len(df.comment_text[df.polarity == 1]) / (len(df.comment_text)*1.))*100))


# THIS CODE GIVES US AN IDEA OF HOW MANY COMMENTS IN THE FINAL SPLIT ARE NEGATIVE, HOW MANY ARE POSITIVE AND THE LEFT OUT PERCENTAGE IS OF THE COMMENTS WHICH ARE NEUTRAL

