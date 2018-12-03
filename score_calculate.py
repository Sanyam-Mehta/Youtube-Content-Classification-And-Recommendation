import csv
import pandas as pd
import re
from textblob import TextBlob
import codecs
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import re
import collections


csv='clean_comments.csv'
df = pd.read_csv(csv,index_col=0)


def sentiment_analysis(text):

    
    ##final_polarity_subjectivity=[0,0]
    ##if text="":
      ## final_polarity_subjectivity[0][0]=0
       ##final_polarity_subjectivity[0][1]=0
       ##return final_polarity_subjectivity

    #text = str([text.encode('utf-8')])

    blob = TextBlob(text)
    result=(blob.sentiment)
    
    #to find only number(polarity and subjectivity from result and store them
    final_polarity_subjectivity=[[float(s) for s in re.findall(r'-?\d+\.?\d*', str(result))]]
    
    if final_polarity_subjectivity[0][0]<0:
       final_polarity_subjectivity[0][0]=-1
    elif final_polarity_subjectivity[0][0]==0:
         final_polarity_subjectivity[0][0]=0
    else:
         final_polarity_subjectivity[0][0]=1
    return final_polarity_subjectivity



#clear null values again
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
df.info()

polarity = []
subjectivity = []
for i in range(0,700599):
    final_polarity_subjectivity= sentiment_analysis(df['comment_text'][i])                                                         
    polarity.append(final_polarity_subjectivity[0][0])
    subjectivity.append(final_polarity_subjectivity[0][1])


df1 = pd.DataFrame(polarity,columns=['polarity'])
df2=pd.DataFrame(subjectivity,columns=['subjectivity'])
df_with_score = pd.concat([df1, df2], axis=1)

#df_with_score = pd.DataFrame(subjectivity,columns=['subjectivity'])
df_with_score ['comment_text'] = df.comment_text
df_with_score ['video_id'] = df.video_id
df_with_score ['likes'] = df.likes

#final with score
df_with_score.to_csv('scored_comments.csv', encoding='utf-8')

