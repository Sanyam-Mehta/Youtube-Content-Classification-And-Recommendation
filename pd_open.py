import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import re
import collections


cols = ['video_id','comment_text','likes','replies']
df = pd.read_csv("GBcomments.csv",header=None,names=cols, low_memory=False)
# above line will be different depending on where you saved your data, and your file name

#print(df.comment_text[2])
#example1 = BeautifulSoup(df.comment_text[2], 'lxml')
#print(example1.get_text())

tok = WordPunctTokenizer()

pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def comment_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()

#test_result = []
#for t in testing:
 #   test_result.append(comment_cleaner(t))

#clearing null entries
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
df.info()
# clean data without any null Values


# a check for null value
for i in df.comment_text:
    if not isinstance(i, collections.Iterable):
        print(type(i), i)
#prints nothing as there is nonull value


clean_comment_texts = []
for i in range(0,718429):
                                                                        
    clean_comment_texts.append(comment_cleaner(df['comment_text'][i]))

clean_df = pd.DataFrame(clean_comment_texts,columns=['comment_text'])
clean_df['video_id'] = df.video_id
clean_df['likes'] = df.likes
print(clean_df.head(5))

#clean comments are stored in csv file finally
clean_df.to_csv('clean_comments.csv', encoding='utf-8')
csv='clean_comments.csv'

#check if it is being read
my_df = pd.read_csv(csv,index_col=0)
print(my_df.head(5))





