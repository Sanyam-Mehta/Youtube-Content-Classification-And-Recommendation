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



def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == -1]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == -1]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == -1]) / (len(x_test)*1.))
    t0 = time()
    sentiment_fit = pipeline.fit(x_train, y_train)
    print("YE LEEEE")
    print(sentiment_fit)
    y_pred = sentiment_fit.predict(x_test)
    


##CHECKING PART START

# SAME AS DONE BELOW< JUST TO BE DOUBLE SURE
    p=0
    q=0
    r=0
    for i in range(0,len(x_test)):
        if y_pred[i]==-1:
           p+=1
        if y_pred[i]==0:
           q+=1
        if y_pred[i]==1:
           r+=1

    f=p+q+r


   #TO CHECK WHETHER ALL THE THREE CLASSES (-1 0 and 1) are predicted or not
    a=len(x_test)
    b=len(x_test[y_pred == -1])+len(x_test[y_pred == 0])+len(x_test[y_pred == 1])
    print(a-b)
    print(a==b)
    print(f==a)
    print(f-a)
    print(f)
    print(a)
    print(b)
    #AS THE OUTPUT ON THE SCREEN IS TRUE HENCE IT IMPLIES THAT WE HAVE PREDICTED y IN ALL THE THREE CLASSES. NOW JUST FURTHER TUNING IS LEFT
##CHECKING PART ENDS



    train_test_time = time() - t0
    accuracy = accuracy_score(y_test, y_pred)
    print("null accuracy: {0:.2f}%".format(null_accuracy*100))
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    if accuracy > null_accuracy:
        print("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100))
    elif accuracy == null_accuracy:
        print("model has the same accuracy with the null accuracy")
    else:
        print("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
    print("train and test time: {0:.2f}s".format(train_test_time))
    print("-"*80)
    return(accuracy, train_test_time)

cvec = CountVectorizer()
lr = LogisticRegression(multi_class='multinomial', solver='lbfgs') #multiclass classification  #check this also : newton-cg
n_features = np.arange(10000,100001,10000)

def nfeature_accuracy_checker(vectorizer=cvec, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=lr):
    result = []
    print(classifier)
    print("\n")
    for n in n_features:
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print("Validation result for {} features".format(n))
        nfeature_accuracy,tt_time = accuracy_summary(checker_pipeline, tf.comment_text, tf.polarity, vf.comment_text, vf.polarity)
        result.append((n,nfeature_accuracy,tt_time))
    return result


#prints the most frequent words occuring
csv = 'term_frequency.csv'
term_freq_df = pd.read_csv(csv,index_col=0)
print(term_freq_df.sort_values(by='total', ascending=False).iloc[:10])

#Check if it is sklearn's stop words list
a = frozenset(list(term_freq_df.sort_values(by='total', ascending=False).iloc[:10].index))
b = text.ENGLISH_STOP_WORDS
print(set(a).issubset(set(b)))


#contains the first 10 stop words
my_stop_words = frozenset(list(term_freq_df.sort_values(by='total', ascending=False).iloc[:10].index))


#training
print("RESULT FOR UNIGRAM WITHOUT STOP WORDS\n")
feature_result_wosw = nfeature_accuracy_checker(stop_words='english')

print("RESULT FOR UNIGRAM WITH STOP WORDS\n")
feature_result_ug = nfeature_accuracy_checker()

print("RESULT FOR UNIGRAM WITHOUT CUSTOM STOP WORDS (Top 10 frequent words)\n")
feature_result_wocsw = nfeature_accuracy_checker(stop_words=my_stop_words)



print("RESULT FOR BIGRAM WITH STOP WORDS\n")
feature_result_bg = nfeature_accuracy_checker(ngram_range=(1, 2))

print("RESULT FOR TRIGRAM WITH STOP WORDS\n")
feature_result_tg = nfeature_accuracy_checker(ngram_range=(1, 3))


#visualisation
nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ug_wocsw = pd.DataFrame(feature_result_wocsw,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ug_wosw = pd.DataFrame(feature_result_wosw,columns=['nfeatures','validation_accuracy','train_test_time'])


plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, 'r', label='with stop words')

plt.plot(nfeatures_plot_ug_wocsw.nfeatures, nfeatures_plot_ug_wocsw.validation_accuracy,'b',label='without custom stop words')

plt.plot(nfeatures_plot_ug_wosw.nfeatures, nfeatures_plot_ug_wosw.validation_accuracy,'g', label='without stop words')

plt.title("Without stop words VS With stop words (Unigram): Accuracy")

plt.xlabel("Number of features")

plt.ylabel("Validation set accuracy")

plt.legend()
plt.show()
plt.hold()

##COMPARISION BETWEEN UNIGRAM TRIGRAM BIGRAM WITH STOP WORDS

nfeatures_plot_tg = pd.DataFrame(feature_result_tg,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_bg = pd.DataFrame(feature_result_bg,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])

plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,'r',label='trigram')
plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,'g',label='bigram')
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy,'b',label='unigram')
plt.title("N-gram(1~3) test result : Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()
plt.show()

## COMPARISION BETWEEN UNIGRAM TRIGRAM BIGRAM WITH CUSTOM STOP WORDS
'''print("RESULT FOR TRIGRAM WITH CUSTOM STOP WORDS (Top 10 frequent words)\n")
feature_result_tgwocsw = nfeature_accuracy_checker(stop_words=my_stop_words,ngram_range=(1, 3))

print("RESULT FOR BIGRAM WITHOUT CUSTOM STOP WORDS (Top 10 frequent words)\n")
feature_result_bgwocsw = nfeature_accuracy_checker(stop_words=my_stop_words,ngram_range=(1, 2))

print("RESULT FOR UNIGRAM WITHOUT CUSTOM STOP WORDS (Top 10 frequent words)\n")
feature_result_ugwocsw = nfeature_accuracy_checker(stop_words=my_stop_words)'''


nfeatures_plot_tg = pd.DataFrame(feature_result_tgwocsw,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_bg = pd.DataFrame(feature_result_bgwocsw,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ug = pd.DataFrame(feature_result_ugwocsw,columns=['nfeatures','validation_accuracy','train_test_time'])

plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,'r',label='trigram')
plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,'g',label='bigram')
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy,'b',label='unigram')
plt.title("N-gram(1~3) test result : Accuracy(WITHOUT CUSTOM STOP WORDS)")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()
plt.show()


