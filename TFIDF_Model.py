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
from sklearn.feature_extraction.text import TfidfVectorizer



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





tvec = TfidfVectorizer()

print("\nUnigram with stop words\n")
feature_result_ugt = nfeature_accuracy_checker(vectorizer=tvec)
print("\nBigram with stop words\n")
feature_result_bgt = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 2))
print("\nTrigram with stop words\n")
feature_result_tgt = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 3))


nfeatures_plot_tgt = pd.DataFrame(feature_result_tgt,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_bgt = pd.DataFrame(feature_result_bgt,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ugt = pd.DataFrame(feature_result_ugt,columns=['nfeatures','validation_accuracy','train_test_time'])
plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_tgt.nfeatures, nfeatures_plot_tgt.validation_accuracy,label='trigram tfidf vectorizer',color='royalblue')
#plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram count vectorizer',linestyle=':', color='royalblue')
plt.plot(nfeatures_plot_bgt.nfeatures, nfeatures_plot_bgt.validation_accuracy,label='bigram tfidf vectorizer',color='orangered')
#plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram count vectorizer',linestyle=':',color='orangered')
plt.plot(nfeatures_plot_ugt.nfeatures, nfeatures_plot_ugt.validation_accuracy, label='unigram tfidf vectorizer',color='gold')
#plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram count vectorizer',linestyle=':',color='gold')
plt.title("TFIDF WITH STOP WORDS N-gram(1~3) test result : Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()
plt.show()
