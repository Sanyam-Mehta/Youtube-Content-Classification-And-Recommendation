import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from nltk.corpus import stopwords as stp

global category_map, clf, columns, out

def Category(c_id):
    return category_map.get(str(c_id))

def run():
    global category_map, clf, columns
    usComments = pd.read_csv("/home/nikita/Desktop/FlaskProject/GBcomments.csv", low_memory = False, error_bad_lines = False, encoding = 'utf-8')
    usVideos = pd.read_csv("/home/nikita/Desktop/FlaskProject/GBvideos .csv", low_memory = False, error_bad_lines = False, encoding = 'utf-8')
        
    category_map = {}
    with open('/home/nikita/Desktop/FlaskProject/GB_category_id.json') as f:
        contents = json.load(f)
        for i in range(len(contents['items'])):
            key = contents['items'][i]['id']
            category_map[key] = contents['items'][i]['snippet']['title']

    df = pd.DataFrame()
    df["tags"] = usVideos["tags"]
    df["target"] = usVideos["category_id"]
    df['category'] = df['target'].apply(Category)

    tags_list = list()
    target_list = list()
    for i in range(len(df)):
        tags_list.append(df.iloc[i,0].split("|"))
        target_list.append(df.iloc[i,1])
        
    
    x_train,x_test,y_train,y_test = train_test_split(tags_list, target_list, test_size = 0.2, random_state = 1)

    stopwords = stp.words('english')

    stopwords_i = [word.title() for word in stopwords] # list of all stopwords appearing as Title words(eg. Abc )
    stopwords_u = [word.upper() for word in stopwords] # list of all stopwords appearing as CAPITALISED WORDS(eg. ABC)
    others = [str(i) for i in range(101)] + [str(i) + "." for i in range(101)]
    sp = ["''",'""',"'",'"','!','~','`','!','@','#','$','%','^','&','*','/','-','+','_',':)','**','***','_/',
          '----------------------------------------------------------------------',
          '[]','{}','()','\\','|','[',']','(','|>','>>','<<',
    ')','{','}',',','.','?','=','<=','>=','<','>',':',';','A','B','Q','W','E','R','T','Y','U','I','O','P','S','D','F',
          'G','H','J','K','L','Z','X','C','V','N','M','A.','B.','Q.','W.','E.','R.','T.','Y.','U.','I.','O.','P.',
          'S.','D.','F.','G.','H.','J.','K.','L.','Z.','X.','C.','V.','N.','M.','None','none','NONE']
    stopwords = stopwords + stopwords_i + stopwords_u + others + sp
    vocab = {}
    for i in range(len(x_train)):
            for j in range(len(x_train[i])):
                if x_train[i][j] not in stopwords:
                    if vocab.get(x_train[i][j]) != None: # if the feature already exists in the vocab dictionary already built
                        vocab[x_train[i][j]] += 1
                    else:
                        vocab[x_train[i][j]] = 1
        
    voc = vocab
    voc = sorted(voc.items(), key=lambda x: x[1], reverse=True) #Sorting the words according to their frequencies in the dataset
    col = voc[0:2000] #Choosing top 2000 most occuring words
    col = dict(col) 
    columns = list(col.keys()) #list of all 2000 words

    x_train_ = np.array([[0] * len(col) for i in range(len(x_train))]) 

    for i in range(len(x_train)):
        for j in range(len(x_train[i])):
            if x_train[i][j] in columns: 
                pos = columns.index(x_train[i][j])
                x_train_[i][pos] += 1

    y_train_ = np.array(y_train)

    x_test_ = np.array([[0] * len(col) for i in range(len(x_test))])

    for i in range(len(x_test)):
        for j in range(len(x_test[i])):
            if x_test[i][j] in columns:
                pos = columns.index(x_test[i][j])
                x_test_[i][pos] += 1


    y_test_ = np.array(y_test)
    clf = MultinomialNB()
    clf.fit(x_train_, y_train_)
    y_pred_sklearn = clf.predict(x_test_)

    #Multinomial Logistic Regression    
    clfLR = LogisticRegression(solver = "sag", multi_class = "multinomial")
    clfLR.fit(x_train_, y_train_)
    y_pred_LR = clfLR.predict(x_test_)

    clfKNN = KNeighborsClassifier(n_neighbors = 3)
    clfKNN.fit(x_train_, y_train_)
    y_pred_KNN = clfKNN.predict(x_test_)
    return category_map, columns

def getCategory(searchQuery):
    global out
    global columns, clf, category_map
    tags = searchQuery.split("|")
    for tag in tags:
        tag = tag.strip()
    input_predict = [0 for i in range(len(columns))]
    for word in tags:
        if word in columns:
            pos = columns.index(word)
            input_predict[pos] += 1
            
    input_predict = [input_predict]
    
    output_predict = np.asscalar(clf.predict(input_predict))
    out = category_map.get(str(output_predict))
    return output_predict

