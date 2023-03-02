import os
from datasets import temp_seed
import pandas as pd
from sklearn import preprocessing 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

import numpy as np
import random
import Levenshtein.StringMatcher as lev 
from sklearn.cluster import dbscan 
from sklearn.cluster import OPTICS

import time
import math

import threading

import ClusteringStep as cs


def classification(ML_method = 'Text', selection_method="Clustering", trainingSize = 5, vectorizer = "Count"):
    # our test: 200 websites 
    dir_list = get_Websites()
    # five test for determining signifance
    for test_no in range(4):
        for theDir in dir_list:
            theDir_temp =  theDir.split("-")
            train_web_page_count = -1

            if selection_method == "Clustering": #the other random selection
                train_web_page_count = trainingSize
                trainingSize = -1

            train, test, l = flat_train_test_df(ProjectName=theDir_temp[0], train_web_page_count = train_web_page_count, method = ML_method, filename = theDir, trainingSize=trainingSize)
            train = train.drop(['Number'], axis=1)
            test = test.drop(['Number'], axis=1)
            try:
                if ML_method == 'Text':
                    # Number, WebSite: additional information fro test
                    X_train_text, y_train = [doc for doc in train.iloc[:,0]], [doc for doc in train.iloc[:,1]]
                    if vectorizer == "Count":
                        # CountVectorizer, preferred vectorizer
                        my_vectorizer = CountVectorizer(analyzer = 'word', tokenizer=my_tokenizer) #
                        X_train_cnt = my_vectorizer.fit_transform(X_train_text)
                    else: #tfidf
                        my_vectorizer = TfidfVectorizer(analyzer = 'word', tokenizer=my_tokenizer) #
                        X_train_cnt = my_vectorizer.fit_transform(X_train_text)
                    
                    # testing
                    X_test_text, y_test = [doc for doc in test.iloc[:,0]], [doc for doc in test.iloc[:,1]]
                    X_test_cnt = my_vectorizer.transform(X_test_text)      
                else:
                    X_train, y_train = train.drop('main_image', axis=1), train['main_image']
                    X_test, y_test = test.drop('main_image', axis=1), test['main_image']
            except:
                continue
            
            #you can add the other classifiers, if you have time :) 
            clf_names = ["AdaBoost"] 
            # , "Random Forest", "Decision Tree", "RBF SVM", "Nearest Neighbors", "Neural Net", "ADA-DT"
            # "Naive Bayes",
            classifiers = [AdaBoostClassifier()]
            # , RandomForestClassifier(), DecisionTreeClassifier(), KNeighborsClassifier(), SVC(), MLPClassifier(), AdaBoostClassifier(DecisionTreeClassifier())
            #GaussianNB(),
            
            resampling = False # no impact
            for clf_name, clf in zip(clf_names, classifiers):
                try:
                    if ML_method == 'Text':
                        classifier, c_time, stop = create_model(X_train_cnt, y_train, clf)    
                        if not stop:  
                            y_pred_cnt, p_time_cnt = model_predict(X_test_cnt, classifier)
                            y0, y1 = y_train.count(0), y_train.count(1)
                            calculate_res(clf_name, test_no, "count", y_test, y_pred_cnt, ML_method, theDir_temp[0], train_web_page_count, resampling, y0, y1, X_train_cnt.shape[0], c_time, p_time_cnt, l, trainingSize)
                    else:
                        classifier, c_time, stop = create_model(X_train, y_train, clf)    
                        if not stop:  
                            y_pred, p_time = model_predict(X_test, classifier)
                            y0, y1 = y_train.tolist().count(0), y_train.tolist().count(1)
                            calculate_res(clf_name, test_no, "no_vector", y_test, y_pred, ML_method, theDir_temp[0], train_web_page_count, resampling, y0, y1, X_train.shape[0], c_time, p_time, l, trainingSize)
                except:
                    print('Error - 1')
    return

def get_Websites():
    path = 'websites/'
    dir_list = os.listdir(path)
    dir_list = [w.replace('map_url_to_number_', '') for w in dir_list]
    return [w.replace('.txt', '') for w in dir_list]

def create_model(X, y, clf):
    ts = time.time()
    #if no relevent or irrelevant image in dataset, stop create model
    if y.count(0) == 0 or y.count(1) == 0:
        stop = True
    else:
        clf.fit(X, y)
        stop = False
    te = time.time()
    return clf, te - ts, stop

# prediction step
def model_predict(X_test, clf):
    ts = time.time()
    y_pred = clf.predict(X_test)
    te = time.time()
    return y_pred, te - ts

# create output for evuating models, parameters etc...
def calculate_res(clf_name, test_no, vector_method, y, pred, method, theWebsite, train_web_page_count, resampling, neg, pos, record_count, c_time, p_time, l, trainingSize):
    tn, fp, fn, tp = confusion_matrix(y,  pred).ravel()
    logL = log_loss(y,  pred)
    acc, recall, pre, fmeasure = (tn+tp)/(tn + fp + fn + tp), (tp)/(fn + tp), (tp)/(fp + tp), (2*tp) / (2*tp + fp + fn)
    print('{} {} {}: Accuracy={} Recall={} Precision={} F-Measure={} logLoss={} Page_Count={} neg={} pos={} resampling={} recordCount={}'.format(method , theWebsite, clf_name, acc, recall, pre, fmeasure, logL, train_web_page_count, neg, pos, resampling, record_count))
    if train_web_page_count == -1:
        tmp = "cluster"
    else:
        tmp = "nocluster"

    lock = threading.Lock()
    lock.acquire()
    with open('tester_adaboost_' + tmp + '.txt', 'a') as a_writer:
        a_writer.write(method + '\t' + str(test_no) + '\t' + vector_method + '\t' + theWebsite + '\t' + clf_name + '\t' + str(acc) + '\t' + str(recall) + '\t' + str(pre) + '\t' + str(fmeasure)+ '\t' + str(logL) + '\t' + str(train_web_page_count) + '\t' + str(neg) + '\t' + str(pos) + '\t' + str(resampling) + '\t' + str(record_count)  + '\t' + str(c_time)  + '\t' + str(p_time) + '\t' + str(len(l)) + '\t' + str(l) + '\t' + str(trainingSize)  + '\n')
    lock.release()

def flat_train_test_df(ProjectName = 'adalet', train_web_page_count = 3, method = 'Text', filename = 'adalet-az', trainingSize=5):
    df = pd.read_csv('training_Dataset_AllValues_News.csv', sep='\t')
    df = df[df['WebSite'].str.contains(ProjectName)]
    if method == 'Text':
        df['Text'] = df['theImg'] + ' ' + df['Parent1'] + ' ' + df['Parent2']
        df = pd.concat([df['Number'], df['Text'], df['main_image']], axis=1)
    else:
        df = df.drop(['WebSite', 'theImg', 'Parent1', 'Parent2', 'Scr', 'ratio_theimg_allimgs'], axis=1)
        le = preprocessing.LabelEncoder()
        le.fit(df['file_ext'])
        df['file_ext'] = le.transform(df['file_ext'])
        df = df.fillna(0)

    l = []
    if train_web_page_count == -1:
        l = cs.find_clusters_given_website(filename, trainingSize)
        print(l)
    else:
        l = random.sample(range(1, 101), train_web_page_count)

    return df[df['Number'].isin(l)], df[~df['Number'].isin(l)], l

# tokenizer for html...
def my_tokenizer(s):
    s = s.replace('<', '').replace('>','').strip()
    s = s.replace('//', ' ').replace('/', ' ').replace('.', ' ').replace('?', ' ').replace(';', ' ').replace('  ',' ').replace('   ',' ') #scr,
    return s.split(' ')

# no impact on performance results...
from imblearn.over_sampling import RandomOverSampler
def reSampling(X, y):
    ros = RandomOverSampler(random_state=42)
    return ros.fit_sample(X, y)

#test, range web page count
#classfication function use train_web_page_count: selection of web pages 
# for i in range(4, 5):
#     thr1 = threading.Thread(target=classification, args=("Text", "Clustering", i, "Count")) #clustering
#     thr2 = threading.Thread(target=classification, args=("Text", "Random Selection", i, "Count")) #random page selection
#     thr1.start()
#     thr2.start()

#use cluster size for selection
classification("Text", "Clustering", -1, "Count")