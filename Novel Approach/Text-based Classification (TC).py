import os
import pandas as pd
from sklearn import preprocessing 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
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

def classification(method = 'Text', train_web_page_count = -1, clus_size = 5):
    # path = 'DS/News/'
    # dir_list = os.listdir(path)
    path = 'websites/'
    dir_list = os.listdir(path)
    dir_list = [w.replace('map_url_to_number_', '') for w in dir_list]
    dir_list = [w.replace('.txt', '') for w in dir_list]

    # dir_list = dir_list[0:4]  
    for theDir in dir_list:
        theDir_temp =  theDir.split("-")
        train, test, l = flat_train_test_df(ProjectName=theDir_temp[0], train_web_page_count = train_web_page_count, method = method, filename = theDir, clus_size=clus_size)
        train = train.drop(['Number'], axis=1)
        test = test.drop(['Number'], axis=1)

        if method == 'Text':
            # Number, WebSite: additional information fro test
            X_train_text, y_train = [doc for doc in train.iloc[:,0]], [doc for doc in train.iloc[:,1]]
            vectorizer = CountVectorizer(analyzer = 'word', tokenizer=my_tokenizer) #
            X_train = vectorizer.fit_transform(X_train_text)
            print(X_train.shape[0], y_train.count(1), y_train.count(0))
            X_test_text, y_test = [doc for doc in test.iloc[:,0]], [doc for doc in test.iloc[:,1]]
            X_test = vectorizer.transform(X_test_text)         
        else:
            X_train, y_train = train.drop('main_image', axis=1), train['main_image']
            X_test, y_test = test.drop('main_image', axis=1), test['main_image']
        
        clf_names = ["AdaBoost", "Random Forest", "Decision Tree", "RBF SVM", "Nearest Neighbors", "Neural Net", 
          "ADA-DT"] #"Naive Bayes",
        classifiers = [AdaBoostClassifier(), RandomForestClassifier(), DecisionTreeClassifier(), KNeighborsClassifier(), SVC(), MLPClassifier(), AdaBoostClassifier(DecisionTreeClassifier())]
            #GaussianNB(),
        
        resampling = False
        for clf_name, clf in zip(clf_names, classifiers):
            try:
                classifier, c_time, stop = create_model(X_train, y_train, clf)    
                if not stop:  
                    y_pred, p_time = model_predict(X_test, classifier)
                    if method == 'Text':
                        y0, y1 = y_train.count(0), y_train.count(1)
                    else:
                        y0, y1 = y_train.tolist().count(0), y_train.tolist().count(1)
                    calculate_res(clf_name, y_test, y_pred, method, theDir_temp[0], train_web_page_count, resampling, y0, y1, X_train.shape[0], c_time, p_time, l, clus_size)
            except:
                print('Hata Oluştu - 1')
        
        try:
            resampling = True
            X_train, y_train = reSampling(X_train, y_train)
            for clf_name, clf in zip(clf_names, classifiers):
                try:
                    classifier, c_time, stop = create_model(X_train, y_train, clf)    
                    if not stop:  
                        y_pred, p_time = model_predict(X_test, classifier)
                        if method == 'Text':
                            y0, y1 = y_train.count(0), y_train.count(1)
                        else:
                            y0, y1 = y_train.tolist().count(0), y_train.tolist().count(1)
                        calculate_res(clf_name, y_test, y_pred, method, theDir_temp[0], train_web_page_count, resampling, y0, y1, X_train.shape[0], c_time, p_time, l, clus_size)
                except:
                    print('Hata Oluştu - 2')
        except:
            print('Resampling hatası... - 3')

    return

def create_model(X, y, clf):
    ts = time.time()
    # yy = y.tolist()
    yy = y
    print(X.shape, yy.count(0), yy.count(1), "=", yy.count(0) + yy.count(1))
    if yy.count(0) == 0 or yy.count(1) == 0:
        stop = True
    else:
        clf.fit(X, y)
        stop = False
    te = time.time()
    return clf, te - ts, stop

def model_predict(X_test, clf):
    ts = time.time()
    y_pred = clf.predict(X_test)
    te = time.time()
    return y_pred, te - ts

def calculate_res(clf_name, y, pred, method, theWebsite, train_web_page_count, resampling, neg, pos, record_count, c_time, p_time, l, clus_size):
    tn, fp, fn, tp = confusion_matrix(y,  pred).ravel()
    logL = log_loss(y,  pred)
    acc, recall, pre, fmeasure = (tn+tp)/(tn + fp + fn + tp), (tp)/(fn + tp), (tp)/(fp + tp), (2*tp) / (2*tp + fp + fn)
    print('{} {} {}: Accuracy={} Recall={} Precision={} F-Measure={} logLoss={} Page_Count={} neg={} pos={} resampling={} recordCount={}'.format(method , theWebsite, clf_name, acc, recall, pre, fmeasure, logL, train_web_page_count, neg, pos, resampling, record_count))
    with open('tester_random_cluster2.txt', 'a') as a_writer:
        a_writer.write(method + '\t' + theWebsite + '\t' + clf_name + '\t' + str(acc) + '\t' + str(recall) + '\t' + str(pre) + '\t' + str(fmeasure)+ '\t' + str(logL) + '\t' + str(train_web_page_count) + '\t' + str(neg) + '\t' + str(pos) + '\t' + str(resampling) + '\t' + str(record_count)  + '\t' + str(c_time)  + '\t' + str(p_time) + '\t' + str(len(l)) + '\t' + str(l) + '\t' + str(clus_size)  + '\n')

def flat_train_test_df(ProjectName = 'adalet', train_web_page_count = 3, method = 'Text', filename = 'adalet-az', clus_size=5):
    df = pd.read_csv('training_Dataset_AllValues_News.csv', sep='\t')
    df = df[df['WebSite'].str.contains(ProjectName)]
    if method == 'Text':
        df['Text'] = df['theImg'] + ' ' + df['Parent1'] + ' ' + df['Parent2']
        df = pd.concat([df['Number'], df['Text'], df['main_image']], axis=1)
        # print(df.loc[df['main_image'] == 1])
    else:
        df = df.drop(['WebSite', 'theImg', 'Parent1', 'Parent2', 'Scr', 'ratio_theimg_allimgs'], axis=1)
        le = preprocessing.LabelEncoder()
        le.fit(df['file_ext'])
        df['file_ext'] = le.transform(df['file_ext'])
        df = df.fillna(0)

    l = []
    if train_web_page_count == -1:
        l = find_clusters(filename, clus_size, 'random')
    else:
        l = random.sample(range(1, 101), train_web_page_count)

    print(l)

    return df[df['Number'].isin(l)], df[~df['Number'].isin(l)], l

def my_tokenizer(s):
    s = s.replace('<', '').replace('>','').strip()
    s = s.replace('//', ' ').replace('/', ' ').replace('.', ' ').replace('?', ' ').replace(';', ' ').replace('  ',' ').replace('   ',' ') #scr,
    return s.split(' ')

from imblearn.over_sampling import RandomOverSampler
def reSampling(X, y):
    ros = RandomOverSampler(random_state=42)
    return ros.fit_sample(X, y)


def find_clusters(website, clus_size, selection):
    def clus_to_dict(clustering):
        d = {}
        i = 0
        for clus in clustering[1]:
            if clus in d:
                d[clus].append(i)
            else:
                d[clus] = [i]
            i += 1
        return d

    def lev_metric(x, y):
        i, j = int(x[0]), int(y[0])     # extract indices
        return lev.distance(data[i], data[j])
    
    # dir_list = os.listdir('DS/News/' + website + '/')
    df = pd.read_csv('websites/map_url_to_number_' + website +'.txt', sep='\t', names=['Number', 'WebSite', 'Url', 'Filesize', 'ImgCount'], encoding='latin1')
    data = df['Url'].tolist()
    d_list = df['Number'].tolist()
    d_filesize = df['Filesize'].tolist()
    dict_filesize = dict(zip(d_list, d_filesize))

    X = np.arange(len(data)).reshape(-1, 1)

    pos_eps, pos_minsamples, upSize = 0.5, 1, 0.5
    selected_clus_d = {}
    while True:
        clustering = dbscan(X, metric=lev_metric, eps=pos_eps, min_samples=pos_minsamples, algorithm='auto')
        d = clus_to_dict(clustering)

        # print(len(d), " : ", len(selected_clus_d), ", ", pos_eps, ", ", pos_minsamples)
        if clus_size == len(d):
            selected_clus_d = d.copy()
            break #clusters ok
        elif len(selected_clus_d) == 0:
            selected_clus_d = d.copy()
        elif abs(len(d) - clus_size) < abs(len(selected_clus_d) - clus_size): #closer cluster
            selected_clus_d = d.copy()
        else:
            upSize += 0.5
            if upSize > 3: #
                pos_eps = 30.5
        
        if pos_eps > 30:
            pos_minsamples += 1
            if pos_minsamples == 6: 
                break
            pos_eps = 0.5
            upSize = 0.5
        else:
            pos_eps += upSize
 
    selected_pages = []
    d_count = {}
    d_greater = {}
    for key in selected_clus_d:
        d_count[key] = len(selected_clus_d[key])
        if selection == "random":
            selected_pages.append(d_list[random.choice(selected_clus_d[key])])
        else: #biggest value
            d_list_num = [d_list[i] for i in selected_clus_d[key]]
            dict_temp = { theNum: dict_filesize[theNum] for theNum in d_list_num }
            sorted_tuples = sorted(dict_temp.items(), key=lambda item: item[1])
            print(sorted_tuples)
            items = sorted_tuples.pop()
            d_greater[key] = sorted_tuples
            selected_pages.append(items[0])
    
    print(d_count)
    rest_clus = clus_size - len(selected_pages)
    lenData = len(data)
    if rest_clus > 0:
        for key in d_count.keys():
            d_count[key] = d_count[key] * rest_clus / lenData
        # print(d_count)
        sort_orders = sorted(d_count.items(), key=lambda x: x[1], reverse=True)
        for i in sort_orders:
            # print(i[0], i[1])
            if i[1] > 0:
                for j in range(math.ceil(i[1])):
                    if selection == "random":
                        selected_pages.append(d_list[random.choice(selected_clus_d[i[0]])])
                    else:
                        items = d_greater[i[0]].pop()
                        selected_pages.append(items[0])
                    if clus_size == len(selected_pages):
                        break
            if clus_size == len(selected_pages):
                break

    print(selected_pages)
    return selected_pages

for i in range(16, 22):
    classification(method = 'Text', train_web_page_count = i, clus_size = -1)
    classification(method = 'Text', train_web_page_count = -1, clus_size = i)

# for i in range(17, 21):
#     classification(method = 'Features', train_web_page_count = i, clus_size = -1)

# for i in range(17, 21):
#     classification(method = 'Features', train_web_page_count = -1, clus_size = i)


# classification(method = 'Text', train_web_page_count = -1, resampling = False)
# find_clusters('sanspo-jp', 10, 'biggest')
# find_clusters('oamarumail-nz', 10, 'biggest')
# find_clusters('thesaigontimes-vn', 10, 'biggest')
# find_clusters('abendzeitung-muenchen-de', 3, 'biggest')