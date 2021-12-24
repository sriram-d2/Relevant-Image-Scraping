import os
import pandas as pd
import random

from sklearn import preprocessing 

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

from sklearn import svm
import time

def classification(n = 190):
    train, test = flat_train_test_df(n)
    X_train, y_train = train.drop('relevantImg', axis=1), train['relevantImg']
    X_test, y_test = test.drop('relevantImg', axis=1), test['relevantImg']
    names = ["AdaBoost", "Random Forest", "Decision Tree", "RBF SVM", "Nearest Neighbors", "Neural Net"]
    classifiers = [
        AdaBoostClassifier(),
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        SVC(),    
        KNeighborsClassifier(weights='distance'),
        MLPClassifier()
    ]

    for name, clf in zip(names, classifiers):
        start_time = time.time()
        clf = clf.fit(X_train, y_train)
        elapsed_time_model = time.time() - start_time
        start_time = time.time()
        y_pred = clf.predict(X_test)
        elapsed_time_prediction = time.time() - start_time
        # prediction
        calculate_res(y_test, y_pred, name, n, elapsed_time_model, elapsed_time_prediction)
    return

def calculate_res(y, pred, test_name, theWebsite, modeltime, predtime):
    tn, fp, fn, tp = confusion_matrix(y,  pred).ravel()
    logL = log_loss(y,  pred)
    acc, recall, pre, fmeasure = (tn+tp)/(tn + fp + fn + tp), (tp)/(fn + tp), (tp)/(fp + tp), (2*tp) / (2*tp + fp + fn)
    print('{} {}: Accuracy={} Recall={} Precision={} F-Measure={} logLoss={} modeltime={} predtime={} '.format(test_name ,theWebsite, acc, recall, pre, fmeasure, logL, modeltime, predtime))
 
def flat_train_test_df(n = 10):
    df = pd.read_csv('training_Dataset_AllValues_News.csv', sep='\t')

    df['parent1tag'] = df['Parent1'].apply(lambda x: x[1: x.find(' ')] if x.find(' ') != -1 else x[1: x.find('>')])
    df['parent2tag'] = df['Parent2'].apply(lambda x: x[1: x.find(' ')] if x.find(' ') != -1 else x[1: x.find('>')])

    df = df.drop(columns=['theImg', 'Parent1', 'Parent2', 'Scr'])

    le = preprocessing.LabelEncoder()
    le.fit(df['fileext'].astype(str))
    df['fileext'] = le.transform(df['fileext'])
    le.fit(df['parent1tag'].astype(str))
    df['parent1tag'] = le.transform(df['parent1tag'])
    le.fit(df['parent2tag'].astype(str))
    df['parent2tag'] = le.transform(df['parent2tag'])

    ls = df['WebSite'].unique()

    df1 = df.loc[(df['WebSite'].isin(ls[:n]))]
    df2 = df.loc[(df['WebSite'].isin(ls[n:]))]
    df1 = df1.drop(columns=['WebSite'])
    df2 = df2.drop(columns=['WebSite'])
    return df1, df2


for n in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]:
    classification(n)


