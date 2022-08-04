import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.feature_extraction.text import CountVectorizer

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

from sklearn import svm
import time

def classification(n = 190):
    train, test = flat_train_test_df(n)
    X_train, y_train = train.drop('relevantImg', axis=1), train['relevantImg']
    X_test, y_test = test.drop('relevantImg', axis=1), test['relevantImg']

    X_train_text, y_train = [doc for doc in train.iloc[:,0]], [doc for doc in train.iloc[:,1]]
    vectorizer = CountVectorizer(analyzer = 'word', tokenizer=my_tokenizer) #
    X_train = vectorizer.fit_transform(X_train_text)
    X_test_text, y_test = [doc for doc in test.iloc[:,0]], [doc for doc in test.iloc[:,1]]
    X_test = vectorizer.transform(X_test_text)         

    names = ["AdaBoost", "Random Forest", "Decision Tree", "RBF SVM", "Nearest Neighbors", "Neural Net"]
    classifiers = [AdaBoostClassifier(),RandomForestClassifier(), DecisionTreeClassifier(), SVC(),  KNeighborsClassifier(weights='distance'), MLPClassifier()]

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
    # Number of websites
    df = pd.read_csv('training_Dataset_AllValues_News.csv', sep='\t')

    df['Text'] = df['theImg'] + ' ' + df['Parent1'] + ' ' + df['Parent2']
    df = pd.concat([df['WebSite'], df['Text'], df['relevantImg']], axis=1)

    ls = df['WebSite'].unique()

    df1 = df.loc[(df['WebSite'].isin(ls[:n]))]
    df2 = df.loc[(df['WebSite'].isin(ls[n:]))]
    df1 = df1.drop(columns=['WebSite'])
    df2 = df2.drop(columns=['WebSite'])
    return df1, df2

def my_tokenizer(s):
    s = s.replace('<', '').replace('>','').strip()
    s = s.replace('//', ' ').replace('/', ' ').replace('.', ' ').replace('?', ' ').replace(';', ' ').replace('  ',' ').replace('   ',' ') #scr,
    return s.split(' ')

for n in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]:
    classification(n)