import os
import pandas as pd
import numpy as np
import time

from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

df1 = pd.read_csv('test_relimage_csv.csv', sep=',')
df2 = pd.read_csv('train_relimage_csv.csv', sep=',')

df = pd.concat([df1, df2])

test_names = ['WidthHeight>120000', 'FileSize', 'Width', 'Height', 'WidthHeight']
tests = [df['WidthHeight'] > 120000, df['orderFileSize'] == 1, df['orderWidth'] == 1, df['orderHeight'] == 1, df['orderWidthHeight'] == 1]

print('Method, Accuracy, Recall, F-Measure, Log Loss, tp, fp, fn, tp')
i = 0
for test in tests:
    start_time = time.time()
    df['Prediction'] = np.where(test, 1, 0)
    tn, fp, fn, tp = confusion_matrix(df['relevantImg'], df['Prediction']).ravel()
    elapsed_time_model = time.time() - start_time
    print(test_names[i], (tn+tp)/(tn + fp + fn + tp), (tp)/(fn + tp), (tp)/(fp + tp), (2 * tp / (2 * tp + fp + fn)), log_loss(df['relevantImg'], df['Prediction']), tn, fp, fn, tp, elapsed_time_model)
    i += 1



