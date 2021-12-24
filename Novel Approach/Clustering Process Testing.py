import pandas as pd
import numpy as np
import os
import random

import Levenshtein.StringMatcher as lev 
from sklearn.cluster import dbscan 

def find_clusters(website):
    def lev_metric(x, y):
        i, j = int(x[0]), int(y[0])     # extract indices
        return lev.distance(data[i], data[j])
    
    dir_list = os.listdir('DS/News/' + website + '/')
    df = pd.read_csv('DS/News/' + website + '/map_url_to_number.txt', sep='\t', names=['Number', 'WebSite', 'Url', 'Filzesize', 'ImgCount'], encoding='latin1')
    data = df['Url'].tolist()
    X = np.arange(len(data)).reshape(-1, 1)
    clustering = dbscan(X, metric=lev_metric, eps=5, min_samples=2, algorithm='brute')
    d = {}
    i = 1
    for clus in clustering[1]:
        if clus in d:
            d[clus].append(i)
        else:
            d[clus] = [i]
        i += 1
    selected_pages = []
    for key in d:
        selected_pages.append(random.choice(d[key]))

    print(selected_pages)
    return selected_pages
