import pandas as pd
import numpy as np
import math
import random

import Levenshtein.StringMatcher as lev 
from sklearn.cluster import dbscan 

def find_clusters_given_website(website, trainingSize=5):
    URLs = get_URLS(website)
    return find_clusters(URLs, trainingSize)

def get_URLS(website):
    df = pd.read_csv('websites/map_url_to_number_'+ website +'.txt', sep='\t', names=['Number', 'WebSite', 'Url', 'Filzesize', 'ImgCount'], encoding='latin1')
    data = df['Url'].tolist()
    d_list = df['Number'].tolist()
    return data, d_list

def find_clusters(URLs, trainingSize):
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
    
    #prepare URL data: data -> urls, d_list -> id
    data, d_list = URLs
    lenURLs = len(data)

    X = np.arange(len(data)).reshape(-1, 1)

    clustering = dbscan(X, metric=lev_metric, eps=5, min_samples=2, algorithm='brute')
    selected_clus_d = clus_to_dict(clustering)

    selected_pages = []
    info_clusters = {}

    for key in selected_clus_d:
        info_clusters[key] = len(selected_clus_d[key])
        selected_pages.append(d_list[random.choice(selected_clus_d[key])])
        if trainingSize != -1 and len(selected_pages) == trainingSize:
            break
    
    if trainingSize != -1:
        rest_trainingsize = trainingSize - len(selected_pages)
    else:
        rest_trainingsize = -1
    
    if rest_trainingsize > 0:
        for key in info_clusters.keys():
           info_clusters[key] = info_clusters[key] * rest_trainingsize / lenURLs

        sort_orders = sorted(info_clusters.items(), key=lambda x: x[1], reverse=True)
        for i in sort_orders:
            if i[1] > 0:
                for j in range(math.ceil(i[1])):
                    selected_pages.append(d_list[random.choice(selected_clus_d[i[0]])])
                    if len(selected_pages) == trainingSize:
                        break
            if len(selected_pages) == trainingSize:
                break

    return selected_pages

output = find_clusters_given_website("bd-pratidin-bd", 3)
print(output)