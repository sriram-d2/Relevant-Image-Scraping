import random
import Levenshtein.StringMatcher as lev 
import numpy as np
from sklearn.cluster import dbscan
import math

import ClusteringStep as cs

#output: clustersize: 5 or 6 (we recommend these values with our tests)
pages, urls, clus_size = cs.find_clusters_given_website(5)
print(urls, clus_size)

#result: 
# 'https://www.dailysabah.com/world/europe/royal-ceremony-proclaims-charles-as-king-queens-funeral-on-sept-19', 
# 'https://www.dailysabah.com/search?qsection=world&pgno=95', 
# 'https://www.dailysabah.com/search?qsection=world&pgno=105', 
# 'https://www.dailysabah.com/world/new-covid-cases-deaths-on-decline-worldwide-who/news', 
# 'https://www.dailysabah.com/world/mid-east/iran-iaea-standoff-last-major-hurdle-in-reviving-nuke-deal'
# 5
