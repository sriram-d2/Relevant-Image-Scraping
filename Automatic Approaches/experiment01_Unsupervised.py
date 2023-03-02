import os
import pandas as pd
import numpy as np
import time

from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

# df1 = pd.read_csv('test_relimage_csv.csv', sep=',')
# df2 = pd.read_csv('train_relimage_csv.csv', sep=',')
# df = pd.concat([df1, df2])

df = pd.read_csv('training_Dataset_AllValues_News.csv', sep='\t')

test_names = ['Bhardwaj and Mangat, 2014: WidthHeight>120000', 'Helfman and Hollan, 200: FileSize', 'Width', 'Height', 'WidthHeight', 'W>400, H>400', 'W>400, H>300', 
                'Gali et al., 2015', 'Fazal et al., 2019']

tests = [(df['Width'] * df['Height'] > 120000), df['order_filesize'] == 1, df['order_width'] == 1, 
         df['order_height'] == 1, df['order_width_height'] == 1, (df['Width'] > 400) & (df['Height'] > 400), 
         ((df['Width'] > 400) & (df['Height'] > 300) | (df['Width'] > 300) & (df['Height'] > 400)),
         ((df['Width'] * df['Height'] > 10000) & ((df['Width'] / df['Height'] <= 1.8) | (df['Height'] | df['Width'] <= 1.8)) &
         ~df["theImg"].str.contains("logo|banner|header|footer|buttonfree|adserver|now|buy|join|click|affiliate|adv|hits|counterbackground|bg|spirit|templates")),
         (df['Width'] > 400) & (df['Height'] > 400 &
         ~df["theImg"].str.contains("free|ads|now|buy|join|click|affiliate|adv|hits|counter|sprite|logo|banner|header|footer|button"))]

print('Method, Accuracy, Precision, Recall, F-Measure, Log Loss, tp, fp, fn, tp')
i = 0
for test in tests:
    start_time = time.time()
    df['Prediction'] = np.where(test, 1, 0)
    tn, fp, fn, tp = confusion_matrix(df['relevantImg'], df['Prediction']).ravel()
    elapsed_time_model = time.time() - start_time
    print(test_names[i], (tn+tp)/(tn + fp + fn + tp), (tp)/(fn + tp), (tp)/(fp + tp), (2 * tp / (2 * tp + fp + fn)), log_loss(df['relevantImg'], df['Prediction']), tn, fp, fn, tp, elapsed_time_model)
    i += 1

# Aanshi Bhardwaj and Veenu Mangat. 2014. An improvised algorithm for relevant content extraction from web pages. Journal of Emerging Technologies in Web Intelligence 6, 2 (may 2014), 226–230. https://doi.org/10.4304/jetwi.6.2.226-230
# Jonathan I. Helfman and James D. Hollan. 2000. Image representations for accessing and organizing Web information. In Internet Imaging II, Giordano B. Beretta and Raimondo Schettini (Eds.), Vol. 4311. International Society for Optics and Photonics, SPIE, 91 – 101. https://doi.org/10.1117/12.411880
# Najlah Gali., Andrei Tabarcea., and Pasi Fränti. 2015. Extracting Representative Image from Web Page. In Proceedings of the 11th International Conference on Web Information Systems and Technologies - WEBIST,. INSTICC, SciTePress, Portugal, 411–419. https://doi.org/10.5220/0005438704110
# Nancy Fazal, Khue Nguyen, and Pasi Fränti. 2019. Efficiency of Web Crawling for Geotagged Image Retrieval. Webology 16 (2019), 16–39. https://doi.org/10.14704/WEB/V16I1/a177

