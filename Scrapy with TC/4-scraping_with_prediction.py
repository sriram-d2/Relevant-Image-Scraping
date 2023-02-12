import scrapy
import pandas as pd
# we recommend adaBoostClassifier after our tests
from sklearn.ensemble import AdaBoostClassifier
# we recommend CountVectorizer after our tests, also different vectorizers can be avulated
from sklearn.feature_extraction.text import CountVectorizer
import time
import re

class DailySabahSpider(scrapy.Spider):
    name = 'DailySabahSpider'
    start_urls = ['https://www.dailysabah.com/search?qsection=world&pgno=1']

    def __init__(self):
        #prepare the model for the training dataset prepared in the previous step
        train_df = pd.read_csv('training_dataset.csv', sep='\t')
        print(train_df.columns)
        X_train_text, y_train = [doc for doc in train_df['texts']], [doc for doc in train_df['relevant']]
        self.vectorizer = CountVectorizer(analyzer = 'word', tokenizer=self.my_tokenizer) #
        X_train = self.vectorizer.fit_transform(X_train_text)
        clf = AdaBoostClassifier()
        self.classifier_model, c_time, stop = self.create_model(X_train, y_train, clf)  
        self.vectorizer

    #crawl web pages for a website, and predict the relevant/irrevant of the images in the web pages
    def parse(self, response):
        html_body = response.xpath("//body").extract()
        html_body = str(html_body).replace('\\n', '').replace('\\t', '').replace('\\r', '')
        html_body = html_body.replace('\n', '').replace('\t', '').replace('\r', '')
        html_body = re.sub('<script.*?</script>', '', html_body); 
        
        imgs = re.findall("<img.*?>", html_body)
        par_imgs = []
        for theImg in imgs:
            textual_data = self.find_parents(str(html_body), theImg)
            text_vector = self.vectorizer.transform([textual_data])
            pred = self.classifier_model.predict(text_vector)[0]
            with open('prediction.csv', 'a') as f:
                f.write(theImg + "\t" + str(pred) + '\n')
            par_imgs.append(str(pred))
        yield {'imgs': par_imgs}

        for next_page in response.css('h3 a'):
            yield response.follow(next_page, self.parse)

    # find two parents of the image
    def find_parents(self, html_body, img_url):
        pos_img = html_body.find(img_url)
        start = pos_img 
        open_ch, close_ch = 0, 0
        while open_ch - close_ch != 2:
            pos_parent_img = html_body[0:start].rfind("<")
            if pos_parent_img != -1:
                all_ch = html_body[pos_parent_img:pos_img].count("<")
                close_ch = html_body[pos_parent_img:pos_img].count("</")
                open_ch = all_ch - close_ch
                start = pos_parent_img
            else:
                break
        return html_body[start:pos_img + len(img_url)]
    
    #tokenize the html text
    def my_tokenizer(self, s):
        s = s.replace('<', '').replace('>','').strip()
        s = s.replace('//', ' ').replace('/', ' ').replace('.', ' ').replace('?', ' ').replace(';', ' ').replace('  ',' ').replace('   ',' ') #scr,
        return s.split(' ')
    
    def create_model(self, X, y, clf):
        ts = time.time()
        yy = y
        if yy.count(0) == 0 or yy.count(1) == 0:
            stop = True
        else:
            clf.fit(X, y)
            stop = False    
        te = time.time()
        return clf, te - ts, stop




