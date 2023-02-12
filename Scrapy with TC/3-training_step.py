import scrapy
import re
import os.path

# Model 
class DailySabahSpider_CreateModel(scrapy.Spider):
    name = 'DailySabahSpider'

    # get from 2-prepare-URLs.py
    start_urls = [
        'https://www.dailysabah.com/world/europe/royal-ceremony-proclaims-charles-as-king-queens-funeral-on-sept-19', 
        'https://www.dailysabah.com/search?qsection=world&pgno=95', 
        'https://www.dailysabah.com/search?qsection=world&pgno=105', 
        'https://www.dailysabah.com/world/new-covid-cases-deaths-on-decline-worldwide-who/news', 
        'https://www.dailysabah.com/world/mid-east/iran-iaea-standoff-last-major-hurdle-in-reviving-nuke-deal']
    
    # open a browser, search relevant images, copy the src of the images and paste them here
    # we find three relevant images for 5 web pages
    given_relevant_images = [
        'src="https://idsb.tmgrup.com.tr/ly/uploads/images/2022/09/10/thumbs/800x531/230123.jpg?v=1662838000"',
        'src="https://idsb.tmgrup.com.tr/ly/uploads/images/2022/08/31/thumbs/800x531/228120.jpg?v=1661966273"',
        'src="https://idsb.tmgrup.com.tr/ly/uploads/images/2022/08/26/thumbs/800x531/227103.jpg?v=1661507864"'
    ]

    # parse web pages and prepare the training dataset for images of 5 web pages
    def parse(self, response):
        html_body = response.xpath("//body").extract()
        html_body = str(html_body).replace('\\n', '').replace('\\t', '').replace('\\r', '')
        html_body = html_body.replace('\n', '').replace('\t', '').replace('\r', '')
        html_body = re.sub('<script.*?</script>', '', html_body);
        html_body = re.sub('<noscript.*?>', '', html_body); 
        html_body = re.sub(' +', ' ', html_body); 
        
        imgs = re.findall("<img.*?>", html_body)
        par_imgs = []
        texts = []
        relevant = []
        for theImg in imgs:
            textual_data = self.find_parents(str(html_body), theImg)
            texts.append(textual_data)

            temp = 0
            if any(ext in theImg for ext in self.given_relevant_images):
                temp = 1
            relevant.append(temp)

            if os.path.exists('training_dataset.csv'):
                with open('training_dataset.csv', 'a') as f:
                    f.write(textual_data + '\t' + str(temp) + '\n')
            else:
                with open('training_dataset.csv', 'w') as f:
                    f.write('texts\trelevant\n')

        yield {'imgs': par_imgs}

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
