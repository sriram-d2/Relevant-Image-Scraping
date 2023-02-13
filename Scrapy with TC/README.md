
# Scrapy with TC Approach
Expert-user-driven methods such as CSS Selector, XPath, or Regular Expression can be utilized to extract images using web scraping tools like Scrapy. AI-based techniques can eliminate the need for expert users, but they necessitate a substantial number of features. Our proposed approach focuses on text data, similar to expert-user methods. By annotating a small number of relevant images, we can identify the relevant image without expert intervention in crawling process. We applied this approach to Scrapy, a well-known web scraping tool.

The approach we propose involves four steps:

## 1. Preparing URLs
First, a minimum of 100 URLs from a website are collected using a simple code written in Scrapy (1-prepare-URLs.py, output: URLs.csv).

## 2. Clustering step
In the clustering step, the user is prompted to choose the desired number of web pages. (2-clustering_selection_step.py)

Example output for 5 web pages:
```javascript
'https://www.dailysabah.com/world/europe/royal-ceremony-proclaims-charles-as-king-queens-funeral-on-sept-19', 
'https://www.dailysabah.com/search?qsection=world&pgno=95', 
'https://www.dailysabah.com/search?qsection=world&pgno=105', 
'https://www.dailysabah.com/world/new-covid-cases-deaths-on-decline-worldwide-who/news', 
'https://www.dailysabah.com/world/mid-east/iran-iaea-standoff-last-major-hurdle-in-reviving-nuke-deal'
```

## 3. Training step
The user selects the relevant image from among these pages by right-clicking and inspect on it from the browser, which takes them to the HTML and its textual data. Simply copying the "src" data from here is sufficient. (3-training_step.py, output: training_dataset.csv)

We found 3 related images among the suggested URLs. Here are the src outputs of these three images found by the user:
```javascript
'src="https://idsb.tmgrup.com.tr/ly/uploads/images/2022/09/10/thumbs/800x531/230123.jpg?v=1662838000"',
'src="https://idsb.tmgrup.com.tr/ly/uploads/images/2022/08/31/thumbs/800x531/228120.jpg?v=1661966273"',
'src="https://idsb.tmgrup.com.tr/ly/uploads/images/2022/08/26/thumbs/800x531/227103.jpg?v=1661507864"'
```
In this step, we create a machine learning model for the website by classifying the text of the images obtained from the web pages and the src data of the related images. To do this, we use a text classification technique and in our tests, we found that Adaboost classifier was successful. The resulting model is then utilized in the prediction step."


## 4. Scraping with Relevant Image Prediction
The final step involves making predictions for relevant images on web pages obtained through crawling, using the model established in the previous step. (output: prediction.csv: predictions for all images in web pages...)




