# Relevant Image Scraping

This repository includes codes for current methods, new features, and new techniques in extracting images from related web pages. The codes also include tests on a dataset that includes 100 web pages from each of the 200 websites.

You can download the datasets via https://adys.nku.edu.tr/Datasets/.

## Novel features for Automatic Approaches

In this study, tests were conducted using the features gathered from web pages during the crawling of a website. It was found that the cached feature was particularly noteworthy. The results and evaluations of the tests were compared with related literature (Uzun et al., 2020).

## Novel approach (TC - Text-based Classfication) 

In this study, we focused exclusively on the text data on web pages. By doing so, we showed that web image scraping can be accomplished using only text data (the text of images and their parent elements) without downloading the images and without needing to gather many features. Our approach is easily adaptable to existing web scraping tools, as demonstrated by its integration with the Scrapy application (Python - Directory: Scrapy with TC). Our approach offers a highly resource-efficient and time-efficient solution for web image scraping.

Hap(py)codes...

# References
Uzun, E., Ozhan, E., Agun, H. V., Yerlikaya, T., & Bulus, H. N. (2020). Automatically Discovering Relevant Images From Web Pages. IEEE Access, 8, 208910–208921. Retrieved from https://doi.org/10.1109/access.2020.3039044

# External links
<a href="https://erdincuzun.com/yayinlar/" target="_blank">Click for bibtex, downloads, all publications...</a><br />
<a href="https://scrapy.org/" target="_blank">Click for Scrapy</a>
