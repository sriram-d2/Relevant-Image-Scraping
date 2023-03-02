import scrapy

# get min 100 URLs for using in clustering based URL selection step
class DailySabahSpider_CreateModel(scrapy.Spider):
    name = 'DailySabahSpider'
    start_urls = ['https://www.dailysabah.com/search?qsection=world&pgno=100']

    def parse(self, response):
        #get minimum 100 urls for using in the clustering step

        #urls 1, store in the file ("URLs.csv")
        urls = response.css('h3 a::attr(href)').extract()
        for theUrl in urls:
            with open('URLs.csv', 'a') as f:
                f.write(str(theUrl) + '\n')
        
        #urls 2, store in the file ("URLs.csv")
        urls = response.css('a.arrow_link.hover_arrow_link::attr(href)').extract()
        for theUrl in urls:
            with open('URLs.csv', 'a') as f:
                f.write(str(theUrl) + '\n')
        
        #You can specify the number of URLs you want to get (our setting minimum 100 URLs)
        with open('URLs.csv', 'r') as f:
            if len(f.readlines())>=100:
                print("The End...")
                return None
            
        #goto next page, if the number of URLs is less than 100
        for next_page in response.css('a.arrow_link.hover_arrow_link'):
            yield response.follow(next_page, self.parse)