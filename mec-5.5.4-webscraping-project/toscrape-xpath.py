import scrapy

class QuotesSpider(scrapy.Spider):
    name = "toscrape-xpath"
    start_urls = [
        'http://quotes.toscrape.com/page/1/',]

    def parse(self, response):
        for index, quote in enumerate(response.xpath('//div[contains(@class, "quote")]')):
            yield {
                'text': quote.xpath('span/text()').extract_first(),
                'author': quote.xpath('span//small/text()').extract_first(),
                'tags': quote.xpath('div[contains(@class, "tags")]//a/text()').extract()
            }
        next_page = response.xpath('//li[contains(@class, "next")]//a/@href').extract_first()
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)
