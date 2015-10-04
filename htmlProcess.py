from __future__ import division
__author__ = 'wangjiewen'

import BeautifulSoup
import nltk, re, pprint
from urllib import urlopen


def test_parse_html():
    urlpath = 'http://www.superlib.cn'
    html = urlopen(urlpath).read()
    soup = BeautifulSoup.BeautifulSoup(html.decode('utf-8'))
    title = soup.find('title')
    print soup.title.text
    print(title.name)
    print(title.text)

    detail = soup.find('div', {'class': 'detail'})
    span = detail.find('p')
    print span.text
    return

if __name__ == '__main__':
    test_parse_html()