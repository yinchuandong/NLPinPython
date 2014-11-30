from __future__ import division
__author__ = 'wangjiewen'

import BeautifulSoup
import nltk, re, pprint
from urllib import urlopen


urlpath = 'http://www.superlib.cn'
html = urlopen(urlpath).read()

soup = BeautifulSoup.BeautifulSoup(html.decode('utf-8'))

title = soup.find('title')


print(title.name)
print(title.text)