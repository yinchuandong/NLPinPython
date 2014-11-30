from __future__ import division
__author__ = 'wangjiewen'

import BeautifulSoup
import nltk, re, pprint
from urllib import urlopen

nltk.download()
text = nltk.word_tokenize("I am a programmer")
list = nltk.pos_tag(text)

print(list)