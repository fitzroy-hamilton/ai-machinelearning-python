#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 16:45:27 2021

@author: Jeremy Levens
"""

# from nltk import *
# from nltk.book import *
import requests
from bs4 import BeautifulSoup


class HTMLScapingHelper:
    def __init__(self, urls):
        self.urls = urls
        self.docs = list()

    def web_read(self, _raw=0, _words=2):
        for url in self.urls:
            self.docs.append(self.web_read_url(url, _raw, _words))

    def web_read_url(self, url, _raw=0, _words=2):
        """
        -----------------------------------------------------
        This function returns the text of a website for a
        given url
        -----------------------------------------------------
        OPTIONS
        -----------------------------------------------------
        - _raw = option to return raw text from HTML
                - 0 = no (default)
                - 1 = yes, return raw text
        -----------------------------------------------------
        - words = option to return word tokens from HTML
                - 1 = return all words (default)
                - 2 = return only alphanumeric words
        -----------------------------------------------------
        """

        print('SOURCE:', url, '\n')

        headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; \
                   rv:52.0) Gecko/20100101 Firefox/52.0'}
        # fetching the url, raising error if operation fails
        try:
            response = requests.get(url, headers=headers)
        except requests.exceptions.RequestException as e:
            print(e)
            exit()
#        response = request.urlopen(self.url)
#        html = response.read().decode('utf-8')
        soup = BeautifulSoup(response.text, "html5lib")

        # Get Text from HTML
#        soup = BeautifulSoup(html, "html5lib")
        whitelist = [
            'p',
            'h1',
            'h2',
            'h3'
        ]
        sentences = [t for t in soup.find_all(text=True)
                     if t.parent.name in whitelist]
        doc = ' '.join(sentences)

#        print('DOC:', doc[:500], ' (truncated)...\n')
#        print('SENTENCES:', sentences, '\n')

        # Options
        # Raw Text Option
#        if _raw == 0:
#            pass
#        else:
#            pass
            # return raw

        # Get Tokens
#        tokens = word_tokenize(doc)

        # Word Options
#        print('WORDS:')
#        # All Words
#        if _words == 1:
#            print(tokens[:20])
            # return tokens
        # Alphanumeric Words
 #       elif _words == 2:
 #           words = [w for w in tokens if w.isalnum()]
 #           print(words[:20])
            # return words
 #       print('\n')

        return doc

    def get_doc_list(self):
        return self.docs
