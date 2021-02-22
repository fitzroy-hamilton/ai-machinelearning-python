#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 08:58:31 2021

@author: Jeremy Levens
"""

import re
import pandas as pd

# Gensim
import gensim
from gensim.utils import simple_preprocess

# SpaCy
import spacy

# NLTK Stop words
from nltk.corpus import stopwords


class NLPHelper:
    def __init__(self):
        self.nlp = None
        self.stop_words = stopwords.words('english')
        self.stop_words.extend([
            'from', 'subject', 're', 'edu', 'use', 'not', 'would', 'take',
            'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'come',
            'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'may', 'also',
            'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'even',
            'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line'])

    def get_stop_words(self):
        return self.stop_words

    # Convert sentence to words with an initial cleansing and tokenization using
    # Gensim simple_preprocess
    def sentence_to_words(self, sentences):
        for sent in sentences:
            sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
            sent = re.sub('\s+', ' ', sent)  # remove newline chars
            sent = re.sub("\'", "", sent)  # remove single quotes
            # remove ponctuation (with deacc set to True)
            sent = gensim.utils.simple_preprocess(str(sent), deacc=True)
            yield(sent)

    # Remove Stopwords, Form Bigrams, Trigrams and Lemmatization
    # python3 -m spacy download en  # run in terminal once
    def process_words(self, texts,
                      bigram_mod, trigram_mod,
                      allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        print('Start processing words...')

        # Remove stopwords (le, la, les, ...)
        texts = [[word for word in simple_preprocess(str(doc))
                  if word not in self.stop_words] for doc in texts]

        # Form bigrams and trigrams
        texts = [bigram_mod[doc] for doc in texts]
        texts = [trigram_mod[bigram_mod[doc]] for doc in texts]

        # Lemmatization (petits, petites, petit -> petit)
        texts_out = []
        nlp = spacy.load('en', disable=['parser', 'ner'])
        for sentence in texts:
            doc = nlp(" ".join(sentence))
            texts_out.append([token.lemma_ for token in doc
                              if token.pos_ in allowed_postags])

        # Remove stopwords once more after lemmatization
        texts_out = [[word for word in simple_preprocess(str(doc))
                      if word not in self.stop_words] for doc in texts_out]

        print('Example of text without stopword and lemmatized:', texts_out[:1])
        print('Done.\n')
        return texts_out

    # Get the main topic in each document concatenated to original text
    def format_topics_sentences(self, ldamodel, corpus, texts):
        sent_topics_df = pd.DataFrame()
        for i, row_list in enumerate(ldamodel[corpus]):
            row = row_list[0] if ldamodel.per_word_topics else row_list
            row = sorted(row, key=lambda x: (x[1]), reverse=True)

            # Get the Dominant topic, Perc Contrib and Keywords for each doc
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num),
                                   round(prop_topic, 4),
                                   topic_keywords]),
                        ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contrib',
                                  'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return(sent_topics_df)

    def topics_per_document(self, model, corpus, start=0, end=1):
        corpus_sel = corpus[start:end]
        dominant_topics = []
        topic_percentages = []
        for i, corp in enumerate(corpus_sel):
            topic_percs, wordid_topics, wordid_phivalues = model[corp]
            dominant_topic = sorted(topic_percs, key=lambda x: x[1],
                                    reverse=True)[0][0]
            dominant_topics.append((i, dominant_topic))
            topic_percentages.append(topic_percs)
        return(dominant_topics, topic_percentages)
