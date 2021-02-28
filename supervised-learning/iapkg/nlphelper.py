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
from nltk.tokenize.regexp import regexp_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from alive_progress import alive_bar
from tqdm import tqdm


class NLPHelper:
    def __init__(self, stop_words_extension=[]):
        self.nlp = None
        self.stop_words = stopwords.words('french')
        self.stop_words.extend(stop_words_extension)

    def get_stop_words(self):
        return self.stop_words

    # performs a few cleaning steps to remove non-alphabetic characters
    def clean_text(self, text):
        # replace new line, carriage return and tabulation with space
        try:
            text= str(text)
            text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        except AttributeError:
            print('ERREUR:', text, text.dtype)
        text = text.lower()

        # replace the numbers and punctuation with space
        punc_list = '!"#$%&()*+,-.©–/:;<=>?@[\\]^_{}~' + '0123456789' + "'’"

        # replace accentuation with simple characters
        # FIXME: remove accents

        t = str.maketrans(dict.fromkeys(punc_list, ' '))
        text = text.translate(t)

        return text

    def custom_tokenizer(self, text, tokenizer='nltk_regexp'):
        text = self.clean_text(text)

        if (tokenizer == 'nltk'):
            tokens = word_tokenize(text)
        elif (tokenizer == 'gensim'):
            tokens = gensim.utils.simple_preprocess(str(tokens), deacc=True)
        else:
            tokens = regexp_tokenize(text, pattern = '\s+', gaps = True)

        for token in tokens:
            if len(token) > 512:
                print('TOKER > 512', token)
                token = token[:512]

        return tokens

    # Convert docs to tokenized docs with an initial cleansing and tokenization
    def tokenize_docs(self, docs):
        print('Start tokenizing docs...')
        for doc in docs:
            doc = self.custom_tokenizer(doc)
            yield(doc)
        print('Docs tokenized.')

    # Remove Stopwords, Form Bigrams, Trigrams and Lemmatization
    # python3 -m spacy download en  # run in terminal once
    # REMOVE STOPWORDS
    def lemmatize_docs(self, docs,
                       allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        print('Start lemmatizing docs...')
        print('\tRemoving stop words...')
        # Remove stopwords (le, la, les, ...)
        docs = [[word for word in simple_preprocess(str(doc))
                 if word not in self.stop_words] for doc in docs]

        # Form bigrams and trigrams (higher threshold fewer phrases)
        # Après un découpage en mots, on ne considère plus l’ordre avec une
        # approche bag-of-words. Si l’information contenu par l’ordre des mots
        # s’avère importante, il faut considérer un découpage en couple de mots
        # (bi-grammes), triplets de mots (3-grammes)...
        print('\tForming bigrams and trigrams...')
        bigram = gensim.models.Phrases(docs, min_count=5, threshold=100)
        trigram = gensim.models.Phrases(bigram[docs], threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        # print('Trigram example:', trigram_mod[bigram_mod[docs[0]]], '\n')
        docs = [bigram_mod[doc] for doc in docs]
        docs = [trigram_mod[bigram_mod[doc]] for doc in docs]

        # Lemmatization (petits, petites, petit -> petit)
        print('\tLemmatizing...')
        texts_out = []
        # pip install --upgrade spacy # to get v3.0
        # python -m spacy download fr_dep_news_trf
        # fr_dep_news_trf = French transformer pipeline (camembert-base).
        # Components: transformer, morphologizer, parser, attribute_ruler,
        # lemmatizer.
        nlp = spacy.load('fr_dep_news_trf', disable=['parser', 'ner'])
        with alive_bar(len(docs), force_tty=1, spinner='ball_bouncing') as bar:
            for sentence in docs:
                doc = nlp(" ".join(sentence))
    #            spacy.displacy.render(doc, style='ent', jupyter=True)
                texts_out.append([token.lemma_ for token in doc
                                  if token.pos_ in allowed_postags])
                bar()

        # Remove stopwords once more after lemmatization
        print('\tRemoving stop words after lemmatization...')
        texts_out = [[word for word in simple_preprocess(str(doc))
                      if word not in self.stop_words] for doc in texts_out]
        print('Docs lemmatized.')
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
