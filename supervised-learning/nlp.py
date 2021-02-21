#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 20:00:51 2021

@author: kama
"""

# !{sys.executable} -m spacy download en
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import spacy
import logging
import warnings
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# sklearn
from sklearn.manifold import TSNE

# visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import pyLDAvis.gensim

from iapkg.datavizhelper import DatavizHelper

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would',
                   'say', 'could', '_', 'be', 'know', 'good', 'go', 'get',
                   'do', 'done', 'try', 'many', 'some', 'nice', 'thank',
                   'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack',
                   'make', 'want', 'seem', 'run', 'need', 'even', 'right',
                   'line', 'even', 'also', 'may', 'take', 'come'])

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.ERROR)

# Import Dataset
df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
df = df.loc[df.target_names.isin(['soc.religion.christian',
                                  'rec.sport.hockey',
                                  'talk.politics.mideast',
                                  'rec.motorcycles']), :]
print(df.columns)
print(df.shape)
# (2361, 3)
print(df.head(), '\n')


# Convert sentence to words with an initial cleansing and tokenization using
# Gensim simple_preprocess
def sentence_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        # remove ponctuation (with deacc set to True)
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True)
        yield(sent)


# Remove Stopwords, Form Bigrams, Trigrams and Lemmatization
# !python3 -m spacy download en  # run in terminal once
def process_words(texts, stop_words=stop_words,
                  allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    print('Start processing words...')

    # Remove stopwords (le, la, les, ...)
    texts = [[word for word in simple_preprocess(str(doc))
              if word not in stop_words] for doc in texts]

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
                  if word not in stop_words] for doc in texts_out]

    print('Example of text without stopword and lemmatized:', texts_out[:1])
    print('Done.\n')
    return texts_out


# Get the main topic in each document concatenated to original text
def format_topics_sentences(ldamodel, corpus, texts):
    sent_topics_df = pd.DataFrame()
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each doc
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


def topics_per_document(model, corpus, start=0, end=1):
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


# Convert the 'content' column to list
data = df.content.values.tolist()

# Get an array (doc) of array (words) based on each doc's sentences
# [['from', 'irwin', 'arnstein', 'subject', 're', 'recommendation', 'on', ...]]
data_words = list(sentence_to_words(data))
print('First doc\'s sentences to words:')
print(data_words[:1], '\n')

# Build the bigram and trigram models (higher threshold fewer phrases)
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
print('Trigram example:', trigram_mod[bigram_mod[data_words[0]]], '\n')

# Process data to get a for each document a list of words without stopwords
# and already lemmatized (fixed, fixing, fix => fix). A same word may appear
# several times (which will be used to create the corpus with Term Document
# Frequency)
# [['worth', 'expire', 'ducati', 'much', 'ducatif', ...], ['article', ...]]
data_lemmatized = process_words(data_words)

# Create a common Dictionary for all the documents with Gensim. It encapsulates
# the unique normalized words with their integer ids (position in the
# Dictionary): ['dog', 'bike', 'bit', ...]
id2word = corpora.Dictionary(data_lemmatized)
print('Dictionary:', id2word, '\n')

# Create Corpus with Term Document Frequency: [[(id, freq), (id, freq), ...]]
# doc2bow converts document (a list of words) into the bag-of-words
# format = list of (token_id, token_count) 2-tuples in an array for each doc
corpus = [id2word.doc2bow(text) for text in data_lemmatized]
# print('Corpus example:')
# pprint(corpus[:1])
# print('\n')
print('Human readable format of corpus (term-frequency):')
pprint([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])
print('\n')

# Build the Topic Model using LDA model (Latent Dirichlet Allocation)
# chunksize is the number of documents to be used in each training chunk.
# update_every determines how often the model parameters should be updated
# and passes is the total number of training passes.
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=4,  # default: 100
                                            random_state=100,
                                            update_every=1,
                                            chunksize=10,  # default: 2000
                                            passes=10,
                                            alpha='symmetric',
                                            iterations=100,
                                            per_word_topics=True)

# The above LDA model is built with 4 different topics where each topic is a
# combination of keywords and each keyword contributes a certain weightage to
# the topic.
print('Topics with the weightage (importance) of each keyword:')
pprint(lda_model.print_topics())
print(' ')
# [(0,
#   '0.019*"team" + 0.019*"game" + 0.013*"hockey" + 0.010*"player" + '
#   '0.009*"play" + 0.009*"win" + 0.009*"nhl" + 0.009*"year" + 0.009*"hawk" + '
#   '0.009*"season"'),
#  (1,
#   '0.008*"christian" + 0.008*"believe" + 0.007*"god" + 0.007*"law" + '
#   '0.006*"state" + 0.006*"israel" + 0.006*"israeli" + 0.005*"exist" + '
#   '0.005*"way" + 0.004*"bible"'),
#  ...
# ]

# Compute Perplexity: a measure of how good the model is. lower the better
print('Perplexity: ', lda_model.log_perplexity(corpus), '\n')

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized,
                                     dictionary=id2word, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('Coherence Score: ', coherence_lda, '\n')

# Get the main topic in each document concatenated to original text
df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model,
                                                  corpus=corpus,
                                                  texts=data_lemmatized)
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No',
                             'Dominant_Topic',
                             'Topic_Perc_Contrib',
                             'Keywords',
                             'Text']
print('Dominant topics:')
print(df_dominant_topic.head(10))
print('\n')

pd.options.display.max_colwidth = 100
sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                             grp.sort_values(['Perc_Contrib'],
                                                             ascending=False)
                                             .head(1)],
                                            axis=0)

sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
sent_topics_sorteddf_mallet.columns = ['Topic_Num',
                                       "Topic_Perc_Contrib",
                                       "Keywords",
                                       "Representative Text"]
print('Main dominant topics:')
print(sent_topics_sorteddf_mallet.head(10))
print('\n')

doc_lengths = [len(d) for d in df_dominant_topic.Text]

dominant_topics, topic_percentages = topics_per_document(model=lda_model,
                                                         corpus=corpus, end=-1)

# Distribution of Dominant Topics in Each Document
df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
df_dominant_topic_in_docs = dominant_topic_in_each_doc.\
    to_frame(name='count').reset_index()

# Total Topic Distribution by actual weight
topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
df_topic_weightage_by_doc = topic_weightage_by_doc.sum().\
    to_frame(name='count').reset_index()

# Top 5 Keywords for each Topic
topic_top5words = [(i, topic) for i, topics in lda_model.
                   show_topics(formatted=False)
                   for j, (topic, wt) in enumerate(topics) if j < 5]

df_top5words_stacked = pd.DataFrame(topic_top5words, columns=['topic_id',
                                                              'words'])
df_top5words = df_top5words_stacked.groupby('topic_id').agg(', \n'.join)
df_top5words.reset_index(level=0, inplace=True)

# Dataviz
datavizHelper = DatavizHelper('nlp')
datavizHelper.word_count_distribution(doc_lengths)
datavizHelper.dominant_topic_distribution(df_dominant_topic, mcolors)
datavizHelper.wordcloud(lda_model, stop_words, mcolors)
# datavizHelper.word_count_importance_merged(lda_model, data_lemmatized,mcolors)
datavizHelper.word_count_importance(lda_model, data_lemmatized, mcolors)
datavizHelper.sentences_chart(lda_model, corpus, 0, 11, mcolors)
datavizHelper.topic_distribution_by_dominant_topics(df_dominant_topic_in_docs,
                                                    df_topic_weightage_by_doc,
                                                    df_top5words,
                                                    mcolors)

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
vis
print(vis)

# Get topic weights and dominant topics
# Get topic weights
topic_weights = []
for i, row_list in enumerate(lda_model[corpus]):
    topic_weights.append([w for i, w in row_list[0]])

# Array of topic weights
arr = pd.DataFrame(topic_weights).fillna(0).values

# Keep the well separated points (optional)
arr = arr[np.amax(arr, axis=1) > 0.35]

# Dominant topic number in each doc
topic_num = np.argmax(arr, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99,
                  init='pca')
tsne_lda = tsne_model.fit_transform(arr)

# Plot the Topic Clusters using Bokeh
output_notebook()
n_topics = 4
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics),
              plot_width=900, plot_height=700)
plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[topic_num])
show(plot)
