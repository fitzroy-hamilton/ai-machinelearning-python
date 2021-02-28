#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 20:00:51 2021

@author: Jeremy Levens
"""

import numpy as np
import pandas as pd
from pprint import pprint
import logging
import warnings
from os import path
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import pyLDAvis.gensim
import seaborn as sns
from iapkg.datavizhelper import DatavizHelper
from iapkg.nlphelper import NLPHelper
from iapkg.htmlparallelprocessing import HTMLScapingHelper
import time
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count


warnings.filterwarnings("ignore", category=DeprecationWarning)
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
#                    level=logging.ERROR)

CURDIR = path.dirname(__file__)
CORPUS = path.join(CURDIR, "corpus/pluvalor.html")
OUTDIR = path.join(CURDIR, "tagged")
urls = ['https://www.myriamlevens.fr/coaching/',
        'https://www.feedebeauxreves.fr/philosophie/',
        'https://www.thefrenchworkingmum.fr/podcast/une-role-modele/',
        'https://ghost.fnh.re/assemblage-des-parois-de-derriere/',
        'https://ghost.fnh.re/assemblages-a-queues-daronde/',
        'https://ghost.fnh.re/batis-a-petit-cadre/',
        'https://ghost.fnh.re/pivots-droits/'
        ]

if __name__ == '__main__':
    # Import Dataset
    # df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
    # df = df.loc[df.target_names.isin(['soc.religion.christian',
    #                                  'rec.sport.hockey',
    #                                  'talk.politics.mideast',
    #                                  'rec.motorcycles']), :]
    # print(df.columns)
    # print(df.shape)
    # (2361, 3)
    # print(df.head(), '\n')
    # docs = df.content.values.tolist()

    # htmlproc = HTMLScapingHelper(urls)
    # htmlproc.web_read()
    # docs = htmlproc.get_doc_list()

    df = pd.read_csv('Incidents AEC janvier 2021.csv',
                     engine='python',
                     encoding='utf8',
                     quotechar='"',
                     sep=';',
                     header=0,
                     error_bad_lines=True)
    print(df.columns)
    print(df.shape)
    print(df.head(5), '\n')
    df1 = df.replace(np.nan, '', regex=True)
    docs = df.commentaires.values.tolist()
    #pprint(docs)

    # Get an array (doc) of array (words) based on each doc's sentences
    stop_words_extension = ['cet', 'comment', 'tous', 'ca', 'aussi', 'alors', 'si',
                            'plus', 'fig', 'note', 'travail', 'commentaire',
                            'system', 'rappeler', 'merci', 'oui', 'non',
                            'intervenir', 'additionnel', 'automatically',
                            'incident', 'closed', 'after', 'days', 'resolved',
                            'state', 'note', 'ok', 'ko', 'action', 'utilisateur',
                            'demande', 'numéro', 'bien', 'faire', 'rappeler',
                            'indique', 'description', 'conseiller', 'arrive',
                            'el', 'tel', 'mme', 'poste', 'probleme', 'test',
                            'toujours', 'inc', 'pj', 'lors', 'fr', 'itsm',
                            'euro', 'renseigner', 'réaliser', 'voir', 'mesnage',
                            'memebr', 'suite', 'id', 'résoudre', 'jamais',
                            'refaire']
    nlphlp = NLPHelper(stop_words_extension)
    tokenized_docs = list(nlphlp.tokenize_docs(docs))
    print('First doc\'s tokens: [', tokenized_docs[0][:20], ', ...]\n')

    # Process data to get a for each document a list of words without stopwords
    # and already lemmatized (fixed, fixing, fix => fix). A same word may appear
    # several times (which will be used to create the corpus with Term Document
    # Frequency)
    # [['worth', 'expire', 'ducati', 'much', 'ducatif', ...], ['article', ...]]

    start = timer()
#    tokenized_lemmatized_docs = None
#    with Pool() as pool:
#    tokenized_lemmatized_docs = pool.map(nlphlp.lemmatize_docs, tokenized_docs)
#        print(res)
    tokenized_lemmatized_docs = nlphlp.lemmatize_docs(tokenized_docs)
    print('First doc\'s lemmatized tokens (without stop words): [',
          tokenized_lemmatized_docs[0][:20], ', ...]\n')
    end = timer()
    print(f'Elapsed time for lemmatizing docs: {end - start}')

    # Create a common Dictionary for all the documents with Gensim. It
    # encapsulates
    # the unique normalized words with their integer ids (position in the
    # Dictionary): ['dog', 'bike', 'bit', ...]
    id2word = corpora.Dictionary(tokenized_lemmatized_docs)
    print('Dictionary:', id2word, '\n')

    # Create Corpus with Term Document Frequency: [[(id, freq), (id, freq), ...]]
    # doc2bow converts document (a list of words) into the bag-of-words
    # format = list of (token_id, token_count) 2-tuples in an array for each doc
    corpus = [id2word.doc2bow(text) for text in tokenized_lemmatized_docs]
    print('First doc\'s corpus with Term Document Frequency (truncated): [',
          corpus[0][:10], ', ...]')
    print('Human readable format of corpus (truncated): [',
          [[(id2word[id], freq) for id, freq in cp] for cp in corpus][0][:10],
          ', ...]\n')

    # Build the Topic Model using LDA model (Latent Dirichlet Allocation)
    # chunksize is the number of documents to be used in each training chunk.
    # update_every determines how often the model parameters should be updated
    # and passes is the total number of training passes.
    n_topics = 6
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=n_topics,  # default: 100
                                                random_state=100,
                                                update_every=1,
                                                chunksize=2,  # default: 2000
                                                passes=10,
                                                alpha='symmetric',
                                                iterations=100,
                                                per_word_topics=True)

    # The above LDA model is built with 4 different topics where each topic is a
    # combination of keywords and each keyword contributes a certain weightage to
    # the topic.
    print('Topics with the weightage (importance) of each keyword:')
    pprint(lda_model.print_topics(), width=200)
    print(' ')

    # Compute Perplexity: a measure of how good the model is
    print('Perplexity (lower is better): ', lda_model.log_perplexity(corpus), '\n')

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model,
                                         texts=tokenized_lemmatized_docs,
                                         dictionary=id2word, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    # print('Coherence Score: ', coherence_lda, '\n')

    # Get the main topic in each document concatenated to original text
    df_topic_sents_keywords = nlphlp.format_topics_sentences(ldamodel=lda_model,
                                                             corpus=corpus,
                                                             texts=tokenized_lemmatized_docs)
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic',
                                 'Topic_Perc_Contrib', 'Keywords', 'Text']
    print('Dominant topics:', df_dominant_topic.head(10), '\n')

    pd.options.display.max_colwidth = 100
    sent_topics_sorteddf_mallet = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                 grp.sort_values(['Perc_Contrib'],
                                                                 ascending=False)
                                                 .head(1)], axis=0)

    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib",
                                           "Keywords", "Representative Text"]
    print('Main dominant topics:', sent_topics_sorteddf_mallet.head(10), '\n')

    doc_lengths = [len(d) for d in df_dominant_topic.Text]

    dominant_topics, topic_percentages = nlphlp.topics_per_document(model=lda_model,
                                                                    corpus=corpus,
                                                                    end=-1)

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
    # datavizHelper.dominant_topic_distribution(df_dominant_topic, mcolors)
    datavizHelper.wordcloud(lda_model, nlphlp.get_stop_words(), mcolors)
    # datavizHelper.word_count_importance_merged(lda_model, tokenized_lemmatized_docs,mcolors)
    datavizHelper.word_count_importance(lda_model, tokenized_lemmatized_docs, mcolors)
    datavizHelper.sentences_chart(lda_model, corpus, 0, len(urls)-1, mcolors)
    datavizHelper.topic_distribution_by_dominant_topics(df_dominant_topic_in_docs,
                                                        df_topic_weightage_by_doc,
                                                        df_top5words,
                                                        mcolors)

    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    vis
    print(vis)

    # Get topic weights and dominant topics
    topic_weights = []
    for i, row_list in enumerate(lda_model[corpus]):
        topic_weights.append([w for i, w in row_list[0]])

    # Array of topic weights
    arr_topic_weights = pd.DataFrame(topic_weights).fillna(0).values
    # Keep the well separated points (opt.)
    arr_topic_weights = arr_topic_weights[np.amax(arr_topic_weights, axis=1) > 0.35]
    # Dominant topic number in each doc
    topic_num = np.argmax(arr_topic_weights, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=0, random_state=0, angle=.99,
                      init='pca')
    tsne_lda = tsne_model.fit_transform(arr_topic_weights)
    # X = arr_topic_weights
    # Y = tsne_lda

    cluster = pd.DataFrame({'PC1_2d': tsne_lda[:, 0],
                            'PC2_2d': tsne_lda[:, 1],
                            'topic': topic_num})
    datavizHelper.clustering(2, pd.DataFrame(data=cluster), topic_num)

    tsne_df = pd.DataFrame({'X': tsne_lda[:, 0],
                            'Y': tsne_lda[:, 1],
                            'topic': topic_num})
    print('TNSE:')
    pprint(tsne_df.head(10))
    print('\n')

    colors = ['red', 'orange', 'blue', 'green']
    colors = colors[:tsne_df.shape[1]]
    print(colors)
    sns.scatterplot(x="X", y="Y",
                    hue="topic",
                    palette=colors,
                    legend='full',
                    data=tsne_df)

    datavizHelper.topic_clusters_bokeh(tsne_lda, topic_num, n_topics, mcolors)
