#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 21:20:32 2021

@author: Jeremy Levens
"""

import warnings
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import set_config
from sklearn.metrics import plot_confusion_matrix
from scipy.interpolate import make_interp_spline

from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from wordcloud import WordCloud
from collections import Counter

from bokeh.plotting import figure, show
from bokeh.io import output_notebook


class DatavizHelper:

    def __init__(self, ml_type):
        warnings.filterwarnings("ignore")
        self.max_features = 25
        self.plt_h = 6
        self.plt_cols = 2
        sns.set_context("talk", font_scale=0.8)
        plt.rcParams.update({'font.size': 20})
        set_config(display='diagram')
        pd.options.display.float_format = '{:,.3f}'.format
        self.ml_type = ml_type

    def describe(self, X, y):
        pd.options.display.float_format = '{:.1f}'.format

        # percentile list
        perc = [.10, .50, .80, .95]

        # list of dtypes to include
        include = ['object', 'float', 'int', 'category', np.number, np.object]
        dataset = pd.concat([X, y], axis=1, sort=False)

        return dataset.describe(percentiles=perc, include=include)

    def emptyValues(self, X, y, verbose=False):
        plt.figure(figsize=(9, 8), dpi=300, tight_layout=True)
        plt.clf()

        if (y is None):
            dataset = X
        else:
            dataset = pd.concat([X.reset_index(drop=True),
                                 y.reset_index(drop=True)],
                                axis=1, sort=False)
        count_missing = dataset.isna().sum()
        percent_missing = dataset.isna().sum() * 100 / dataset.shape[0]
        should_drop = percent_missing >= 90
        missing_value_df = pd.DataFrame({'count_missing': count_missing,
                                         'percent_missing': percent_missing,
                                         'should_drop': should_drop})
        missing_value_df.sort_values('percent_missing', inplace=True,
                                     ascending=False)
        if (verbose):
            print('Empty values analysis:')
            print(missing_value_df)

        cmap = sns.color_palette("Blues", as_cmap=True)
        ax = plt.axes()
        sns.heatmap(dataset.isnull(), yticklabels=False, cbar=False,
                    cmap=cmap, ax=ax)
        ax.set_title('Empty values')
        plt.show()

    def distributionNumerical(self, X, y):
        distribdataset = pd.concat([X, y], axis=1, sort=False)
        subdataset = distribdataset.select_dtypes([np.number])

        cmap = sns.color_palette("Blues", as_cmap=True)
        sns.set(style='white')
        sns.set_style(rc={'axes.spines.top': False, 'axes.spines.right': False,
                          'axes.edgecolor': 'lightgrey'})

        nblines = math.ceil(len(subdataset.columns) / self.plt_cols)
        i = 1
        for feature in subdataset:
            if ((i - 1) % 6 == 0):
                plt.figure(figsize=(10, 20),
                           dpi=150, tight_layout=True)
                plt.clf()

            plt.subplot(nblines, self.plt_cols, i)

            sns.histplot(subdataset[feature], kde=True, palette=cmap)
            i = i + 1

        plt.show()

    def distributionCategorical(self, X, y):
        data = pd.concat([X, y], axis=1, sort=False)
        subdataset = data.select_dtypes('category')

        i = 1
        nblines = math.ceil(len(subdataset.columns) / self.plt_cols)
        plt.figure(figsize=(35, nblines*self.plt_h), dpi=150, tight_layout=True)
        plt.clf()

        for feature in subdataset:
            print(f'{feature :-<30} {data[feature].unique()}')
            plt.subplot(nblines, self.plt_cols, i)

            data[feature].value_counts().plot.pie(textprops={'fontsize': 16})
            i = i + 1

        plt.show()

    def relationNumericalFeatureTarget(self, X, y):
        if (y is None):
            print('No target to compare with')
            print(' ')
        else:
            data = pd.concat([X, y], axis=1, sort=False)
            subdataset = data.select_dtypes([np.number])

            if (y.unique().size > self.max_features):
                print('Too many target unique values')
            else:
                target_data = {}
                for target in y.unique():
                    target_data[target] = data[data[data.columns[-1]] == target]

                i = 1
                nblines = math.ceil(len(subdataset.columns) / self.plt_cols)
                plt.figure(figsize=(35, nblines*self.plt_h), dpi=150,
                           tight_layout=True)
                plt.clf()

                for feature in subdataset:
                    plt.subplot(nblines, self.plt_cols, i)

                    for target in y.unique():
                        sns.distplot(target_data[target][feature], label=target)
                    plt.legend()
                    i = i + 1

                plt.show()

    def relationCategoricalFeatureTarget(self, X, y):
        if (y is None):
            print('No target to compare with')
            print(' ')
        else:
            dataset = X
            subdataset = dataset.select_dtypes('category')

            if (subdataset.size < 1):
                print('No categorical feature to show')
            elif (subdataset.columns.size > self.max_features):
                print('Too many categorical features to show:', subdataset.size)
            else:
                i = 1
                nblines = math.ceil(len(subdataset.columns) / self.plt_cols)
                plt.figure(figsize=(35, nblines*self.plt_h), dpi=150,
                           tight_layout=True)
                plt.clf()

                for feature in subdataset:
                    plt.subplot(nblines, self.plt_cols, i)

                    sns.countplot(x=feature, hue=y, data=dataset)
                    plt.legend()
                    i = i + 1

                plt.show()

    def heatmap(self, X, y, mirrored=False):
        data = pd.concat([X, y], axis=1, sort=False)
        if (data.select_dtypes([np.number]).columns.size < self.max_features):
            corr = data.corr()

            plt.figure(figsize=(12, 11), dpi=300, tight_layout=True)
            plt.clf()
            ax = plt.axes()
            colormap = sns.color_palette("RdBu_r", 7)

            dropSelf = None
            if mirrored is False:
                # Drop self-correlations
                dropSelf = np.zeros_like(corr)
                dropSelf[np.triu_indices_from(dropSelf)] = True

            sns.heatmap(corr, cmap=colormap, annot=True, fmt=".1f",
                        mask=dropSelf)
            ax.set_title('Correlation between Numerical features')
            plt.show()
        else:
            print('Too many features to draw a heatmap')

    def clustermap(self, X, y):
        data = pd.concat([X, y], axis=1, sort=False)
        if (data.select_dtypes([np.number]).columns.size < self.max_features):
            corr = data.corr()

            colormap = sns.color_palette("RdBu_r", 7)

            sns.clustermap(corr, cmap=colormap, annot=True, fmt=".2f")
            plt.show()
        else:
            print('Too many features to draw a heatmap')

    def pairplot(self, X, y):
        data = pd.concat([X, y], axis=1, sort=False)
        if ((self.ml_type == 'classification') and
           (data.select_dtypes([np.number]).columns.size < self.max_features)):
            pairplot_data = data.fillna(method='ffill')
            sns.pairplot(pairplot_data,
                         vars=pairplot_data.select_dtypes([np.number]).columns,
                         hue=y.name, dropna='true')
        else:
            print('Too many features to draw a pairplot')

    def hexbin(x, y, color, **kwargs):
        cmap = sns.light_palette(color, as_cmap=True)
        plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)

    def facetgrid(self, max_score):
        g = sns.FacetGrid(max_score, hue="estimator",
                          col="estimator", col_wrap=3, height=4)
        g.map(self.hexbin, "mean_score", "max_score", extent=[0, 1, 0, 1])

    def confusionmatrix(self, model, X, y):
        disp = plot_confusion_matrix(model, X, y, cmap=plt.cm.Blues,
                                     normalize=None)
        disp.ax_.set_title("Confusion matrix")
        print(disp.confusion_matrix)

    def learningcurve(self, N, train_score, val_score):
        f, ax = plt.subplots(1)
        plt.plot(N, train_score.mean(axis=1), label='train')
        plt.plot(N, val_score.mean(axis=1), label='validation')
        plt.xlabel('train_sizes')
        plt.ylabel('scores')
        plt.legend()
        ax.set_ylim(ymin=0)
        ax.set_title('Learning curve')

        plt.show()

    def errorhist(self, y, y_pred):
        plt.figure(figsize=(10, 6), tight_layout=True)
        plt.clf()

        err_hist = np.abs(y - y_pred)
        plt.xlabel('Error value')
        plt.ylabel('Quantity')
        plt.hist(err_hist, bins=50)

        plt.show()

    def drawElbow(self, K_range, optimal_nb_cluster, inertia):
        plt.figure(figsize=(9, 5), dpi=300, tight_layout=True)
        plt.clf()

        plt.xticks(K_range)
        K_range_smoothed = np.linspace(1, K_range[-1], 50)
        a_BSpline = make_interp_spline(K_range, inertia)
        inertia_smoothed = a_BSpline(K_range_smoothed)

        plt.plot(K_range_smoothed, inertia_smoothed)
        plt.vlines(optimal_nb_cluster, 0, inertia[0],
                   linestyles='dashed', label='optimal', colors='g')
        plt.xlabel('# clusters')
        plt.ylabel('Model cost (= inertia)')

        plt.show()

    def clustering(self, d, X, y_clusters, k_means_centroids=None):
        # Add the cluster vector to our DataFrame, X
        dataset = X.copy()
        dataset["cluster"] = y_clusters

        plt.figure(figsize=(7, 7), dpi=300, tight_layout=True)
        plt.clf()

        plt.rc('axes', titlesize=10)
        plt.rc('axes', labelsize=10)
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        plt.rc('legend', fontsize=8)
        colors = ['#6e1e78', '#e05206', '#ffb612', '#d2e100', '#82be00',
                  '#009aa6', '#0088ce']

        nb_clus = len(y_clusters)
        if (d == 2):
            self.clustering2D(dataset, colors, nb_clus, k_means_centroids)
        elif (d == 3):
            self.clustering3D(dataset, colors, nb_clus, k_means_centroids)

        plt.legend(scatterpoints=1, ncol=2, markerscale=3.0)
        plt.show()

    def clustering2D(self, dataset, colors, nb_clus, k_means_centroids=None):
        ax = plt.axes()
        ax.set(xlabel='PC1', ylabel='PC2')
#        ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25))
        ax.set_title('Clustering 2D')
        n_clusters = nb_clus
#        n_clusters = len(k_means_cluster_centers)

        # for each cluster we associate a color
        for k, col in zip(range(n_clusters), colors):
            cluster = dataset[dataset["cluster"] == k]
            ax.plot(cluster['PC1_2d'],
                    cluster['PC2_2d'],
                    linewidth=0.25,
                    alpha=0.5,
                    color=col,
                    linestyle='--',
                    antialiased=True,
                    animated=True,
                    markerfacecolor=col,
                    markeredgewidth=0,
                    marker='.', label='Cluster '+str(k+1))
            if (k_means_centroids is not None):
                cluster_center = k_means_centroids[k]
                ax.plot(cluster_center[0], cluster_center[1], 'o',
                        markerfacecolor=col,
                        markeredgecolor='k',
                        antialiased=True,
                        animated=True,
                        markersize=6)

    def clustering3D(self, dataset, colors, nb_clus, k_means_centroids=None):
        ax = plt.axes(projection='3d')
        ax.set(xlabel='PC1', ylabel='PC2', zlabel='PC3')
        ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), zlim=(-1.1, 1.1))
        ax.set_title('Clustering 3D')
        n_clusters = nb_clus
#        n_clusters = len(k_means_cluster_centers)

        # for each cluster we associate a color
        for k, col in zip(range(n_clusters), colors):
            cluster = dataset[dataset["cluster"] == k]
            ax.plot3D(cluster['PC1_3d'],
                      cluster['PC2_3d'],
                      cluster['PC3_3d'],
                      linewidth=0.25,
                      alpha=0.5,
                      color=col,
                      linestyle='--',
                      antialiased=True,
                      animated=True,
                      markerfacecolor=col,
                      markeredgewidth=0,
                      marker='.', label='Cluster '+str(k+1))

            if (k_means_centroids is not None):
                cluster_center = k_means_centroids[k]
                ax.text(cluster_center[0],
                        cluster_center[1],
                        cluster_center[2],
                        'Cluster '+str(k+1),
                        fontsize=8,
                        horizontalalignment='center',
                        verticalalignment='center',
                        bbox=dict(facecolor='white', alpha=0.66))
                ax.plot3D(cluster_center[0],
                          cluster_center[1],
                          cluster_center[2],
                          'o',
                          markerfacecolor=col,
                          markeredgecolor='k',
                          antialiased=True,
                          animated=True,
                          markersize=6)

    # Dataviz for Distribution of document word counts
    def word_count_distribution(self, doc_lengths):
        plt.figure(figsize=(16, 7), dpi=300)
        plt.hist(doc_lengths, bins=1000, color='steelblue')
        plt.text(750, 100, "Mean   :" + str(round(np.mean(doc_lengths))))
        plt.text(750, 90, "Median :" + str(round(np.median(doc_lengths))))
        plt.text(750, 80, "Stdev   :" + str(round(np.std(doc_lengths))))
        plt.text(750, 70, "1%ile    :" + str(round(np.quantile(doc_lengths,
                                                               q=0.01))))
        plt.text(750, 60, "99%ile  :" + str(round(np.quantile(doc_lengths,
                                                              q=0.99))))
        plt.gca().set(xlim=(0, 1000),
                      ylabel='Number of Documents',
                      xlabel='Document Word Count')
        plt.tick_params(size=16)
        plt.xticks(np.linspace(0, 1000, 9))
        plt.title('Distribution of Document Word Counts',
                  fontdict=dict(size=22))
        plt.show()

    # Dataviz for Distribution of document word counts by Dominant topic
    def dominant_topic_distribution(self, df_dominant_topic, mcolors):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=300,
                                 sharex=True,
                                 sharey=True)

        for i, ax in enumerate(axes.flatten()):
            df_dominant_topic_sub = df_dominant_topic.\
                loc[df_dominant_topic.Dominant_Topic == i, :]
            doc_lengths = [len(d) for d in df_dominant_topic_sub.Text]
            ax.hist(doc_lengths, bins=1000, color=cols[i])
            ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
            sns.kdeplot(doc_lengths, color="black", shade=False, ax=ax.twinx())
            ax.set(xlim=(0, 1000), xlabel='Document Word Count')
            ax.set_ylabel('Number of Documents', color=cols[i])
            ax.set_title('Topic: '+str(i), fontdict=dict(size=16,
                                                         color=cols[i]))

        fig.tight_layout()
        fig.subplots_adjust(top=0.90)
        plt.xticks(np.linspace(0, 1000, 9))
        fig.suptitle('Distribution of Document Word Counts by Dominant Topic',
                     fontsize=22)
        plt.show()

    # Dataviz for Wordcloud of Top N words in each topic
    def wordcloud(self, lda_model, stop_words, mcolors):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        cloud = WordCloud(stopwords=stop_words,
                          background_color='white',
                          width=2500,
                          height=1800,
                          max_words=20,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        topics = lda_model.show_topics(formatted=False)

        fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=300,
                                 sharex=True, sharey=True)
        for i, ax in enumerate(axes.flatten()):
            fig.add_subplot(ax)
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
            plt.gca().axis('off')
        fig.suptitle('Word cloud', fontsize=22)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.show()

    # Dataviz for Word count and importance of topic keywords
    def word_count_importance_merged(self, lda_model, data_lemmatized, mcolors):
        topics = lda_model.show_topics(formatted=False)
        data_flat = [w for w_list in data_lemmatized for w in w_list]
        counter = Counter(data_flat)
        out = []
        for i, topic in topics:
            for word, weight in topic:
                out.append([word, i, weight, counter[word]])
        df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance',
                                        'word_count'])

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True, dpi=300)
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        for i, ax in enumerate(axes.flatten()):
            ax.bar(x='word', height="word_count",
                   data=df.loc[df.topic_id == i, :],
                   color=cols[i], width=0.5, alpha=0.3, label='Word Count')
            ax_twin = ax.twinx()
            ax_twin.bar(x='word', height="importance",
                        data=df.loc[df.topic_id == i, :],
                        color=cols[i], width=0.2, label='Weights')
            ax.set_ylabel('Word Count', color=cols[i])
            ax_twin.set_ylim(0, 0.030)
            ax.set_ylim(0, 3500)
            ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
            ax.tick_params(axis='y', left=False)
            ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30,
                               horizontalalignment='right')
            ax.legend(loc='upper left')
            ax_twin.legend(loc='upper right')

        fig.tight_layout(w_pad=2)
        plt.rc('axes', titlesize=16)     # fontsize of the axes title
        plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
        plt.rc('legend', fontsize=16)    # legend fontsize
        plt.rc('figure', titlesize=20)  # fontsize of the figure title
        fig.suptitle('Word Count and Importance of Topic Keywords',
                     fontsize=22, y=1.05)
        plt.show()

    # Dataviz for Word count and importance of topic keywords
    def word_count_importance(self, lda_model, data_lemmatized, mcolors):
        topics = lda_model.show_topics(formatted=False)
        data_flat = [w for w_list in data_lemmatized for w in w_list]
        counter = Counter(data_flat)
        out = []
        for i, topic in topics:
            for word, weight in topic:
                out.append([word, i, weight, counter[word]])
        df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance',
                                        'word_count'])

        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        i = 0
        for top in topics:
            fig = plt.figure(figsize=(16, 10), dpi=300, tight_layout=True)
            ax = plt.axes()
            ax.bar(x='word', height="word_count",
                   data=df.loc[df.topic_id == i, :],
                   color=cols[i], width=0.5, alpha=0.3, label='Word Count')
            ax_twin = ax.twinx()
            ax_twin.bar(x='word', height="importance",
                        data=df.loc[df.topic_id == i, :],
                        color=cols[i], width=0.2, label='Weights')
            ax.set_ylabel('Word Count', color=cols[i])
#            ax_twin.set_ylim(0, 0.030)
#            ax.set_ylim(0, 3500)
            ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
            ax.tick_params(axis='y', left=False)
            ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30,
                               horizontalalignment='right')
            ax.legend(loc='upper left')
            ax_twin.legend(loc='upper right')

            plt.rc('axes', titlesize=16)     # fontsize of the axes title
            plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
            plt.rc('legend', fontsize=16)    # legend fontsize
            plt.rc('figure', titlesize=20)  # fontsize of the figure title
            fig.suptitle('Word Count and Importance of Topic Keywords',
                         fontsize=22, y=1.05)
            plt.show()
            i = i + 1

    # Dataviz for Topic Distribution by Dominant Topics
    def topic_distribution_by_dominant_topics(self,
                                              df_dominant_topic_in_each_doc,
                                              df_topic_weightage_by_doc,
                                              df_top5words,
                                              mcolors):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        fig = plt.figure(figsize=(16, 10), dpi=300, tight_layout=True)
        ax = plt.axes()
        ax.bar(x='Dominant_Topic', height="count",
               data=df_dominant_topic_in_each_doc,
               color=cols[0], width=0.5, alpha=0.3,
               label='By Dominant Topic')
        ax_twin = ax.twinx()
        ax_twin.bar(x='index', height="count",
                    data=df_topic_weightage_by_doc,
                    color=cols[0], width=0.2, label='By Topic Weightage')
        ax.set_ylabel('Number of docs', color=cols[0])
        ax.tick_params(axis='y', left=False)
        ax.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.
                            unique().__len__()))
        tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x) + '\n' +
                                       df_top5words.loc[df_top5words.
                                                        topic_id == x,
                                                        'words'].values[0])
        ax.xaxis.set_major_formatter(tick_formatter)
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')

        plt.rc('axes', titlesize=16)     # fontsize of the axes title
        plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
        plt.rc('legend', fontsize=16)    # legend fontsize
        plt.rc('figure', titlesize=20)  # fontsize of the figure title
        fig.suptitle('Number of docs by Dominant Topic and Topic Weightage',
                     fontsize=22, y=1.05)
        plt.show()

    # Dataviz for sentence Coloring of N Sentences based on topics for each word
    def sentences_chart(self, lda_model, corpus, start, end, mcolors):
        end = end + 2
        corp = corpus[start:end]
        mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

        fig, axes = plt.subplots(end-start, 1, figsize=(20, (end-start)*0.95),
                                 dpi=300)
        axes[0].axis('off')
        for i, ax in enumerate(axes):
            if i > 0:
                corp_cur = corp[i-1]
                topic_percs, wordid_topics, wordid_phival = lda_model[corp_cur]
                topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]),
                                            reverse=True)
                word_dominanttopic = [(lda_model.id2word[wd], topic[0])
                                      for wd, topic in wordid_topics]
                ax.text(0.01, 0.5, "Doc " + str(i-1) +
                        " (topic #" + str(topic_percs_sorted[0][0]) + "): ",
                        verticalalignment='center',
                        fontsize=18, color='black',
                        transform=ax.transAxes,
                        fontweight=700)

                # Draw Rectange
                ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90,
                                       fill=None, alpha=1,
                                       color=mycolors[topic_percs_sorted[0][0]],
                                       linewidth=2))

                word_pos = 0.12
                for j, (word, topics) in enumerate(word_dominanttopic):
                    if j < 14:
                        ax.text(word_pos, 0.5, word,
                                horizontalalignment='left',
                                verticalalignment='center',
                                fontsize=18, color=mycolors[topics],
                                transform=ax.transAxes, fontweight=700)
                        # move the word to next iter
                        word_pos += .009 * len(word)
                        ax.axis('off')
                ax.text(word_pos, 0.5, '. . .',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=18, color='black',
                        transform=ax.transAxes)

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) +
                     ' to ' + str(end-2), fontsize=22, y=0.95, fontweight=700)
        plt.tight_layout()
        plt.show()

    # Plot the Topic Clusters using Bokeh
    # jupyter labextension install @jupyter-widgets/jupyterlab-manager
    # jupyter labextension install @bokeh/jupyter_bokeh
    def topic_clusters_bokeh(self, tsne_lda, topic_num, n_topics, mcolors):
        output_notebook()
        colsarray = mcolors.TABLEAU_COLORS.items()
        cols = np.array([color for name, color in colsarray])
        plot = figure(title="t-SNE cluster. of {} LDA Topics".format(n_topics),
                      plot_width=900,
                      plot_height=700)
        plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=cols[topic_num])
        show(plot)

    def clustering_nlp(self, dataset, k_means_cluster_centers, colors):
        ax = plt.axes()
        ax.set(xlabel='PC1', ylabel='PC2')
        ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25))
        ax.set_title('NLP Topic Modeling')
        n_clusters = len(k_means_cluster_centers)

        # for each cluster we associate a color
        for k, col in zip(range(n_clusters), colors):
            cluster = dataset[dataset["cluster"] == k]
            cluster_center = k_means_cluster_centers[k]
            ax.plot(cluster['PC1_2d'],
                    cluster['PC2_2d'],
                    linewidth=0.25,
                    alpha=0.5,
                    color=col,
                    linestyle='--',
                    antialiased=True,
                    animated=True,
                    markerfacecolor=col,
                    markeredgewidth=0,
                    marker='.', label='Cluster '+str(k+1))
            ax.plot(cluster_center[0], cluster_center[1], 'o',
                    markerfacecolor=col,
                    markeredgecolor='k',
                    antialiased=True,
                    animated=True,
                    markersize=6)
