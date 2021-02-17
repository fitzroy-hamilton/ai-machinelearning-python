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
        plt.figure(figsize=(9, 9), dpi=300, tight_layout=True)
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

            plt.figure(figsize=(12, 12), dpi=300, tight_layout=True)
            plt.clf()
            ax = plt.axes()
            colormap = sns.color_palette("RdBu_r", 7)

            dropSelf = None
            if mirrored is False:
                # Drop self-correlations
                dropSelf = np.zeros_like(corr)
                dropSelf[np.triu_indices_from(dropSelf)] = True

            sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f",
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
        from scipy.interpolate import make_interp_spline
        import matplotlib.pyplot as plt
        import numpy as np

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

    def clustering(self, d, X, y_clusters, k_means_cluster_centers):
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

        if (d == 2):
            self.clustering2D(dataset, k_means_cluster_centers, colors)
        elif (d == 3):
            self.clustering3D(dataset, k_means_cluster_centers, colors)

        plt.legend(scatterpoints=1, ncol=2, markerscale=3.0)
        plt.show()

    def clustering2D(self, dataset, k_means_cluster_centers, colors):
        ax = plt.axes()
        ax.set(xlabel='PC1', ylabel='PC2')
        ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25))
        ax.set_title('Clustering 2D')
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

    def clustering3D(self, dataset, k_means_cluster_centers, colors):
        ax = plt.axes(projection='3d')
        ax.set(xlabel='PC1', ylabel='PC2', zlabel='PC3')
        ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), zlim=(-1.1, 1.1))
        ax.set_title('Clustering 3D')
        n_clusters = len(k_means_cluster_centers)

        # for each cluster we associate a color
        for k, col in zip(range(n_clusters), colors):
            cluster = dataset[dataset["cluster"] == k]
            cluster_center = k_means_cluster_centers[k]
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
