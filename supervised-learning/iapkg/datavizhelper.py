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

    def emptyValues(self, X, y):
        plt.figure(figsize=(10, 6))

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
        subdistribdataset = distribdataset.select_dtypes([np.number])

        cmap = sns.color_palette("Blues", as_cmap=True)
        sns.set(style='white')
        sns.set_style(rc={'axes.spines.top': False, 'axes.spines.right': False,
                          'axes.edgecolor': 'lightgrey'})

        i = 1
        nblines = math.ceil(len(subdistribdataset.columns) / 2)
        plt.figure(figsize=(14, nblines*5), tight_layout=True)

        for feature in subdistribdataset:
            plt.subplot(nblines, 2, i)

            sns.histplot(subdistribdataset[feature], kde=True, palette=cmap)
            i = i + 1

        plt.show()

    def distributionCategorical(self, X, y):
        data = pd.concat([X, y], axis=1, sort=False)
        subdistribdataset = data.select_dtypes('category')

        i = 1
        nblines = math.ceil(len(subdistribdataset.columns) / 2)
        plt.figure(figsize=(14, nblines*5), tight_layout=True)

        for feature in subdistribdataset:
            print(f'{feature :-<30} {data[feature].unique()}')
            plt.subplot(nblines, 2, i)

            data[feature].value_counts().plot.pie(textprops={'fontsize': 16})
            i = i + 1

        plt.show()

    def relationshipNumericalFeatureTarget(self, X, y):
        data = pd.concat([X, y], axis=1, sort=False)
        subdistribdataset = data.select_dtypes([np.number])

        if (y.unique().size > self.max_features):
            print('Too many target unique values')
        else:
            target_dataset = {}
            for target in y.unique():
                target_dataset[target] = data[data[data.columns[-1]] == target]

            i = 1
            nblines = math.ceil(len(subdistribdataset.columns) / 2)
            plt.figure(figsize=(14, nblines*5), tight_layout=True)

            for feature in subdistribdataset:
                plt.subplot(nblines, 2, i)

                for target in y.unique():
                    sns.distplot(target_dataset[target][feature], label=target)
                plt.legend()
                i = i + 1

            plt.show()

    def relationshipCategoricalFeatureTarget(self, X, y):
        dataset = X
        subdataset = dataset.select_dtypes('category')

        if (subdataset.size < 1):
            print('No categorical feature to show')
        elif (subdataset.columns.size > self.max_features):
            print('Too many categorical features to show:', subdataset.size)
        else:
            i = 1
            nblines = math.ceil(len(subdataset.columns) / 2)
            plt.figure(figsize=(14, nblines*5), tight_layout=True)

            for feature in subdataset:
                plt.subplot(nblines, 2, i)

                sns.countplot(x=feature, hue=y, data=dataset)
                plt.legend()
                i = i + 1

            plt.show()

    def heatmap(self, X, y, mirrored=False):
        data = pd.concat([X, y], axis=1, sort=False)
        if (data.select_dtypes([np.number]).columns.size < self.max_features):
            corr = data.corr()

            fig, ax = plt.subplots(figsize=(10, 10))
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
        err_hist = np.abs(y - y_pred)
        plt.xlabel('Error value')
        plt.ylabel('Quantity')
        plt.hist(err_hist, bins=50)
        plt.show()
