#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 21:23:49 2021

@author: Jeremy Levens
"""

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier,
                              GradientBoostingClassifier, BaggingClassifier,
                              ExtraTreesClassifier, StackingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              RandomForestRegressor, ExtraTreesRegressor,
                              StackingRegressor)
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.cluster import KMeans
from sklearn.decomposition import (FastICA, IncrementalPCA, PCA, KernelPCA,
                                   SparsePCA, NMF)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.manifold import (TSNE, Isomap, SpectralEmbedding,
                              LocallyLinearEmbedding)
import numpy as np
import pandas as pd
import math


class ModelSelectionHelper:

    def __init__(self, ml_type):
        self.ml_type = ml_type
        self.models_classif = {
            'SGDClassifier': SGDClassifier(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'GaussianNB': GaussianNB(),
            'SVC': SVC(),
            'MultiLayerPerceptronClassifier': MLPClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'BaggingClassifier': BaggingClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'ExtraTreesClassifier': ExtraTreesClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'StackingClassifier': StackingClassifier(
                [('SGD', SGDClassifier(random_state=42)),
                 ('Tree', DecisionTreeClassifier(random_state=42)),
                 ('KNN', KNeighborsClassifier(n_neighbors=2))],
                final_estimator=KNeighborsClassifier())
        }
        self.hyperparams_classif = {
            'SGDClassifier': {},
            'KNeighborsClassifier': {
                'n_neighbors': [2, 4, 6, 8, 10, 12],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                'leaf_size': [10, 20, 30, 40, 50, 60, 70, 100]
                },
            'DecisionTreeClassifier': {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_features': [None, 'auto', 'sqrt', 'log2']
                },
            'GaussianNB': {},
            'SVC': [
                {'kernel': ['linear'], 'C': np.logspace(0.1, 2, 3)},
                {'kernel': ['poly'], 'C': np.logspace(0.1, 2, 3)},
                {'kernel': ['rbf'], 'C': np.logspace(0.1, 2, 3),
                 'gamma': [0.001, 0.0001]},
                {'kernel': ['sigmoid'], 'C': np.logspace(0.1, 2, 3)}
            ],
            'MultiLayerPerceptronClassifier': {
                'solver': ['lbfgs', 'sgd', 'adam'],
                'hidden_layer_sizes': [(150, 150)],
                'tol': [0.000001, 0.00001, 0.0001, 0.001],
                'max_iter': [500],
                'random_state': [42]
                },
            'AdaBoostClassifier':  {
                'n_estimators': [32, 64, 128],
                'learning_rate': [0.1, 0.5, 1.0],
                'algorithm': ['SAMME', 'SAMME.R']
                },
            'BaggingClassifier': {'base_estimator': [KNeighborsClassifier()]},
            'RandomForestClassifier': {
                'n_estimators': [32, 64, 128],
                'criterion':  ['gini', 'entropy']
                },
            'ExtraTreesClassifier': {'n_estimators': [32, 64, 128]},
            'GradientBoostingClassifier': {
                'n_estimators': [32, 64, 128],
                'learning_rate': [0.1, 0.5, 1.0],
                'criterion': ['friedman_mse', 'mse', 'mae']
                },
            'StackingClassifier': {}
        }
        self.models_regress = {
            'Linear Regression': LinearRegression(),
            'SVR': SVR(),
            'Decision Tree': DecisionTreeRegressor(),
            'MultiLayerPerceptronRegressor': MLPRegressor(),
            'Lasso': Lasso(alpha=0.1),
            'Ridge': Ridge(),
            'ElasticNet': ElasticNet(),
            'AdaBoostRegressor': AdaBoostRegressor(),
            'BaggingRegressor': BaggingRegressor(),
            'RandomForestRegressor': RandomForestRegressor(),
            'ExtraTreesRegressor': ExtraTreesRegressor(),
            'StackingRegressor': StackingRegressor(
                [('SGD', LinearRegression()),
                 ('Tree', DecisionTreeRegressor(random_state=42)),
                 ('MLP', MLPRegressor())],
                final_estimator=AdaBoostRegressor())
        }
        self.hyperparams_regress = {
            'Linear Regression': {},
            'SVR': [
                {'kernel': ['linear'], 'C': np.logspace(0.1, 2, 3),
                 'epsilon': [0.1, 1]},
                {'kernel': ['poly'], 'C': np.logspace(0.1, 2, 3),
                 'degree': [2, 3, 4],
                 'gamma': ['scale', 'auto', 0.001, 0.0001],
                 'epsilon': [0.1, 1]},
                {'kernel': ['rbf'], 'C': np.logspace(0.1, 2, 3),
                 'gamma': ['scale', 'auto', 0.001], 'epsilon': [0.1, 1]},
                {'kernel': ['sigmoid'], 'C': np.logspace(0.1, 2, 3),
                 'gamma': ['scale', 'auto', 0.001], 'epsilon': [0.1, 1]}
            ],
            'Decision Tree': {
                'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                'splitter': ['best', 'random'],
                'max_features': [None, 'auto', 'sqrt', 'log2']
                },
            'MultiLayerPerceptronRegressor': {
                'solver': ['lbfgs', 'sgd', 'adam'],
                'hidden_layer_sizes': [(150, 150)],
                'tol': [0.000001, 0.00001, 0.0001, 0.001],
                'max_iter': [500],
                'random_state': [42]
                },
            'Lasso': {},
            'Ridge': {
                'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'saga'],
                'alpha': [0.1, 1.0, 10.0]
                },
            'ElasticNet': {
                'l1_ratio': [0.1, 0.5, 1.0], 'alpha': [0.1, 1.0, 10.0]
                },
            'AdaBoostRegressor': {},
            'BaggingRegressor': {},
            'RandomForestRegressor': {},
            'ExtraTreesRegressor': {},
            'StackingRegressor': {}
        }
        self.models_cluster = {
            'K-Means': KMeans()
        }
        self.hyperparams_cluster = {
            'K-Means': {}
        }
        self.models_dimrec = {
            'Fast ICA': FastICA(n_components=7),
            'Incremental PCA': IncrementalPCA(n_components=7, batch_size=200),
            'PCA': PCA(n_components=2),
            'Kernel PCA': KernelPCA(n_components=7, kernel='linear'),
            'Sparse PCA': SparsePCA(n_components=5),
            'NMF': NMF(n_components=2, init='random'),
            'LDA': LinearDiscriminantAnalysis(n_components=2),
            'Neighborhood CA': NeighborhoodComponentsAnalysis(n_components=2),
            't-SNE': TSNE(n_components=2),
            'Isomap': Isomap(n_components=2),
            'SpectralEmbedding': SpectralEmbedding(n_components=2),
            'LLE': LocallyLinearEmbedding(n_components=2)
        }
        self.hyperparams_dimrec = {
            'Fast ICA': {},
            'Incremental PCA': {},
            'PCA': {},
            'Kernel PCA': {},
            'Sparse PCA': {},
            'NMF': {},
            'LDA': {},
            'Neighborhood Component Analysis': {},
            't-SNE': {},
            'Isomap': {},
            'SpectralEmbedding': {},
            'LLE': {}
        }

        if (self.ml_type == 'classification'):
            self.scoring = 'f1'
            self.models = self.models_classif
            self.params = self.hyperparams_classif
        elif (self.ml_type == 'regression'):
            self.scoring = 'r2'
            self.models = self.models_regress
            self.params = self.hyperparams_regress
        elif (self.ml_type == 'clustering'):
            self.scoring = None
            self.models = self.models_cluster
            self.params = self.hyperparams_cluster
        elif (self.ml_type == 'dimensionality-reduction'):
            self.scoring = 'accuracy'
            self.models = self.models_dimrec
            self.params = self.hyperparams_dimrec

        self.keys = self.models.keys()
        self.grid_searches = {}
        self.preprocessor = None
        self.best_cluster_model = None

    def gs(self, X, y, cv=3, n_jobs=-1, verbose=1, refit=True):
        print('\n', "Benchmarking estimators & hyperparameter optimization:")
        for key in self.keys:
            model = self.models[key]
            params = self.params[key]

            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=self.scoring,
                              refit=refit, return_train_score=True)
            gs.fit(X, y)
            self.grid_searches[key] = gs
            print("- %s" % key, math.ceil(gs.best_score_*100)/100,
                  "after", len(gs.cv_results_['params']), "combinations")
        print(" ")

    def get_optimal_nb_clusters(self, X, datavizHelper, threshold=0.05):
        inertia = []
        gaps = []
        K_range = range(1, 20)
        i = 0
        for k in K_range:
            model = KMeans(n_clusters=k).fit(X)
            inertia.append(model.inertia_)
            if (i == 0):
                gaps.append(0)
            else:
                gaps.append((inertia[i] - inertia[i-1]) / inertia[i-1])
            i = i + 1

        # Calculate the optimal number of cluster (= increase the number of
        # clusters won't decrease the inertia more than 'threshold' (e.g.: 0.05)
        optimal_nb_cluster = 0
        while (optimal_nb_cluster < len(gaps)):
            optimal_nb_cluster = optimal_nb_cluster + 1
            if (gaps[optimal_nb_cluster] > -threshold):
                break

        datavizHelper.drawElbow(K_range, optimal_nb_cluster, inertia)

        return optimal_nb_cluster

    def gs_cluster(self, X, datavizHelper, threshold):
        optim_model = self.models_cluster['K-Means']
        optimal_nb_cluster = self.get_optimal_nb_clusters(X,
                                                          datavizHelper,
                                                          threshold)
        optim_model.set_params(n_clusters=optimal_nb_cluster)
        self.best_cluster_model = optim_model

    def get_best_model(self):
        if (self.ml_type == 'clustering'):
            return self.best_cluster_model, None

        best_score = 0
        best_estimator = ""

        for k in self.grid_searches:
            if (self.grid_searches[k].best_score_ > best_score):
                best_score = self.grid_searches[k].best_score_
                best_estimator = k

        optim_model = self.grid_searches[best_estimator].best_estimator_
        print("Best model with %.2f%% : " % (best_score*100))
        print('-> ', optim_model, '\n')

        return optim_model, best_score

    def get_scoring_method(self):
        return self.scoring

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            # print(k, "- Best estim:", self.grid_searches[k].best_estimator_)
            # print("- Score: %.2f%" % (self.grid_searches[k].best_score_))
            # print("- Best Params :", self.grid_searches[k].best_params_)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score',
                   'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
