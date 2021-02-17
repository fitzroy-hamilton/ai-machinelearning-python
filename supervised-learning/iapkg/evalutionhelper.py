#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 21:24:45 2021

@author: Jeremy Levens
"""

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class EvaluationHelper:

    def __init__(self, pipeline, preprocHlp, datavizHlp):
        self.pipeline = pipeline
        self.preprocHlp = preprocHlp
        self.datavizHelper = datavizHlp
        self.model = pipeline.named_steps['estimator']

    def features_importance(self, feature_names, threshold=0.01):
        show_feat_imp = False
        print('Features importance:')
        try:
            importance = np.abs(self.model.coef_)
            show_feat_imp = True
        except (AttributeError, TypeError, NameError, ValueError):
            print("- No feature importances via coef_ available for this model")
            pass

        if (show_feat_imp is False):
            try:
                importance = self.model.feature_importances_
                show_feat_imp = True
            except (AttributeError, TypeError, NameError, ValueError):
                print("- No feature importances via feature_importances_ "
                      + "available for this model")
                pass
            return []

        if (show_feat_imp is True):
            plt.figure(figsize=(14, feature_names.size*0.8), tight_layout=True)
            plt.xlabel('Importance', {'size': '22'})
            plt.ylabel('Features', {'size': '22'})
            feat_imp = pd.DataFrame({'x': importance, 'y': feature_names})
            feat_imp = feat_imp.sort_values('x')
            keep_feat = feat_imp[feat_imp['x'] > threshold].sort_values('x')
            drop_feat = feat_imp[feat_imp['x'] <= threshold].sort_values('x')

            plt.barh(width=drop_feat['x'], y=drop_feat['y'], color='r')
            plt.barh(width=keep_feat['x'], y=keep_feat['y'], color='g')
            plt.title("Feature importances via coefficients")
            plt.show()
            print(' ')

            print('Try dropping these columns:', drop_feat['y'].values.tolist())
            print(' ')
            return drop_feat['y'].values.tolist()

    def evaluation(self, ml_type, X_test, y_test):
        y_test_pred = self.pipeline.predict(X_test)

        X_test_pp, y_test_pp = self.preprocHlp.preprocess(X_test, y_test,
                                                          ml_type)
        y_test_pred_pp = self.preprocHlp.preprocessY(y_test_pred, y_test.name)

        # Classification report
        if (ml_type == "classification"):
            print("Confusion matrix:")
            print(confusion_matrix(y_test, y_test_pred))
            print(' ')
            print('Classification report:')
            print(classification_report(y_test_pp, y_test_pred_pp))

        # Validation Learning curve
        train_size = np.linspace(0.1, 1.0, 10)
        N, test_score, val_score = learning_curve(self.model,
                                                  X_test_pp,
                                                  y_test_pp,
                                                  train_sizes=train_size,
                                                  cv=5,
                                                  verbose=0)
        self.datavizHelper.learningcurve(N, test_score, val_score)
        self.datavizHelper.errorhist(y_test_pp, y_test_pred_pp)

        # Features importance and return columns which could be dropped
        return self.features_importance(np.array(X_test_pp.columns),
                                        threshold=0.02)
