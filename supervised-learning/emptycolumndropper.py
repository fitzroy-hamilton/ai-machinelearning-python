#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 21:22:32 2021

@author: kama
"""

from sklearn.base import BaseEstimator, TransformerMixin

class EmptyColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold = 0.9):
        self.threshold = threshold
        self.cols_dropped = None
        self.X_original_cols_name = None
        
    def fit(self, X, y = None):
        self.X_original_cols_name = X.columns
        
        return self
        
    def transform(self, X, y = None):
        X_ = X.copy()
        self.cols_dropped = [] #FIXME
        return X_ #FIXME
        
        if (y != None):
            y_ = y.copy()
        else:
            y_ = None
            
        # We only keep columns with no more than 10% of missing values
        self.cols_dropped = X_.columns[X_.isna().sum()/X_.shape[0] >= self.threshold]
        #print('Columns dropped by CustomTransformer:')
        #print(self.cols_name_dropped)
        X_ = X_[X_.columns[X_.isna().sum()/X_.shape[0] < self.threshold]]

        return X_
    
    def get_support(self, cols_name = False):
        # return directly the label of each dropped columns
        if (cols_name):
            return self.cols_dropped
        
        # return a list of Boolean which indicate for each feature if
        # the column has been dropped (True) or not (False)
        bool_list = []
        for col in self.X_original_cols_name:
            if (col in self.cols_dropped):
                bool_list.append(True)
            else:
                bool_list.append(False)
        return bool_list