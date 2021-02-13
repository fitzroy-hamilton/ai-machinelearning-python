#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 21:23:05 2021

@author: Jeremy Levens
"""

from pandas.api.types import is_numeric_dtype
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (RobustScaler, OneHotEncoder, LabelEncoder,
                                   Normalizer)
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import math

from iapkg.emptycolumndropper import EmptyColumnDropper


class PreprocessingPipelineHelper:
    def __init__(self, dataset,
                 int_feat=None, cont_feat=None, cat_nom_feat=None,
                 cat_ord_feat=None, date_feat=None, obj_feat=None):
        X_data = self.getX(dataset)
        self.preprocessor = None

        self.int_feat, self.cont_feat, self.cat_nom_feat, self.cat_ord_feat, \
            self.date_feat, self.obj_feat = self.splitXByType(X_data,
                                                              int_feat,
                                                              cont_feat,
                                                              cat_nom_feat,
                                                              cat_ord_feat,
                                                              date_feat,
                                                              obj_feat)

        # Steps for preprocessing :
        # 1. Custom transformations
        # 2. Missing values
        # 3. Polynomial features
        # 4. Encoding (for categorical Features)
        # 5. Discretization / binarization (for numerical features)
        # 6. Feature scaling (e.g. Standardization)
        # 7. Normalization
        continuous_pipeline = Pipeline(steps=[
            ('customtransfo', EmptyColumnDropper(threshold=0.9)),
            ('missing-values', SimpleImputer(strategy='median')),
            # ('poly-features', PolynomialFeatures(include_bias=False)),
            # ('discretizer', KBinsDiscretizer(encode='ordinal')),
            ('scaler', RobustScaler()),
            ('normalizer', Normalizer(norm='l2')),
            ('select-kbest', SelectKBest(f_classif,
                                         k=math.ceil(len(self.cont_feat)*0.66)
                                         )
             )
            ])

        categorical_nominal_pipeline = Pipeline(steps=[
                ('customtransfo', EmptyColumnDropper(threshold=0.9)),
                ('missing-values', SimpleImputer(strategy='constant',
                                                 fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='error',
                                         drop='if_binary'))])

        self.preprocessor = ColumnTransformer(
            transformers=[
                    # ('Integer Features', None, self.int_feat),
                    ('Continuous Feat.', continuous_pipeline, self.cont_feat),
                    ('Categ. (nominal) Feat.', categorical_nominal_pipeline,
                     self.cat_nom_feat)],
                    # ('Object Features', None, self.obj_feat)])
            remainder='drop')

    # Features:
    # - Numerical
    # -- Discrete (int64)
    # -- Continuous (float64)
    # - Categorical
    # -- Nominal
    # -- Ordinal
    # -- Date / time
    def splitXByType(self, X, int_feat, cont_feat, cat_nom_feat, cat_ord_feat,
                     date_feat, obj_feat):
        # FIXME : if params are not null, it overrides the line below

        ints = list(X.columns[X.dtypes == 'int64'])
        conts = list(X.columns[X.dtypes == 'float64'])
        cat_nom = list(X.columns[X.dtypes == 'category'])
        cat_ord = list()
        dates = list()
        objs = list(X.columns[X.dtypes == 'object'])

        return ints, conts, cat_nom, cat_ord, dates, objs

    def getX(self, dataset):
        X = dataset.drop(dataset.columns[len(dataset.columns)-1], axis=1)
        return X

    def getY(self, dataset):
        y = dataset[dataset.columns[len(dataset.columns)-1]]
        return y

    def preprocessY(self, y, yname=None):
        # preprocessing for y
        # if the target value are not in number format, we encode them
        if (not is_numeric_dtype(y.dtype)):
            le = LabelEncoder()
            if (yname is None):
                name = y.name
            else:
                name = yname
            y_pp = pd.Series(data=le.fit_transform(y), name=name)
        else:
            y_pp = y
        return y_pp

    def get_pp_colsname(self):
        # gather all the pipelines...
        # int_pipeline  = self.preprocessor.transformers_[0][1]
        cont_pipeline = self.preprocessor.transformers_[0][1]
        cat_nom_pipeline = self.preprocessor.transformers_[1][1]
        # obj_pipeline  = self.preprocessor.transformers_[3][1]

        # ... and original columns names
        # int_cols_name = self.int_feat
        cont_cols_name = self.cont_feat
        cat_nom_cols_name = self.cat_nom_feat
        # obj_cols_name = self.obj_feat

        # Integer features
        # --> TbD

        # Continuous features
        if ((cont_cols_name is not None) and len(cont_cols_name) > 0):
            # --> EmptyColumnDropper
            custotrans_mask = cont_pipeline.named_steps['customtransfo']\
                .get_support()
            inverted_mask = [not elem for elem in custotrans_mask]
            remaining_feat = pd.DataFrame(cont_cols_name)[inverted_mask]
            cont_cols_name = remaining_feat[0].to_list()

            # --> Select KBest Transformer
            selectkbest_mask = cont_pipeline.named_steps['select-kbest']\
                                            .get_support()
            remaining_feat = pd.DataFrame(cont_cols_name)[selectkbest_mask]
            cont_cols_name = remaining_feat[0].to_list()

        # Categorical features - Nominal
        if ((cat_nom_cols_name is not None) and len(cat_nom_cols_name) > 0):
            # --> Customer Transformer
            custotrans_mask = cat_nom_pipeline.named_steps['customtransfo']\
                .get_support()
            inverted_mask = [not elem for elem in custotrans_mask]
            remaining_feat = pd.DataFrame(cat_nom_cols_name)[inverted_mask]
            cat_nom_cols_name = remaining_feat[0].to_list()

            # --> Select OneHotEncoder
            cat_nom_cols_name = cat_nom_pipeline.named_steps['onehot']\
                .get_feature_names(cat_nom_cols_name).tolist()

        # Categorical features - Ordinal
        # --> Customer Transformer

        # --> Select OrdinalEncoder

        # Object features
        # --> TbD

        # Polynomial Features Transformer
#        polyfeatured_feat = self.preprocessor.transformers_[0][1]\
#                            .named_steps['poly-features']\
#                            .get_feature_names(self.continuous_features)

#        cols_name = int_cols + cont_cols + cat_nom_cols + obj_cols
        cols_name = cont_cols_name + cat_nom_cols_name

        return cols_name

    def preprocess(self, X, y, verbose=True):
        if (verbose):
            print('Columns changes with preprocessing:')
            print('- Before:', X.columns.to_list())

        # preprocessing for y
        y_pp = self.preprocessY(y)

        # preprocessing for X
        data = self.preprocessor.fit_transform(X, y_pp)
        cols_name = self.get_pp_colsname()

        if (verbose):
            print('- After:', cols_name)
            print(' ')

        X_pp = pd.DataFrame(data=data,
                            columns=cols_name)

        return X_pp, y_pp
