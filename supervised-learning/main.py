#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 21:08:21 2021

@author: kama
"""

import sys
import pandas as pd
import numpy as np

from sklearn import __version__
from sklearn import set_config
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline

from .datavizhelper import DatavizHelper
from .preprocessingpipelinehelper import PreprocessingPipelineHelper
from .modelselectionhelper import ModelSelectionHelper
from .evalutionhelper import EvaluationHelper

def export(dataset, openml_source, openml_version):
#    dataset = pd.concat([X, y], axis=1, sort=False)
    excel_filename = 'datasource-' + openml_source + '-v' 
    + str(openml_version) + '-' + '.xlsx'
    writer = pd.ExcelWriter(excel_filename)  

    dataset.to_excel(writer)
    writer.save()

    print('DataFrame is written successfully to:', excel_filename)  

def import_data(openml_source, export_excel = False):
    # 'fetch_openml' can return directly 'X' and 'y' but alternatively
    # 'X' and 'y' can be obtained directly from the frame attribute:
    # X = dataset.frame.drop('survived', axis=1)
    # y = dataset.frame['survived']
    openml_version = 1
    original_X, original_y = fetch_openml(openml_source, 
                                          version=openml_version, 
                                          return_X_y = True, as_frame=True)

    df = pd.concat([original_X.reset_index(drop=True), 
                    original_y.reset_index(drop=True)], 
                   axis=1, sort=False)
    
    if (export_excel):
        export(df, openml_source, openml_version)
        
    return df

def main(ml_type, openml_source):
    required_version = "0.22"
    try:
        if (__version__ < required_version):
            raise Exception()
    except Exception:
        print("***")
        print("*** /!\\ scikit-learn >=", required_version, 
              "required while", __version__, "found")
        print("***")
        sys.exit(1)

    np.random.seed(42)
    set_config(display='diagram')
    pd.options.display.float_format = '{:,.3f}'.format
    
    return import_data(openml_source, export_excel = False)

def featureEngineering(df):
    return df

def ialoop(df, ml_type):
    datavizHelper = DatavizHelper(ml_type)
    
    # Split between train and test dataset
    from sklearn.model_selection import train_test_split
    trainset, testset = train_test_split(df, test_size=0.2)

    # Feature dropping / Feature engineering
    trainset = featureEngineering(trainset)
    testset = featureEngineering(testset)

    # Preprocessing
    preprocHelper = PreprocessingPipelineHelper(trainset)
    X_train = preprocHelper.getX(trainset)
    y_train = preprocHelper.getY(trainset)
    X_train_pp, y_train_pp = preprocHelper.preprocess(X_train, y_train)

    # Dataviz
    datavizHelper.emptyValues(X_train_pp, y_train_pp)
    datavizHelper.distributionNumerical(X_train, y_train)
    datavizHelper.distributionNumerical(X_train_pp, y_train_pp)
    datavizHelper.heatmap(X_train_pp, y_train_pp)
    
    # Model selection
    modelSelectionHelper = ModelSelectionHelper(ml_type)
    modelSelectionHelper.fit(X_train_pp, y_train_pp, n_jobs=5, verbose=0)

    # Pipeline
#    final_pipeline = Pipeline(steps=[
#                        ('preprocessor', preprocHelper.preprocessor),
#                        ('estimator', best_model)
#                    ])
#    final_pipeline.fit(X_train, y_train)

    # Evaluation
    X_test = preprocHelper.getX(testset)
    y_test = preprocHelper.getY(testset)
    if hasattr(y_test, 'cat'):
        y_test.value_counts().plot.pie()
        print(y_test.value_counts())
    else:
        y_test.head()
        
    final_pipeline = Pipeline(steps=[
                    ('preprocessor', preprocHelper.preprocessor),
                    ('estimator', modelSelectionHelper.get_best_model())
                ])
    final_pipeline.fit(X_train, y_train) # we train with the trainset

    
    evaluationHelper = EvaluationHelper(final_pipeline,
                                        preprocHelper, 
                                        modelSelectionHelper,
                                        datavizHelper)
    evaluationHelper.evaluation(X_test, y_test)

    final_pipeline
    # [Loop to Feature dropping / Feature engineering]

    # Make predictions

ml_type = "classification" 
openml_source = 'titanic' # https://www.openml.org/d/40945
#openml_source = 'diabetes' # https://www.openml.org/d/37
#openml_source = 'credit-g' # https://www.openml.org/d/31

#ml_type = "regression" 
#openml_source = 'stock' # https://www.openml.org/d/223
#openml_source = 'auto_price' # https://www.openml.org/d/195

df = main(ml_type, openml_source)

ialoop(df, ml_type)