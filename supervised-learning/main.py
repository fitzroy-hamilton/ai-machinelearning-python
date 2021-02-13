#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 21:08:21 2021

@author: Jeremy Levens
"""

import sys
import pandas as pd

from sklearn import __version__
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline

from iapkg.datavizhelper import DatavizHelper
from iapkg.preprocessingpipelinehelper import PreprocessingPipelineHelper
from iapkg.modelselectionhelper import ModelSelectionHelper
from iapkg.evalutionhelper import EvaluationHelper


def export(dataset, openml_source, openml_version):
    # dataset = pd.concat([X, y], axis=1, sort=False)
    excel_filename = 'datasource-' + openml_source + '-v'
    + str(openml_version) + '-' + '.xlsx'
    writer = pd.ExcelWriter(excel_filename)

    dataset.to_excel(writer)
    writer.save()

    print('DataFrame is written successfully to:', excel_filename)


def import_data(openml_source, export_excel=False):
    # 'fetch_openml' can return directly 'X' and 'y' but alternatively
    # 'X' and 'y' can be obtained directly from the frame attribute:
    # X = dataset.frame.drop('survived', axis=1)
    # y = dataset.frame['survived']
    openml_version = 1
    original_X, original_y = fetch_openml(openml_source,
                                          version=openml_version,
                                          return_X_y=True, as_frame=True)

    df = pd.concat([original_X.reset_index(drop=True),
                    original_y.reset_index(drop=True)],
                   axis=1, sort=False)

    if (export_excel):
        export(df, openml_source, openml_version)

    return df


def build_pipeline(ml_type, df, pipeline):
    print('Columns of the dataset:', list(df.columns))
    print(' ')

    # Split between train and test dataset
    from sklearn.model_selection import train_test_split
    trainset, testset = train_test_split(df, test_size=0.2)

    # Preprocessing
    preprocHelper = PreprocessingPipelineHelper(trainset)
    X_train = preprocHelper.getX(trainset)
    y_train = preprocHelper.getY(trainset)
    X_train_pp, y_train_pp = preprocHelper.preprocess(X_train, y_train,
                                                      verbose=False)
    print('ypp:', y_train_pp)
    score = 0
    datavizHelper = DatavizHelper(ml_type)
    if (pipeline is None):
        # Dataviz
        datavizHelper.emptyValues(X_train_pp, y_train_pp)
        # datavizHelper.distributionNumerical(X_train, y_train)
        datavizHelper.distributionNumerical(X_train_pp, y_train_pp)
        datavizHelper.heatmap(X_train_pp, y_train_pp)

        # Model selection
        modelSelectionHelper = ModelSelectionHelper(ml_type)
        modelSelectionHelper.fit(X_train_pp, y_train_pp, n_jobs=5, verbose=0)

        # Pipeline
        best_model, score = modelSelectionHelper.get_best_model()
        pipeline = Pipeline(steps=[
                        ('preprocessor', preprocHelper.preprocessor),
                        ('estimator', best_model)
                    ])

    # Training
    pipeline.fit(X_train, y_train_pp)

    # Evaluation
    X_test = preprocHelper.getX(testset)
    y_test = preprocHelper.getY(testset)
    evaluationHelper = EvaluationHelper(pipeline,
                                        preprocHelper,
                                        datavizHelper)
    cols_to_drop = evaluationHelper.evaluation(ml_type, X_test, y_test)

    return pipeline, score, cols_to_drop


def featureEngineering(df, cols_to_drop):
    if len(cols_to_drop) < 1:
        print(' ')
        print('Feature dropping:', cols_to_drop)
        print(' ')
        df = df.drop(cols_to_drop, axis=1)

    return df


def main(ml_type, openml_source):
    required_version = "0.22"
    try:
        if (__version__ < required_version):
            raise Exception()
    except Exception:
        print("*** /!\\ scikit-learn >=", required_version,
              "required while", __version__, "found")
        sys.exit(1)

    # Import data
    df = import_data(openml_source, export_excel=False)

    # Define the preprocessing and find the best estimator to build pipeline
    pipeline = None
    cols_to_drop = []
    i = 0
    while (i < 2):
        i = i+1

        # Feature dropping / Feature engineering
        df = featureEngineering(df, cols_to_drop)

        print('*************')
        print('*** ROUND #', i)
        print('*************')
        pipeline, score, cols_to_drop = build_pipeline(ml_type, df, pipeline)

        print(' ')
        print("*** RESULT: %.2f%%" % (score*100))
        print(' ')
        print(pipeline)
        print(' ')
        print(' ')

        if len(cols_to_drop) < 1:
            break

    # Make predictions
    print('*************************')
    print('*** READY FOR PREDICTION')
    print('*************************')


ml_type = "classification"
# openml_source = 'titanic'     # https://www.openml.org/d/40945
openml_source = 'diabetes'    # https://www.openml.org/d/37
# openml_source = 'credit-g'      # https://www.openml.org/d/31

# ml_type = "regression"
# openml_source = 'stock'       # https://www.openml.org/d/223
# openml_source = 'auto_price'  # https://www.openml.org/d/195

main(ml_type, openml_source)
