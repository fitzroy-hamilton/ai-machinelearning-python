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


def export_data(dataset, openml_source, openml_version):
    # dataset = pd.concat([X, y], axis=1, sort=False)
    excel_filename = 'datasource-' + openml_source + '-v'
    + str(openml_version) + '-' + '.xlsx'
    writer = pd.ExcelWriter(excel_filename)

    dataset.to_excel(writer)
    writer.save()

    print('DataFrame is written successfully to:', excel_filename)


def import_data(source, export_excel=False):
    df = None

    if (source.endswith('.csv')):
        df = pd.read_csv(source)
    else:
        # 'fetch_openml' can return directly 'X' and 'y' but alternatively
        # 'X' and 'y' can be obtained directly from the frame attribute:
        # X = dataset.frame.drop('survived', axis=1)
        # y = dataset.frame['survived']
        openml_version = 1
        original_X, original_y = fetch_openml(source,
                                              version=openml_version,
                                              return_X_y=True, as_frame=True)

        df = pd.concat([original_X.reset_index(drop=True),
                        original_y.reset_index(drop=True)],
                       axis=1, sort=False)

        if (export_excel):
            export_data(df, source, openml_version)

    return df


def build_pipeline(ml_type, df, pipeline):
    print('Columns of the dataset:', list(df.columns), '\n')

    # Split between train and test dataset
    from sklearn.model_selection import train_test_split
    trainset, testset = train_test_split(df, test_size=0.2)

    # Preprocessing
    preprocHelper = PreprocessingPipelineHelper(trainset, ml_type)
    X_train = preprocHelper.getX(trainset, ml_type)
    y_train = preprocHelper.getY(trainset, ml_type)
    X_train_pp, y_train_pp = preprocHelper.preprocess(X_train, y_train,
                                                      ml_type,
                                                      verbose=False)

    score = 0
    datavizHelper = DatavizHelper(ml_type)
    if (pipeline is None):
        # Dataviz
        datavizHelper.emptyValues(X_train_pp, y_train_pp)
        datavizHelper.distributionNumerical(X_train, y_train)
        # datavizHelper.distributionNumerical(X_train_pp, y_train_pp)
        datavizHelper.relationNumericalFeatureTarget(X_train_pp, y_train_pp)
        datavizHelper.heatmap(X_train_pp, y_train_pp)

        # Model selection performing a grid search
        modelSelectionHelper = ModelSelectionHelper(ml_type)
        if ((ml_type == 'classification') or (ml_type == 'regression')):
            modelSelectionHelper.gs(X_train_pp, y_train_pp, n_jobs=5, verbose=0)
        best_model, score = modelSelectionHelper.get_best_model()

        # Pipeline
        pipeline = Pipeline(steps=[
                        ('preprocessor', preprocHelper.preprocessor),
                        ('estimator', best_model)
                    ])

    # Training
    if ((ml_type == 'classification') or (ml_type == 'regression')):
        pipeline.fit(X_train, y_train)
    elif (ml_type == 'clustering'):  # FIXME
        # FIXME CLUSTER SPECIFIC : mettre le PCA dans le preprocessor ?
        # en prenant en entree X_train -> X_train_pp -> X_train_pp_reduced
        # ou comme nouvelle step du pipeline entre preprocessor et estimator
        X_train_pp_reduced2 = preprocHelper.PCAreduct(2, X_train_pp)
        best_model2 = best_model
        best_model2.fit(X_train_pp_reduced2)
        y_clusters = best_model2.predict(X_train_pp_reduced2)
        datavizHelper.clustering(2, X_train_pp_reduced2, y_clusters,
                                 best_model2.cluster_centers_)
        X_train_pp_reduced3 = preprocHelper.PCAreduct(3, X_train_pp)
        best_model3 = best_model
        best_model3.fit(X_train_pp_reduced3)

    # Evaluation
    cols_to_drop = []
    if ((ml_type == 'classification') or (ml_type == 'regression')):
        X_test = preprocHelper.getX(testset)
        y_test = preprocHelper.getY(testset)
        evaluationHelper = EvaluationHelper(pipeline,
                                            preprocHelper,
                                            datavizHelper)
        cols_to_drop = evaluationHelper.evaluation(ml_type, X_test, y_test)
    elif (ml_type == 'clustering'):  # FIXME
        y_clusters = best_model3.predict(X_train_pp_reduced3)
        datavizHelper.clustering(3, X_train_pp_reduced3, y_clusters,
                                 best_model3.cluster_centers_)

    return pipeline, score, cols_to_drop


def featureEngineering(df, cols_to_drop):
    if len(cols_to_drop) > 0:
        print('\n', 'Feature dropping:', cols_to_drop, '\n')
        df = df.drop(cols_to_drop, axis=1)

    return df


def main(ml_type, datasource):
    max_rounds = 2
    required_version = "0.22"
    try:
        if (__version__ < required_version):
            raise Exception()
    except Exception:
        print("*** /!\\ scikit-learn >=", required_version,
              "required while", __version__, "found")
        sys.exit(1)

    # Import data
    df = import_data(datasource, export_excel=False)

    # Define the preprocessing and find the best estimator to build pipeline
    pipeline = None
    cols_to_drop = []
    i = 0
    while (i < max_rounds):
        i = i+1

        print('*************')
        print('***', ml_type, '- ROUND #', i)
        print('*************')
        pipeline, score, cols_to_drop = build_pipeline(ml_type, df, pipeline)

        if ((ml_type == 'classification') or (ml_type == 'regression')):
            print('\n', "*** RESULT: %.2f%%" % (score*100))
        print('\n', pipeline, '\n')

        if len(cols_to_drop) < 1:
            break

        # Feature dropping / Feature engineering
        df = featureEngineering(df, cols_to_drop)

    # Make predictions
    print('*************************')
    print('*** READY FOR PREDICTION')
    print('*************************')


# ml_type = "classification"
# openml_source = 'titanic'     # https://www.openml.org/d/40945
# openml_source = 'diabetes'    # https://www.openml.org/d/37
# openml_source = 'credit-g'      # https://www.openml.org/d/31

# ml_type = "regression"
# openml_source = 'stock'       # https://www.openml.org/d/223
# openml_source = 'auto_price'  # https://www.openml.org/d/195

ml_type = "clustering"
openml_source = 'data/clustering-credit-card-customer-segmentation.csv'

main(ml_type, openml_source)
