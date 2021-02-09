#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 21:24:45 2021

@author: kama
"""
class EvaluationHelper:

    def __init__(self, pipeline, preprocHelper, modelSelectionHelper, datavizHelper):
        self.pipeline = pipeline
        self.preprocHelper = preprocHelper
        self.modelSelectionHelper = modelSelectionHelper
        self.datavizHelper = datavizHelper
        self.model = pipeline.named_steps['estimator']
        
    def features_importance(self, feature_names, threshold = 0.01):
        show_feat_imp = False
        try:
            importance = np.abs(self.model.coef_)
            show_feat_imp = True
        except (RuntimeError, AttributeError, TypeError, NameError, ValueError):
            print("No feature importances via coef_ available for this model")
            pass

        if (show_feat_imp == False):
            try:
                importance = self.model.feature_importances_
                show_feat_imp = True
            except (RuntimeError, AttributeError, TypeError, NameError, ValueError):
                print("No feature importances via feature_importances_ available for this model")
                pass

        if (show_feat_imp == True):
            fig = plt.figure(figsize=(14, feature_names.size*0.8), tight_layout = True)
            plt.xlabel('Importance', {'size':'22'})
            plt.ylabel('Features', {'size':'22'})
            feat_imp_df = pd.DataFrame({'x' : importance , 'y' : feature_names})
            feat_imp_df = feat_imp_df.sort_values('x')
            feat_to_keep = feat_imp_df[feat_imp_df['x'] > threshold].sort_values('x')
            feat_to_drop = feat_imp_df[feat_imp_df['x'] <= threshold].sort_values('x')

            plt.barh(width=feat_to_drop['x'], y=feat_to_drop['y'], color='r')
            plt.barh(width=feat_to_keep['x'], y=feat_to_keep['y'], color='g')
            plt.title("Feature importances via coefficients")
            plt.show()

            print('Try dropping these columns:', feat_to_drop['y'].values.tolist())

    def evaluation(self, X_test, y_test):
        y_test_pred = self.pipeline.predict(X_test)

        # pas ailleurs...?
        X_test_pp, y_test_pp = self.preprocHelper.preprocess(X_test, y_test)
        y_test_pred_pp = self.preprocHelper.preprocessY(y_test_pred, y_test.name)

        # Classification report
        if (ml_type == "classification"):
            print(confusion_matrix(y_test, y_test_pred))
            print(classification_report(y_test_pp, y_test_pred_pp))

        # Validation Learning curve
        N, test_score, val_score = learning_curve(self.model, X_test_pp, y_test_pp, 
                                                  train_sizes = np.linspace(0.1, 1.0, 10), 
                                                  cv=5, verbose=0)
        self.datavizHelper.learningcurve(N, test_score, val_score)

        model_scoring_method = self.modelSelectionHelper.get_scoring_method()
        self.datavizHelper.errorhist(y_test_pp, y_test_pred_pp)

        # Features importance
        self.features_importance(np.array(X_test_pp.columns), threshold = 0.02)