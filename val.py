from modelling import models, models_param_space
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, cross_validate, StratifiedKFold, KFold, RepeatedKFold
from numpy import mean
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import make_scorer
from sklearn import base

scoring_opts = ['accuracy', 'f1_weighted', 'precision_weighted','recall_weighted',"roc_auc_ovr"]

def true_positive_rate(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return (tp)/(tp+fn)

def false_positive_rate(y_true, y_pred):
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    return fp / (fp + tn)

def n_predict_value(y_true, y_pred):
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return (tn)/(tn+fn)

def calc_score(classifier, X, y):
    scores = []
    for scoring_opt in scoring_opts:
        score = cross_val_score(classifier, X, y, scoring=scoring_opt)
        score = (mean(score))
        scores.append(round((score),3)) 

    score_fpr = cross_validation(false_positive_rate, classifier, X, y)
    score_tpr = cross_validation(true_positive_rate, classifier, X, y)
    score_npv = cross_validation(n_predict_value, classifier, X, y)

    return (*scores, score_fpr, score_tpr, score_npv)

def cross_validation(scorer_param, classifier, X, y):
    scorer = make_scorer(scorer_param)
    rate = cross_validate(classifier, X, y, scoring=scorer)
    rate = np.mean(rate['test_score'])
    return round((rate),3)

def holdout(classifier, X, y, k_fold, repeat):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=k_fold/100, random_state=42)
    classifier.fit(X_train, y_train)
    scores_tuple = calc_score(classifier, X_test, y_test)

    scores_dict = {
        'Accuracy': scores_tuple[0],
        'Precision': scores_tuple[1],
        'Recall': scores_tuple[2],
        'F1': scores_tuple[3],
        'False Positive Rate': scores_tuple[4],
        'True Positive Rate': scores_tuple[5],
        'Negative Predictive Value': scores_tuple[6]
    }

    return scores_dict

def repeated_holdout(classifier, X, y, k_fold, repeat):
    results = []

    for _ in range(repeat):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=k_fold)
        classifier_cloned = base.clone(classifier)
        classifier_cloned.fit(X_train, y_train)
        results.append(calc_score(classifier_cloned, X_test, y_test))

    # Transpose the results to get an array with rows corresponding to metrics and columns to repetitions
    results_array = np.array(results).T

    # Calculate the mean for each metric
    mean_scores = {metric: np.mean(scores) for metric, scores in zip(['Accuracy', 'Precision', 'Recall', 'F1', 'False Positive Rate', 'True Positive Rate', 'Negative Predictive Value'], results_array)}

    return mean_scores

def StratKFoldCV(classifier, X, y, k_fold, repeat):
    
    cross_validation = StratifiedKFold(n_splits=k_fold)

    scores = {}

    # Calculate and store scores for each custom metric
    score_fpr = cross_validate(classifier, X, y, cv=cross_validation, scoring=false_positive_rate)
    score_tpr = cross_validate(classifier, X, y, cv=cross_validation, scoring=true_positive_rate)
    score_npv = cross_validate(classifier, X, y, cv=cross_validation, scoring=n_predict_value)

    scores['False Positive Rate'] = np.mean(score_fpr['test_score'])
    scores['True Positive Rate'] = np.mean(score_tpr['test_score'])
    scores['Negative Predictive Value'] = np.mean(score_npv['test_score'])

    # Calculate and store scores for each scoring option
    for scoring_opt in scoring_opts:
        accuracy_scores = cross_validate(classifier, X, y, cv=cross_validation, scoring=scoring_opt)
        test_scores = accuracy_scores['test_score']
        score = np.mean(test_scores)
        scores[scoring_opt] = round(score, 3)

    return scores

def LeaveOneOutCV(classifier, X, y, k_fold, repeat):
    
    cross_validation = LeaveOneOut()

    scores ={}

    score_fpr = cross_validate(classifier, X, y, cv=cross_validation, scoring=false_positive_rate)
    score_tpr = cross_validate(classifier, X, y, cv=cross_validation, scoring=true_positive_rate)
    score_npv = cross_validate(classifier, X, y, cv=cross_validation, scoring=n_predict_value)

    scores['False Positive Rate'] = np.mean(score_fpr['test_score'])
    scores['True Positive Rate'] = np.mean(score_tpr['test_score'])
    scores['Negative Predictive Value'] = np.mean(score_npv['test_score'])

    # Calculate and store scores for each scoring option
    for scoring_opt in scoring_opts:
        accuracy_scores = cross_validate(classifier, X, y, cv=cross_validation, scoring=scoring_opt)
        test_scores = accuracy_scores['test_score']
        score = np.mean(test_scores)
        scores[scoring_opt] = round(score, 3)

    return scores

def KFoldCV(classifier, X, y, k_fold, repeat):
  
    cross_validation = KFold(k_fold)

    scores ={}

    score_fpr = cross_validate(classifier, X, y, cv=cross_validation, scoring=false_positive_rate)
    score_tpr = cross_validate(classifier, X, y, cv=cross_validation, scoring=true_positive_rate)
    score_npv = cross_validate(classifier, X, y, cv=cross_validation, scoring=n_predict_value)

    scores['False Positive Rate'] = np.mean(score_fpr['test_score'])
    scores['True Positive Rate'] = np.mean(score_tpr['test_score'])
    scores['Negative Predictive Value'] = np.mean(score_npv['test_score'])

    # Calculate and store scores for each scoring option
    for scoring_opt in scoring_opts:
        accuracy_scores = cross_validate(classifier, X, y, cv=cross_validation, scoring=scoring_opt)
        test_scores = accuracy_scores['test_score']
        score = np.mean(test_scores)
        scores[scoring_opt] = round(score, 3)

    return scores

def RepeatedKFoldCV(classifier, X, y, k_fold, repeat):

    cross_validation = RepeatedKFold(n_splits=k_fold, n_repeats= repeat)

    scores ={}

    score_fpr = cross_validate(classifier, X, y, cv=cross_validation, scoring=false_positive_rate)
    score_tpr = cross_validate(classifier, X, y, cv=cross_validation, scoring=true_positive_rate)
    score_npv = cross_validate(classifier, X, y, cv=cross_validation, scoring=n_predict_value)

    scores['False Positive Rate'] = np.mean(score_fpr['test_score'])
    scores['True Positive Rate'] = np.mean(score_tpr['test_score'])
    scores['Negative Predictive Value'] = np.mean(score_npv['test_score'])

    # Calculate and store scores for each scoring option
    for scoring_opt in scoring_opts:
        accuracy_scores = cross_validate(classifier, X, y, cv=cross_validation, scoring=scoring_opt)
        test_scores = accuracy_scores['test_score']
        score = np.mean(test_scores)
        scores[scoring_opt] = round(score, 3)

    return scores

def classification_function(X, y, model_name):
    classifier = models[model_name]
    classifier.fit(X, y)
    return classifier

def none_valid(classifier, X, y):
    pass

val_methods = {
    'None' : none_valid,
    'Holdout': holdout,
    'Repeated Holdout': repeated_holdout,
    'Stratified K-fold Cross Validation': StratKFoldCV,
    'Leave One Out Cross Validation': LeaveOneOutCV,
    'Repeated Cross Validation': KFoldCV,
    'Nested Cross Validation': RepeatedKFoldCV,
}

def optimized_classification_function(X, y, model_name, validation_method):
    parameter_space = models_param_space[model_name]
    # model =  models[model_name]

    # if validation_method ==