from modelling import models, models_param_space
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, cross_validate, StratifiedKFold, KFold, RepeatedKFold
from numpy import mean
import numpy as np
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
    scoring_opts = ['accuracy', 'f1_weighted', 'precision_weighted','recall_weighted',"roc_auc_ovr"]
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

def holdout(classifier, X, y, train_size = 0.8):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
    classifier.fit(X_train, y_train)
    return calc_score(classifier, X_test, y_test)

def repeated_holdout(classifier, X, y, num_of_repeats = 3, train_size = 0.8):
    results = []  

    for _ in range(num_of_repeats):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        classifier_cloned = base.clone(classifier)
        classifier_cloned.fit(X_train, y_train)
        results.append(calc_score(classifier, X_test, y_test))

    return np.mean(np.array(results),axis = 1)

def StratKFoldCV(classifier, X, y, kfold = 3):
    
    cross_validation = StratifiedKFold(n_splits=kfold)

    score_fpr = cross_validation(false_positive_rate, classifier, X, y)
    score_tpr = cross_validation(true_positive_rate, classifier, X, y)
    score_npv = cross_validation(n_predict_value, classifier, X, y)

    return cross_validate(classifier, X, y, cv=cross_validation, scoring=scoring_opts)['test_score'],\
            score_fpr, score_tpr, score_npv 

def LeaveOneOutCV(classifier, X, y):
    
    loo = LeaveOneOut()

    score_fpr = cross_validation(false_positive_rate, classifier, X, y)
    score_tpr = cross_validation(true_positive_rate, classifier, X, y)
    score_npv = cross_validation(n_predict_value, classifier, X, y)

    return cross_validate(classifier, X, y, cv=loo, scoring=scoring_opts)['test_score'],\
            score_fpr, score_tpr, score_npv 


def KFoldCV(classifier, X, y, kys = 3):
    
    loo = KFold(kys)

    score_fpr = cross_validation(false_positive_rate, classifier, X, y)
    score_tpr = cross_validation(true_positive_rate, classifier, X, y)
    score_npv = cross_validation(n_predict_value, classifier, X, y)

    return cross_validate(classifier, X, y, cv=loo, scoring=scoring_opts)['test_score'],\
            score_fpr, score_tpr, score_npv 

def RepeatedKFoldCV(classifier, X, y, kysagain=3):
    
    loo = RepeatedKFold(kysagain)

    score_fpr = cross_validation(false_positive_rate, classifier, X, y)
    score_tpr = cross_validation(true_positive_rate, classifier, X, y)
    score_npv = cross_validation(n_predict_value, classifier, X, y)

    return cross_validate(classifier, X, y, cv=loo, scoring=scoring_opts)['test_score'],\
            score_fpr, score_tpr, score_npv 

def classification_function(X, y, model_name):
    classifier = models[model_name]
    classifier.fit(X, y)
    return classifier

val_methods = {
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