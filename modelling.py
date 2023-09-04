from pyexpat import model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_curve,auc
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from itertools import cycle
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold,cross_validate
from numpy import mean
from numpy import std
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import catboost as cb
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import streamlit as st


models = {
    'AdaBoost': AdaBoostClassifier(), 
    'CatBoost': cb.CatBoostClassifier(), 
    'Decision Tree': DecisionTreeClassifier(), 
    'Gaussian Naive Bayes': GaussianNB(), 
    'Gradient Boosting': GradientBoostingClassifier(), 
    'LightGBM': LGBMClassifier(objective = 'binary'), 
    'Logistic Regression': LogisticRegression(), 
    'Multilayer Perceptron (MLP)': RandomForestClassifier(), 
    'Random Forest': MLPClassifier(), 
    'Support Vector Machine': SVC(probability=True), 
    'XGBoost': XGBClassifier()
}

models_param_space = {
    'AdaBoost': [{'n_estimators': [10, 100,200, 300,500, 700, 900, 1000],
                 'learning_rate': [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]}],

    'CatBoost': [{'learning_rate': [0.0001, 0.001,0.03, 0.01, 0.1, 1.0], 
                  'subsample': [0.7,0.9, 1.0], 
                  'depth': [1,2, 3, 4, 5, 6, 7,8,9,10], 
                  'iterations': [50,100,200,300,400]}],

    'Decision Tree': [{'splitter' : ['best', 'random'],
                       'criterion' : ['gini', 'entropy'],
                       'max_features': ['log2', 'sqrt','auto'],
                       'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10][1:],
                       'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],}],

    'Gaussian Naive Bayes': [{'var_smoothing': np.logspace(0,-9, num=100)}],

    'Gradient Boosting': [{'n_estimators': [10, 100,200, 300,500, 700, 900, 1000],
                           'learning_rate': [0.0001, 0.001,0.03, 0.01, 0.1, 1.0],
                           'subsample': [0.5, 0.7,0.9, 1.0],
                           'max_depth': [1,2, 3, 4, 5, 6, 7,8,9,10],}],

    'LightGBM': [{'num_leaves': [31, 127,155],
                  "max_depth" : [1,2, 3, 4, 5, 6, 7,8,9,10],
                  'min_child_weight': [0.03,0.001,0.01,0.1],
                  'subsample' : [0.5, 0.7,0.9, 1.0],
                  "learning_rate" : [0.0001, 0.001,0.03, 0.01, 0.1, 1.0],
                  "n_estimators" : [10, 100,200, 300,500, 700, 900, 1000]}],
                   
    'Logistic Regression': [{'C' : (np.logspace(-4, 4, 20)),
                             'penalty' : ('l1', 'l2', 'elasticnet', 'none'),
                             'solver' : ('lbfgs','newton-cg','liblinear','sag','saga')}],

    'Multilayer Perceptron (MLP)': [{'criterion': ['gini', 'entropy'],
                                     'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                     'max_depth':  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                     'min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10][1:]}],

    'Random Forest': [{'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
                       'activation': ['tanh', 'relu', 'logistic'],
                       'solver': ['sgd', 'adam', 'lbfgs'],
                       'alpha': [0.0001, 0.05],
                       'learning_rate': ['constant','adaptive'],
                       'max_iter': [50, 100, 150,200,300]}],

    'Support Vector Machine': [{'C':(0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10,11,15,20),
                                'gamma':(0.0001,0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10)}],

    'XGBoost': [{'n_estimators': [10, 100,200, 300,500, 700, 900, 1000], 
                 'learning_rate': [0.0001, 0.001,0.03, 0.01, 0.1, 1.0], 
                 'subsample': [0.5, 0.7,0.9, 1.0], 
                 'max_depth': [1,2, 3, 4, 5, 6, 7,8,9,10]}]
}

val_methods = {
    'Holdout': None,
    'Repeated Holdout': None,
    'Stratified K-fold Cross Validation': None,
    'Leave One Out Cross Validation': None,
    'Repeated Cross Validation': None,
    'Nested Cross Validation': None,
}

@st.cache_data
def get_model(model_name):
    return models[model_name]

def get_model_param_space(model_name):
    return models_param_space[model_name]

# def validate(val_method, model, opt):

#     if val_method == 'Holdout':
#         return holdout(model, opt)
#     elif val_method == 'Repeated Holdout':
#         return repeated_holdout(model, opt)
#     elif val_method == 'Stratified K-fold Cross Validation':
#         return stratified_kfold_cv(model, opt)
#     elif val_method == 'Leave One Out Cross Validation':
#         return leave_one_out_cv(model, opt)
#     elif val_method == 'Repeated Cross Validation':
#         return repeated_cv(model, opt)
#     elif val_method == 'Nested Cross Validation':
#         return nested_cv(model, opt)
    

#def holdout(model, opt):
