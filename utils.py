import streamlit as st
import pandas as pd
import numpy as np
#from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
from sklearn.impute import SimpleImputer
#from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import RandomizedSearchCV
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.svm import SVC
#from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTENC
import io
import pandas as pd
#from sklearn.model_selection import RepeatedStratifiedKFold
import fontawesome as fa
#from lightgbm import LGBMClassifier
#import pyreadstat
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#import base64
import io
from imblearn.over_sampling import SMOTE
import json
#import plotly.figure_factory as ff
#from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import RandomizedSearchCV
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
#import plotly.graph_objs as go
#import os
#from plotly import tools
from collections import Counter
#import datetime
import fontawesome as fa
#import math
#from sklearn.model_selection import LeaveOneOut
#from sklearn.model_selection import ShuffleSplit
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import cross_val_predict
import sys
# import sklearn.neighbors._base
# sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
# from missingpy import MissForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from boruta import BorutaPy
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
##randomforest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import sklearn_json as skljson
import shap
import sklearn
import matplotlib
import matplotlib.pyplot as pl
import matplotlib.pylab as plt
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets, ensemble, model_selection
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import plot_confusion_matrix
from io import BytesIO
import sys
from sklearn.model_selection import RepeatedKFold

def has_nulls(df):
    nulls = df.isna().sum().sum()
    if nulls > 0:
        return True
    return False
    
def null_count(df):
    return df.isna().sum().sum()

def has_categ_columns(df):
    for col in df.columns:
        if df[col].dtype.name in ["category", "object"]:
            return True
    return False

def categ_columns(df):
    if(not has_categ_columns):
        return []
    categ_columns = []
    for col in df.columns:
        if df[col].dtype.name in ["category", "object"]:
            categ_columns.append(col)
    return categ_columns


def encode_categorical_columns(df):
    for col in df.columns:
        if df[col].dtype.name in ["category", "object"]:
            df[col] = df[col].astype("category")
            df[col] = df[col].cat.codes
    return df

def numerical_columns(df):
    numerical_columns = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_columns.append(col)
    return numerical_columns   




def check_for_outliers(df):
    df = encode_categorical_columns(df)
    df = df.dropna()
    
    numeric_columns = numerical_columns(df)
    if not numeric_columns:
        return []

    lof = LocalOutlierFactor()
    lof.fit(df[numeric_columns])

    yhat = lof.fit_predict(df[numeric_columns])
    outliers = np.where(yhat == -1)[0]

    return len(outliers)


def is_imbalance(y):
    # count the frequency of each class
    count = Counter(y)
    aa=list(count.values())
    deger = False
    for i in range(len(aa)-1):
        if aa[i] >= (3*aa[i+1]) or 3*aa[i] <= aa[i+1]:
            deger=True
    return deger


def impute_missing(df):
    # missing data
    cols = df.columns
    def cleanveri(veri):
        try:
            veri = float(veri)
        except:
            veri = veri
        if isinstance(veri, str):
            return np.nan
        elif isinstance(veri, float):
            return veri
    for i in df.columns:
        df[i] = df[i].apply(lambda x : cleanveri(x))

    #df = df.to_numpy()
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean' )
    imp_mean.fit(df)
    df = pd.DataFrame(imp_mean.transform(df))
    df.columns = cols
    return df

def drop_outliers(x):
    
    dff = encode_categorical_columns(x)
    dff = impute_missing(dff)

    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(dff)
    mask = yhat != -1 ##yine burada aykırı değer içeren satırları bulduk
    aykirideger= 0
    aykiridegerlist=[]
    for i in range(len(mask)):
        if mask[i] != True:
            aykirideger = aykirideger+1
            aykiridegerlist.append(i)
    x = x.drop(aykiridegerlist, axis=0)
    x = x.reset_index(drop=True)
    return x  

def attr(df, X, y, a):
    if a==0:
        model = LogisticRegression(solver='lbfgs')
        rfe = RFE(model)
        fit = rfe.fit(X, y)
        df=rfe.transform(X)
        new_columns = list(X.columns[rfe.support_])

    elif a==1:
        clf = ExtraTreesClassifier(n_estimators=50)
        fit = clf.fit(X, y)
        clf.feature_importances_
        model = SelectFromModel(clf, prefit=True)
        feature_idx = model.get_support()
        new_columns = list(X.columns[feature_idx])

    elif a==2:
        sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
        sel.fit(X, y)
        sel.get_support()
        feature_idx = sel.get_support()
        new_columns = list(X.columns[feature_idx])
        df = sel.transform(X)

    elif a==3:
        sel = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
        sel.fit(X, np.ravel(y,order='C'))
        feature_idx = sel.get_support()
        X=pd.DataFrame(X,  columns=X.columns)
        new_columns = list(X.columns[feature_idx])
        df = sel.transform(X)

    elif a==4:
        rfc = RandomForestClassifier(random_state=1, n_estimators=1000, max_depth=5)
        boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2, random_state=1)
        boruta_selector.fit(np.array(X), np.array(y))
        a=X.columns[boruta_selector.support_]
        new_columns=list(a)
        df = boruta_selector.transform(np.array(X))

    return new_columns

def transform_features(x, a):
    # data transformation function

    try:
        dff = encode_categorical_columns(x)
    except:
        dff =x
    # dff = kayipveriler(dff)
    if a==0:
        normalizer = preprocessing.Normalizer()
        list = []
        for i in dff.columns:
            threshold = 10
            if dff[i].nunique() < threshold:
                list.append(i)
        adf = dff.copy()
        normalizer = preprocessing.Normalizer().fit(adf)
        adf= normalizer.transform(adf)
        xcolumns = dff.columns.values
        adf = pd.DataFrame(adf)
        for i in range(len(xcolumns)):
            adf= adf.rename(columns={i:xcolumns[i]})
        for i in list:
            adf[i] = dff[i]
        dff = adf

    if a==1:
        std = MinMaxScaler()
        list = []
        for i in dff.columns:
            threshold = 10
            if dff[i].nunique() < threshold:
                list.append(i)



        adf = dff.copy()
        std = std.fit(adf)
        adf= std.transform(adf)
        xcolumns = dff.columns.values
        adf = pd.DataFrame(adf)
        for i in range(len(xcolumns)):
            adf= adf.rename(columns={i:xcolumns[i]})
        for i in list:
            adf[i] = dff[i]
        dff = adf

    if a==2:
        std = StandardScaler()
        list = []
        for i in dff.columns:
            threshold = 10
            if dff[i].nunique() < threshold:
                list.append(i)

        adf = dff.copy()
        std = std.fit(adf)
        adf= std.transform(adf)
        xcolumns = dff.columns.values
        adf = pd.DataFrame(adf)
        for i in range(len(xcolumns)):
            adf= adf.rename(columns={i:xcolumns[i]})
        for i in list:
            adf[i] = dff[i]
        dff = adf

    if a==3:
        std = RobustScaler()
        list = []
        for i in dff.columns:
            threshold = 10
            if dff[i].nunique() < threshold:
                list.append(i)

        adf = dff.copy()
        std = std.fit(adf)
        adf= std.transform(adf)
        xcolumns = dff.columns.values
        adf = pd.DataFrame(adf)
        for i in range(len(xcolumns)):
            adf= adf.rename(columns={i:xcolumns[i]})
        for i in list:
            adf[i] = dff[i]
        dff = adf


    return dff

#def missing_forest_impute(x):
#    imputer = MissForest()
#    x = imputer.fit_transform(x,cat_vars=None)
#    return x



def smote_function(df,X,y, a):

    X =  X.to_numpy()

    categorical_features = np.argwhere(np.array([len(set(X[:,x])) for x in range(X.shape[1])]) <= 9).flatten()
    ##normalde buradaki <=10 du, bu haliyle bazı sayısalları kategorik yapıyor veya cinsiyeti kategorik görmüyor vs, bağlanıp bakmamız gerek aslında

    if a==0:
        sm = SMOTE()
        X, y = sm.fit_resample(X, y)
    elif a==1:
        sm = SMOTETomek(sampling_strategy='not majority')
        X, y = sm.fit_resample(X, y)

    elif a==3 and len(categorical_features) !=0:
        sm = SMOTENC(categorical_features=categorical_features, sampling_strategy='not majority')
        X, y = sm.fit_resample(X, y)
    elif a==3 and len(categorical_features) ==0:
        sm = SMOTE()
        X, y = sm.fit_resample(X, y)


    return X,y