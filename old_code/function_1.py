from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTENC
import io
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
import fontawesome as fa
from lightgbm import LGBMClassifier
import pyreadstat
import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash import dash_table

import base64
import io
from imblearn.over_sampling import SMOTE
import dash_daq as daq
from dash.dependencies import Input, Output, State
import json
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import plotly.graph_objs as go
import os
from plotly import tools
import dash_daq as daq
from collections import Counter
import datetime
import fontawesome as fa
import math
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
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
def dosyakaydet(content, filename):
    try:
        UPLOAD_DIRECTORY = "/var/www/html/xai/uploaded_files"
        dosyayolu = os.path.join(UPLOAD_DIRECTORY, filename)
        if not os.path.exists(UPLOAD_DIRECTORY):
            os.makedirs(UPLOAD_DIRECTORY)
        data = content.encode('utf-8').split(b";base64,")[1]
        with open(dosyayolu, "wb") as fp:
            fp.write(base64.decodebytes(data))

        return dosyayolu
    except Exception as e:
        print(e)
        return('a')

def dosyaoku(contents, filename):
    # read file function
    # path = save file
    dosyayolu = dosyakaydet(contents, filename)
    try:
        if 'csv' in filename:
            df = pd.read_csv(dosyayolu)
        elif 'xls' in filename:
            df = pd.read_excel(dosyayolu)
        elif 'sav' in filename:
            df= pd.read_spss(dosyayolu)
        adf = df.to_json(date_format='iso', orient='split')
        return adf
    except Exception as e:
        print(e)
        return html.Div([ html.Hr(),
            # wrong file type
            'HATALI DOSYA TÜRÜ'
        ])


def veriyazdir1(df):
    # data print1
    return html.Div([ dash_table.DataTable(
    id='datatable-interactivity',
    columns=[
        {"name": i, "id": i} for i in df.columns
    ],
    data=df.to_dict('records'),
    editable=True,
    column_selectable="multi",
    selected_columns=df.columns,
    page_action="native",
    page_current= 0,
    page_size= 5, ) ])


def veriyazdir(veri):
    # dataprint
    if veri is not None:
        dff=pd.read_json(veri , orient='split')
        return html.Div([ dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i} for i in dff.columns
        ],
        data=dff.to_dict('records'),
        editable=True,
        column_selectable="multi",
        selected_columns=dff.columns,
        page_action="native",
        #page_current= 0,
        #page_size= 5,
        style_table={
            'width': '100%',
                 'height': '400px',
            'overflowY': 'scroll',
            'overflowX': 'scroll',
            'textAlign': 'center',

        },style_header=
                                {
                                    'fontWeight': 'bold',
                                    'border': 'thin lightgrey solid',
                                    'backgroundColor': 'rgb(100, 100, 100)',
                                    'color': 'white',
                                    'textAlign': 'center',
                                },
                                style_cell={
                                    "font-family": "Bahnschrift",
                                    'textAlign': 'center',
                                    'width': '150px',
                                    'minWidth': '180px',
                                    'maxWidth': '180px',
                                    'whiteSpace': 'no-wrap',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',

                                },
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': 'rgb(248, 248, 248)'
                                    },
                                    {
                                        'if': {'column_id': 'country'},
                                        'backgroundColor': '#FF8F00',
                                        'color': 'black',
                                        'fontWeight': 'bold',
                                        'textAlign': 'center'
                                    }],
    ) ])


# html.Hr(),
    # html.H6('Dosya Başarılı Şekilde Okundu Veri Seçme işlemine Geçebilirsiniz.')
    # ])



####################### sınıflandırma alt programları


def kayipveriler(df):
    # missing data
    def cleanveri(veri):
        try:
            veri = float(veri)
        except:
            veri=veri
        if isinstance(veri, str):
            return np.nan
        elif isinstance(veri, float):
            return veri
    for i in df.columns:
        df[i] = df[i].apply(lambda x : cleanveri(x))

    df = df.to_numpy()
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean' )
    imp_mean.fit(df)
    df = imp_mean.transform(df)

    return df


def kategorikveriler(df):
    # categorical data in R / spss format where NaN is -1
    def is_categoric(x):
        is_str = True
        for i in range(len(x)):
            try:
                float(x[i])
                is_str = False
            except:
                str(x[i])

        return is_str


    for i in df.columns:
        if is_categoric(df[i]):
            df[i]=pd.Categorical(df[i]).codes

    return df

from sklearn import preprocessing
def stringtofloat(df):
    # label encoder
    def is_str1(x):
        is_str = True
        for i in range(len(x)):
            try:
                float(x[i])
                is_str = False
            except:
                str(x[i])

        return is_str

    le = preprocessing.LabelEncoder()
    for i in df.columns:
        if is_str1(df[i]):
            le.fit(df[i])
            df[i]=le.transform(df[i])
            # df[i]=pd.Categorical(df[i]).codes

    return df









def is_kayipveri(x):
    # is missing data
    a = x.isnull().sum()
    a = a.to_dict()
    # sums missing values
    toplamkayipdeger = 0
    for i in a.values():
        toplamkayipdeger = toplamkayipdeger+i
    return toplamkayipdeger




def is_aykiriveri(x):
    # is outlier data
    # categorical data
    df = kategorikveriler(x)
    # missing data
    df = kayipveriler(df)

    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(df)
    mask = yhat != -1
    aykirideger= 0
    for i in range(len(mask)):
        if mask[i] != True:
            aykirideger = aykirideger+1

    return aykirideger



def kayipveritamamla(x):
    # missing data
    # total missing values = # is missing data
    toplamkayipdeger = is_kayipveri(x)

    if toplamkayipdeger != 0 :
        imputer = MissForest()
        x = imputer.fit_transform(x,cat_vars=None)

    return x

from sklearn.preprocessing import OneHotEncoder


def categoric1(df):
    category_features = []
    threshold = 10
    for each in df.columns:
        #df[each] = round(df[each],4)
        if df[each].nunique() < threshold:
            category_features.append(each)

    for each in category_features:
        df[each] = df[each].astype('category')
    #print(df.columns)
    return df , category_features




def onehotfonk(x , onehot_liste):
    xcopy= x.copy()
    for i in onehot_liste:
        if i in list(x.columns):
            encoder=OneHotEncoder(handle_unknown="ignore", sparse=False)
            a = pd.DataFrame(encoder.fit_transform(x[[i]]))
            a.columns = encoder.get_feature_names([str(i)])
            xcopy.drop(columns=[i],  inplace=True)
            xcopy= pd.concat([xcopy, a ], axis=1)
    return xcopy



def aykiriveritamamla(x):
    # is outlier data

    #df ilk gelen orijinal
    #dff geçici dataframe
    # categorical data
    dff = kategorikveriler(x)
    # missing data
    dff = kayipveriler(dff)
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


from sklearn.preprocessing import MinMaxScaler


def veridonusumfonk(x, a):
    # data transformation function

    try:
        dff = kategorikveriler(x)
    except:
        dff =x
    # dff = kayipveriler(dff)
    if a==str(0):
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

    if a==str(1):
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

    if a==str(2):
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

    if a==str(3):
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



def ozelliksecimfonk(df,X,y, a):
    # feature selection function
    if a==str(0):
        model = LogisticRegression(solver='lbfgs')
        rfe = RFE(model)
        fit = rfe.fit(X, y)
        df=rfe.transform(X)


        new_columns = list(X.columns[rfe.support_])

    elif a==str(1):
        clf = ExtraTreesClassifier(n_estimators=50)
        fit = clf.fit(X, y)
        clf.feature_importances_
        model = SelectFromModel(clf, prefit=True)
        # print(X.columns[clf.support_])
        feature_idx = model.get_support()
        new_columns = list(X.columns[feature_idx])
        # print(feature_name)
        # df = model.transform(X)


    elif a==str(2):
        sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
        sel.fit(X, y)
        sel.get_support()

        feature_idx = sel.get_support()
        new_columns = list(X.columns[feature_idx])

        df = sel.transform(X)

    elif a==str(3):
        sel = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
        sel.fit(X, np.ravel(y,order='C'))
        feature_idx = sel.get_support()
        X=pd.DataFrame(X,  columns=X.columns)
        new_columns = list(X.columns[feature_idx])

        df = sel.transform(X)

    elif a==str(4):
        rfc = RandomForestClassifier(random_state=1, n_estimators=1000, max_depth=5)
        boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2, random_state=1)
        boruta_selector.fit(np.array(X), np.array(y))
        #selected_rf_features = pd.DataFrame({'Feature':list(X.columns),
                                      # 'Ranking':boruta_selector.ranking_})
        #print(selected_rf_features)
        a=X.columns[boruta_selector.support_]
        new_columns=list(a)

        df = boruta_selector.transform(np.array(X))

    else:
        new_columns=list(df.columns)
    return new_columns



def smotefonk(df,X,y, a):

    X =  X.to_numpy()

    categorical_features = np.argwhere(np.array([len(set(X[:,x])) for x in range(X.shape[1])]) <= 9).flatten()
    ##normalde buradaki <=10 du, bu haliyle bazı sayısalları kategorik yapıyor veya cinsiyeti kategorik görmüyor vs, bağlanıp bakmamız gerek aslında

    if a==str(0):
        sm = SMOTE()
        X, y = sm.fit_resample(X, y)
    elif a==str(1):
        sm = SMOTETomek(sampling_strategy='not majority')
        X, y = sm.fit_resample(X, y)

    elif a==str(3) and len(categorical_features) !=0:
        sm = SMOTENC(categorical_features=categorical_features, sampling_strategy='not majority')
        X, y = sm.fit_resample(X, y)
    elif a==str(3) and len(categorical_features) ==0:
        sm = SMOTE( )
        X, y = sm.fit_resample(X, y)


    return X,y









def siniflandirma_1(xtrain, ytrain, xtest, ytest, model_ismi):
    # classification function
    model_isimleri=['SVM','LR','RF','DT','LGBM','GNB','ADA','GBT','XGB']

    if model_ismi==model_isimleri[0]:
        classifier = SVC(probability=True)
    elif model_ismi==model_isimleri[1]:
        classifier = LogisticRegression()
    elif model_ismi==model_isimleri[2]:
        classifier = RandomForestClassifier()
    elif model_ismi==model_isimleri[3]:
        classifier = DecisionTreeClassifier()
    elif model_ismi==model_isimleri[4]:
        classifier = lgb.LGBMClassifier()
    elif model_ismi==model_isimleri[5]:
        classifier = GaussianNB()
    elif model_ismi==model_isimleri[6]:
        classifier = AdaBoostClassifier()
    elif model_ismi==model_isimleri[7]:
        classifier = GradientBoostingClassifier(0)
    else:
        classifier = SVC(probability=True)

    classifier.fit(xtrain, ytrain)
    score = accuracy_score(ytest, classifier.predict(xtest))
    score = round(score,3)
    return score

def siniflandirma(x, y , model_ismi):
    # classification
    model_isimleri=['SVM','LR','RF','DT','LGBM','GNB','ADA','XGB']
    # categoricaldata
    X = kategorikveriler(x)
    # missingdata
    X= kayipveriler(X)

    try:
        Y=y.to_frame()
    except:
        Y=y
    Y=kategorikveriler(Y)
    Y=kayipveriler(Y)
    Y = np.ravel(Y)
    normalizasyon = StandardScaler()
    X = normalizasyon.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

    if model_ismi==model_isimleri[0]:
        classifier = SVC(probability=True)
    elif model_ismi==model_isimleri[1]:
        classifier = LogisticRegression()
    elif model_ismi==model_isimleri[2]:
        classifier = RandomForestClassifier()
    elif model_ismi==model_isimleri[3]:
        classifier = DecisionTreeClassifier()
    elif model_ismi==model_isimleri[4]:
        classifier = lgb.LGBMClassifier()
    elif model_ismi==model_isimleri[5]:
        classifier = GaussianNB()
    elif model_ismi==model_isimleri[6]:
        classifier = AdaBoostClassifier()
    elif model_ismi==model_isimleri[7]:
        classifier = GradientBoostingClassifier()
    else:
        classifier = SVC(probability=True)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)


    return classifier , X, Y


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




############################################################################################################
##optimizasyonlu sınıflandırma

# optimized classification

def siniflandirma_2(xtrain, ytrain, xtest, ytest,kfold, model_ismi):
    model_isimleri=['SVM','LR','RF','DT','LGBM','GNB','ADA','GBT','CB','XGB', 'MLP']

    if model_ismi==model_isimleri[0]:

        model = SVC(probability=True)
        space_svm = [{'C':(0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10,11,15,20),'gamma':(0.0001,0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10)}]
        params = {"learning_rate" : [0.03,0.001,0.01,0.1,0.2,0.3],
                  "n_estimators" : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                  "max_depth" : [2, 3, 4, 5, 6, 7, 8, 9, 10]}
        cv = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=1, random_state=1)
        search = RandomizedSearchCV(model, space_svm,  scoring='accuracy',  cv=cv)


    elif model_ismi==model_isimleri[1]:

        model = LogisticRegression()
        param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        space_lr = [{'C' : (np.logspace(-4, 4, 20)),'penalty' : ('l1', 'l2', 'elasticnet', 'none'),'solver' : ('lbfgs','newton-cg','liblinear','sag','saga')}]
        cv = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=1, random_state=1)
        search = RandomizedSearchCV(model, space_lr,  scoring='accuracy',  cv=cv)

    elif model_ismi==model_isimleri[2]:
        print(model_ismi)
        model = RandomForestClassifier()
        param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        space_rf = [{'criterion': ['gini', 'entropy'],'min_samples_leaf': param_range,'max_depth':  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           'min_samples_split': param_range[1:]}]
        cv = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=1, random_state=1)
        search = RandomizedSearchCV(model, space_rf,  scoring='accuracy',  cv=cv)

    elif model_ismi==model_isimleri[3]:
        print(model_ismi)
        model = DecisionTreeClassifier()
        param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        space_dt =[{'splitter' : ['best', 'random'],
                          'criterion' : ['gini', 'entropy'],
                          'max_features': ['log2', 'sqrt','auto'],
                          'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          'min_samples_split': param_range[1:],
                          'min_samples_leaf': param_range,

                          }]
        cv = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=1, random_state=1)
        search = RandomizedSearchCV(model, space_dt,  scoring='accuracy',  cv=cv)

    elif model_ismi==model_isimleri[4]:
        print(model_ismi)
        model = LGBMClassifier()
        space_lgbm =[{
            'num_leaves': [31, 127,155],
            "max_depth" : [1,2, 3, 4, 5, 6, 7,8,9,10],
            
           'min_child_weight': [0.03,0.001,0.01,0.1],
             'subsample' : [0.5, 0.7,0.9, 1.0],
            "learning_rate" : [0.0001, 0.001,0.03, 0.01, 0.1, 1.0],"n_estimators" : [10, 100,200, 300,500, 700, 900, 1000]
            }]
        cv = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=1, random_state=1)
        search = RandomizedSearchCV(model, space_lgbm,  scoring='accuracy',  cv=cv)

    elif model_ismi==model_isimleri[5]:
        print(model_ismi)
        model = GaussianNB()
        space_gnb =[{'var_smoothing': np.logspace(0,-9, num=100)}]
        cv = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=1, random_state=1)
        search = RandomizedSearchCV(model, space_gnb,  scoring='accuracy',  cv=cv)

    elif model_ismi==model_isimleri[6]:
        print(model_ismi)
        model = AdaBoostClassifier()
        space_ada = dict()
        space_ada['n_estimators'] = [10, 100,200, 300,500, 700, 900, 1000]
        space_ada['learning_rate'] = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        
        cv = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=1, random_state=1)
        search = RandomizedSearchCV(model, space_ada,  scoring='accuracy',  cv=cv)

    elif model_ismi==model_isimleri[7]:
        print(model_ismi)
        model = GradientBoostingClassifier()
        n_estimators = [10, 100,200, 300,500, 700, 900, 1000]
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [0.5, 0.7,0.9, 1.0]
        max_depth = [1,2, 3, 4, 5, 6, 7,8,9,10]
        space_gb = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        cv = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=1, random_state=1)
        search = RandomizedSearchCV(model, space_gb,  scoring='accuracy',  cv=cv)
    elif model_ismi==model_isimleri[8]:
        print(model_ismi)
        model = cb.CatBoostClassifier()
        
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [0.7,0.9, 1.0]
        depth = [1,2, 3, 4, 5, 6, 7,8,9,10]
        iterations=[50,100,200,300,400]
        space_gb = dict(learning_rate=learning_rate,  subsample=subsample, depth=depth,iterations=iterations)
        cv = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=3, random_state=1)
        search = RandomizedSearchCV(model, space_gb,  scoring='accuracy',  cv=cv)
    elif model_ismi==model_isimleri[9]:
        print(model_ismi)
        model = XGBClassifier()
        n_estimators = [10, 100,200, 300,500, 700, 900, 1000]
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [0.5, 0.7,0.9, 1.0]
        max_depth = [1,2, 3, 4, 5, 6, 7,8,9,10]
        space_gb = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        cv = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=1, random_state=1)
        search = RandomizedSearchCV(model, space_gb,  scoring='accuracy',  cv=cv)
    elif model_ismi==model_isimleri[10]:
        print(model_ismi)
        model = MLPClassifier()
        parameter_space = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],'max_iter': [50, 100, 150,200,300]}

        cv = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=1, random_state=1)
        search = RandomizedSearchCV(model, parameter_space,  scoring='accuracy',  cv=cv)
    else:
        print(model_ismi)
        model = SVC(probability=True)
        space_svm = [{'C':(0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10,11,15,20),'gamma':(0.0001,0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10)}]
        params = {"learning_rate" : [0.03,0.001,0.01,0.1,0.2,0.3],
                  "n_estimators" : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                  "max_depth" : [2, 3, 4, 5, 6, 7, 8, 9, 10]}
        cv = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=1, random_state=1)
        search = RandomizedSearchCV(model, space_svm,  scoring='accuracy',  cv=cv)

    classifier = search.fit(xtrain, ytrain)
    yhat=classifier.predict(xtest)
    
    n_classes= ytest.explode().nunique()
    if n_classes==2:
        score = accuracy_score(ytest, classifier.predict(xtest))
        score = round((score),3)
        f1= f1_score(ytest, classifier.predict(xtest))
        f1 = round((f1),3)
        preci=precision_score(ytest, classifier.predict(xtest))
        preci = round((preci),3)
        recal=recall_score(ytest, classifier.predict(xtest))
        recal = round((recal),3)
        classifier_probs = classifier.predict_proba(xtest)
        lr_probs= classifier_probs[:,1]
        roc_auc_table = roc_auc_score(ytest, lr_probs)
        lr_fpr, lr_tpr, _ = roc_curve(ytest, lr_probs)
        roc=roc_auc_table
        roc = round((roc),3)
        cm = confusion_matrix(ytest, yhat)
        TP, FP, FN, TN = (cm[0][0], cm[0][1], cm[1][0], cm[1][1])
        confusionmat = {'TP':TP, 'FP':FP, 'FN':FN,'TN':TN}
        cmlist=list(confusionmat.values())
        FPR= (FP)/(FP+TN)
        FPR = round((FPR),3)
        TPR = (TP)/(TP+FN)
        TPR = round((TPR),3)
        NPV= (TN)/(TN+FN)
        NPV = round((NPV),3)
    else:
        score = accuracy_score(ytest, classifier.predict(xtest))
        score = round((score),3)
        f1= f1_score(ytest, classifier.predict(xtest),average='micro')
        f1 = round((f1),3)
        preci=precision_score(ytest, classifier.predict(xtest),average='micro')
        preci = round((preci),3)
        recal=recall_score(ytest, classifier.predict(xtest),average='micro')
        recal = round((recal),3)
        classifier_probs = classifier.predict_proba(xtest)
        roc=roc_auc_table = roc_auc_score(ytest, classifier_probs, multi_class="ovr")
        roc = round((roc),3)
        cm = confusion_matrix(ytest, yhat)
        TP, FP, FN, TN = (cm[0][0], cm[0][1], cm[1][0], cm[1][1])
        confusionmat = {'TP':TP, 'FP':FP, 'FN':FN,'TN':TN}
        cmlist=list(confusionmat.values())
        FPR= (FP)/(FP+TN)
        FPR = round((FPR),3)
        TPR = (TP)/(TP+FN)
        TPR = round((TPR),3)
        NPV= (TN)/(TN+FN)
        NPV = round((NPV),3)


    return classifier, score, f1, preci, recal,roc, FPR,TPR, NPV





##############################################################################
##optimizasyonsuz sınıflandırma

# unoptimized classification

from sklearn.model_selection import cross_val_predict,StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import catboost as cb
def siniflandirma_3(xtrain, ytrain, xtest, ytest, model_ismi):
    model_isimleri=['SVM','LR','RF','DT','LGBM','GNB','ADA','GBT','CB','XGB', 'MLP']

    if model_ismi==model_isimleri[0]:
        classifier = SVC(probability=True)
    elif model_ismi==model_isimleri[1]:
        classifier = LogisticRegression()
    elif model_ismi==model_isimleri[2]:
        classifier = RandomForestClassifier()
    elif model_ismi==model_isimleri[3]:
        classifier = DecisionTreeClassifier()
    elif model_ismi==model_isimleri[4]:
        classifier = LGBMClassifier()
    elif model_ismi==model_isimleri[5]:
        classifier = GaussianNB()
    elif model_ismi==model_isimleri[6]:
        classifier = AdaBoostClassifier()
    elif model_ismi==model_isimleri[7]:
        classifier = GradientBoostingClassifier()
    elif model_ismi==model_isimleri[8]:
        classifier=cb.CatBoostClassifier(one_hot_max_size=10)
    elif model_ismi==model_isimleri[9]:
        classifier = XGBClassifier()
    elif model_ismi==model_isimleri[10]:
        classifier = MLPClassifier()


    else:
        classifier = SVC(probability=True)

    class_fit=classifier.fit(xtrain, ytrain)

    yhat=class_fit.predict(xtest)
    n_classes= ytest.explode().nunique()
    if n_classes==2:
        score = accuracy_score(ytest, classifier.predict(xtest))
        score = round((score),3)
        f1= f1_score(ytest, classifier.predict(xtest))
        f1 = round((f1),3)
        preci=precision_score(ytest, classifier.predict(xtest))
        preci = round((preci),3)
        recal=recall_score(ytest, classifier.predict(xtest))
        recal = round((recal),3)
        classifier_probs = classifier.predict_proba(xtest)
        lr_probs= classifier_probs[:,1]
        roc_auc_table = roc_auc_score(ytest, lr_probs)
        
        roc=roc_auc_table
        roc = round((roc),3)
        cm = confusion_matrix(ytest, yhat)
        TP, FP, FN, TN = (cm[0][0], cm[0][1], cm[1][0], cm[1][1])
        confusionmat = {'TP':TP, 'FP':FP, 'FN':FN,'TN':TN}
        cmlist=list(confusionmat.values())
        FPR= (FP)/(FP+TN)
        FPR = round((FPR),3)
        TPR = (TP)/(TP+FN)
        TPR = round((TPR),3)
        NPV= (TN)/(TN+FN)
        NPV = round((NPV),3)
    else:
        score = accuracy_score(ytest, classifier.predict(xtest))
        score = round((score),3)
        f1= f1_score(ytest, classifier.predict(xtest),average='micro')
        f1 = round((f1),3)
        preci=precision_score(ytest, classifier.predict(xtest),average='micro')
        preci = round((preci),3)
        recal=recall_score(ytest, classifier.predict(xtest),average='micro')
        recal = round((recal),3)
        classifier_probs = classifier.predict_proba(xtest)
        roc=roc_auc_table = roc_auc_score(ytest, classifier_probs, multi_class="ovr")
        roc = round((roc),3)
        cm = confusion_matrix(ytest, yhat)
        TP, FP, FN, TN = (cm[0][0], cm[0][1], cm[1][0], cm[1][1])
        confusionmat = {'TP':TP, 'FP':FP, 'FN':FN,'TN':TN}
        cmlist=list(confusionmat.values())
        FPR= (FP)/(FP+TN)
        FPR = round((FPR),3)
        TPR = (TP)/(TP+FN)
        TPR = round((TPR),3)
        NPV= (TN)/(TN+FN)
        NPV = round((NPV),3)

    return classifier, score, f1, preci, recal,roc, FPR,TPR, NPV
from sklearn.metrics import make_scorer

def false_positive_rate(y_true, y_pred):
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    return fp / (fp + tn)

def true_pozitive_rate(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return (tp)/(tp+fn)

def n_predict_value(y_true, y_pred):

    tn = ((y_pred == 0) & (y_true == 0)).sum()
    # false negative
    fn = np.sum((y_pred == 0) & (y_true == 1))
    # false positive rate
    return (tn)/(tn+fn)



def repeatedholdout(X, y,nsplits, splitsize, model_ismi):
    model_isimleri=['SVM','LR','RF','DT','LGBM','GNB','ADA','GBT','CB','XGB', 'MLP']
    if model_ismi==model_isimleri[0]:
        classifier = SVC(probability=True )
    elif model_ismi==model_isimleri[1]:
        classifier = LogisticRegression(max_iter=3000)
    elif model_ismi==model_isimleri[2]:
        classifier = RandomForestClassifier()
    elif model_ismi==model_isimleri[3]:
        classifier = DecisionTreeClassifier()
    elif model_ismi==model_isimleri[4]:
        classifier = LGBMClassifier()
    elif model_ismi==model_isimleri[5]:
        classifier = GaussianNB()
    elif model_ismi==model_isimleri[6]:
        classifier = AdaBoostClassifier()
    elif model_ismi==model_isimleri[7]:
        classifier = GradientBoostingClassifier( )
    elif model_ismi==model_isimleri[8]:
        classifier=cb.CatBoostClassifier(one_hot_max_size=31)
    elif model_ismi==model_isimleri[9]:
        classifier=XGBClassifier()
    elif model_ismi==model_isimleri[10]:
        classifier = MLPClassifier()
    else:
        classifier = SVC( probability=True)
    class_fit=classifier.fit(X, y)

    ssplit=StratifiedShuffleSplit(n_splits=nsplits,test_size = splitsize)
    score = cross_val_score(classifier, X, y, scoring='accuracy',cv=ssplit)
    score= (mean(score))
    score = round((score),3)

    f1 = cross_val_score(classifier, X, y, scoring='f1_weighted',cv=ssplit)
    f1= mean(f1)
    f1 = round((f1),3)
    preci = cross_val_score(classifier, X, y, scoring='precision_weighted',cv=ssplit)
    preci= mean(preci)
    preci = round((preci),3)
    recal = cross_val_score(classifier, X, y, scoring='recall_weighted',cv=ssplit)
    recal= mean(recal)
    recal = round((recal),3)
    roc=cross_val_score(classifier,X, y, scoring="roc_auc_ovr",cv=ssplit)
    print(roc)

    roc=mean(roc)
    roc = round((roc),3)
    n_classes= y.explode().nunique()


    scorefpr=make_scorer(false_positive_rate)
    FPR = cross_validate(classifier, X, y,cv=ssplit, scoring=scorefpr)
    FPR=FPR['test_score']
    print(FPR)
    print("fprliste")
    FPR=np.mean(FPR)
    FPR = round((FPR),3)
    scoretpr=make_scorer(true_pozitive_rate)
    TPR = cross_validate(classifier, X, y, cv=ssplit,scoring=scoretpr)
    TPR=TPR['test_score']
    print(TPR)
    print("tprliste")
    TPR=np.mean(TPR)
    TPR = round((TPR),3)
    scorenpv=make_scorer(n_predict_value)
    NPV = cross_validate(classifier, X, y, cv=ssplit,scoring=scorenpv)
    NPV=NPV['test_score']
    print(NPV)
    print("npvliste")
    NPV=np.mean(NPV)
    NPV = round((NPV),3)

    return classifier, score, f1,preci,recal,roc, FPR,TPR, NPV


def stratifiedcrossvalidation(X, y,kfold, model_ismi):
    model_isimleri=['SVM','LR','RF','DT','LGBM','GNB','ADA','GBT','CB','XGB', 'MLP']
    if model_ismi==model_isimleri[0]:
        classifier = SVC(probability=True)
    elif model_ismi==model_isimleri[1]:
        classifier = LogisticRegression(max_iter=3000)
    elif model_ismi==model_isimleri[2]:
        classifier = RandomForestClassifier()
    elif model_ismi==model_isimleri[3]:
        classifier = DecisionTreeClassifier()
    elif model_ismi==model_isimleri[4]:
        classifier = LGBMClassifier()
    elif model_ismi==model_isimleri[5]:
        classifier = GaussianNB()
    elif model_ismi==model_isimleri[6]:
        classifier = AdaBoostClassifier()
    elif model_ismi==model_isimleri[7]:
        classifier = GradientBoostingClassifier()
    elif model_ismi==model_isimleri[8]:
        classifier=cb.CatBoostClassifier(one_hot_max_size=31)
    elif model_ismi==model_isimleri[9]:
        classifier=XGBClassifier()
    elif model_ismi==model_isimleri[10]:
        classifier = MLPClassifier()
    else:
        classifier = SVC(probability=True)
    class_fit=classifier.fit(X, y)

    skfold=StratifiedKFold(n_splits=kfold)
    score = cross_val_score(classifier, X, y, scoring='accuracy',cv=skfold)
    score=mean(score)
    score = round((score),3)
    f1 = cross_val_score(classifier, X, y, scoring='f1_macro',cv=skfold)
    f1=mean(f1)
    f1 = round((f1),3)
    preci = cross_val_score(classifier, X, y, scoring='precision_macro',cv=skfold)
    preci=mean(preci)
    preci = round((preci),3)
    recal = cross_val_score(classifier, X, y, scoring='recall_macro',cv=skfold)
    recal=mean(recal)
    recal = round((recal),3)
    roc=cross_val_score(classifier,X, y, scoring="roc_auc",cv=skfold)
    roc=mean(roc)
    roc = round((roc),3)
    y_pred = classifier.predict(X)
    cm = confusion_matrix(y, y_pred)
    scorefpr=make_scorer(false_positive_rate)
    FPR = cross_validate(classifier, X, y, scoring=scorefpr,cv=skfold)
    FPR=FPR['test_score']
    FPR=np.mean(FPR)
    FPR = round((FPR),3)
    scoretpr=make_scorer(true_pozitive_rate)
    TPR = cross_validate(classifier, X, y, scoring=scoretpr,cv=skfold)
    TPR=TPR['test_score']
    TPR=np.mean(TPR)
    TPR = round((TPR),3)
    scorenpv=make_scorer(n_predict_value)
    NPV = cross_validate(classifier, X, y, scoring=scorenpv,cv=skfold)
    NPV=NPV['test_score']
    NPV=np.mean(NPV)
    NPV = round((NPV),3)
    return classifier, score, f1,preci,recal,roc, FPR,TPR, NPV






def repeatedcrossvalidation(X, y,kfold,nrepeat, model_ismi):
    model_isimleri=['SVM','LR','RF','DT','LGBM','GNB','ADA','GBT','CB','XGB', 'MLP']
    if model_ismi==model_isimleri[0]:
        classifier = SVC(probability=True)
    elif model_ismi==model_isimleri[1]:
        classifier = LogisticRegression( max_iter=3000)
    elif model_ismi==model_isimleri[2]:
        classifier = RandomForestClassifier()
    elif model_ismi==model_isimleri[3]:
        classifier = DecisionTreeClassifier()
    elif model_ismi==model_isimleri[4]:
        classifier = LGBMClassifier()
    elif model_ismi==model_isimleri[5]:
        classifier = GaussianNB()
    elif model_ismi==model_isimleri[6]:
        classifier = AdaBoostClassifier()
    elif model_ismi==model_isimleri[7]:
        classifier = GradientBoostingClassifier()
    elif model_ismi==model_isimleri[8]:
        classifier=cb.CatBoostClassifier(one_hot_max_size=31)
    elif model_ismi==model_isimleri[9]:
        classifier=XGBClassifier()
    elif model_ismi==model_isimleri[10]:
        classifier = MLPClassifier()
    else:
        classifier = SVC(kernel='linear', probability=True)
    class_fit=classifier.fit(X, y)
    classifier_probs = classifier.predict_proba(xtest)
    skfold = RepeatedKFold(n_splits=kfold, n_repeats=nrepeat)
    score = cross_val_score(classifier, X, y, scoring='accuracy',cv=skfold)
    score=mean(score)
    score = round((score),3)
    f1 = cross_val_score(classifier, X, y, scoring='f1_macro',cv=skfold)
    f1=mean(f1)
    f1 = round((f1),3)
    preci = cross_val_score(classifier, X, y, scoring='precision_macro',cv=skfold)
    preci=mean(preci)
    preci = round((preci),3)
    recal = cross_val_score(classifier, X, y, scoring='recall_macro',cv=skfold)
    recal=mean(recal)
    recal = round((recal),3)
    roc=cross_val_score(classifier,X, y, scoring="roc_auc",cv=skfold)
    roc=mean(roc)
    roc = round((roc),3)
    y_pred = classifier.predict(X)
    cm = confusion_matrix(y, y_pred)
    scorefpr=make_scorer(false_positive_rate)
    FPR = cross_validate(classifier, X, y, scoring=scorefpr,cv=skfold)
    FPR=FPR['test_score']
    FPR=np.mean(FPR)
    FPR = round((FPR),3)
    scoretpr=make_scorer(true_pozitive_rate)
    TPR = cross_validate(classifier, X, y, scoring=scoretpr,cv=skfold)
    TPR=TPR['test_score']
    TPR=np.mean(TPR)
    TPR = round((TPR),3)
    scorenpv=make_scorer(n_predict_value)
    NPV = cross_validate(classifier, X, y, scoring=scorenpv,cv=skfold)
    NPV=NPV['test_score']
    NPV=np.mean(NPV)
    NPV = round((NPV),3)
    return classifier, score, f1,preci,recal,roc, FPR,TPR, NPV


def leaveoneout(X, y, model_ismi):
    model_isimleri=['SVM','LR','RF','DT','LGBM','GNB','ADA','GBT','CB','XGB', 'MLP']
    if model_ismi==model_isimleri[0]:
        classifier = SVC(probability=True)
    elif model_ismi==model_isimleri[1]:
        classifier = LogisticRegression( max_iter=3000)
    elif model_ismi==model_isimleri[2]:
        classifier = RandomForestClassifier()
    elif model_ismi==model_isimleri[3]:
        classifier = DecisionTreeClassifier()
    elif model_ismi==model_isimleri[4]:
        classifier = LGBMClassifier()
    elif model_ismi==model_isimleri[5]:
        classifier = GaussianNB()
    elif model_ismi==model_isimleri[6]:
        classifier = AdaBoostClassifier()
    elif model_ismi==model_isimleri[7]:
        classifier = GradientBoostingClassifier()
    elif model_ismi==model_isimleri[8]:
        classifier=cb.CatBoostClassifier(one_hot_max_size=31)
    elif model_ismi==model_isimleri[9]:
        classifier=XGBClassifier()
    elif model_ismi==model_isimleri[10]:
        classifier = MLPClassifier()
    else:
        classifier = SVC(probability=True)
    class_fit=classifier.fit(X, y)

    leave_validation=LeaveOneOut()
    score = cross_val_score(classifier, X, y, scoring='accuracy',cv=leave_validation)
    score=mean(score)
    score = round((score),3)
    f1 = cross_val_score(classifier, X, y, scoring='f1_macro',cv=leave_validation)
    f1=mean(f1)
    f1 = round((f1),3)
    preci = cross_val_score(classifier, X, y, scoring='precision_macro',cv=leave_validation)
    preci=mean(preci)
    preci = round((preci),3)
    recal = cross_val_score(classifier, X, y, scoring='recall_macro',cv=leave_validation)
    recal=mean(recal)
    recal = round((recal),3)
    roc=cross_val_score(classifier,X, y, scoring="roc_auc",cv=leave_validation)
    roc=mean(roc)
    roc = round((roc),3)
    y_pred = classifier.predict(X)
    cm = confusion_matrix(y, y_pred)
    scorefpr=make_scorer(false_positive_rate)
    FPR = cross_validate(classifier, X, y, scoring=scorefpr,cv=leave_validation)
    FPR=FPR['test_score']
    FPR=np.mean(FPR)
    FPR = round((FPR),3)
    scoretpr=make_scorer(true_pozitive_rate)
    TPR = cross_validate(classifier, X, y, scoring=scoretpr,cv=leave_validation)
    TPR=TPR['test_score']
    TPR=np.mean(TPR)
    TPR = round((TPR),3)
    scorenpv=make_scorer(n_predict_value)
    NPV = cross_validate(classifier, X, y, scoring=scorenpv,cv=leave_validation)
    NPV=NPV['test_score']
    NPV=np.mean(NPV)
    NPV = round((NPV),3)
    return classifier, score, f1,preci,recal,roc, FPR,TPR, NPV




def nestedcross(X, y,innerkfol,outerkfold,model_ismi):
    model_isimleri=['SVM','LR','RF','DT','LGBM','GNB','ADA','GBT','CB','XGB', 'MLP']

    if model_ismi==model_isimleri[0]:
        print(model_ismi)
        classifier = SVC(probability=True)
        grid_params_svm = [{'C':(0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10,11,15,20),'gamma':(0.0001,0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10)}]
        params = {"learning_rate" : [0.03,0.001,0.01,0.1,0.2,0.3],
                  "n_estimators" : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                  "max_depth" : [2, 3, 4, 5, 6, 7, 8, 9, 10]}
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_svm, n_iter=250, scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = KFold(n_splits=outerkfold, shuffle=True, random_state=1)

    elif model_ismi==model_isimleri[1]:
        print(model_ismi)
        classifier = LogisticRegression(max_iter=3000)
        param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        grid_params_lr = [{'C' : (np.logspace(-4, 4, 20)),'penalty' : ('l1', 'l2', 'elasticnet', 'none'),'solver' : ('lbfgs','newton-cg','liblinear','sag','saga')}]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_lr, n_iter=250, scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = KFold(n_splits=outerkfold, shuffle=True, random_state=1)

    elif model_ismi==model_isimleri[2]:
        print(model_ismi)
        classifier = RandomForestClassifier()
        param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        grid_params_rf = [{'criterion': ['gini', 'entropy'],'min_samples_leaf': param_range,'max_depth':  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           'min_samples_split': param_range[1:]}]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_rf, n_iter=250, scoring='accuracy',  cv=cv_inner, random_state=1)

        cv_outer = KFold(n_splits=outerkfold, shuffle=True, random_state=1)

    elif model_ismi==model_isimleri[3]:
        print(model_ismi)
        classifier = DecisionTreeClassifier()
        grid_params_dt =[{'splitter' : ['best', 'random'],
                          'criterion' : ['gini', 'entropy'],
                          'max_features': ['log2', 'sqrt','auto'],
                          'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          'min_samples_split': param_range[1:],
                          'min_samples_leaf': param_range,

                          }]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_dt,n_iter=250, scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = KFold(n_splits=outerkfold, shuffle=True, random_state=1)

    elif model_ismi==model_isimleri[4]:
        print(model_ismi)
        classifier = LGBMClassifier()
        grid_params_lgbm =[{
           'num_leaves': [31, 127,155],
            "max_depth" : [1,2, 3, 4, 5, 6, 7,8,9,10],
            
           'min_child_weight': [0.03,0.001,0.01,0.1],
             'subsample' : [0.5, 0.7,0.9, 1.0],
            "learning_rate" : [0.0001, 0.001,0.03, 0.01, 0.1, 1.0],"n_estimators" : [10, 100,200, 300,500, 700, 900, 1000]
            }]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_lgbm, n_iter=250, scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = KFold(n_splits=outerkfold, shuffle=True, random_state=1)

    elif model_ismi==model_isimleri[5]:
        print(model_ismi)
        classifier = GaussianNB()
        grid_params_gnb =[{'var_smoothing': np.logspace(0,-9, num=100)}]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_gnb, n_iter=250, scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = KFold(n_splits=outerkfold, shuffle=True, random_state=1)

    elif model_ismi==model_isimleri[6]:
        print(model_ismi)
        classifier = AdaBoostClassifier()
        space_ada = dict()
        space_ada['n_estimators'] = [10, 100,200, 300,500, 700, 900, 1000]
        space_ada['learning_rate'] = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
       
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, space_ada, n_iter=250, scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = KFold(n_splits=outerkfold, shuffle=True, random_state=1)

    elif model_ismi==model_isimleri[7]:
        print(model_ismi)
        classifier = GradientBoostingClassifier()
        n_estimators = [10, 100,200, 300,500, 700, 900, 1000]
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [ 0.7,0.9, 1.0]
        max_depth = [1,2, 3, 4, 5, 6, 7,8,9,10]
        space_gb = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, space_gb, n_iter=250, scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = KFold(n_splits=outerkfold, shuffle=True, random_state=1)
    elif model_ismi==model_isimleri[8]:
        print(model_ismi)
        classifier=cb.CatBoostClassifier(one_hot_max_size=31)
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [0.5, 0.7,0.9, 1.0]
        depth = [1,2, 3, 4, 5, 6, 7,8,9,10]
        iterations=[50,100,200,300,400]
        grid = dict(learning_rate=learning_rate,  subsample=subsample, depth=depth,iterations=iterations)
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid, n_iter=250, scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = KFold(n_splits=outerkfold, shuffle=True, random_state=1)
    elif model_ismi==model_isimleri[9]:
        print(model_ismi)
        classifier = XGBClassifier()
        n_estimators = [10, 100,200, 300,500, 700, 900, 1000]
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [0.5, 0.7,0.9, 1.0]
        max_depth = [1,2, 3, 4, 5, 6, 7,8,9,10]
        grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid, n_iter=250, scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = KFold(n_splits=outerkfold, shuffle=True, random_state=1)
    elif model_ismi==model_isimleri[10]:
        print(model_ismi)
        model = MLPClassifier()
        parameter_space = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],'max_iter': [50, 100, 150,200,300]}
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, parameter_space, n_iter=250, scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = KFold(n_splits=outerkfold, shuffle=True, random_state=1)
    else:
        print(model_ismi)
        classifier = SVC(probability=True)
        grid_params_svm = [{'C':(0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10,11,15,20),'gamma':(0.0001,0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10)}]
        params = {"learning_rate" : [0.03,0.001,0.01,0.1,0.2,0.3],
                  "n_estimators" : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                  "max_depth" : [2, 3, 4, 5, 6, 7, 8, 9, 10]}
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_svm, n_iter=250, scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = KFold(n_splits=outerkfold, shuffle=True, random_state=1)

    grid_result = search.fit(X, y)
    score = cross_val_score(search, X, y, scoring='accuracy',cv=cv_outer)
    score=mean(score)
    score = round((score),3)
    f1 = cross_val_score(search, X, y, scoring='f1_macro',cv=cv_outer)
    f1=mean(f1)
    f1 = round((f1),3)
    preci = cross_val_score(search, X, y, scoring='precision_macro',cv=cv_outer)
    preci=mean(preci)
    preci = round((preci),3)
    recal = cross_val_score(search, X, y, scoring='recall_macro',cv=cv_outer)
    recal=mean(recal)
    recal = round((recal),3)
    roc=cross_val_score(search,X, y, scoring="roc_auc",cv=cv_outer)
    roc=mean(roc)
    roc = round((roc),3)
    y_pred = search.predict(X)
    cm = confusion_matrix(y, y_pred)
    scorefpr=make_scorer(false_positive_rate)
    FPR = cross_validate(search, X, y, scoring=scorefpr,cv=cv_outer)
    FPR=FPR['test_score']
    FPR=np.mean(FPR)
    FPR = round((FPR),3)
    scoretpr=make_scorer(true_pozitive_rate)
    TPR = cross_validate(search, X, y, scoring=scoretpr,cv=cv_outer)
    TPR=TPR['test_score']
    TPR=np.mean(TPR)
    TPR = round((TPR),3)
    scorenpv=make_scorer(n_predict_value)
    NPV = cross_validate(search, X, y, scoring=scorenpv,cv=cv_outer)
    NPV=NPV['test_score']
    NPV=np.mean(NPV)
    NPV = round((NPV),3)

    return classifier, score, f1,preci,recal,roc, FPR,TPR, NPV


def repeatholdoutoptimize(X, y, innerkfol, nsplits, splitsize, model_ismi):
    model_isimleri=['SVM','LR','RF','DT','LGBM','GNB','ADA','GBT','CB','XGB', 'MLP']

    if model_ismi==model_isimleri[0]:
        print(model_ismi)
        classifier = SVC(probability=True)
        grid_params_svm = [{'C':(0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10,11,15,20),'gamma':(0.0001,0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10)}]
        params = {"learning_rate" : [0.03,0.001,0.01,0.1,0.2,0.3],
                  "n_estimators" : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                  "max_depth" : [2, 3, 4, 5, 6, 7, 8, 9, 10]}
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_svm,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = ShuffleSplit(n_splits=nsplits,test_size = splitsize)

    elif model_ismi==model_isimleri[1]:
        print(model_ismi)
        classifier = LogisticRegression(max_iter=3000)
        param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        grid_params_lr = [{'C' : (np.logspace(-4, 4, 20)),'penalty' : ('l1', 'l2', 'elasticnet', 'none'),'solver' : ('lbfgs','newton-cg','liblinear','sag','saga')}]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_lr,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = ShuffleSplit(n_splits=nsplits,test_size = splitsize)

    elif model_ismi==model_isimleri[2]:
        print(model_ismi)
        classifier = RandomForestClassifier()
        param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        grid_params_rf = [{'criterion': ['gini', 'entropy'],'min_samples_leaf': param_range,'max_depth':  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           'min_samples_split': param_range[1:]}]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_rf,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = ShuffleSplit(n_splits=nsplits,test_size = splitsize)

    elif model_ismi==model_isimleri[3]:
        print(model_ismi)
        classifier = DecisionTreeClassifier()
        grid_params_dt =[{'splitter' : ['best', 'random'],
                          'criterion' : ['gini', 'entropy'],
                          'max_features': ['log2', 'sqrt','auto'],
                          'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          'min_samples_split': param_range[1:],
                          'min_samples_leaf': param_range,

                          }]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_dt,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = ShuffleSplit(n_splits=nsplits,test_size = splitsize)

    elif model_ismi==model_isimleri[4]:
        print(model_ismi)
        classifier = LGBMClassifier()
        grid_params_lgbm =[{
            'num_leaves': [31, 127,155],
            "max_depth" : [1,2, 3, 4, 5, 6, 7,8,9,10],
            
           'min_child_weight': [0.03,0.001,0.01,0.1],
             'subsample' : [0.5, 0.7,0.9, 1.0],
            "learning_rate" : [0.0001, 0.001,0.03, 0.01, 0.1, 1.0],"n_estimators" : [10, 100,200, 300,500, 700, 900, 1000]
            }]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_lgbm,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = ShuffleSplit(n_splits=nsplits,test_size = splitsize)

    elif model_ismi==model_isimleri[5]:
        print(model_ismi)
        classifier = GaussianNB()
        grid_params_gnb =[{'var_smoothing': np.logspace(0,-9, num=100)}]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_gnb,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = ShuffleSplit(n_splits=nsplits,test_size = splitsize)

    elif model_ismi==model_isimleri[6]:
        print(model_ismi)
        classifier = AdaBoostClassifier()
        space_ada = dict()
        space_ada['n_estimators'] = [10, 100,200, 300,500, 700, 900, 1000]
        space_ada['learning_rate'] = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, space_ada,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = ShuffleSplit(n_splits=nsplits,test_size = splitsize)

    elif model_ismi==model_isimleri[7]:
        print(model_ismi)
        classifier = GradientBoostingClassifier()
        n_estimators = [10, 100,200, 300,500, 700, 900, 1000]
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [0.5, 0.7,0.9, 1.0]
        max_depth = [1,2, 3, 4, 5, 6, 7,8,9,10]
        space_gb = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, space_gb,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = ShuffleSplit(n_splits=nsplits,test_size = splitsize)
    elif model_ismi==model_isimleri[8]:
        print(model_ismi)
        classifier=cb.CatBoostClassifier(one_hot_max_size=31)
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [ 0.7,0.9, 1.0]
        depth = [1,2, 3, 4, 5, 6, 7,8,9,10]
        iterations=[50,100,200,300,400]
        grid = dict(learning_rate=learning_rate,  subsample=subsample, depth=depth,iterations=iterations)
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = ShuffleSplit(n_splits=nsplits,test_size = splitsize)
    elif model_ismi==model_isimleri[9]:
        print(model_ismi)
        classifier = XGBClassifier()
        n_estimators = [10, 100,200, 300,500, 700, 900, 1000]
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [0.5, 0.7,0.9, 1.0]
        max_depth = [1,2, 3, 4, 5, 6, 7,8,9,10]
        grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = ShuffleSplit(n_splits=nsplits,test_size = splitsize)
    elif model_ismi==model_isimleri[10]:
        print(model_ismi)
        model = MLPClassifier()
        parameter_space = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],'max_iter': [50, 100, 150,200,300]}
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, parameter_space,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = ShuffleSplit(n_splits=nsplits,test_size = splitsize)
    else:
        print(model_ismi)
        classifier = SVC(probability=True)
        grid_params_svm = [{'C':(0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10,11,15,20),'gamma':(0.0001,0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10)}]
        params = {"learning_rate" : [0.03,0.001,0.01,0.1,0.2,0.3],
                  "n_estimators" : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                  "max_depth" : [2, 3, 4, 5, 6, 7, 8, 9, 10]}
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, params,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = ShuffleSplit(n_splits=nsplits,test_size = splitsize)

    grid_result = search.fit(X, y)
    score = cross_val_score(search, X, y, scoring='accuracy',cv=cv_outer)
    score=mean(score)
    score = round((score),3)
    f1 = cross_val_score(search, X, y, scoring='f1_macro',cv=cv_outer)
    f1=mean(f1)
    f1 = round((f1),3)
    preci = cross_val_score(search, X, y, scoring='precision_macro',cv=cv_outer)
    preci=mean(preci)
    preci = round((preci),3)
    recal = cross_val_score(search, X, y, scoring='recall_macro',cv=cv_outer)
    recal=mean(recal)
    recal = round((recal),3)
    roc=cross_val_score(search,X, y, scoring="roc_auc",cv=cv_outer)
    roc=mean(roc)
    roc = round((roc),3)
    y_pred = search.predict(X)
    cm = confusion_matrix(y, y_pred)
    scorefpr=make_scorer(false_positive_rate)
    FPR = cross_validate(search, X, y, scoring=scorefpr,cv=cv_outer)
    FPR=FPR['test_score']
    FPR=np.mean(FPR)
    FPR = round((FPR),3)
    scoretpr=make_scorer(true_pozitive_rate)
    TPR = cross_validate(search, X, y, scoring=scoretpr,cv=cv_outer)
    TPR=TPR['test_score']
    TPR=np.mean(TPR)
    TPR = round((TPR),3)
    scorenpv=make_scorer(n_predict_value)
    NPV = cross_validate(search, X, y, scoring=scorenpv,cv=cv_outer)
    NPV=NPV['test_score']
    NPV=np.mean(NPV)
    NPV = round((NPV),3)
    return classifier, score, f1,preci,recal,roc, FPR,TPR, NPV

def skfoldcrossoptimize(X, y,innerkfol,kfold,model_ismi):
    model_isimleri=['SVM','LR','RF','DT','LGBM','GNB','ADA','GBT','CB','XGB', 'MLP']

    if model_ismi==model_isimleri[0]:
        print(model_ismi)
        classifier = SVC(probability=True)
        grid_params_svm = [{'C':(0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10,11,15,20),'gamma':(0.0001,0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10),'degrees' :(0, 1, 2, 3, 4, 5, 6)}]
        params = {"learning_rate" : [0.03,0.001,0.01,0.1,0.2,0.3],
                  "n_estimators" : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                  "max_depth" : [1,2, 3, 4, 5, 6, 7, 8, 9, 10]}
        cv_inner = KFold(n_splits=innerkfol, shuffle=True,random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_svm, scoring='accuracy',  cv=cv_inner)
        cv_outer=StratifiedKFold(n_splits=kfold)


    elif model_ismi==model_isimleri[1]:
        print(model_ismi)
        classifier = LogisticRegression(max_iter=3000)
        param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        grid_params_lr = [{'C' : (np.logspace(-4, 4, 20)),'penalty' : ('l1', 'l2', 'elasticnet', 'none'),'solver' : ('lbfgs','newton-cg','liblinear','sag','saga')}]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True,random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_lr,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer=StratifiedKFold(n_splits=kfold)

    elif model_ismi==model_isimleri[2]:
        print(model_ismi)
        classifier = RandomForestClassifier()
        param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        grid_params_rf =  [{'criterion': ['gini', 'entropy'],'min_samples_leaf': param_range,'max_depth': [1,2, 3, 4, 5, 6, 7, 8, 9, 10],
                           'min_samples_split': param_range[1:]}]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True,random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_rf,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer=StratifiedKFold(n_splits=kfold)

    elif model_ismi==model_isimleri[3]:
        print(model_ismi)
        classifier = DecisionTreeClassifier()
        grid_params_dt =[{'splitter' : ['best', 'random'],
                          'criterion' : ['gini', 'entropy'],
                          'max_features': ['log2', 'sqrt','auto'],
                          'max_depth': [1,2, 3, 4, 5, 6, 7, 8, 9, 10],
                          'min_samples_split': param_range[1:],
                          'min_samples_leaf': param_range,

                          }]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True,random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_dt, scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer=StratifiedKFold(n_splits=kfold)

    elif model_ismi==model_isimleri[4]:
        print(model_ismi)
        classifier = LGBMClassifier()
        grid_params_lgbm =[{
             'num_leaves': [31, 127,155],
            "max_depth" : [1,2, 3, 4, 5, 6, 7,8,9,10],
            
           'min_child_weight': [0.03,0.001,0.01,0.1],
             'subsample' : [0.5, 0.7,0.9, 1.0],
            "learning_rate" : [0.0001, 0.001,0.03, 0.01, 0.1, 1.0],"n_estimators" : [10, 100,200, 300,500, 700, 900, 1000]
            }]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True,random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_lgbm,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer=StratifiedKFold(n_splits=kfold)

    elif model_ismi==model_isimleri[5]:
        print(model_ismi)
        classifier = GaussianNB()
        grid_params_gnb =[{'var_smoothing': np.logspace(0,-9, num=100)}]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True,random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_gnb,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer=StratifiedKFold(n_splits=kfold)

    elif model_ismi==model_isimleri[6]:
        print(model_ismi)
        classifier = AdaBoostClassifier()
        space_ada = dict()
        space_ada['n_estimators'] = [10, 100,200, 300,500, 700, 900, 1000]
        space_ada['learning_rate'] = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
       
        cv_inner = KFold(n_splits=innerkfol, shuffle=True,random_state=1)
        search = RandomizedSearchCV(classifier, space_ada, scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer=StratifiedKFold(n_splits=kfold)

    elif model_ismi==model_isimleri[7]:
        print(model_ismi)
        classifier = GradientBoostingClassifier()
        n_estimators = [10, 100,200, 300,500, 700, 900, 1000]
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [0.5, 0.7,0.9, 1.0]
        max_depth = [1,2, 3, 4, 5, 6, 7, 8, 9, 10],
        space_gb = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        cv_inner = KFold(n_splits=innerkfol, shuffle=True,random_state=1)
        search = RandomizedSearchCV(classifier, space_gb,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer=StratifiedKFold(n_splits=kfold)
    elif model_ismi==model_isimleri[8]:
        print(model_ismi)
        classifier=cb.CatBoostClassifier(one_hot_max_size=31)
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [ 0.7,0.9, 1.0]
        depth = [1,2, 3, 4, 5, 6, 7,8,9,10]
        iterations=[50,100,200,300,400]
        grid = dict(learning_rate=learning_rate,  subsample=subsample, depth=depth,iterations=iterations)
        cv_inner = KFold(n_splits=innerkfol, shuffle=True,random_state=1)
        search = RandomizedSearchCV(classifier, grid,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer=StratifiedKFold(n_splits=kfold)
    elif model_ismi==model_isimleri[9]:
        print(model_ismi)
        classifier = XGBClassifier()
        n_estimators = [10, 100,200, 300,500, 700, 900, 1000]
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [0.5, 0.7,0.9, 1.0]
        max_depth = [1,2, 3, 4, 5, 6, 7, 8, 9, 10],
        grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer=StratifiedKFold(n_splits=kfold)
    elif model_ismi==model_isimleri[10]:
        print(model_ismi)
        model = MLPClassifier()
        parameter_space = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],'max_iter': [50, 100, 150,200,300]}
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, parameter_space,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer=StratifiedKFold(n_splits=kfold)
    else:
        print(model_ismi)
        classifier = SVC(probability=True)
        grid_params_svm = [{'C':(0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10,11,15,20),'gamma':(0.0001,0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10)}]
        params = {"learning_rate" : [0.03,0.001,0.01,0.1,0.2,0.3],
                  "n_estimators" : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                  "max_depth" : [1,2, 3, 4, 5, 6, 7, 8, 9, 10],}
        cv_inner = KFold(n_splits=innerkfol, shuffle=True,random_state=1)
        search = RandomizedSearchCV(classifier, params,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer=StratifiedKFold(n_splits=kfold)

    grid_result = search.fit(X, y)
    score = cross_val_score(search, X, y, scoring='accuracy',cv=cv_outer)
    score=mean(score)
    score = round((score),3)
    f1 = cross_val_score(search, X, y, scoring='f1_macro',cv=cv_outer)
    f1=mean(f1)
    f1 = round((f1),3)
    preci = cross_val_score(search, X, y, scoring='precision_macro',cv=cv_outer)
    preci=mean(preci)
    preci = round((preci),3)
    recal = cross_val_score(search, X, y, scoring='recall_macro',cv=cv_outer)
    recal=mean(recal)
    recal = round((recal),3)
    roc=cross_val_score(search,X, y, scoring="roc_auc",cv=cv_outer)
    roc=mean(roc)
    roc = round((roc),3)
    y_pred = search.predict(X)
    cm = confusion_matrix(y, y_pred)
    scorefpr=make_scorer(false_positive_rate)
    FPR = cross_validate(search, X, y, scoring=scorefpr,cv=cv_outer)
    FPR=FPR['test_score']
    FPR=np.mean(FPR)
    FPR = round((FPR),3)
    scoretpr=make_scorer(true_pozitive_rate)
    TPR = cross_validate(search, X, y, scoring=scoretpr,cv=cv_outer)
    TPR=TPR['test_score']
    TPR=np.mean(TPR)
    TPR = round((TPR),3)
    scorenpv=make_scorer(n_predict_value)
    NPV = cross_validate(search, X, y, scoring=scorenpv,cv=cv_outer)
    NPV=NPV['test_score']
    NPV=np.mean(NPV)
    NPV = round((NPV),3)
    return classifier, score, f1,preci,recal,roc, FPR,TPR, NPV

def loocvoptimize(X, y,innerkfol,model_ismi):
    model_isimleri=['SVM','LR','RF','DT','LGBM','GNB','ADA','GBT','CB','XGB', 'MLP']

    if model_ismi==model_isimleri[0]:
        print(model_ismi)
        classifier = SVC(probability=True)
        grid_params_svm = [{'C':(0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10,11,15,20),'gamma':(0.0001,0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10)}]
        params = {"learning_rate" : [0.03,0.001,0.01,0.1,0.2,0.3],
                  "n_estimators" : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                  "max_depth" : [1,2, 3, 4, 5, 6, 7, 8, 9, 10]}
        cv_inner = KFold(n_splits=innerkfol, shuffle=True,random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_svm,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer =LeaveOneOut()

    elif model_ismi==model_isimleri[1]:
        print(model_ismi)
        classifier = LogisticRegression(max_iter=3000)
        param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        grid_params_lr = [{'C' : (np.logspace(-4, 4, 20)),'penalty' : ('l1', 'l2', 'elasticnet', 'none'),'solver' : ('lbfgs','newton-cg','liblinear','sag','saga')}]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_lr,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer =LeaveOneOut()

    elif model_ismi==model_isimleri[2]:
        print(model_ismi)
        classifier = RandomForestClassifier()
        param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        grid_params_rf = [{'criterion': ['gini', 'entropy'],'min_samples_leaf': param_range,'max_depth':  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           'min_samples_split': param_range[1:]}]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_rf,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer =LeaveOneOut()

    elif model_ismi==model_isimleri[3]:
        print(model_ismi)
        classifier = DecisionTreeClassifier()
        grid_params_dt =[{'splitter' : ['best', 'random'],
                          'criterion' : ['gini', 'entropy'],
                          'max_features': ['log2', 'sqrt','auto'],
                          'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          'min_samples_split': param_range[1:],
                          'min_samples_leaf': param_range,

                          }]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_dt,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer =LeaveOneOut()

    elif model_ismi==model_isimleri[4]:
        print(model_ismi)
        classifier = LGBMClassifier()
        grid_params_lgbm =[{
             'num_leaves': [31, 127,155],
            "max_depth" : [1,2, 3, 4, 5, 6, 7,8,9,10],
            
           'min_child_weight': [0.03,0.001,0.01,0.1],
             'subsample' : [0.5, 0.7,0.9, 1.0],
            "learning_rate" : [0.0001, 0.001,0.03, 0.01, 0.1, 1.0],"n_estimators" : [10, 100,200, 300,500, 700, 900, 1000]
            }]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_lgbm,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer =LeaveOneOut()

    elif model_ismi==model_isimleri[5]:
        print(model_ismi)
        classifier = GaussianNB()
        grid_params_gnb =[{'var_smoothing': np.logspace(0,-9, num=100)}]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_gnb,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer =LeaveOneOut()

    elif model_ismi==model_isimleri[6]:
        print(model_ismi)
        classifier = AdaBoostClassifier()
        space_ada = dict()
        space_ada['n_estimators'] = [10, 100,200, 300,500, 700, 900, 1000]
        space_ada['learning_rate'] = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
       
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, space_ada,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer =LeaveOneOut()

    elif model_ismi==model_isimleri[7]:
        print(model_ismi)
        classifier = GradientBoostingClassifier()
        n_estimators = [10, 100,200, 300,500, 700, 900, 1000]
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [0.5, 0.7,0.9, 1.0]
        max_depth = [1,2, 3, 4, 5, 6, 7,8,9,10]
        space_gb = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, space_gb,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer =LeaveOneOut()
    elif model_ismi==model_isimleri[8]:
        print(model_ismi)
        classifier=cb.CatBoostClassifier(one_hot_max_size=31)
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [ 0.7,0.9, 1.0]
        depth = [1,2, 3, 4, 5, 6, 7,8,9,10]
        iterations=[50,100,200,300,400]
        grid = dict(learning_rate=learning_rate,  subsample=subsample, depth=depth,iterations=iterations)
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer =LeaveOneOut()
    elif model_ismi==model_isimleri[9]:
        print(model_ismi)
        classifier = XGBClassifier()
        n_estimators = [10, 100,200, 300,500, 700, 900, 1000]
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [0.5, 0.7,0.9, 1.0]
        max_depth = [1,2, 3, 4, 5, 6, 7,8,9,10]
        grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer =LeaveOneOut()
    elif model_ismi==model_isimleri[10]:
        print(model_ismi)
        model = MLPClassifier()
        parameter_space = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],'max_iter': [50, 100, 150,200,300]}
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, parameter_space,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer =LeaveOneOut()
    else:
        print(model_ismi)
        classifier = SVC(probability=True)
        grid_params_svm = [{'C':(0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10,11,15,20),'gamma':(0.0001,0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10)}]
        params = {"learning_rate" : [0.03,0.001,0.01,0.1,0.2,0.3],
                  "n_estimators" : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                  "max_depth" : [1,2, 3, 4, 5, 6, 7, 8, 9, 10]}
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, params,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer =LeaveOneOut()

    grid_result = search.fit(X, y)
    score = cross_val_score(search, X, y, scoring='accuracy',cv=cv_outer)
    score=mean(score)
    score = round((score),3)
    f1 = cross_val_score(search, X, y, scoring='f1_macro',cv=cv_outer)
    f1=mean(f1)
    f1 = round((f1),3)
    preci = cross_val_score(search, X, y, scoring='precision_macro',cv=cv_outer)
    preci=mean(preci)
    preci = round((preci),3)
    recal = cross_val_score(search, X, y, scoring='recall_macro',cv=cv_outer)
    recal=mean(recal)
    recal = round((recal),3)
    roc=cross_val_score(search,X, y, scoring="roc_auc",cv=cv_outer)
    roc=mean(roc)
    roc = round((roc),3)
    y_pred = search.predict(X)
    cm = confusion_matrix(y, y_pred)
    scorefpr=make_scorer(false_positive_rate)
    FPR = cross_validate(search, X, y, scoring=scorefpr,cv=cv_outer)
    FPR=FPR['test_score']
    FPR=np.mean(FPR)
    FPR = round((FPR),3)
    scoretpr=make_scorer(true_pozitive_rate)
    TPR = cross_validate(search, X, y, scoring=scoretpr,cv=cv_outer)
    TPR=TPR['test_score']
    TPR=np.mean(TPR)
    TPR = round((TPR),3)
    scorenpv=make_scorer(n_predict_value)
    NPV = cross_validate(search, X, y, scoring=scorenpv,cv=cv_outer)
    NPV=NPV['test_score']
    NPV=np.mean(NPV)
    NPV = round((NPV),3)
    return classifier, score, f1,preci,recal,roc, FPR,TPR, NPV




def repeatedcvoptimize(X, y,innerkfol,kfold,nrepeat, model_ismi):
    model_isimleri=['SVM','LR','RF','DT','LGBM','GNB','ADA','GBT','CB','XGB', 'MLP']

    if model_ismi==model_isimleri[0]:
        print(model_ismi)
        classifier = SVC(probability=True)
        grid_params_svm = [{'C':(0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10,11,15,20),'gamma':(0.0001,0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10)}]
        params = {"learning_rate" : [0.03,0.001,0.01,0.1,0.2,0.3],
                  "n_estimators" : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                  "max_depth" : [2, 3, 4, 5, 6, 7, 8, 9, 10]}
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_svm,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = RepeatedKFold(n_splits=kfold, n_repeats=nrepeat)

    elif model_ismi==model_isimleri[1]:
        print(model_ismi)
        classifier = LogisticRegression(max_iter=3000)
        param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        grid_params_lr = [{'C' : (np.logspace(-4, 4, 20)),'penalty' : ('l1', 'l2', 'elasticnet', 'none'),'solver' : ('lbfgs','newton-cg','liblinear','sag','saga')}]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_lr,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = RepeatedKFold(n_splits=kfold, n_repeats=nrepeat)

    elif model_ismi==model_isimleri[2]:
        print(model_ismi)
        classifier = RandomForestClassifier()
        param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        grid_params_rf = [{'criterion': ['gini', 'entropy'],'min_samples_leaf': param_range,'max_depth':  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           'min_samples_split': param_range[1:]}]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_rf,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = RepeatedKFold(n_splits=kfold, n_repeats=nrepeat)

    elif model_ismi==model_isimleri[3]:
        print(model_ismi)
        classifier = DecisionTreeClassifier()
        grid_params_dt =[{'splitter' : ['best', 'random'],
                          'criterion' : ['gini', 'entropy'],
                          'max_features': ['log2', 'sqrt','auto'],
                          'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          'min_samples_split': param_range[1:],
                          'min_samples_leaf': param_range,

                          }]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_dt,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = RepeatedKFold(n_splits=kfold, n_repeats=nrepeat)

    elif model_ismi==model_isimleri[4]:
        print(model_ismi)
        classifier = LGBMClassifier()
        grid_params_lgbm =[{
            'num_leaves': [31, 127,155],
            "max_depth" : [1,2, 3, 4, 5, 6, 7,8,9,10],
            
           'min_child_weight': [0.03,0.001,0.01,0.1],
             'subsample' : [0.5, 0.7,0.9, 1.0],
            "learning_rate" : [0.0001, 0.001,0.03, 0.01, 0.1, 1.0],"n_estimators" : [10, 100,200, 300,500, 700, 900, 1000]
            }]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_lgbm,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = RepeatedKFold(n_splits=kfold, n_repeats=nrepeat)

    elif model_ismi==model_isimleri[5]:
        print(model_ismi)
        classifier = GaussianNB()
        grid_params_gnb =[{'var_smoothing': np.logspace(0,-9, num=100)}]
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_gnb,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = RepeatedKFold(n_splits=kfold, n_repeats=nrepeat)

    elif model_ismi==model_isimleri[6]:
        print(model_ismi)
        classifier = AdaBoostClassifier()
        space_ada = dict()
        space_ada['n_estimators'] = [10, 100,200, 300,500, 700, 900, 1000]
        space_ada['learning_rate'] = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
       
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, space_ada,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = RepeatedKFold(n_splits=kfold, n_repeats=nrepeat)

    elif model_ismi==model_isimleri[7]:
        print(model_ismi)
        classifier = GradientBoostingClassifier()
        n_estimators = [10, 100,200, 300,500, 700, 900, 1000]
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [0.5, 0.7,0.9, 1.0]
        max_depth = [1,2, 3, 4, 5, 6, 7,8,9,10]
        space_gb = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, space_gb,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = RepeatedKFold(n_splits=kfold, n_repeats=nrepeat)
    elif model_ismi==model_isimleri[8]:
        print(model_ismi)
        classifier=cb.CatBoostClassifier(one_hot_max_size=31)
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [ 0.7,0.9, 1.0]
        depth = [1,2, 3, 4, 5, 6, 7,8,9,10]
        iterations=[50,100,200,300,400]
        grid = dict(learning_rate=learning_rate,  subsample=subsample, depth=depth,iterations=iterations)
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = RepeatedKFold(n_splits=kfold, n_repeats=nrepeat)
    elif model_ismi==model_isimleri[9]:
        print(model_ismi)
        classifier = XGBClassifier()
        n_estimators = [10, 100,200, 300,500, 700, 900, 1000]
        learning_rate = [0.0001, 0.001,0.03, 0.01, 0.1, 1.0]
        subsample = [0.5, 0.7,0.9, 1.0]
        max_depth = [1,2, 3, 4, 5, 6, 7,8,9,10]
        grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = RepeatedKFold(n_splits=kfold, n_repeats=nrepeat)
    elif model_ismi==model_isimleri[10]:
        print(model_ismi)
        model = MLPClassifier()
        parameter_space = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],'max_iter': [50, 100, 150,200,300]}
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, parameter_space,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = RepeatedKFold(n_splits=kfold, n_repeats=nrepeat)
    else:
        print(model_ismi)
        classifier = SVC(probability=True)
        grid_params_svm = [{'C':(0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10,11,15,20),'gamma':(0.0001,0.001,0.002,0.005,0.01,0.015,0.02,0.1,1,10)}]
        params = {"learning_rate" : [0.03,0.001,0.01,0.1,0.2,0.3],
                  "n_estimators" : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                  "max_depth" : [2, 3, 4, 5, 6, 7, 8, 9, 10]}
        cv_inner = KFold(n_splits=innerkfol, shuffle=True, random_state=1)
        search = RandomizedSearchCV(classifier, grid_params_svm,  scoring='accuracy',  cv=cv_inner, random_state=1)
        cv_outer = RepeatedKFold(n_splits=kfold, n_repeats=nrepeat)

    grid_result = search.fit(X, y)
    score = cross_val_score(search, X, y, scoring='accuracy',cv=cv_outer)
    score=mean(score)
    score = round((score),3)
    f1 = cross_val_score(search, X, y, scoring='f1_macro',cv=cv_outer)
    f1=mean(f1)
    f1 = round((f1),3)
    preci = cross_val_score(search, X, y, scoring='precision_macro',cv=cv_outer)
    preci=mean(preci)
    preci = round((preci),3)
    recal = cross_val_score(search, X, y, scoring='recall_macro',cv=cv_outer)
    recal=mean(recal)
    recal = round((recal),3)
    roc=cross_val_score(search,X, y, scoring="roc_auc",cv=cv_outer)
    roc=mean(roc)
    roc = round((roc),3)
    y_pred = search.predict(X)
    cm = confusion_matrix(y, y_pred)
    scorefpr=make_scorer(false_positive_rate)
    FPR = cross_validate(search, X, y, scoring=scorefpr,cv=cv_outer)
    FPR=FPR['test_score']
    FPR=np.mean(FPR)
    FPR = round((FPR),3)
    scoretpr=make_scorer(true_pozitive_rate)
    TPR = cross_validate(search, X, y, scoring=scoretpr,cv=cv_outer)
    TPR=TPR['test_score']
    TPR=np.mean(TPR)
    TPR = round((TPR),3)
    scorenpv=make_scorer(n_predict_value)
    NPV = cross_validate(search, X, y, scoring=scorenpv,cv=cv_outer)
    NPV=NPV['test_score']
    NPV=np.mean(NPV)
    NPV = round((NPV),3)
    return classifier, score, f1,preci,recal,roc, FPR,TPR, NPV


from collections import Counter
def isdengesiz(y):
    # count the frequency of each class
    count = Counter(y)
    print(count)
    aa=list(count.values())
    print(aa)
    deger = False
    for i in range(len(aa)-1):
        if aa[i] >= (3*aa[i+1]) or 3*aa[i] <= aa[i+1]:
            deger=True
    print(deger)
    return deger
