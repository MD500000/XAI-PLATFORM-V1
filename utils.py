from regex import R
from sklearn.pipeline import make_pipeline
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from imblearn.over_sampling import SMOTENC
import io
import pandas as pd
import fontawesome as fa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import io
from imblearn.over_sampling import SMOTE, RandomOverSampler
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from collections import Counter
import fontawesome as fa
import sys
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
from io import BytesIO
import sys
from sklearn.model_selection import RepeatedKFold
from mrmr import mrmr_classif

#Function to check if the DF contains any nulls
def has_nulls(df):
    nulls = df.isna().sum().sum()
    if nulls > 0:
        return True
    return False

#Returns the count of nulls in the DF  
def null_count(df):
    return df.isna().sum().sum()

#Function to check if the column is categorical by checking if the type is of category or object
def has_categ_columns(df):
    for col in df.columns:
        if df[col].dtype.name in ["category", "object"]:
            return True
    return False

#Function to find a return the list of categorical columns (has object or category as type)
def categ_columns(df):
    if(not has_categ_columns):
        return []
    categ_columns = []
    for col in df.columns:
        if df[col].dtype.name in ["category", "object"]:
            categ_columns.append(col)
    return categ_columns

#Function that takes the df and transforms the categorical columns with encoded numerical labels
def encode_categorical_columns(df):
    for col in df.columns:
        if df[col].dtype.name in ["category", "object"]:
            le = preprocessing.LabelEncoder()
            df[col] = le.fit_transform(df[col])
            #df[col] = df[col].astype("category")
            #df[col] = df[col].cat.codes
    return df

#Function that takes the df and returns the list of columns that have numerical data
def numerical_columns(df):
    numerical_columns = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_columns.append(col)
    return numerical_columns   

#Function that checks for outliers in the DF
def check_for_outliers(df):

    #outlier count is initialized at 0 to keep track of the total
    outlier_count = 0
    numerical_columns_ = numerical_columns(df)
    #Iterates through each numerical column and calculates the 25th and 75th percentile
    for col in numerical_columns_:
        percentile25 = df[col].quantile(0.25)
        percentile75 = df[col].quantile(0.75)
        #Calculates the iqr noted as iqr = percentile-75 - percentile 25
        iqr = percentile75 - percentile25 #error here?
        #Upper limit and lower limits calculated with the forumla
        upper_limit = percentile75 + 1.5 * iqr
        lower_limit = percentile25 - 1.5 * iqr
        #Replaces the upper and lower limit respectively with the current column using the np.where() function
        ul = np.where(df[col] > upper_limit, upper_limit, df[col])
        ll = np.where(df[col] < lower_limit, lower_limit, df[col])
        #Adds on to the outlier count based on the length of the upper and lower limtis combined
        outlier_count += len(ul + ll)

    if outlier_count > 0:
        return True


#Function to check for imbalance in the df
def is_imbalance(y):
    #Returns a list of unique counts from the parameter Y
    value_c = list(y.value_counts())
    #Checks if the count of the minority class is 3 times smaller than the majority class
    #Or checks if the count of the majority class is at least three times smaller than the count of the minority class
    #If either condition is true then there is an imbalance and the function returns true
    if value_c[0] >= (3*value_c[1]) or 3*value_c[0] <= value_c[1]:
        return True
    return False

#Function to drop outliers from a specific column in the dataframe
def drop_outlier(df, field_name):
    #Calculates the IQR of a specific column with the formula ' 1.5 * (Q3-Q1) ' where Q1 refers to the 25th percentile and Q3 refers to the 7th percentile
    distance = 1.5 * (np.percentile(df[field_name], 75) - np.percentile(df[field_name], 25))
    #Considers data points greater than Q3 + distance or smaller than Q1 - distance as outliers and drops them from the df
    df.drop(df[df[field_name] > distance + np.percentile(df[field_name], 75)].index, inplace=True)
    df.drop(df[df[field_name] < np.percentile(df[field_name], 25) - distance].index, inplace=True)


#Function to drop outliers from the dataframe's numerical columns
def drop_outliers(df):
    numerical_columns_ = numerical_columns(df)
    #Iterates through each numerical column and drops the outliers using the previous function
    for col in numerical_columns_:
        drop_outlier(df, col)
    return df.reset_index(drop=True)

#Function that takes in a feature matrix X, a target variable y and a string a that specifies the selected feature
def attr(X, y, a):
    #User selects RFE : uses logistic regression as the base model and RFE to recursively remove features, fit the model and select the beast features
    if a=='Recursive Feature Elimination':
        model = LogisticRegression(solver='lbfgs')
        rfe = RFE(model)
        fit = rfe.fit(X, y)
        df=rfe.transform(X)
        new_columns = list(X.columns[rfe.support_])

    #User selects feature selection based on extra trees classifier
    #SelectFromModel used to select features based on importance and store them in new_columns
    elif a=='Based on Extra Trees Classifier':
        #defines the number of trees in the ensemble to be 50
        clf = ExtraTreesClassifier(n_estimators=50)
        fit = clf.fit(X, y)
        clf.feature_importances_
        model = SelectFromModel(clf, prefit=True)
        feature_idx = model.get_support()
        new_columns = list(X.columns[feature_idx])

    #User selects feature selection based on random forest classifier
    #Random forest classifiers are used to determine feature importance
    elif a=='Based on Random Forest Classifier':
        #defines the number of trees in the ensemble to be 100
        sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
        sel.fit(X, y)
        sel.get_support()
        feature_idx = sel.get_support()
        new_columns = list(X.columns[feature_idx])
        df = sel.transform(X)

    #User selects LASSO which uses logistic regression with L1 penalty for feature selection
    elif a=='LASSO':
        #Features with non-zero coefficients after L1 regularization are selected and stored in new_columns
        sel = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
        sel.fit(X, np.ravel(y,order='C'))
        feature_idx = sel.get_support()
        X=pd.DataFrame(X,  columns=X.columns)
        new_columns = list(X.columns[feature_idx])
        df = sel.transform(X)

    #User selects mRMR (minimum redundancy - maximum relevance)
    elif a=='mRMR (minimum Redundancy - Maximum Relevance)':
        #Uses the mRMR algorithm for feature selection
        new_columns = mrmr_classif(X=X, y=y, K=round(X.shape[1]/2))

    return new_columns
    #returns the new columns based on the feature selected

#Function that takes in the df as input and fills in the missing values using most frequent imputation
def simple_imputer(df):
    #creates list of column names
    cols = list(df.columns.values)
    #Uses imuter to replace the missing values with the most frequently occuring value in each column
    imp_mean = SimpleImputer(missing_values= np.nan, strategy='most_frequent')
    df = pd.DataFrame(imp_mean.fit_transform(df), columns=cols)
    return df

#Function that performs transformations on extracted categorical and numerical columns from the df
def transform(df, categ_columns, numerical_columns, transformation):
    categ = df[categ_columns]
    numerical = df[numerical_columns]

    transformations = {
        'Normalization': preprocessing.Normalizer(), 
        'Min-max Standardization': preprocessing.MinMaxScaler(), 
        'Standardization': preprocessing.StandardScaler(), 
        'Robust Standardization': preprocessing.RobustScaler(),
    }
    #If transformation is selected creates a pipeline and applies transformation to the numerical columns and replaces them in the df
    if transformation is not 'None':
        numerical_pipeline = make_pipeline(transformations[transformation])
        numerical = pd.DataFrame(numerical_pipeline.fit_transform(numerical), 
                                 columns=numerical_columns)

    df = pd.concat([categ, numerical], axis=1)

    return df



def transform_features(x, a):
    #Data transformation function
    try:
        dff = encode_categorical_columns(x)
    except:
        dff =x

    #If a is 0 the function does normalization
    if a==0:
        normalizer = preprocessing.Normalizer()
        list = []
        #Adds the unique values below 10 to the list
        for i in dff.columns:
            threshold = 10
            if dff[i].nunique() < threshold:
                list.append(i)
        #Creates a copy of the encoded df with adf        
        adf = dff.copy()
        #Creates normalizer object to fit into the df
        normalizer = preprocessing.Normalizer().fit(adf)
        adf= normalizer.transform(adf)
        xcolumns = dff.columns.values
        adf = pd.DataFrame(adf)
        #Renames the columns of adf to match the original df
        for i in range(len(xcolumns)):
            adf= adf.rename(columns={i:xcolumns[i]})
        #Retains the specific columns from the original df in the new one
        for i in list:
            adf[i] = dff[i]
        dff = adf

    #If a is 1, create an instance of the minmaxscaler object
    if a==1:
        std = MinMaxScaler()
        list = []
        #Iterates over each column and makes a list of unique values with a threshold of 10
        for i in dff.columns:
            threshold = 10
            if dff[i].nunique() < threshold:
                list.append(i)


        #Creates a copy of the original df in adf
        adf = dff.copy()
        std = std.fit(adf)
        adf= std.transform(adf)
        xcolumns = dff.columns.values
        adf = pd.DataFrame(adf)
        #Renames the columns of adf to match the original df
        for i in range(len(xcolumns)):
            adf= adf.rename(columns={i:xcolumns[i]})
        #Retains specific columns in the df
        for i in list:
            adf[i] = dff[i]
        dff = adf

    #If a is 2, create an instance of the standard scaler object
    if a==2:
        #Iterates over each column and makes a list of unique values with a threshold of 10
        std = StandardScaler()
        list = []
        for i in dff.columns:
            threshold = 10
            if dff[i].nunique() < threshold:
                list.append(i)

        #Creates a copy of the original df in adf
        adf = dff.copy()
        std = std.fit(adf)
        adf= std.transform(adf)
        xcolumns = dff.columns.values
        adf = pd.DataFrame(adf)
        #Renames the columns of adf to match the original df
        for i in range(len(xcolumns)):
            adf= adf.rename(columns={i:xcolumns[i]})
        #Retains specific columns in the df
        for i in list:
            adf[i] = dff[i]
        dff = adf

    #If a is 3, create an instance of the robust scaler object
    if a==3:
        std = RobustScaler()
        list = []
        #Iterates over each column and makes a list of unique values with a threshold of 10
        for i in dff.columns:
            threshold = 10
            if dff[i].nunique() < threshold:
                list.append(i)

        #Creates a copy of the original df in adf
        adf = dff.copy()
        std = std.fit(adf)
        adf= std.transform(adf)
        xcolumns = dff.columns.values
        adf = pd.DataFrame(adf)
        #Renames the columns of adf to match the original df
        for i in range(len(xcolumns)):
            adf= adf.rename(columns={i:xcolumns[i]})
        #Retains specific columns in the df
        for i in list:
            adf[i] = dff[i]
        dff = adf


    return dff

#def missing_forest_impute(x):
#    imputer = MissForest()
#    x = imputer.fit_transform(x,cat_vars=None)
#    return x


#Applies smote to address class imbalance
def smote_function(X,y, smote):
    categorical_features = np.argwhere(np.array([len(set(X.iloc[:,x])) for x in range(X.shape[1])]) <= 9).flatten()
    ##normalde buradaki <=10 du, bu haliyle bazı sayısalları kategorik yapıyor veya cinsiyeti kategorik görmüyor vs, bağlanıp bakmamız gerek aslında
    #For each column in feature matrix X, calculate the number of unique values with set() and len() giving us an array having the count of unique values in each column
    #Returns indices of elements that are below or equal to 9
    #Flattens the array of indices to get 1D array that contains the indicee of columns in X that are considered categorical by having a lower count of unique values
    #The comment in turkish is saying that originally the value was supposed to be <=9, but since the code was modified the new version is leading to some numerical
    #features being incorrectly treated as categorical/ not being able to recognize certain features like gender as categorical


    #Rest of code checks which SMOTE parameter was chosen to apply the oversampling technique
    if smote=='SMOTE':
        try:
            sm = SMOTE()
            X, y = sm.fit_resample(X, y)
        except:
            ros = RandomOverSampler()
            X, y = ros.fit_resample(X, y)
    #oversampling strategy relies on the minority class (not majority)
    elif smote=='SMOTETomek':
        try:
            sm = SMOTETomek()
            X, y = sm.fit_resample(X, y)
        except:
            ros = RandomOverSampler(sampling_strategy='not majority')
            X, y = ros.fit_resample(X, y)

    #If there are categorical features apply SMOTE-NC and use minority class again
    elif smote=='SMOTE-NC' and len(categorical_features) !=0:
        try:
            sm = SMOTENC(categorical_features=categorical_features, 
                        sampling_strategy='not majority')
            X, y = sm.fit_resample(X, y)
        except:
            ros = RandomOverSampler(sampling_strategy='not majority')
            X, y = ros.fit_resample(X, y)

    #If SMOTE is NC and there are no categorical features apply regular SMOTE
    elif smote=='SMOTE-NC' and len(categorical_features) == 0:
        try:
            sm = SMOTE()
            X, y = sm.fit_resample(X, y)
        except:
            ros = RandomOverSampler()
            X, y = ros.fit_resample(X, y)


    return X,y

def grid_search(model, param_space, X_train, Y_train):
    grid_search = GridSearchCV(estimator=model, param_grid= param_space, n_jobs=-1, scoring='accuracy')
    grid_result = grid_search.fit(X_train, Y_train)

    #results_df = pd.DataFrame(grid_result.cv_results_)

    return grid_result





