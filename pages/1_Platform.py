import streamlit as st
import pandas as pd
import numpy as np
import csv
import spss_converter
import tempfile
import os
import utils
import modelling

st.title('XAI Platform')

tab1, tab2, tab3, tab4, tab5 = st.tabs(["File Upload", "Data Preprocessing", "Modelling", "LIME", "SHAP"])

df = None

with tab1:
    uploaded_file = st.file_uploader('Choose a file', 
                                type=['xls', 'xlsx', 'sav', 'csv', 'txt'])
    
    if uploaded_file is not None:
        file_name = uploaded_file.name

        extention = uploaded_file.name.split('.')[1]

        if extention in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)

        elif extention == 'sav':
            st.write('If you are having issues, you can convert SPSS files here: https://secure.ncounter.de/spssconverter')
            df, metadata = spss_converter.to_dataframe(uploaded_file.getvalue())

        elif extention == 'csv':
            st.write('If your file is not working correctly, specify the delimeter here:')
            seperator = st.text_input('Seperator', ',')
            df = pd.read_csv(uploaded_file, sep=seperator)

        elif extention == 'txt':
            st.write('If your file is not working correctly, specify the delimeter here:')
            seperator = st.text_input('Seperator', ',')
            df = pd.read_csv(uploaded_file, sep=seperator)

        st.subheader('Data preview')
        if df is not None:
            st.dataframe(df)
            cols = df.columns.tolist()
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write('Select predictive attributes')
                pred = st.multiselect('Predictive attributes', cols, default=cols[:-1])

            cols = [x for x in cols if x not in pred]

            with col2:
                st.write('Select target/output attribute')
                target = st.selectbox('Target attribute', cols)
                
            with col3:
                st.write('Select the class of interest')
                i_c = st.selectbox('Class of interest', df[target].unique().tolist())


            new_df = df[pred + [target]]
            class_of_interest = i_c

with tab2:
        if df is not None:        
            with st.expander('Missing Data Analysis Results', expanded=True):
                missing = utils.has_nulls(df)
                m_c = 'No missing values'
                if(missing):
                    st.write('Total Missing Values:', utils.null_count(df))
                    m_c = st.radio('Missing Value Imputation Method:', 
                    ['None',
                    'Remove rows with missing values', ])
                    #'Let the assignment be made with the Random Forest method.'])
                else:
                    st.write('No missing values found.')
            
            with st.expander('Outlier Value Analysis Result', expanded=True):
                if utils.check_for_outliers(df) == 0:
                    outliers = 'No outliers'
                    st.write('No outlier values were detected in the data set.')
                else:
                    st.write('Outlier values are detected in the data set. Total Outlier Values:', utils.check_for_outliers(df))
                    outliers = st.radio('Remove outliers?',['No', 'Yes'])
            
            with st.expander('Transformation Methods', expanded=True):
                transformations = st.radio('Please choose one of the following methods for data transformation.',
                ['None',
                'Normalization', 
                'Min-max Standardization', 
                'Standardization', 
                'Robust Standardization'])
            
            with st.expander('Attribute Selection Methods', expanded=True):
                attribute_selection = st.radio('Please choose one of the following methods for attribute selection.',
                ['None',
                'Recursive Feature Elimination', 
                'Based on Extra Trees Classifier',
                'Based on Random Forest Classifier', 
                'LASSO',])
                # 'Boruta'])

            with st.expander('Class Imbalance Analysis', expanded=True):
                onehot_list = utils.categ_columns(df)
                df = utils.encode_categorical_columns(df)
                class_imbalance = 'None'
                is_imb = utils.is_imbalance(df[target])
                if is_imb and len(onehot_list) == 0:
                    class_imbalance = st.radio('There is a class imbalance problem in the dataset. Select one of the following methods to resolve the class imbalance problem.',
                    ['None','SMOTE','SMOTETomek'])
                elif is_imb and len(onehot_list) != 0:
                    class_imbalance = st.radio('There is a class imbalance problem in the dataset. Select one of the following methods to resolve the class imbalance problem.',
                    ['None','SMOTE-NC'])
                else:
                    st.write('There is no class imbalance problem in the dataset.')


            with st.expander('Preprocessing Pipeline', expanded=True):
                st.write('**Missing data:**', m_c)
                if m_c == 'Remove rows with missing values.':
                    df = df.dropna()
                #if m_c == 'Let the assignment be made with the Random Forest method.':
                    # df = utils.missing_forest_impute(df)
                
                st.write('**Remove outliers:**', outliers)
                if outliers == 'Yes':
                    df = utils.drop_outliers(df)
                
                new_columns = df.columns[:-1]
                st.write('**Attribute selection method:**', attribute_selection)
                if attribute_selection == 'None':
                    new_columns = df.columns[:-1]

                
                st.write('**Data transformation:**', transformations)
                if transformations == 'Normalization':
                    df = df.dropna()
                    transformed_x = utils.transform_features(df[list(new_columns)], 0)
                    df = pd.concat([transformed_x, df[target]], axis=1)
                if transformations == 'Min-max Standardization':
                    df = df.dropna()
                    transformed_x = utils.transform_features(df[list(new_columns)], 1)
                    df = pd.concat([transformed_x, df[target]], axis=1)
                if transformations == 'Standardization':
                    df = df.dropna()
                    transformed_x = utils.transform_features(df[list(new_columns)], 2)
                    df = pd.concat([transformed_x, df[target]], axis=1)
                if transformations == 'Robust Standardization':
                    df = df.dropna()
                    transformed_x = utils.transform_features(df[list(new_columns)], 3)
                    df = pd.concat([transformed_x, df[target]], axis=1)
                
                target_col = df[target]
                if attribute_selection == 'Recursive Feature Elimination':
                    df = df.dropna()
                    new_columns = utils.attr(df, df.drop([target], axis=1), df[target], 0)
                    df = df[new_columns]
                    df = pd.concat([df, target_col], axis=1)
                    st.write('**Important Attributes:**', ', '.join(new_columns))
                if attribute_selection == 'Based on Extra Trees Classifier':
                    df = df.dropna()
                    new_columns = utils.attr(df, df.drop([target], axis=1), df[target], 1)
                    df = df[new_columns]
                    df = pd.concat([df, target_col], axis=1)
                    st.write('**Important Attributes:**', ', '.join(new_columns))
                if attribute_selection == 'Based on Random Forest Classifier':
                    df = df.dropna()
                    new_columns = utils.attr(df, df.drop([target], axis=1), df[target], 2)
                    df = df[new_columns]
                    df = pd.concat([df, target_col], axis=1)
                    st.write('**Important Attributes:**', ', '.join(new_columns))
                if attribute_selection == 'LASSO':
                    df = df.dropna()
                    new_columns = utils.attr(df, df.drop([target], axis=1), df[target], 3)
                    df = df[new_columns]
                    df = pd.concat([df, target_col], axis=1)
                    st.write('**Important Attributes:**', ', '.join(new_columns))
                if attribute_selection == 'Boruta':
                    df = df.dropna()
                    new_columns = utils.attr(df, df.drop([target], axis=1), df[target], 4)
                    df = df[new_columns]
                    df = pd.concat([df, target_col], axis=1)
                    st.write('**Important Attributes:**', ', '.join(new_columns))

                columns_ = df.columns
                X, y = df.drop([target], axis=1), df[target]
                df = pd.concat([X, y], axis=1)

                if class_imbalance != 'None':
                    st.write('**Class imbalance handling strategy:**', class_imbalance)
                    df = df.dropna()
                    if class_imbalance == 'SMOTE':
                        X, y = utils.smote_function(df, df.drop([target], axis=1), df[target], 0)
                    if class_imbalance == 'SMOTETomek':
                        X, y = utils.smote_function(df, df.drop([target], axis=1), df[target], 1)
                    if class_imbalance == 'SMOTE-NC':
                        X, y = utils.smote_function(df, df.drop([target], axis=1), df[target], 2)

                columns_ = df.columns
                X, y = df.drop([target], axis=1), df[target]
                df = pd.concat([X, y], axis=1)

                st.dataframe(df)

                csv = df.to_csv().encode('utf-8')
                st.download_button(
                label="Download Preprocessed Data",
                data=csv,
                file_name=f'{file_name.split(".")[0]}_preprocessed.csv',
                mime='text/csv')



with tab3:
    if df is not None:
        with st.expander('Modelling', expanded=True):
            models = st.multiselect('Select Model', ['AdaBoost', 'CatBoost', 'Decision Tree', 'Gaussian Naive Bayes', 'Gradient Boosting', 'LightGBM', 'Logistic Regression', 
                                                    'Multilayer Perceptron (MLP)', 'Random Forest', 'Support Vector Machine', 'XGBoost'])
        
            hyperparameter = st.radio('Hyperparameter Optimization', ['Yes', 'No'])

        with st.expander('Validation', expanded=True): 
            val = st.radio('Select Validation Method', ['Holdout', 'Repeated Holdout', 'Stratified K-fold Cross Validation', 'Leave One Out Cross Validation', 
                                                                'Repeated Cross Validation', 'Nested Cross Validation'])
            
        with st.expander('Train Test Split', expanded=True):
            test_size = st.slider('Test size', 0.5, 1.0, 0.8, 0.05)

        
        with st.expander('Modelling Options', expanded=True):  
            st.write('**Models:**', ', '.join(models))
            st.write('**Hyperparameter Optimization:**', hyperparameter)
            st.write('**Validation Method:**', val)
            st.write('**Test Size:**', test_size)

        model_list = {}
        for model in models:
            if hyperparameter:
                m, h = modelling.get_config(model)
                model_list[model] = (m,h)
        st.write(model_list)
