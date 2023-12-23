from hmac import new
import matplotlib
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, cross_validate, StratifiedKFold, KFold, RepeatedKFold
from sklearn.model_selection import GridSearchCV
import streamlit as st
import pandas as pd
import numpy as np
import csv
import spss_converter
import tempfile
import os
import utils
import modelling
import shap
import streamlit.components.v1 as components
import lime
import val as vt

from val import calc_score

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('XAI Platform')

tab1, tab2, tab3, tab4, tab5 = st.tabs(["File Upload", "Data Preprocessing", "Modelling", "SHAP", "LIME"])

df = None
new_df = None
preprocessed_df = None
uploaded_file = None
validation_function = None
hyperparameter = None
cross_val = None
k_fold = None
repeat = None
labels = ["accuracy", "f1_weighted", "precision_weighted","recall_weighted","roc_auc_ovr", "false_positive_rate", "true_positive_rate", "negative_predictive_value"]
model_count = 0
results_dict = {}
models = []
target = []

#Upload file using first Tab
with tab1:
    uploaded_file = st.file_uploader('Choose a file', 
                                type=['xls', 'xlsx', 'sav', 'csv', 'txt'])
    
    if uploaded_file is None:
            st.write('Please upload a file first.')
    
    #Get file name and extension
    if uploaded_file is not None:
        file_name = uploaded_file.name

        extention = uploaded_file.name.split('.')[1]

        #Directly read dataframe if file type is excel
        if extention in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)

        #Converts sav file to a dataframe
        elif extention == 'sav':
            st.write('If you are having issues, you can convert SPSS files here: https://secure.ncounter.de/spssconverter')
            df, metadata = spss_converter.to_dataframe(uploaded_file.getvalue())

        #Converts csv file to a dataframe
        elif extention == 'csv':
            st.write('If your file is not working correctly, specify the delimeter here:')
            seperator = st.text_input('Seperator', ',')
            df = pd.read_csv(uploaded_file, sep=seperator)

        #Converts txt file to a dataframe
        elif extention == 'txt':
            st.write('If your file is not working correctly, specify the delimeter here:')
            seperator = st.text_input('Seperator', ',')
            df = pd.read_csv(uploaded_file, sep=seperator)


        st.subheader('Data preview')

        if df is not None:
            #Turn the columns into a python list
            cols = df.columns.tolist()
            
            #Takes in the columns made into a list and makes them into selective attributes
            st.write('Select predictive attributes')
            pred = st.multiselect('Predictive attributes', cols, default=cols[:-1])



            #Excludes the columns that were selected in the pred variable
            cols = [x for x in cols if x not in pred]
            col1, col2 = st.columns(2)

            #Allows you to select the output attribute from the selected predictive attributes
            with col1:
                st.write('Select target/output attribute')
                target = st.selectbox('Target attribute', cols)
                
            #Allows you to select row/Class of interest from the attribute   
            with col2:
                st.write('Select the class of interest')
                i_c = st.selectbox('Class of interest', df[target].dropna().unique().tolist())


            #prints each unique value to the console
            for cls in df[target].unique().tolist():
                print(cls)

            #creates a new dataframe of the specificed prediction with the target attribute selected
            new_df = df[pred + [target]]
            class_of_interest = i_c

            #Uses a function to check whether the given column is cateogrical or numerical from the utils file to then add it onto a variable
            categ_columns = [x for x in utils.categ_columns(new_df) if x not in [target]]
            numerical_columns = [x for x in utils.numerical_columns(new_df) if x not in [target]]


            #Displays the new_df created alongside information from the given df
            with st.expander('Data Information', expanded=True):
                st.dataframe(new_df)
                col1, col2 = st.columns(2)
                with col1:
                    st.write('**Number of Instances:**', new_df.shape[0])
                    st.write('**Number of Predictive Attributes:**', len(pred))
                    st.write('**Number of Target Attributes:**', len([target]))
                    
                with col2:
                    st.write('**Number of Attributes:**', new_df.shape[1])
                    st.write('**Number of Classes:**', len(new_df[target].dropna().unique().tolist()))
                    st.write('**Class of Interest:**', i_c)  

#Data Preprocessing tab
with tab2:
        if new_df is None:
            st.write('Please upload a file first.')
    

        if new_df is not None:
            #Uses util function to check if the new_df has any null variables and prints the result for both cases       
            with st.expander('Missing Data Analysis Results', expanded=True):
                missing = utils.has_nulls(new_df)
                m_c = 'No missing values'
                if(missing):
                    st.write('Total Missing Values:', utils.null_count(new_df))
                    m_c = st.radio('Missing Value Imputation Method:', 
                    ['Remove rows with missing values',])
                    #'Most-frequent imputation'])
                else:
                    st.write('No missing values found.')
            
            #Uses util function to analyze class imbalance and gives the user options to address it
            with st.expander('Class Imbalance Analysis', expanded=True):
                onehot_list = utils.categ_columns(new_df)
                class_imbalance = 'None'
                is_imb = utils.is_imbalance(new_df[target])
                if is_imb and len(onehot_list) == 0:
                    class_imbalance = st.radio('There is a class imbalance problem in the dataset. Select one of the following methods to resolve the class imbalance problem.',
                    ['None','SMOTE','SMOTETomek'])
                elif is_imb and len(onehot_list) != 0:
                    class_imbalance = st.radio('There is a class imbalance problem in the dataset. Select one of the following methods to resolve the class imbalance problem.',
                    ['None','SMOTE-NC'])
                else:
                    st.write('There is no class imbalance problem in the dataset.')
                
            
            #Uses util function to check for outliers and asks the user if they'd like them to be removed
            with st.expander('Outlier Value Analysis Result', expanded=True):
                if utils.check_for_outliers(new_df) == 0:
                    outliers = 'No outliers'
                    st.write('No outlier values were detected in the data set.')
                else:
                    st.write('Outlier values are detected in the data set. Total Outlier Values:', utils.check_for_outliers(df))
                    outliers = st.radio('Remove outliers?',['No', 'Yes'])
                
            #Gives the user options for which methods to perform data transformation
            with st.expander('Transformation Methods', expanded=True):
                transformations = st.radio('Please choose one of the following methods for data transformation.',
                ['None',
                'Normalization', 
                'Min-max Standardization', 
                'Standardization', 
                'Robust Standardization'])
                
            #Gives the user options for which attribute selection method to use
            with st.expander('Attribute Selection Methods', expanded=True):
                attribute_selection = st.radio('Please choose one of the following methods for attribute selection.',
                ['None',
                'Recursive Feature Elimination', 
                'Based on Extra Trees Classifier',
                'Based on Random Forest Classifier', 
                'LASSO',
                'mRMR (minimum Redundancy - Maximum Relevance)'])
                

            #Preprocessing Pipeline
            with st.expander('Preprocessing Pipeline', expanded=True):
                
                #Copies the inputed dataframe for manipulation 
                new_df_ = new_df.copy()

                #Makes unique columns into a list and prints them 
                for cls in new_df_[target].unique().tolist():
                    print(cls)
                #Removes missing values from the df and resets the index
                st.write('**Missing data:**', m_c)
                if m_c == 'Remove rows with missing values':
                    new_df = new_df_.copy()
                    new_df = new_df.dropna().reset_index(drop=True)
                #if m_c == 'Most-frequent imputation':
                #    new_df = new_df_.copy()
                #    new_df = utils.simple_imputer(new_df)


                #Takes the df and converts the categorical columns to numerical values for ML usage
                encoded_df = utils.encode_categorical_columns(new_df)


                #If there is class imbalance SMOTE is used to balance the distribution
                #It creates a copy of a feature matrix X with a target vector Y and uses SMOTE to create synthetic samples
                #This creates a balanced DF
                X, y = encoded_df.drop([target], axis=1), encoded_df[target]
                X_, y_ = X.copy(), y.copy()
                if class_imbalance != 'None':
                    st.write('**Class imbalance handling strategy:**', class_imbalance)
                    X, y = X_, y_
                    X, y = utils.smote_function(encoded_df.drop([target], axis=1), encoded_df[target], class_imbalance)

                balanced_df = pd.concat([X, y], axis=1)
                balanced_df_ = balanced_df.copy()
                
                #Removes outliers from the df if the user choses to do so with a function from utils
                st.write('**Remove outliers:**', outliers)
                if outliers == 'Yes':
                     balanced_df = balanced_df_
                     balanced_df = utils.drop_outliers(balanced_df)
                if outliers == 'No':
                    balanced_df = balanced_df_


                #Splits the DF into X which holds all the columns except for "target" and Y which contains the target column in order to prepare the DF for training
                X, y = balanced_df.drop([target], axis=1), balanced_df[target]


                #Shows the attribute selection method chosen by the user
                new_columns = X.columns
                st.write('**Attribute selection method:**', attribute_selection)
                if attribute_selection == 'None':
                    new_columns = cols
                

                #If data transformation is selected, the transform function from utils will seperate the categorical and numerical columns while creating a dictionary.
                #The dictionary will map methods to their coressponding scalers.
                #The transformed df will also be concatenated into a new DF.
                st.write('**Data transformation:**', transformations)
                if transformations != 'None':
                    X = utils.transform(X, categ_columns, numerical_columns, transformations)

                transformed_df = pd.concat([X, y], axis=1)

                #Target column is extracted and copied onto a variable
                target_col = transformed_df[target]
                transformed_df_ = transformed_df.copy()
                #Applies attribute selection based on the option chosen by the user and returns the selected attributes as a list
                if attribute_selection != 'None':
                    transformed_df = transformed_df_
                    new_columns = utils.attr(transformed_df.drop([target], axis=1), 
                                             transformed_df[target], 
                                             attribute_selection)
                    transformed_df = transformed_df[new_columns]
                    transformed_df = pd.concat([transformed_df, target_col], axis=1)


                #Writes the number of instances, predictive attributes and target attributes
                col1, col2 = st.columns(2)
                with col1:
                    st.write('**Number of Instances:**', transformed_df.shape[0])
                    st.write('**Number of Predictive Attributes:**', len(pred))
                    st.write('**Number of Target Attributes:**', len([target]))

                #writes the number of attributes, cloasses and classes of interest
                with col2:
                    st.write('**Number of Attributes:**', transformed_df.shape[1])
                    st.write('**Number of Classes:**', len(transformed_df[target].dropna().unique().tolist()))
                    st.write('**Class of Interest:**', i_c)
                #Writes the important attributes
                st.write('**Important Attributes:**', ', '.join(new_columns))

                #Displays the preprocessed data
                preprocessed_columns = transformed_df.columns
                X_preprocessed, y_preprocessed = transformed_df.drop([target], axis=1), transformed_df[target]
                preprocessed_df = pd.concat([X_preprocessed, y_preprocessed], axis=1)

                #Allows the user to download the preprocessed data
                try:
                    st.dataframe(preprocessed_df)
                    csv = transformed_df.to_csv().encode('utf-8')
                    st.download_button(
                    label="Download Preprocessed Data",
                    data=csv,
                    file_name=f'{file_name.split(".")[0]}_preprocessed.csv',
                    mime='text/csv')
                except:
                    st.write("Make sure you've selected your entire pipeline.")
#Modelling Tab
with tab3:
    if preprocessed_df is None:
        st.write('Please ensure you have preprocessed your data.')
    #If df has been preprocessed from the previous tab, allows you to use a model for the dataset
    if preprocessed_df is not None:
        X, y = preprocessed_df.drop([target], axis=1), preprocessed_df[target]
        with st.expander('Modelling', expanded=True):
            models = st.multiselect('Select Model', ['AdaBoost', 'CatBoost', 'Decision Tree', 'Gaussian Naive Bayes', 'Gradient Boosting', 'LightGBM', 'Logistic Regression', 
                                                    'Multilayer Perceptron (MLP)', 'Random Forest', 'Support Vector Machine', 'XGBoost'])
            

    grid_search = st.radio('**Would you like to perform Grid Search?**', ['Yes', 'No'])

    cross_val = st.radio('**Would you like to perform Cross Validation?**', ['Yes', 'No'])
    if cross_val == 'Yes':
        with st.expander('Validation', expanded=True): 
            validation_function = st.radio('Select Validation Method', ['None', 'Holdout', 'Repeated Holdout', 'Stratified K-fold Cross Validation', 'Leave One Out Cross Validation', 
                                                         'Repeated Cross Validation', 'Nested Cross Validation'])
            
            if validation_function == 'None':
                st.empty()

            if validation_function == 'Holdout':
                k_fold = st.slider('Select the training dataset percentage:', 50, 100, 50, 5)

            if validation_function == 'Repeated Holdout':
                k_fold = st.slider('Select split size:', 50, 100, 50, 5)
                repeat = st.slider('Select the number of repeats:', 5, 50, 5, 1)

            if validation_function == 'Stratified K-fold Cross Validation':
                k_fold = st.slider('Select k-fold:', 2, 10, 2, 1)

            if validation_function == 'Leave One Out Cross Validation':
                pass

            if validation_function == 'Repeated Cross Validation':
                k_fold = st.slider('Select k-fold:', 5, 10, 5, 1)

            if validation_function == 'Nested Cross Validation':
                k_fold = st.slider('Select the inner k-fold:', 5, 10, 5, 1)
                repeat = st.slider('Select the outer k-fold:', 5, 10, 5, 1)

    else:
        st.empty()

    generate = st.button("Create models")
    
    if models and generate:
        with st.spinner('Please wait while we initiate modelling.'):
            model_list = {}
            models_created = []
            #If model is from the listed choices below, display results with scores and labels
            for model in models:
                #Check if model is a supported choice
                #if model in ['AdaBoost', 'Decision Tree', 'Gaussian Naive Bayes', 'Gradient Boosting', 'Logistic Regression', 
                        #'Multilayer Perceptron (MLP)', 'Random Forest', 'Support Vector Machine']:
                        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.30)
                        classifier, param_space = modelling.initialize_model(models[0])
                        best_model = None
                        model_name = None
                        #If grid search is selected, do grid search and depending on the cross validation selection either print a dataframe or add the results to a dict
                        if grid_search == 'Yes':
                            results = utils.grid_search(classifier, param_space, X_train, Y_train)
                            best_model = results.best_estimator_
                            print(best_model)
                            model_name = type(best_model).__name__
                            results_dict[model] = results
                            if cross_val!= 'Yes':
                                 st.write(f"**Grid Search Results:**")
                                 model_list[model] = best_model.fit(X, y)
                                 models_created.append(best_model)
                                 scores = calc_score(model_list[model], X, y)
                                 scores = dict(zip(labels,scores))
                                 st.dataframe(pd.DataFrame.from_dict(scores, orient='index', columns=[f"Model : {best_model}"]))
                                 model_count += 1
                        #if cross validation is selected, get cross validation method from dict and perform it
                        if cross_val == 'Yes' and validation_function!= "None" and grid_search!= 'Yes':
                            validation_method = vt.val_methods.get(validation_function, None)
                            cross_vali = validation_method(classifier, X_train, Y_train, k_fold, repeat)
                            st.write('**Cross validation score:**')
                            df = pd.DataFrame.from_dict(cross_vali, orient='index', columns=['Score'])
                            st.dataframe(df)

                        #if no cross validation or grid search do modelling as normal
                        if cross_val =='No' or grid_search == 'No':
                            st.write(f"**{model} Results:**")
                            model_list[model] = modelling.get_model(model).fit(X, y)
                            models_created.append(model)
                            scores = calc_score(model_list[model], X, y)
                            scores = dict(zip(labels, scores))
                            st.dataframe(pd.DataFrame.from_dict(scores, orient='index', columns=[f"Model : {model}"]))
                            model_count += 1
                        #if grid search and cross validation is selected do cross validation based on grid search best model and show scores
                        if cross_val== 'Yes' and grid_search == 'Yes':
                            validation_method = vt.val_methods.get(validation_function, None)
                            cross_vali = validation_method(best_model, X_train, Y_train, k_fold, repeat)
                            st.write(f"**{model_name} Results:**")
                            st.write('**Cross validation and grid search best model score:**')

                            model_list[model] = best_model.fit(X, y)
                            models_created.append(best_model)
                            df = pd.DataFrame.from_dict(cross_vali, orient='index', columns=['Score'])
                            st.dataframe(df)
                            model_count+=1

            #If model is from the listed choices below, display results with scores and labels
            #for model in models:
                #if model in ['XGBoost', 'LightGBM', 'CatBoost']:
                    #try:
                        #st.write(f"**{model} Results:**")
                        #model_list[model] = modelling.get_model(model).fit(X, y)
                        #models_created.append(model)
                        #scores = calc_score(model_list[model], X, y)
                        #labels = ["accuracy", "f1_weighted", "precision_weighted","recall_weighted","roc_auc_ovr"]
                        #scores = dict(zip(labels, scores))
                        #st.dataframe(pd.DataFrame.from_dict(scores, orient='index', columns=['Score']))
                        #model_count += 1
                    #except:
                        #st.write(f'**{model}** is not supported for the data you uploaded.')        

        #if validation_function!= "None":    
            #with st.expander('Hyper-Parameter tuning with cross validation results', expanded=True):
                #validation_method = vt.val_methods.get(validation_function, None)
                #st.write('**Model:**', models[-1])
                #st.write('**Hyperparameter Optimization:**', hyperparameter)
                #st.write('**Validation Method:**', validation_function)
                #if models:
                        #with st.spinner('Please wait while we do hyper-parameter tuning.'):            
                            #X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.30)
                            #model, param_space = modelling.initialize_model(models[0])

                            #validation_method = vt.val_methods.get(validation_function, None)
                            #print('this is the ' + str(validation_method))
                            #cross_validate = validation_method(model, X_train, Y_train)
                            

                            #results = utils.grid_search(model, param_space, X_train, Y_train)
                            #st.dataframe(results)

        #with st.expander('Modelling Options', expanded=True):  
             #st.write('**Models:**', ', '.join(models))
             #st.write('**Hyperparameter Optimization:**', hyperparameter)
             #st.write('**Validation Method:**', val)
             #st.dataframe(preprocessed_df)


        # if model_count != 0:
        #     st.write('**Models:**', ', '.join(models_created))
        #     for model in model_list.keys():
        #         with st.expander(f'{model} Results', expanded=True):
        #             scores = calc_score(model_list[model], X, y)
        #             labels = ["accuracy", "f1_weighted", "precision_weighted","recall_weighted","roc_auc_ovr"]
        #             scores = dict(zip(labels, scores))
        #             st.dataframe(pd.DataFrame.from_dict(scores, orient='index', columns=['Score']))




#Shap tab
with tab4:
    if preprocessed_df is None:
            st.write('Please upload a file first.')
    with st.spinner('Please while we explain the predictions.'):
        if model_count != 0:
            for model in model_list.keys():
                #Generates shap values for the following models and ensures that values are still generated even if model is not specified
                if model in ['XGBoost', 'CatBoost', 'Decision Tree', 'Gradient Boosting', 'Random Forest']:
                    st.write(model)
                    shap_values = shap.TreeExplainer(model_list[model]).shap_values(X)
                    st.pyplot(shap.summary_plot(shap_values, X))

                else:
                    st.write(model)
                    shap_values = shap.KernelExplainer(model_list[model].predict_proba, X).shap_values(X)
                    st.pyplot(shap.summary_plot(shap_values, X))


#LIME tab
with tab5:
    if preprocessed_df is None:
            st.write('Please upload a file first.')
    #Provides a prediction explanation interface
    with st.spinner('Please while we explain the predictions.'):
        if model_count != 0:
            st.dataframe(X)
            for model in model_list.keys():
                explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X), feature_names=list(X.columns), categorical_features=categ_columns)
                instances = X.shape[0]
                inc_int = None
                #If random instance button is clicked a random instance index between 0 and the total number of indices is generated.
                #This index will allow the user to explore a random data point for explanation.
                if st.button('Random Instace', key=f'new_instance_{model}'):
                    inc_int = np.random.randint(0, instances)
                
                #Users can also manually input an instance index for testing.
                record = st.number_input('Instance', min_value=0, max_value=instances-1, value=0, step=1, key=f'instance_{model}')
                #If explain button is clicked, the instance is used to explain a prediction made by the model.
                if st.button('Explain', key=f'explain_{model}'):
                    inc_int = record
                

                st.write(model)
                #Displays prediction probabilities for the model.
                if inc_int is not None:
                    ic, pc = st.columns(2)
                    with ic:
                        st.write('**Instance:**', X.iloc[inc_int])
                    with pc:
                        st.write('**Prediction:**', model_list[model].predict_proba(np.array(X.iloc[inc_int]).reshape(1, -1)))
                    #uses LIME to generate plot explanations for the selected instance as well as details for said instance and prediction probabilities.
                    exp = explainer.explain_instance(np.array(X.iloc[inc_int]), model_list[model].predict_proba, num_features=5)
                    st.pyplot(exp.as_pyplot_figure())
