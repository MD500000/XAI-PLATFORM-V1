import streamlit as st
import pandas as pd
import numpy as np

st.title('Explainable Artificial Intelligence (xAI)')
st.image(r'..\streamlit_app\assets\classification.jpg')
st.markdown('''
Some methods were needed in order to make the results obtained as a result of modeling with machine learning methods more interpretable and explainable. Based on these requirements, the concept of explicable artificial intelligence was introduced. It is a set of methods developed to make the model more understandable by revealing the relationships between output and input variables. The use of classification models to diagnose disease in the field of health largely depends on the ability of the models created to be interpreted and explained by the researcher. There are many different  ways to increase the explainability of artificial intelligence models created in the field of health and variable significance is one of them. Explainable AI methods used for this purpose provide a patient-specific explanation for a particular classification so that any i allows for a simpler explanation of a complex classifier in the clinical setting.
''')

st.subheader('Local Interpretable Model-Agnostic Explanations (LIME)')
st.image(r'..\streamlit_app\assets\lime.png')
with st.expander('Local Interpretable Model-Agnostic Explanations (LIME)', expanded=True):
    st.markdown('''
LIME is a post-hoc model-free annotation technique that aims to approximate any black box machine learning model with a native, interpretable model to explain each individual prediction. As a result, LIME works locally, which means it is observation-specific and provides explanations for the prediction for each observation. What LIME does is try to fit a local model using sample data points similar to the observation described.
''')
st.subheader('Shapley Additive Explanations (SHAP)')
st.image(r'..\streamlit_app\assets\shap.png')
with st.expander('Shapley Additive Explanations (SHAP)', expanded=True):
    st.markdown('''
The main idea of SHAP is to calculate the Shapley values for each feature of the sample to be interpreted, where each Shapley value represents the predictive impact of the feature to which it is associated.
''')
