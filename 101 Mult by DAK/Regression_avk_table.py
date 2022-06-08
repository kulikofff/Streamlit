import streamlit as st
import pandas as pd
import numpy as np
import os
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

st.write("""
# Possum Prediction
""")
st.write('---')

# Loads the Possum Dataset


possum = pd.read_csv("./tableintest.csv")
possum.dropna(inplace=True)
st.header('Mul dataset')
possum.info()

#Population and Sex are not numerical variables, lets convert them to numerical ones in order to 
#apply Multiple Linear Regression using them
#sex_dict = {"m": 0, "f":1}
#possum["numeric_sex"]=possum["sex"].map(sex_dict)

#pop_dict = {"Vic": 0, "other": 1}
#possum["numeric_pop"]=possum["Pop"].map(pop_dict)
#st.write(possum)

A = possum[['x', 'y']]
B = possum['z']


st.write(A)

 #Sidebar
 #Header of Specify Input Parameters

st.sidebar.header('Specify Input Parameters')

# #BUG -> added float

def user_input_features():
      x = st.sidebar.slider('x', float(A.x.min()), float(A.x.max()), float(A.x.mean()))
      y = st.sidebar.slider('y', float(A.y.min()), float(A.y.max()), float(A.y.mean()))
#      skullw = st.sidebar.slider('skullw', float(X.skullw.min()), float(X.skullw.max()), float(X.skullw.mean()))
#      taill = st.sidebar.slider('taill', float(X.taill.min()), float(X.taill.max()), float(X.taill.mean()))
#      footlgth = st.sidebar.slider('footlgth', float(X.footlgth.min()), float(X.footlgth.max()), float(X.footlgth.mean()))
#      earconch = st.sidebar.slider('earconch', float(X.earconch.min()), float(X.earconch.max()), float(X.earconch.mean()))
#      eye = st.sidebar.slider('eye', float(X.eye.min()), float(X.eye.max()), float(X.eye.mean()))
#      chest = st.sidebar.slider('chest', float(X.chest.min()), float(X.chest.max()), float(X.chest.mean()))
#      belly = st.sidebar.slider('belly', float(X.belly.min()), float(X.belly.max()), float(X.belly.mean()))
#      numeric_sex = st.sidebar.slider('numeric_sex', float(X.numeric_sex.min()), float(X.numeric_sex.max()), float(X.numeric_sex.mean()))
#      numeric_pop = st.sidebar.slider('numeric_pop', float(X.numeric_pop.min()), float(X.numeric_pop.max()), float(X.numeric_pop.mean()))
      data = {'x': x,
             'y': y}
#             'skullw': skullw,
#             'taill': taill,
#             'footlgth': footlgth,
#             'earconch': earconch,
#             'eye': eye,
#             'chest': chest,
#             'belly': belly,
#             'numeric_sex': numeric_sex,
#             'numeric_pop': numeric_pop}
      features = pd.DataFrame(data, index=[0])
      return features

df = user_input_features()

# # Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# # Build Regression Model
model = RandomForestRegressor()
#model = LinearRegression()
model.fit(A, B)
# # Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of totlngth')
st.write(prediction)
st.write('---')

# Bug fix
st.set_option('deprecation.showPyplotGlobalUse', False)

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(A)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, A)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, A, plot_type="bar")
st.pyplot(bbox_inches='tight')