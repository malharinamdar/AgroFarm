import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import pickle 


df=pd.read_csv('Crop_recommendation (3).csv')

with open('harvestify.pkl', 'rb') as file:
    ml_model=pickle.load(file)

columns = ['N','P','K','temperature','humidity','ph','rainfall']

st.title('Welcome to Harvestify') 
st.write('Your agricultutral crop companion') 

# Add some explanation or instructions
st.write("""
    ### Instructions
    1. Fill in the agricultural data below.
    2. Click 'Predict' to see the best crop suitable for your farm.
    """)

# Footer with disclaimer or additional info
st.write("""
    --- 
    Developed by Malhar Inamdar. Powered by Streamlit, Plotly, Scikit-learn.
    """)

# data input fields for agricultural crops

N =st.slider('Nitrogen',0,50,8)
P=st.slider('phosphorus',0,70,10)
K=st.slider('potassium',0,60,9)
temperature=st.slider('temperature',0,100,20)
humidity=st.slider('humidity',0,20,5)
ph=st.slider('ph level',0,12,3)
rainfall=st.slider('rainfall',0,300,80)

predict_button=st.button('Predict')

def user_input():
    return pd.DataFrame({
        'N':[N],
        'P':[P],
        'K':[K],
        'temperature':[temperature],
        'humidity':[humidity],
        'ph':[ph],
        'rainfall':[rainfall]
    })

if predict_button:
    user_data=user_input()
    st.subheader('User farm details: ')
    st.write(user_data)

    def predict(adata):
        return ml_model.predict(adata)
    
    user_result=ml_model.predict(user_data)

    st.subheader('The suitable crop prediction is: ')
    st.write(user_result)

