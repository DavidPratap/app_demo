import streamlit as st
st.title("Medical Diagnostic Web App")
import pandas as pd
import numpy as np
import pickle

st.subheader("Is the person diabetic or not?")
# Step1 : Load the picked pipeline
model=open("rfc.pickle", 'rb')
clf=pickle.load(model)
model.close()


# Step2: Get the input from the front end user
pregs=st.number_input('Pregnancies',0,20,0)
glucose=st.slider('Glucose',44,199,44)
bp=st.slider('BloodPressure',20,140,20) 
skin=st.slider('SkinThickness',7.0,99.0,7.0)
insulin=st.slider('Insulin',10, 900,10)
bmi=st.slider('BMI',15, 70,15)
dpf=st.slider('DiabetesPedigreeFunction',0.50, 2.50,0.05)
age=st.slider('Age',20, 90, 20)

# Step3: Get the model input
data={
    'Pregnancies':pregs,
    'Glucose':glucose, 
    'BloodPressure':bp,
    'SkinThickness':skin, 
    'Insulin':insulin,
    'BMI':bmi, 
    'DiabetesPedigreeFunction':dpf, 
    'Age':age}

input_data=pd.DataFrame([data])

# Step4: Get the prediction and print the result
prediction=clf.predict(input_data)[0]
st.write(prediction)
if st.button("Predict"):
    if prediction==1:
        st.error("Diabetic")
    if prediction==0:
        st.success("Healthy")
