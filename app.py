import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
st.title("Medical Diagnostics Web App")
st.subheader("Is the candidate diabetic?")

# Step1: Load the model
model=open('rfc.pickle', 'rb')
clf=pickle.load(model)
model.close()

# Step2 : Get the front end user inout
pregs=st.number_input('Pregnancies',0,20,0)
glucose=st.slider('Glucose',40.0, 200.0, 40.0)
bp=st.slider('BloodPressure',20.0, 140.0, 20.0)
skin=st.slider('SkinThickness',7.0, 99.0, 7.0)
insulin=st.slider('Insulin',14.0, 850.0, 14.0)
bmi=st.slider('BMI',18.0, 70.0, 18.0)
dpf=st.slider('DiabetesPedigreeFunction',0.05, 2.50, 0.05)
age=st.slider('Age', 20, 85, 20)

# Step3: Get the inout as model input
data={
    'Pregnancies':pregs, 
    'Glucose':glucose, 
    'BloodPressure':bp, 
    'SkinThickness':skin, 
    'Insulin':insulin,
    'BMI':bmi, 
    'DiabetesPedigreeFunction':dpf,
    'Age':age
}
input_data=pd.DataFrame([data])

# Step4 : Get the predictions and print result
preds=clf.predict(input_data)[0]
if st.button("Predict"):
    if preds==1:
        st.subheader("Diabetic")
    if preds==0:
        st.subheader("Non-diabetic")
