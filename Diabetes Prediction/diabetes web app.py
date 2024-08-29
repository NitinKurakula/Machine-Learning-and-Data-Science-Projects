# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 23:04:55 2024

@author: NITIN
"""
import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('D:/ML Projects/Diabetes Prediction/trained_model.sav', 'rb'))

def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    
def main():
    #title for web app
    st.title("Diabetes Prediction Web Application")
    
    #inputs for users
    
    Pregnancies = st.text_input("Number of pregnancies")
    
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure value")
    SkinThickness = st.text_input("Skin thickness value")
    Insulin = st.text_input("Insulin level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes prediction function value")
    Age = st.text_input("Age of person")
    
    #code for prediction
    
    diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()