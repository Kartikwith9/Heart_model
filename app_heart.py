import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load('model1.pkl')
scaler = joblib.load('scaler_17.pkl')

st.title("❤️ Heart Disease Prediction Website")

# User inputs
Age = st.number_input("Enter Age", min_value=0, max_value=120, value=25, step=1)
Sex = st.radio("Gender", ('M', 'F'))
RestingBP = st.number_input("Resting BP", min_value=50, max_value=250, value=120)
Cholesterol = st.number_input("Cholesterol", min_value=50, max_value=500, value=200)
MaxHR = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=150)
ExerciseAngina = st.radio("Exercise Angina", ('Y','N'))
Oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0)

# Example for one-hot features (Yes=1, No=0)
ChestPainType_ASY = st.radio("ChestPainType_ASY", ('Y','N'))
ChestPainType_ATA = st.radio("ChestPainType_ATA", ('Y','N'))
ChestPainType_NAP = st.radio("ChestPainType_NAP", ('Y','N'))
ChestPainType_TA = st.radio("ChestPainType_TA", ('Y','N'))
ST_Slope_Down = st.radio("ST_Slope_Down", ('Y','N'))
ST_Slope_Flat = st.radio("ST_Slope_Flat", ('Y','N'))
ST_Slope_Up = st.radio("ST_Slope_Up", ('Y','N'))
RestingECG_LVH = st.radio("RestingECG_LVH", ('Y','N'))
RestingECG_Normal = st.radio("RestingECG_Normal", ('Y','N'))
RestingECG_ST = st.radio("RestingECG_ST", ('Y','N'))

# Convert categorical inputs to 0/1
Sex = 1 if Sex == 'M' else 0
ExerciseAngina = 1 if ExerciseAngina == 'Y' else 0
ChestPainType_ASY = 1 if ChestPainType_ASY == 'Y' else 0
ChestPainType_ATA = 1 if ChestPainType_ATA == 'Y' else 0
ChestPainType_NAP = 1 if ChestPainType_NAP == 'Y' else 0
ChestPainType_TA = 1 if ChestPainType_TA == 'Y' else 0
ST_Slope_Down = 1 if ST_Slope_Down == 'Y' else 0
ST_Slope_Flat = 1 if ST_Slope_Flat == 'Y' else 0
ST_Slope_Up = 1 if ST_Slope_Up == 'Y' else 0
RestingECG_LVH = 1 if RestingECG_LVH == 'Y' else 0
RestingECG_Normal = 1 if RestingECG_Normal == 'Y' else 0
RestingECG_ST = 1 if RestingECG_ST == 'Y' else 0

if st.button("Predict"):
    # Make input array
    data = np.array([[Age, Sex, RestingBP, Cholesterol, MaxHR, ExerciseAngina,
                      Oldpeak, ChestPainType_ASY, ChestPainType_ATA,
                      ChestPainType_NAP, ChestPainType_TA, ST_Slope_Down,
                      ST_Slope_Flat, ST_Slope_Up, RestingECG_LVH,
                      RestingECG_Normal, RestingECG_ST]])
    
    # Scale input
    scaled_input = scaler.transform(data)
    
    # Predict
    prediction = model.predict(scaled_input)[0]
    
    result = "You have heart disease ❤️" if prediction == 1 else "You don't have heart disease ✅"
    st.success(result)
