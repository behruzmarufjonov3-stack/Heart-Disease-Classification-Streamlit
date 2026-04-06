import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('models/heart_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("Heart Disease Diagnostic Tool")
st.write("This tool uses Machine Learning to assist in intelligent healthcare decision-making.")

# Create columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=40)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

with col2:
    restecg = st.slider("Resting ECG Results (0-2)", 0, 2, 0)
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
    slope = st.slider("Slope of Peak Exercise ST", 0, 2, 1)
    ca = st.slider("Number of Major Vessels (0-4)", 0, 4, 0)
    thal = st.slider("Thalassemia (0-3)", 0, 3, 1)

if st.button("Predict Heart Health Status"):
    # Arrange features in the exact same order as the CSV
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Scale the input
    scaled_features = scaler.transform(features)
    
    # Prediction
    prediction = model.predict(scaled_features)
    
    if prediction[0] == 1:
        st.error("HIGH RISK: The model predicts a high probability of heart disease.")
    else:
        st.success("LOW RISK: The model predicts a low probability of heart disease.")
