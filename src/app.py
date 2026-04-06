import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
# Ensure these files are in your 'models' folder on GitHub
model = joblib.load('models/heart_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("Heart Disease Diagnostic Tool")
st.write("This tool uses Machine Learning to assist in intelligent healthcare decision-making.")

# Create columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=40)
    sex_choice = st.selectbox("Sex", options=["Male", "Female"])
    sex = 1 if sex_choice == "Male" else 0
    
    cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    
    fbs_choice = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
    fbs = 1 if fbs_choice == "Yes" else 0

with col2:
    restecg = st.slider("Resting ECG Results (0-2)", 0, 2, 0)
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    
    exang_choice = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])
    exang = 1 if exang_choice == "Yes" else 0
    
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
    
    # BASED ON OUR TEST: 
    # Category 0 = Heart Disease (High Risk)
    # Category 1 = Healthy (Low Risk)
    
    if prediction[0] == 0:
        st.error("HIGH RISK: The model predicts a high probability of heart disease.")
    else:
        st.success("LOW RISK: The model predicts a low probability of heart disease.")

    # (Optional) Remove this once you are sure everything is perfect
    st.caption(f"Technical Output: Classified as Category {prediction[0]}")
