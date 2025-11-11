import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Diabetes Prediction App")
st.write("Enter the details below to check diabetes risk.")

# Input fields
preg = st.number_input("Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose", 0, 300, 0)
bp = st.number_input("Blood Pressure", 0, 200, 0)
skin = st.number_input("Skin Thickness", 0, 100, 0)
insulin = st.number_input("Insulin", 0, 900, 0)
bmi = st.number_input("BMI", 0.0, 70.0, 0.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.0)
age = st.number_input("Age", 1, 120, 25)

# Predict button
if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.error("❗ High chance of Diabetes")
    else:
        st.success("✅ No Diabetes Detected")
