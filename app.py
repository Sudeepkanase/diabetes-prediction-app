import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Page Title
st.title("🩺 Diabetes Risk Prediction App")
st.markdown("""
This app predicts the likelihood of diabetes using a Logistic Regression model trained on medical data.
Enter your health parameters below to check your risk level.
""")

st.markdown("---")

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", 0, 20, 0)
    glucose = st.number_input("Glucose Level", 0, 300, 0)
    bp = st.number_input("Blood Pressure", 0, 200, 0)
    skin = st.number_input("Skin Thickness", 0, 100, 0)

with col2:
    insulin = st.number_input("Insulin Level", 0, 900, 0)
    bmi = st.number_input("BMI", 0.0, 70.0, 0.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.0)
    age = st.number_input("Age", 1, 120, 25)

st.markdown("---")

# Predict button
if st.button("🔍 Predict Diabetes Risk"):
    # Prepare and scale input
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1] * 100  # % score

    st.markdown("### 🧾 Prediction Result")

    if prediction[0] == 1:
        st.error(f"❗ High chance of Diabetes")
        st.write(f"**Estimated Risk Probability: `{probability:.2f}%`**")
        
        st.markdown("""
        ### Interpretation
        Your medical indicators align with individuals diagnosed with diabetes.

        **Possible contributing factors:**
        - High glucose or BMI  
        - Elevated insulin resistance  
        - Higher age or genetic risk (DPF)  
        - Other metabolic signs

        ### Recommendation
        This model suggests a *higher risk*.  
        Please consult a healthcare professional for accurate medical evaluation.
        """)
    
    else:
        st.success("✅ No Diabetes Detected")
        st.write(f"**Estimated Risk Probability: `{probability:.2f}%`**")

        st.markdown("""
        ### Interpretation
        Your values resemble individuals with low diabetes risk.

        **Positive signs from your inputs:**
        - Normal glucose range  
        - Healthy BMI and blood pressure  
        - Low hereditary risk indicators  

        ### Note
        This prediction is not a medical diagnosis.  
        Maintain good health habits and periodic checkups.
        """)

st.markdown("---")
st.caption("Developed for learning and demonstration purposes.")
