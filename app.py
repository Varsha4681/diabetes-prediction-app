import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your saved model
logreg_model = joblib.load('logreg_pipeline.pkl')
xgb_model = joblib.load('xgb_pipeline.pkl')

st.title("Diabetes Risk Prediction App")

st.markdown("### Please fill in the following health information:")

with st.form("diabetes_form"):
    weight = st.number_input("Weight (in kg)", min_value=30.0, max_value=200.0, value=70.0)
    height_cm = st.number_input("Height (in cm)", min_value=100.0, max_value=250.0, value=170.0)

    height_m = height_cm / 100
    BMI = round(weight / (height_m ** 2), 2)
    st.markdown(f"**Calculated BMI:** `{BMI}`")

    GenHlth = st.slider("General Health (1 = Excellent, 5 = Poor)", 1, 5, 3)
    MentHlth = st.slider("Mental Health (bad days in last 30)", 0, 30, 0)
    PhysHlth = st.slider("Physical Health (bad days in last 30)", 0, 30, 0)
    Age = st.slider("Age Category (1 = 18-24, 13 = 80+)", 1, 13, 5)
    Education = st.slider("Education Level (1 = Never attended, 6 = College grad)", 1, 6, 4)
    Income = st.slider("Income Level (1 = Less than $10,000, 8 = $75,000+)", 1, 8, 5)

    HighBP = st.selectbox("High Blood Pressure", [0, 1])
    HighChol = st.selectbox("High Cholesterol", [0, 1])
    CholCheck = st.selectbox("Had cholesterol check in last 5 years", [0, 1])
    Smoker = st.selectbox("Smoker", [0, 1])
    Stroke = st.selectbox("Ever had a stroke", [0, 1])
    HeartDiseaseorAttack = st.selectbox("Heart Disease or Heart Attack", [0, 1])
    PhysActivity = st.selectbox("Physical Activity in past 30 days", [0, 1])
    Fruits = st.selectbox("Eats fruits 1 or more times/day", [0, 1])
    Veggies = st.selectbox("Eats vegetables 1 or more times/day", [0, 1])
    HvyAlcoholConsump = st.selectbox("Heavy Alcohol Consumption", [0, 1])
    AnyHealthcare = st.selectbox("Has any form of health coverage", [0, 1])
    NoDocbcCost = st.selectbox("Couldn’t see doctor due to cost", [0, 1])
    DiffWalk = st.selectbox("Difficulty Walking or Climbing Stairs", [0, 1])
    Sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])

    submitted = st.form_submit_button("Predict Diabetes Risk")

# Prediction
if submitted:
    input_dict = {
        'BMI': BMI, 'GenHlth': GenHlth, 'MentHlth': MentHlth, 'PhysHlth': PhysHlth,
        'Age': Age, 'Education': Education, 'Income': Income,
        'HighBP': HighBP, 'HighChol': HighChol, 'CholCheck': CholCheck, 'Smoker': Smoker,
        'Stroke': Stroke, 'HeartDiseaseorAttack': HeartDiseaseorAttack,
        'PhysActivity': PhysActivity, 'Fruits': Fruits, 'Veggies': Veggies,
        'HvyAlcoholConsump': HvyAlcoholConsump, 'AnyHealthcare': AnyHealthcare,
        'NoDocbcCost': NoDocbcCost, 'DiffWalk': DiffWalk, 'Sex': Sex
    }

    input_df = pd.DataFrame([input_dict])

    xgb_proba = xgb_model.predict_proba(input_df)[:, 1]
    logreg_proba = logreg_model.predict_proba(input_df)[:, 1]

    xgb_pred = (xgb_proba >= 0.4).astype(int)
    logreg_pred = (logreg_proba >= 0.5).astype(int)

    final_prediction = int(np.logical_and(xgb_pred == 1, logreg_pred == 1))

    st.subheader("🔍 Prediction Result")
    if final_prediction == 1:
        st.error("⚠️ High Risk of Diabetes. Please consult a doctor.")
    else:
        st.success("✅ Low Risk of Diabetes. Stay healthy!")

    st.markdown(f"**XGBoost Probability:** `{xgb_proba[0]:.2f}`")
    st.markdown(f"**Logistic Regression Probability:** `{logreg_proba[0]:.2f}`")