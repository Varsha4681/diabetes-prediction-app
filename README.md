# 🩺 Diabetes Prediction App

This is a simple web app built using **Streamlit** that predicts the risk of diabetes using machine learning models — **Logistic Regression** and **XGBoost**.

## 📌 Features

- User inputs for health parameters like Glucose, BMI, Age, etc.
- Two ML models for prediction:
  - Logistic Regression
  - XGBoost Classifier
- Displays the prediction result in an easy-to-understand format
- Simple and clean UI with Streamlit

## 🧠 Machine Learning Models

- Trained on the [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Models are saved as `.pkl` files and loaded using `joblib`

## 🛠️ How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/Varsha4681/diabetes-prediction-app.git
   cd diabetes-prediction-app
