import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the trained model and scaler
model_path = './model/classification_model.h5'
scaler_path = './model/scaler.pkl'
model = load_model(model_path)
scaler = joblib.load(scaler_path)

# Define project details
st.title("Customer Churn Prediction App")

st.header("Project Objective")
st.write("The goal of this project is to predict whether a customer will churn based on their demographic, account, and transaction data.")

# Load and display dataset preview
df = pd.read_csv('./data/processed_data.csv')
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Input form for user features
st.subheader("Input Customer Features")
customer_id = st.text_input("Customer ID")
surname = st.text_input("Surname")
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance (in $)", value=50000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Does the Customer have a Credit Card?", ["No", "Yes"])
is_active_member = st.selectbox("Is the Customer an Active Member?", ["No", "Yes"])
estimated_salary = st.number_input("Estimated Salary (in $)", value=70000.0)

# Convert categorical inputs to numerical format
geography_france = 1 if geography == "France" else 0
geography_spain = 1 if geography == "Spain" else 0
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0
gender_male = 1 if gender == "Male" else 0

# Organize input features for prediction
input_features = np.array([[
    credit_score, age, tenure, balance, num_of_products,
    has_cr_card, is_active_member, estimated_salary,
    geography_france, geography_spain, gender_male
]])

# Scale the input features
scaled_features = scaler.transform(input_features)

# Perform prediction
if st.button("Predict Customer Churn"):
    prediction = model.predict(scaled_features)
    churn_probability = prediction[0][0]
    result = "Customer is likely to churn" if churn_probability > 0.5 else "Customer is not likely to churn"

    # Display prediction results
    st.subheader("Prediction Result")
    st.write(f"Prediction: {result}")
    st.write(f"Churn Probability: {churn_probability:.2f}")

