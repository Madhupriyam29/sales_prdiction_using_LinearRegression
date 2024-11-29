import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('C:/Users/Admin/Downloads/salary_prediction_model.pkl')
scaler = joblib.load('C:/Users/Admin/Downloads/scaler.pkl')

# Streamlit Application
st.title('Salary Prediction based on Years of Experience')

# Input: User enters years of experience
years_of_experience = st.number_input('Enter Years of Experience:', min_value=0.0, step=0.1)

# Button to predict salary
if st.button('Predict Salary'):
    # Scale the input data before prediction
    input_data = np.array([[years_of_experience]])
    scaled_input = scaler.transform(input_data)  # Using the same scaler used for training

    # Make the prediction
    predicted_salary = model.predict(scaled_input)

    # Show the predicted salary
    st.write(f"Predicted Salary for {years_of_experience} years of experience is ${predicted_salary[0]:,.2f}")