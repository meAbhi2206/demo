import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np

kmeans_model_path = "KMeans_Model.joblib"
supervised_model_paths = [
    "ANN.joblib",
    "Decision_Tree.joblib",
    "Logistic_Regression.joblib",
    "Random_Forest.joblib",
    "XGBoost.joblib"
]
label_encoders = joblib.load('label_encoders.joblib')
scaler = joblib.load('standard_scaler.joblib')


st.title("Credit Risk Prediction")

# First row of input fields
col1, col2, col3 = st.columns(3)
with col1:
    age = st.text_input("Age", value="35")
with col2:
    credit_amount = st.text_input("Credit amount", value="2000")
with col3:
    duration = st.text_input("Duration (in months)", value="12")

col4, col5, col6 = st.columns(3)
with col4:
    sex = st.selectbox("Sex", ['male', 'female'])
with col5:
    job = st.selectbox("Job", [0, 1, 2, 3])
with col6:
    housing = st.selectbox("Housing", ['own', 'rent', 'free'])

col7, col8, col9 = st.columns(3)
with col7:
    saving_accounts = st.selectbox("Saving accounts", ['none', 'little', 'moderate', 'quite rich', 'rich'])
with col8:
    checking_account = st.selectbox("Checking account", ['none', 'little', 'moderate', 'rich'])
with col9:
    purpose = st.selectbox("Purpose", ['car', 'furniture/equipment', 'radio/TV', 'domestic appliances', 'repairs', 'education', 'business', 'vacation/others'])

if st.button("Predict Risk"):


    input_dict = {
        'Age': float(age),
        'Sex': label_encoders['Sex'].transform([sex])[0],
        'Job': int(job),  # Use the numeric value directly
        'Housing': label_encoders['Housing'].transform([housing])[0],
        'Saving accounts': label_encoders['Saving accounts'].transform([saving_accounts])[0],
        'Checking account': label_encoders['Checking account'].transform([checking_account])[0],
        'Credit amount': float(credit_amount),
        'Duration': float(duration),
        'Purpose': label_encoders['Purpose'].transform([purpose])[0],
    }

    input_df = pd.DataFrame([input_dict])
    input_df_scaled = scaler.transform(input_df)

    with st.expander("Detailed Information"):
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Unsupervised Models:")
            if os.path.exists(kmeans_model_path):
                kmeans_model = joblib.load(kmeans_model_path)
                kmeans_prediction = kmeans_model.predict(input_df_scaled)[0]
                kmeans_label = "Good Risk" if kmeans_prediction == 1 else "Bad Risk"
                st.write(f"KMeans Model: {kmeans_label}")
            else:
                st.write("KMeans model file not found!")

        with col_right:
            st.subheader("Supervised Models:")
            model_predictions = []
            for model_path in supervised_model_paths:
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    prediction = model.predict(input_df_scaled)[0]
                    model_predictions.append((model_path, prediction))
                else:
                    st.write(f"Model file {model_path} not found!")

            predictions = [prediction for _, prediction in model_predictions]
            if predictions:
                most_common_prediction = np.bincount(predictions).argmax()
                best_label = "Good Risk" if most_common_prediction == 1 else "Bad Risk"


                for model_path, prediction in model_predictions:
                    label = "Good Risk" if prediction == 1 else "Bad Risk"
                    st.write(f"**{os.path.basename(model_path).replace('_', ' ').replace('.joblib', '')}**: {label}")

            else:
                st.write("No valid supervised model predictions to display.")

    if predictions:
        st.subheader("Prediction:")
        st.write(f"**{best_label}**")
