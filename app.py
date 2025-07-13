import streamlit as st
import numpy as np
import joblib

model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳")

st.title("💳 Credit Risk Prediction")
st.markdown("Predict whether a person will default on a salary advance loan.")

loan = st.number_input("Loan Amount", min_value=0)
mortdue = st.number_input("Mortgage Due", min_value=0)
value = st.number_input("Home Value", min_value=0)
clage = st.number_input("Credit Age (months)", min_value=0.0)
clno = st.slider("Number of Credit Lines", 0, 50, 10)
debtinc = st.slider("Debt-to-Income Ratio (%)", 0.0, 100.0, 20.0)

if st.button("Predict Default Risk"):
    input_data = np.array([[loan, mortdue, value, clage, clno, debtinc]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("❌ High Risk: Likely to Default")
    else:
        st.success("✅ Low Risk: Likely to Repay")
