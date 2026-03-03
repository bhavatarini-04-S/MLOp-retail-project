import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("xgb_model.pkl")

st.title("Retail Purchase Prediction")
st.write("Predict if a customer will purchase next month")

quantity = st.number_input("Quantity", min_value=1)
price = st.number_input("Price", min_value=0.1)

invoicehour = st.slider("Invoice Hour", 0, 23)
invoiceday = st.slider("Invoice Day", 1, 31)

totalprice = quantity * price

if st.button("Predict"):
    features = np.array([[quantity, price, totalprice,
                          invoicehour, invoiceday]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("Customer Likely to Purchase Next Month ✅")
    else:
        st.error("Customer Not Likely to Purchase ❌")