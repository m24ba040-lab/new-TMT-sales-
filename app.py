"""
Streamlit App — Bill Qty Forecast
==================================
Loads the pre-trained XGBoost model + encoders and lets the user
enter order details in a form to get a predicted Bill Qty.

Required files in the SAME repo folder:
    app.py                     (this file)
    bill_qty_xgb_model.pkl     (trained model)
    bill_qty_encoders.pkl      (label encoders)
    requirements.txt
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(page_title="Bill Qty Forecast", page_icon="📦", layout="centered")
st.title("📦 Bill Qty Forecast")
st.write("Enter order details to predict expected Bill Quantity.")

# -------------------------------------------------------------------
# Load model + encoders (cached so it only loads once per session)
# -------------------------------------------------------------------
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load("bill_qty_xgb_model.pkl")
    encoders = joblib.load("bill_qty_encoders.pkl")
    return model, encoders

try:
    model, encoders = load_model_and_encoders()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure `bill_qty_xgb_model.pkl` and "
        "`bill_qty_encoders.pkl` are in the same folder as app.py."
    )
    st.stop()

# -------------------------------------------------------------------
# Column mapping — must match training script
# -------------------------------------------------------------------
COLUMN_MAP = {
    "Customer Name": "Customer Name  FIRM dilip buildcon consider konnect ",
    "Grade":         "3 grades",
    "Size":          "Size",
    "Zone":          "ZONE",
    "Plant":         "PLANT",
}

CATEGORICAL_FEATURES = list(COLUMN_MAP.values())

# -------------------------------------------------------------------
# Build input form
# -------------------------------------------------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        bill_date = st.date_input("Bill Date")
        gross_value = st.number_input("Gross Value (₹)", min_value=0.0, step=1000.0)

        # Dropdowns populated from the categories the model was trained on
        def get_options(col_key):
            col_name = COLUMN_MAP[col_key]
            if col_name in encoders:
                return sorted(encoders[col_name].classes_.tolist())
            return []

        customer = st.selectbox("Customer Name", get_options("Customer Name"))
        grade = st.selectbox("Grade", get_options("Grade"))

    with col2:
        size = st.selectbox("Size", get_options("Size"))
        zone = st.selectbox("Zone", get_options("Zone"))
        plant = st.selectbox("Plant", get_options("Plant"))

    submitted = st.form_submit_button("Predict Bill Qty")

# -------------------------------------------------------------------
# Predict
# -------------------------------------------------------------------
if submitted:
    input_dict = {
        "bill_month": bill_date.month,
        "bill_year": bill_date.year,
        "bill_dayofweek": bill_date.weekday(),
        "bill_quarter": (bill_date.month - 1) // 3 + 1,
        "month_col": bill_date.month,
        "year_col": bill_date.year,
        COLUMN_MAP["Customer Name"]: customer,
        COLUMN_MAP["Grade"]: grade,
        COLUMN_MAP["Size"]: size,
        COLUMN_MAP["Zone"]: zone,
        COLUMN_MAP["Plant"]: plant,
        "Gross Value": gross_value,
    }

    X = pd.DataFrame([input_dict])

    # Encode categoricals using the saved encoders
    for col in CATEGORICAL_FEATURES:
        le = encoders[col]
        X[col] = le.transform(X[col].astype(str))

    # Encode month_col/year_col if they were label-encoded during training
    for col in ["month_col", "year_col"]:
        if col in encoders:
            le = encoders[col]
            known = set(le.classes_)
            X[col] = X[col].astype(str).map(
                lambda v: le.transform([v])[0] if v in known else -1
            )

    X = X.fillna(-1)
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(-1)

    # Match column order used at training time
    X = X[model.get_booster().feature_names]

    prediction = model.predict(X)[0]

    st.success(f"### Predicted Bill Qty: **{prediction:,.2f}**")
