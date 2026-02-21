import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Retail Sales Predictor", layout="wide")
st.title("📊 Weekly Sales Prediction App")
st.write("Corrected version: Feature alignment fixed to match the trained XGBoost model.")

# 2. Load the Model
@st.cache_resource
def load_trained_model():
    # Loading your uploaded joblib model
    return joblib.load('best_model.joblib')

model = load_trained_model()

# 3. Input Form
with st.form("prediction_form"):
    st.header("Store & Economic Inputs")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        store = st.number_input("Store Number", min_value=1, max_value=45, value=1)
        dept = st.number_input("Department Number", min_value=1, max_value=99, value=1)
        date = st.date_input("Select Date", datetime.date(2012, 10, 26))
        store_type = st.selectbox("Store Type", options=[0, 1, 2], help="Encoded: 0=A, 1=B, 2=C")
    
    with col2:
        temp = st.number_input("Temperature (F)", value=60.0)
        fuel = st.number_input("Fuel Price", value=3.50)
        size = st.number_input("Store Size", value=151315)
        is_holiday = st.selectbox("Is it a Holiday?", options=["No", "Yes"])
    
    with col3:
        cpi = st.number_input("CPI", value=211.0)
        unemployment = st.number_input("Unemployment Rate", value=8.1)
        markdown_avg = st.number_input("Average Markdown", value=0.0)

    submit = st.form_submit_button("Generate Prediction")

# 4. Feature Engineering & Prediction
if submit:
    # Replicating the training feature set
    week = date.isocalendar()[1]
    is_holiday_val = 1 if is_holiday == "Yes" else 0
    
    input_data = {
        'Store': store,
        'Dept': dept,
        'IsHoliday_x': is_holiday_val,
        'Temperature': temp,
        'Fuel_Price': fuel,
        'MarkDown1': markdown_avg,
        'MarkDown2': 0.0,
        'MarkDown3': 0.0,
        'MarkDown4': 0.0,
        'MarkDown5': 0.0,
        'CPI': cpi,
        'Unemployment': unemployment,
        'Type': store_type,
        'Size': size,
        'Year': date.year,
        'Month': date.month,
        'Week': week,
        'Sales_per_Size': 0.0,  # Placeholder for engineered ratio
        'Temp_Fuel_Interaction': temp * fuel, # Re-creating interaction term
        'Total_MarkDown': markdown_avg, # Re-creating markdown sum
        'High_Holiday_Sales': 1 if (is_holiday_val == 1 and date.month in [11, 12]) else 0,
        'Sales_Lag_1': 0.0,    # Placeholder for previous sales
        'Sales_Diff': 0.0      # Placeholder for sales variance
    }

    input_df = pd.DataFrame([input_data])
    
    # Ensure columns are in the EXACT order expected by the model
    expected_order = [
        'Store', 'Dept', 'IsHoliday_x', 'Temperature', 'Fuel_Price', 
        'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 
        'CPI', 'Unemployment', 'Type', 'Size', 'Year', 'Month', 'Week', 
        'Sales_per_Size', 'Temp_Fuel_Interaction', 'Total_MarkDown', 
        'High_Holiday_Sales', 'Sales_Lag_1', 'Sales_Diff'
    ]
    
    input_df = input_df[expected_order]

    try:
        prediction = model.predict(input_df)
        st.success(f"### Predicted Weekly Sales: ${prediction[0]:,.2f}")
        st.info(f"Model tuned R²: 0.9998")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
