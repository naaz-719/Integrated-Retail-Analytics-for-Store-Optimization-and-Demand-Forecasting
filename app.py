import streamlit as st
import pandas as pd
import joblib
import datetime

# 1. Page Setup
st.set_page_config(page_title="Retail Sales Predictor", layout="wide")
st.title("📊 Weekly Sales Prediction App")
st.write("This application predicts weekly retail sales based on historical trends and economic factors.")

# 2. Load the Model 
# The model 'best_model.joblib' was created using a tuned XGBoost regressor
@st.cache_resource
def load_trained_model():
    return joblib.load('best_model.joblib')

model = load_trained_model()

# 3. User Input Form
with st.form("prediction_form"):
    st.header("Input Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        store = st.number_input("Store Number", min_value=1, max_value=45, value=1)
        dept = st.number_input("Department Number", min_value=1, max_value=99, value=1)
        date = st.date_input("Select Date", datetime.date(2012, 10, 26))
    
    with col2:
        temp = st.number_input("Temperature (F)", value=60.0)
        fuel = st.number_input("Fuel Price", value=3.50)
        is_holiday = st.selectbox("Is it a Holiday?", options=["No", "Yes"])
    
    with col3:
        cpi = st.number_input("Consumer Price Index (CPI)", value=215.0)
        unemployment = st.number_input("Unemployment Rate", value=7.5)
        # Store size is a key driver for sales in this dataset
        size = st.number_input("Store Size", value=150000)

    submit = st.form_submit_button("Generate Prediction")

# 4. Data Processing and Prediction
if submit:
    # Feature Engineering (Replicating steps from the notebook)
    # Extracting date components as the model relies on seasonality
    input_df = pd.DataFrame({
        'Store': [store],
        'Dept': [dept],
        'IsHoliday': [1 if is_holiday == "Yes" else 0],
        'Temperature': [temp],
        'Fuel_Price': [fuel],
        'CPI': [cpi],
        'Unemployment': [unemployment],
        'Size': [size],
        'Month': [date.month],
        'Year': [date.year],
        'Day': [date.day]
    })

    try:
        prediction = model.predict(input_df)
        st.success(f"### Estimated Weekly Sales: ${prediction[0]:,.2f}")
        
        # Display insights based on project summary
        st.info("**Analysis Insight:** Your model uses economic indicators and store size to drive this prediction.")
    except Exception as e:
        st.error(f"Error: {e}")
        st.warning("Ensure your model features match the columns: Store, Dept, IsHoliday, Temperature, Fuel_Price, CPI, Unemployment, Size, Month, Year, Day.")
