import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Set page config
st.set_page_config(
    page_title="Smart Meter Prediction",
    page_icon="⚡"
)

# Title
st.title("Smart Meter Energy Prediction")
st.write("This app predicts future energy consumption.")

# Check if model exists
try:
    model = sm.load('sarimax_model2.pkl')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.write("Please ensure the model file 'sarimax_model.pkl' is in the same directory as the app.")
    st.stop()

# Basic inputs
st.sidebar.header("Prediction Settings")
prediction_date = st.sidebar.date_input(
    "Select prediction date",
    datetime.now().date()
)

# Weather inputs
temperature = st.sidebar.slider("Temperature (°C)", -10.0, 40.0, 20.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 30.0, 5.0)

# Simple prediction button
if st.sidebar.button("Predict"):
    try:
        # Basic weather cluster calculation
        features = np.array([[temperature, humidity, wind_speed]])
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=3, max_iter=600)
        weather_cluster = kmeans.fit_predict(features_scaled)[0]
        
        # Prepare prediction data
        exog_data = pd.DataFrame({
            'weather_cluster': [weather_cluster],
            'holiday_ind': [0]  # Default to non-holiday
        }, index=[prediction_date])
        
        # Add constant term
        exog = sm.add_constant(exog_data)
        
        # Make prediction
        prediction = model.predict(start=0, end=0, exog=exog)[0]
        
        # Display results
        st.header("Prediction Results")
        st.write(f"Predicted energy consumption: {prediction:.2f} units")
        st.write(f"Weather cluster: {weather_cluster}")
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Add information about the model
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("This app uses a SARIMAX model to predict energy consumption based on weather conditions.")