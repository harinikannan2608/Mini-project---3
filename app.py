import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# --------------------------
# Load the trained model
# --------------------------
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --------------------------
# Helper: Haversine distance
# --------------------------
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    km = 6371 * c
    return km

# --------------------------
# Streamlit UI
# --------------------------
st.title("NYC Taxi Fare Prediction")

st.markdown("""
🚕 Predict the **total fare amount** for a taxi trip in NYC  
based on your trip details.  
""")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        pickup_lat = st.number_input("Pickup Latitude", value=40.761432)
        pickup_lon = st.number_input("Pickup Longitude", value=-73.979815)
        dropoff_lat = st.number_input("Dropoff Latitude", value=40.641311)
        dropoff_lon = st.number_input("Dropoff Longitude", value=-73.778139)

    with col2:
        passenger_count = st.slider("Passenger Count", 1, 6, value=1)
        pickup_date = st.date_input("Pickup Date", value=datetime.now().date())
        pickup_time = st.time_input("Pickup Time", value=datetime.now().time())

    submitted = st.form_submit_button("Predict Fare")

if submitted:
    # Derived features
    pickup_dt = datetime.combine(pickup_date, pickup_time)
    pickup_hour = pickup_dt.hour
    pickup_day = pickup_dt.strftime("%A")
    is_weekend = 1 if pickup_day in ["Saturday", "Sunday"] else 0
    is_night = 1 if (pickup_hour >= 22 or pickup_hour < 5) else 0

    trip_distance_km = haversine(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
    trip_duration_min = max((trip_distance_km / 20) * 60, 1)  # avoid 0
    pickup_day_enc = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 
                      'Friday':4, 'Saturday':5, 'Sunday':6}[pickup_day]

    input_df = pd.DataFrame([{
        'trip_distance_km': trip_distance_km,
        'trip_duration_min': trip_duration_min,
        'pickup_hour': pickup_hour,
        'is_weekend': is_weekend,
        'is_night': is_night,
        'fare_per_km': 0,  # not known yet
        'fare_per_min': 0, # not known yet
        'pickup_day_enc': pickup_day_enc
    }])

    # Prediction
    pred_fare = model.predict(input_df)[0]
    pred_fare = round(pred_fare, 2)

    st.success(f"💰 Estimated Total Fare: **${pred_fare}**")

    st.markdown("### Trip Features")
    st.dataframe(input_df.T)

