import os, pickle, warnings
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

st.title("NYC Taxi Fare Prediction App 🚖")

MODEL_FILE = "best_model.pkl"
SCALER_FILE = "scaler.pkl"
FEATURES_FILE = "features.pkl"

# ----------------------------
# Step 1: Train Model (if needed)
# ----------------------------
@st.cache_resource
def train_model():
    st.info("Training model for the first time... please wait ⏳")
    url = "https://drive.google.com/uc?id=1VUb9ucTsroGDBOPcwpOfXwzDi-rd4wqQ"
    df = pd.read_csv(url)

    # Clean & engineer
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    df['trip_distance'] = df['trip_distance'].replace(0, np.nan)
    df.dropna(subset=['trip_distance'], inplace=True)

    features = [
        'VendorID', 'RatecodeID', 'payment_type',
        'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
        'passenger_count', 'trip_distance', 'fare_amount', 'extra', 'mta_tax',
        'tip_amount', 'tolls_amount', 'improvement_surcharge'
    ]
    target = 'total_amount'

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Save artifacts
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    with open(FEATURES_FILE, "wb") as f:
        pickle.dump(features, f)

    st.success(f"Model trained. R²: {r2:.3f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    return model, scaler, features


# ----------------------------
# Step 2: Load Model & Artifacts
# ----------------------------
if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(FEATURES_FILE):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURES_FILE, "rb") as f:
        features = pickle.load(f)
else:
    model, scaler, features = train_model()


# ----------------------------
# Step 3: Prediction UI
# ----------------------------

st.header("Enter Trip Details")

vendor_id = st.selectbox("Vendor ID", [1, 2])
ratecode_id = st.selectbox("Rate Code ID", [1, 2, 3, 4, 5, 6])
payment_type = st.selectbox("Payment Type", [1, 2, 3, 4, 5, 6])

pickup_latitude = st.number_input("Pickup Latitude", value=40.7589)
pickup_longitude = st.number_input("Pickup Longitude", value=-73.9851)
dropoff_latitude = st.number_input("Dropoff Latitude", value=40.7612)
dropoff_longitude = st.number_input("Dropoff Longitude", value=-73.9776)

passenger_count = st.slider("Passenger Count", 1, 6, 1)
trip_distance = st.number_input("Trip Distance (km)", min_value=0.1, value=2.0)

fare_amount = st.number_input("Base Fare ($)", min_value=0.0, value=10.0)
extra = st.number_input("Extra Charges ($)", min_value=0.0, value=0.5)
mta_tax = st.number_input("MTA Tax ($)", min_value=0.0, value=0.5)
tip_amount = st.number_input("Tip Amount ($)", min_value=0.0, value=1.5)
tolls_amount = st.number_input("Tolls Amount ($)", min_value=0.0, value=0.0)
improvement_surcharge = st.number_input("Improvement Surcharge ($)", min_value=0.0, value=0.3)

if st.button("Predict Fare"):
    input_data = pd.DataFrame([[
        vendor_id, ratecode_id, payment_type,
        pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude,
        passenger_count, trip_distance, fare_amount, extra, mta_tax,
        tip_amount, tolls_amount, improvement_surcharge
    ]], columns=features)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"🎯 Predicted Total Fare: ${prediction:.2f}")
