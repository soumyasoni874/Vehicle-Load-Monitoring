
import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("models/vehicle_load_model.pkl")

st.set_page_config(page_title="Vehicle Load Monitoring", layout="wide")

st.title("🚚 Vehicle Load Monitoring Dashboard")

st.sidebar.header("Input Vehicle Data")

usage_hours = st.sidebar.slider("Usage Hours", 0, 20000, 5000)
load_capacity = st.sidebar.number_input("Load Capacity", value=20.0)
actual_load = st.sidebar.number_input("Actual Load", value=15.0)
engine_temp = st.sidebar.slider("Engine Temperature", 50, 150, 90)
fuel_consumption = st.sidebar.slider("Fuel Consumption", 1, 50, 10)
vibration = st.sidebar.slider("Vibration Levels", 0, 100, 20)
failure_history = st.sidebar.slider("Failure History", 0, 10, 1)
predictive_score = st.sidebar.slider("Predictive Score", 0.0, 1.0, 0.3)

vehicle_type = st.sidebar.selectbox("Vehicle Type", [0,1,2])
route_info = st.sidebar.selectbox("Route Info", [0,1,2])
weather = st.sidebar.selectbox("Weather", [0,1,2])
road = st.sidebar.selectbox("Road Condition", [0,1,2])
brake = st.sidebar.selectbox("Brake Condition", [0,1,2])

input_data = np.array([[
    usage_hours,
    load_capacity,
    actual_load,
    engine_temp,
    fuel_consumption,
    vibration,
    failure_history,
    predictive_score,
    vehicle_type,
    route_info,
    weather,
    road,
    brake
]])

prediction = model.predict(input_data)[0]

st.subheader("Prediction Result")

if prediction == 1:
    st.error("⚠ Vehicle is OVERLOADED")
else:
    st.success("✅ Vehicle Load is SAFE")

st.subheader("Load Analysis")

load_percentage = (actual_load / load_capacity) * 100

st.metric("Load Utilization", f"{load_percentage:.2f}%")

if load_percentage > 100:
    st.warning("Critical overload detected!")
elif load_percentage > 85:
    st.info("Vehicle nearing capacity.")
else:
    st.success("Normal operating range.")
