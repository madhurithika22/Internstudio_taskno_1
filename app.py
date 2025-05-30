import streamlit as st
import pandas as pd
import pickle

# Load model & scaler
with open("predictive_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🚀 Predictive Maintenance Dashboard")
st.write("NASA Turbofan FD001 dataset based predictions")

# Input sliders
sensor_features = [
    'sensor_measurement_2', 'sensor_measurement_3', 'sensor_measurement_4',
    'sensor_measurement_7', 'sensor_measurement_8', 'sensor_measurement_11',
    'sensor_measurement_12', 'sensor_measurement_13', 'sensor_measurement_15',
    'sensor_measurement_17', 'sensor_measurement_20', 'sensor_measurement_21'
]

input_data = {}
for sensor in sensor_features:
    input_data[sensor] = st.slider(sensor, 0.0, 100.0, 50.0)

# Predict only when button is pressed
if st.button("🔍 Predict Equipment Health"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    rul_prediction = model.predict(input_scaled)[0]
    st.write(f"🛠️ **Predicted RUL:** {rul_prediction:.2f} cycles")

    if rul_prediction < 50:
        st.error("⚠️ Immediate maintenance required!")
    elif rul_prediction < 100:
        st.warning("⚠️ Maintenance due soon.")
    else:
        st.success("✅ Equipment is healthy.")
else:
    st.info("👆 Adjust sliders and click Predict.")

# Show historical trend for reference
st.subheader("📊 Historical data trends")
df = pd.read_csv("data/processed_train.csv")
st.line_chart(df[['sensor_measurement_2', 'sensor_measurement_3', 'sensor_measurement_4']].head(100))