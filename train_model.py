import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle

# Load data
df = pd.read_csv("data/processed_train.csv")

# Features
features = [
    'sensor_measurement_2', 'sensor_measurement_3', 'sensor_measurement_4',
    'sensor_measurement_7', 'sensor_measurement_8', 'sensor_measurement_11',
    'sensor_measurement_12', 'sensor_measurement_13', 'sensor_measurement_15',
    'sensor_measurement_17', 'sensor_measurement_20', 'sensor_measurement_21'
]
X = df[features]
y = df['RUL']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_scaled, y_train)

# Save model & scaler
with open("predictive_model.pkl", "wb") as f:
    pickle.dump(regressor, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Evaluate
y_pred = regressor.predict(X_test_scaled)
print("Sample predictions:", y_pred[:10])
print("Sample actual:", y_test[:10].values)

print("✅ Model trained and saved!")