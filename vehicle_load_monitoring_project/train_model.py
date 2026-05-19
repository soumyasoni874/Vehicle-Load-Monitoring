import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# Load Dataset
# =========================

dataset_path = "data/cleaned_logistics_dataset.csv"

if not os.path.exists(dataset_path):
    print(f"Dataset not found: {dataset_path}")
    exit()

df = pd.read_csv(dataset_path)

print("Dataset Loaded Successfully!")
print(df.head())

# =========================
# Check Required Columns
# =========================

required_columns = [
    'Usage_Hours',
    'Load_Capacity',
    'Actual_Load',
    'Engine_Temperature',
    'Fuel_Consumption',
    'Vibration_Levels',
    'Failure_History',
    'Predictive_Score',
    'Vehicle_Type',
    'Route_Info',
    'Weather_Conditions',
    'Road_Conditions',
    'Brake_Condition'
]

missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    print("Missing Columns:", missing_cols)
    exit()

# =========================
# Feature Engineering
# =========================

df['Overloaded'] = np.where(
    df['Actual_Load'] > df['Load_Capacity'],
    1,
    0
)

# =========================
# Encode Categorical Columns
# =========================

categorical_cols = [
    'Vehicle_Type',
    'Route_Info',
    'Weather_Conditions',
    'Road_Conditions',
    'Brake_Condition'
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# =========================
# Features and Target
# =========================

features = required_columns
target = 'Overloaded'

X = df[features]
y = df[target]

# =========================
# Split Data
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Train Model
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# Predictions
# =========================

preds = model.predict(X_test)

# =========================
# Evaluation
# =========================

accuracy = accuracy_score(y_test, preds)

print("\n=========================")
print("Model Accuracy:", accuracy)
print("=========================\n")

print("Classification Report:\n")
print(classification_report(y_test, preds))

# =========================
# Save Model
# =========================

os.makedirs("models", exist_ok=True)

model_path = "models/vehicle_load_model.pkl"

joblib.dump(model, model_path)

print(f"\nModel saved successfully at: {model_path}")