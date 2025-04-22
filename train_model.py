import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load the Telco Customer Churn dataset
df = pd.read_csv("C:\\Users\\sabar\\Downloads\\archive\\TelcoCustomerChurn.csv")

print("Columns in the dataset:", list(df.columns))  # <--- Add this line

# --- Feature Engineering (must match your app.py) ---
def prepare_features(df):
    df['InternetType'] = df['InternetType'].fillna('No Internet')
    service_columns = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 
                      'OnlineBackup', 'DeviceProtectionPlan', 'PremiumTechSupport', 
                      'StreamingTV', 'StreamingMovies', 'StreamingMusic', 'UnlimitedData']
    for col in service_columns:
        if col not in df.columns:
            df[col] = 'No'
    df['TotalServices'] = df[service_columns].apply(
        lambda x: x.map({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0}).sum(), axis=1)
    df['ChargePerService'] = df['MonthlyCharge'] / (df['TotalServices'] + 1)
    return df

df = prepare_features(df)

# --- Select Features and Target ---
features = [
    'Age', 'TenureinMonths', 'MonthlyCharge', 'TotalCharges',
    'ChargePerService', 'Contract', 'InternetType', 'OnlineSecurity',
    'PaymentMethod', 'AvgMonthlyLongDistanceCharges', 'AvgMonthlyGBDownload',
    'TotalServices', 'SatisfactionScore', 'NumberofDependents',
    'PaperlessBilling', 'MultipleLines', 'StreamingTV', 'StreamingMovies'
]

# Map target column to binary if needed
if 'ChurnLabel' in df.columns:
    if df['ChurnLabel'].dtype == 'O':
        df['ChurnLabel'] = df['ChurnLabel'].map({'Yes': 1, 'No': 0})
    target = 'ChurnLabel'
else:
    raise ValueError("The dataset must have a 'ChurnLabel' column.")

X = pd.get_dummies(df[features])
y = df[target]

# --- Fit Scaler ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train Model ---
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# --- Save Model and Scaler ---
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")
print("Scaler feature names:", X.columns.tolist())