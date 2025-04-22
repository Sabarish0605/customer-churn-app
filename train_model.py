import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load your data
df = pd.read_csv('customer_data.csv')

# --- Feature Engineering (must match your app.py) ---
def prepare_features(df):
    df['InternetType'] = df['InternetType'].fillna('No Internet')
    service_columns = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 
                      'OnlineBackup', 'DeviceProtectionPlan', 'PremiumTechSupport', 
                      'StreamingTV', 'StreamingMovies', 'StreamingMusic', 'UnlimitedData']
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
target = 'Churn'  # Make sure your CSV has this column (1 for churn, 0 for not)

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

# Print feature order for debugging
print("Scaler feature names:", X.columns.tolist())