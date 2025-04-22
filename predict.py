import pandas as pd
import joblib

def prepare_features(df):
    # Handle missing values
    df['InternetType'] = df['InternetType'].fillna('No Internet')
    
    # Feature engineering
    service_columns = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 
                      'OnlineBackup', 'DeviceProtectionPlan', 'PremiumTechSupport', 
                      'StreamingTV', 'StreamingMovies', 'StreamingMusic', 'UnlimitedData']
    
    df['TotalServices'] = df[service_columns].apply(
        lambda x: x.map({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0}).sum(), axis=1)
    
    df['ChargePerService'] = df['MonthlyCharge'] / (df['TotalServices'] + 1)
    
    return df

def predict_customer_churn(customer_data):
    # Load the saved model and scaler
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Prepare features
    features = [
        'Age', 'TenureinMonths', 'MonthlyCharge', 'TotalCharges',
        'ChargePerService', 'Contract', 'InternetType', 'OnlineSecurity',
        'PaymentMethod', 'AvgMonthlyLongDistanceCharges', 'AvgMonthlyGBDownload',
        'TotalServices', 'SatisfactionScore', 'NumberofDependents',
        'PaperlessBilling', 'MultipleLines', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Prepare and transform data
    df = prepare_features(customer_data)
    X = pd.get_dummies(df[features])
    X = X.reindex(columns=scaler.feature_names_in_, fill_value=0)
    X_scaled = scaler.transform(X)
    
    # Make predictions
    churn_prob = model.predict_proba(X_scaled)[:, 1]
    predictions = model.predict(X_scaled)
    
    # Add predictions to the dataframe
    customer_data['Churn_Probability'] = churn_prob
    customer_data['Predicted_Churn'] = predictions
    
    return customer_data

if __name__ == "__main__":
    # Load test data
    test_data_path = "C:\\Users\\sabar\\Desktop\\Customer churn project\\new_customers.csv"
    test_data = pd.read_csv(test_data_path)
    
    # Make predictions
    results = predict_customer_churn(test_data)
    
    # Display results for customers with probability > 0.5
    high_risk = results[results['Churn_Probability'] > 0.5]
    print("\nHigh Risk Customers:")
    print(f"Number of high-risk customers: {len(high_risk)}")
    print("\nSample of high-risk customers:")
    print(high_risk[['CustomerID', 'Churn_Probability']].head())

    # Save all predictions to a CSV file
    results.to_csv("churn_predictions.csv", index=False)
    print("\nAll predictions saved to churn_predictions.csv")