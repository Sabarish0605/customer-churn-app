import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold

# Load and prepare data
def prepare_data():
    # Load Telco dataset
    df = pd.read_csv("C:\\Users\\sabar\\Downloads\\archive\\TelcoCustomerChurn.csv")
    
    print("\nChurnLabel value counts:")
    print(df['ChurnLabel'].value_counts())
    
    # Handle missing values
    df['InternetType'] = df['InternetType'].fillna('No Internet')
    df['Offer'] = df['Offer'].fillna('No Offer')
    df['ChurnCategory'] = df['ChurnCategory'].fillna('Unknown')
    df['ChurnReason'] = df['ChurnReason'].fillna('Unknown')
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharge'])  # Fill with monthly charge if missing
    
    # Convert binary variables
    df['Churn'] = df['ChurnLabel'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    print("\nTransformed Churn value counts:")
    print(df['Churn'].value_counts())
    
    # Feature engineering
    service_columns = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 
                      'OnlineBackup', 'DeviceProtectionPlan', 'PremiumTechSupport', 
                      'StreamingTV', 'StreamingMovies', 'StreamingMusic', 'UnlimitedData']
    
    # Modified service counting to handle different values
    df['TotalServices'] = df[service_columns].apply(
        lambda x: x.map({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0}).sum(), axis=1)
    
    df['ChargePerService'] = df['MonthlyCharge'] / (df['TotalServices'] + 1)
    
    # Select features (adding more diverse features)
    features = [
        'Age', 'TenureinMonths', 'MonthlyCharge', 'TotalCharges',
        'ChargePerService', 'Contract', 'InternetType', 'OnlineSecurity',
        'PaymentMethod', 'AvgMonthlyLongDistanceCharges', 'AvgMonthlyGBDownload',
        'TotalServices', 'SatisfactionScore', 'NumberofDependents',
        'PaperlessBilling', 'MultipleLines', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Create dummy variables
    df_encoded = pd.get_dummies(df[features])
    
    X = df_encoded
    y = df['Churn']
    
    return X, y

def create_visualizations(X, y, model):
    # Create correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Feature importance plot
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importances.head(10))
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def train_model(X, y):
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize XGBoost with adjusted parameters
    model = XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=2.77,
        random_state=42,
        eval_metric=['logloss', 'auc']  # Added AUC metric for better evaluation
    )
    
    # Train the model
    eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
    model.fit(
        X_train_scaled, 
        y_train,
        eval_set=eval_set,
        verbose=10
    )
    
    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    print("\nTest Set Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler

if __name__ == "__main__":
    # Prepare data
    X, y = prepare_data()
    
    # Train and evaluate model
    model, scaler = train_model(X, y)
    
    # Create visualizations
    create_visualizations(X, y, model)
    
    # Save model and scaler
    joblib.dump(model, 'churn_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')