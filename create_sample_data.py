import pandas as pd

data = {
    "CustomerID": ["CUST-001", "CUST-002", "CUST-003"],
    "Age": [45, 32, 28],
    "TenureinMonths": [24, 6, 12],
    "MonthlyCharge": [89.50, 65.25, 70.00],
    "TotalCharges": [2148.00, 391.50, 840.00],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "InternetType": ["Fiber optic", "DSL", "No Internet"],
    "OnlineSecurity": ["No", "Yes", "No"],
    "PaymentMethod": ["Electronic check", "Credit card", "Mailed check"],
    "AvgMonthlyLongDistanceCharges": [15.25, 8.75, 12.00],
    "AvgMonthlyGBDownload": [85.5, 45.2, 60.0],
    "SatisfactionScore": [3, 4, 2],
    "NumberofDependents": [0, 2, 1],
    "PaperlessBilling": ["Yes", "No", "Yes"],
    "MultipleLines": ["Yes", "No", "Yes"],
    "StreamingTV": ["Yes", "No", "No"],
    "StreamingMovies": ["Yes", "No", "Yes"],
    "PhoneService": ["Yes", "Yes", "No"],
    "OnlineBackup": ["No", "Yes", "No"],
    "DeviceProtectionPlan": ["No", "Yes", "No"],
    "PremiumTechSupport": ["No", "Yes", "No"],
    "StreamingMusic": ["Yes", "No", "No"],
    "UnlimitedData": ["Yes", "No", "Yes"],
    "Churn": [1, 0, 0]  # 1 = churned, 0 = not churned
}

df = pd.DataFrame(data)
df.to_csv("customer_data.csv", index=False)
print("Sample customer_data.csv created.")