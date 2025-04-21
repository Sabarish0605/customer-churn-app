import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

def prepare_features(df):
    df['InternetType'] = df['InternetType'].fillna('No Internet')
    service_columns = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 
                      'OnlineBackup', 'DeviceProtectionPlan', 'PremiumTechSupport', 
                      'StreamingTV', 'StreamingMovies', 'StreamingMusic', 'UnlimitedData']
    df['TotalServices'] = df[service_columns].apply(
        lambda x: x.map({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0}).sum(), axis=1)
    df['ChargePerService'] = df['MonthlyCharge'] / (df['TotalServices'] + 1)
    return df

def predict_customer_churn(customer_data):
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = [
        'Age', 'TenureinMonths', 'MonthlyCharge', 'TotalCharges',
        'ChargePerService', 'Contract', 'InternetType', 'OnlineSecurity',
        'PaymentMethod', 'AvgMonthlyLongDistanceCharges', 'AvgMonthlyGBDownload',
        'TotalServices', 'SatisfactionScore', 'NumberofDependents',
        'PaperlessBilling', 'MultipleLines', 'StreamingTV', 'StreamingMovies'
    ]
    df = prepare_features(customer_data)
    X = pd.get_dummies(df[features])
    X = X.reindex(columns=scaler.feature_names_in_, fill_value=0)
    X_scaled = scaler.transform(X)
    churn_prob = model.predict_proba(X_scaled)[:, 1]
    predictions = model.predict(X_scaled)
    customer_data['Churn_Probability'] = churn_prob
    customer_data['Predicted_Churn'] = predictions
    return customer_data

st.title("Customer Churn Prediction App")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["Upload CSV", "Enter Customer Data", "CSV File Guide"])

with tab1:
    st.info("Upload a CSV file with customer data to predict churn probability.")
    uploaded_file = st.file_uploader("Upload customer CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        results = predict_customer_churn(data)
        
        # Display summary statistics with clearer risk thresholds
        st.write("Predictions:")
        high_risk = results[results['Churn_Probability'] > 0.8]
        moderate_risk = results[(results['Churn_Probability'] >= 0.2) & (results['Churn_Probability'] <= 0.8)]
        low_risk = results[results['Churn_Probability'] < 0.2]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("High Risk Customers", f"{len(high_risk)}", f"{len(high_risk)/len(results):.1%}")
        with col2:
            st.metric("Moderate Risk Customers", f"{len(moderate_risk)}", f"{len(moderate_risk)/len(results):.1%}")
        with col3:
            st.metric("Low Risk Customers", f"{len(low_risk)}", f"{len(low_risk)/len(results):.1%}")
        
        # Add segment analysis if segment column exists
        if 'Segment' in results.columns:
            st.subheader("Risk by Customer Segment")
            segment_risk = results.groupby('Segment')['Churn_Probability'].mean().reset_index()
            segment_risk = segment_risk.sort_values('Churn_Probability', ascending=False)
            segment_risk['Risk Level'] = segment_risk['Churn_Probability'].apply(
                lambda x: 'High Risk' if x > 0.7 else ('Moderate Risk' if x >= 0.3 else 'Low Risk'))
            segment_risk['Churn_Probability'] = segment_risk['Churn_Probability'].apply(lambda x: f"{x:.2%}")
            st.dataframe(segment_risk.rename(columns={'Churn_Probability': 'Average Churn Probability'}))
            
            # Create a bar chart for segment risk
            fig_data = results.groupby('Segment')['Churn_Probability'].mean().reset_index()
            st.bar_chart(fig_data.set_index('Segment'))
        
        # Display detailed results with risk level
        results['Risk Level'] = results['Churn_Probability'].apply(
            lambda x: 'High Risk' if x > 0.8 else ('Moderate Risk' if x >= 0.2 else 'Low Risk'))
        
        # Allow filtering by risk level
        risk_filter = st.multiselect("Filter by Risk Level", 
                                    options=['High Risk', 'Moderate Risk', 'Low Risk'],
                                    default=['High Risk', 'Moderate Risk', 'Low Risk'])
        
        filtered_results = results[results['Risk Level'].isin(risk_filter)]
        st.dataframe(filtered_results[['CustomerID', 'Churn_Probability', 'Predicted_Churn', 'Risk Level'] + 
                                     (['Segment'] if 'Segment' in results.columns else [])])
        
        # --- Add download buttons for each risk group ---
        st.subheader("Download Risk Group Details")
        col_hr, col_mr, col_lr = st.columns(3)
        with col_hr:
            st.download_button(
                label="Download High Risk Customers",
                data=high_risk.to_csv(index=False),
                file_name="high_risk_customers.csv",
                mime="text/csv"
            )
        with col_mr:
            st.download_button(
                label="Download Moderate Risk Customers",
                data=moderate_risk.to_csv(index=False),
                file_name="moderate_risk_customers.csv",
                mime="text/csv"
            )
        with col_lr:
            st.download_button(
                label="Download Low Risk Customers",
                data=low_risk.to_csv(index=False),
                file_name="low_risk_customers.csv",
                mime="text/csv"
            )
        # Add risk level visualization
        st.subheader("Risk Distribution")
        risk_data = pd.DataFrame({
            'Risk Level': ['Low Risk', 'Moderate Risk', 'High Risk'],
            'Count': [len(low_risk), len(moderate_risk), len(high_risk)]
        })
        st.bar_chart(risk_data.set_index('Risk Level'))
        
        st.download_button("Download Predictions as CSV", results.to_csv(index=False), file_name="churn_predictions.csv")
        st.subheader("Churn Probability Distribution")
        st.bar_chart(results['Churn_Probability'])
        st.subheader("Churn Probability Histogram")
        fig, ax = plt.subplots()
        ax.hist(results['Churn_Probability'], bins=10, color='skyblue', edgecolor='black')
        st.pyplot(fig)

with tab2:
    st.header("Enter Customer Information")
    
    # Generate a random customer ID
    import random
    import string
    customer_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure = st.number_input("Tenure in Months", min_value=0, max_value=120, value=12)
        monthly_charge = st.number_input("Monthly Charge ($)", min_value=0.0, max_value=200.0, value=70.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=monthly_charge * tenure)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_type = st.selectbox("Internet Type", ["Fiber optic", "DSL", "No Internet"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Bank transfer", "Credit card", "Mailed check", "Electronic check"])
        
    with col2:
        avg_monthly_ld = st.number_input("Avg Monthly Long Distance Charges ($)", min_value=0.0, max_value=100.0, value=10.0)
        avg_monthly_gb = st.number_input("Avg Monthly GB Download", min_value=0.0, max_value=500.0, value=50.0)
        satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    
    # Additional service options
    st.subheader("Additional Services")
    col3, col4 = st.columns(2)
    
    with col3:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No"])
        device_protection = st.selectbox("Device Protection Plan", ["Yes", "No"])
    
    with col4:
        premium_tech = st.selectbox("Premium Tech Support", ["Yes", "No"])
        streaming_music = st.selectbox("Streaming Music", ["Yes", "No"])
        unlimited_data = st.selectbox("Unlimited Data", ["Yes", "No"])
    
    if st.button("Predict Churn"):
        # Create a dataframe with the input data
        customer_data = pd.DataFrame({
            "CustomerID": [customer_id],
            "Age": [age],
            "TenureinMonths": [tenure],
            "MonthlyCharge": [monthly_charge],
            "TotalCharges": [total_charges],
            "Contract": [contract],
            "InternetType": [internet_type],
            "OnlineSecurity": [online_security],
            "PaymentMethod": [payment_method],
            "AvgMonthlyLongDistanceCharges": [avg_monthly_ld],
            "AvgMonthlyGBDownload": [avg_monthly_gb],
            "SatisfactionScore": [satisfaction_score],
            "NumberofDependents": [dependents],
            "PaperlessBilling": [paperless_billing],
            "MultipleLines": [multiple_lines],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "PhoneService": [phone_service],
            "OnlineBackup": [online_backup],
            "DeviceProtectionPlan": [device_protection],
            "PremiumTechSupport": [premium_tech],
            "StreamingMusic": [streaming_music],
            "UnlimitedData": [unlimited_data]
        })
        
        # Make prediction
        results = predict_customer_churn(customer_data)
        
        # Display results
        st.subheader("Prediction Result")
        churn_prob = results["Churn_Probability"].values[0]
        churn_pred = results["Predicted_Churn"].values[0]
        
        # Create a visual risk meter
        st.subheader("Churn Risk Meter")
        
        # Determine risk level
        if churn_prob <= 0.3:
            risk_level = "Low Risk"
            color = "green"
        elif churn_prob <= 0.7:
            risk_level = "Moderate Risk"
            color = "orange"
        else:
            risk_level = "High Risk"
            color = "red"
        
        # Create a progress bar as a meter
        st.progress(float(churn_prob))
        st.markdown(f"<h3 style='color:{color};text-align:center;'>{risk_level}: {churn_prob:.2%}</h3>", unsafe_allow_html=True)
        
        # Display prediction text
        if churn_pred == 1:
            st.error(f"⚠️ Prediction: Customer likely to churn ({churn_prob:.2%} probability)")
        else:
            st.success(f"✅ Prediction: Customer likely to stay ({1-churn_prob:.2%} retention probability)")
        
        # Display all customer data with prediction
        st.subheader("Customer Details")
        st.dataframe(results)
        
        # Option to download the result
        st.download_button("Download Prediction as CSV", results.to_csv(index=False), file_name="single_customer_prediction.csv")

# New tab for CSV file guide
with tab3:
    st.header("CSV File Format Guide")
    st.write("""
    To create a CSV file for batch predictions, include the following columns:
    """)
    
    # Create a sample dataframe with required columns
    sample_columns = [
        "CustomerID", "Age", "TenureinMonths", "MonthlyCharge", "TotalCharges",
        "Contract", "InternetType", "OnlineSecurity", "PaymentMethod", 
        "AvgMonthlyLongDistanceCharges", "AvgMonthlyGBDownload", "SatisfactionScore", 
        "NumberofDependents", "PaperlessBilling", "MultipleLines", "StreamingTV", 
        "StreamingMovies", "PhoneService", "OnlineBackup", "DeviceProtectionPlan", 
        "PremiumTechSupport", "StreamingMusic", "UnlimitedData"
    ]
    
    # Create a sample row
    sample_data = {
        "CustomerID": ["CUST-001", "CUST-002"],
        "Age": [45, 32],
        "TenureinMonths": [24, 6],
        "MonthlyCharge": [89.50, 65.25],
        "TotalCharges": [2148.00, 391.50],
        "Contract": ["Month-to-month", "One year"],
        "InternetType": ["Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "Yes"],
        "PaymentMethod": ["Electronic check", "Credit card"],
        "AvgMonthlyLongDistanceCharges": [15.25, 8.75],
        "AvgMonthlyGBDownload": [85.5, 45.2],
        "SatisfactionScore": [3, 4],
        "NumberofDependents": [0, 2],
        "PaperlessBilling": ["Yes", "No"],
        "MultipleLines": ["Yes", "No"],
        "StreamingTV": ["Yes", "No"],
        "StreamingMovies": ["Yes", "No"],
        "PhoneService": ["Yes", "Yes"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtectionPlan": ["No", "Yes"],
        "PremiumTechSupport": ["No", "Yes"],
        "StreamingMusic": ["Yes", "No"],
        "UnlimitedData": ["Yes", "No"]
    }
    
    # Create sample dataframe
    sample_df = pd.DataFrame(sample_data)
    
    # Display sample dataframe
    st.subheader("Sample CSV Format")
    st.dataframe(sample_df)
    
    # Provide column descriptions
    st.subheader("Column Descriptions")
    column_descriptions = {
        "CustomerID": "Unique identifier for each customer",
        "Age": "Customer's age (18-100)",
        "TenureinMonths": "How long the customer has been with the company (0-120)",
        "MonthlyCharge": "Monthly bill amount ($)",
        "TotalCharges": "Total amount charged to the customer ($)",
        "Contract": "Type of contract (Month-to-month, One year, Two year)",
        "InternetType": "Type of internet service (Fiber optic, DSL, No Internet)",
        "OnlineSecurity": "Whether the customer has online security (Yes, No)",
        "PaymentMethod": "Payment method (Bank transfer, Credit card, Mailed check, Electronic check)",
        "AvgMonthlyLongDistanceCharges": "Average monthly long distance charges ($)",
        "AvgMonthlyGBDownload": "Average monthly download in GB",
        "SatisfactionScore": "Customer satisfaction score (1-5)",
        "NumberofDependents": "Number of dependents (0-10)",
        "PaperlessBilling": "Whether the customer has paperless billing (Yes, No)",
        "MultipleLines": "Whether the customer has multiple lines (Yes, No)",
        "StreamingTV": "Whether the customer has streaming TV (Yes, No)",
        "StreamingMovies": "Whether the customer has streaming movies (Yes, No)",
        "PhoneService": "Whether the customer has phone service (Yes, No)",
        "OnlineBackup": "Whether the customer has online backup (Yes, No)",
        "DeviceProtectionPlan": "Whether the customer has device protection (Yes, No)",
        "PremiumTechSupport": "Whether the customer has premium tech support (Yes, No)",
        "StreamingMusic": "Whether the customer has streaming music (Yes, No)",
        "UnlimitedData": "Whether the customer has unlimited data (Yes, No)"
    }
    
    # Display as a table
    desc_df = pd.DataFrame({"Column": column_descriptions.keys(), "Description": column_descriptions.values()})
    st.table(desc_df)
    
    # Provide download option for template
    csv = sample_df.to_csv(index=False)
    st.download_button(
        label="Download CSV Template",
        data=csv,
        file_name="customer_template.csv",
        mime="text/csv"
    )
    
    st.info("You can open this template in Excel or any spreadsheet software, add your customer data, and save as CSV.")