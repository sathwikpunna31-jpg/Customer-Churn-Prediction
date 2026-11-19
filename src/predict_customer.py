
import joblib
import pandas as pd

# Load trained pipeline
pipeline = joblib.load("models/final_churn_pipeline.pkl")

# Sample customer
customer = pd.DataFrame([{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0,
    "TotalCharges": 285.0
}])

# Predict
prediction = pipeline.predict(customer)[0]
probability = pipeline.predict_proba(customer)[0][1]

print("Customer Churn Prediction")

print("Prediction:", "Churn" if prediction == 1 else "No Churn")
print("Churn Probability:", round(probability * 100, 2), "%")

if probability >= 0.5:
    print("Suggested Action: Prioritize retention outreach.")
else:
    print("Suggested Action: Continue regular engagement.")
