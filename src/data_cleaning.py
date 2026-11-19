
import pandas as pd

# Load raw dataset
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

df = df.drop(columns=["customerID"])
df = df.dropna()

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Check missing values after conversion
print("Missing values after conversion:")
print(df.isnull().sum())

# Handle missing values
df.dropna(inplace=True)

# Save cleaned dataset
df.to_csv("data/processed/cleaned_customer_data.csv", index=False)

print("\nCleaned dataset shape:", df.shape)
print("\nData cleaning completed!")



