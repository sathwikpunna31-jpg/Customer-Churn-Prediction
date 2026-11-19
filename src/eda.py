
import pandas as pd
import matplotlib

# Use a non-interactive backend
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/cleaned_customer_data.csv")

print("Churn Distribution:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# Plot churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("data/processed/churn_distribution.png")
plt.close()

print("\nGraph saved successfully!")

# Churn by contract type
print("\nChurn by Contract:")
print(pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100)

# Visualization
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("data/processed/churn_by_contract.png")
plt.close()

print("\nContract analysis graph saved!")


# Churn by tenure
plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    multiple="stack"
)

plt.title("Customer Churn by Tenure")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("data/processed/churn_by_tenure.png")
plt.close()

# Average tenure by churn status
print("\nAverage Tenure by Churn:")
print(df.groupby("Churn")["tenure"].mean())

print("\nTenure analysis graph saved!")

# Churn by monthly charges
plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="MonthlyCharges",
    hue="Churn",
    bins=30,
    multiple="stack"
)

plt.title("Customer Churn by Monthly Charges")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("data/processed/churn_by_monthly_charges.png")
plt.close()

# Average monthly charges by churn status
print("\nAverage Monthly Charges by Churn:")
print(df.groupby("Churn")["MonthlyCharges"].mean())

print("\nMonthly charges analysis graph saved!")


# Churn by internet service
print("\nChurn by Internet Service:")
print(
    pd.crosstab(
        df["InternetService"],
        df["Churn"],
        normalize="index"
    ) * 100
)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("data/processed/churn_by_internet_service.png")
plt.close()

print("\nInternet service analysis graph saved!")
