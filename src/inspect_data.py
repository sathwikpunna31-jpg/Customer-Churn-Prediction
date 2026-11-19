
import pandas as pd

df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())
