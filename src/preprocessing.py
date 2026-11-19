
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load cleaned data
df = pd.read_csv("data/processed/cleaned_customer_data.csv")

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})

# Convert categorical features into numerical features
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale numerical features
scaler = StandardScaler()

numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

X_train[numerical_columns] = scaler.fit_transform(
    X_train[numerical_columns]
)

X_test[numerical_columns] = scaler.transform(
    X_test[numerical_columns]
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTarget distribution in training data:")
print(y_train.value_counts())

print("\nPreprocessing completed!")
