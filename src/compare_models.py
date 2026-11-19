
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load data
df = pd.read_csv("data/processed/cleaned_customer_data.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})

# Encode categorical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale numerical features
numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(
    X_train[numerical_columns]
)
X_test[numerical_columns] = scaler.transform(
    X_test[numerical_columns]
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42
    )
}

results = []

# Train and evaluate
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })

# Display comparison
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.round(3).to_string(index=False))

results_df.to_csv(
    "data/processed/model_comparison.csv",
    index=False
)

print("\nComparison saved successfully!")
