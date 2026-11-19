
import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load("models/final_churn_pipeline.pkl")

# Extract feature names
preprocessor = pipeline.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()

# Extract Logistic Regression coefficients
coefficients = pipeline.named_steps["classifier"].coef_[0]

# Create feature importance table
importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

importance["Impact"] = importance["Coefficient"].apply(
    lambda x: "Higher churn tendency" if x > 0
    else "Lower churn tendency"
)

importance["Absolute_Impact"] = importance["Coefficient"].abs()

importance = importance.sort_values(
    by="Absolute_Impact",
    ascending=False
)

print("\nTOP 15 INFLUENTIAL FEATURES:")
print(importance.head(15).to_string(index=False))

# Save results
importance.to_csv(
    "data/processed/feature_importance.csv",
    index=False
)

print("\nFeature analysis saved successfully!")
