# Customer Retention Intelligence

### Customer Churn Prediction & Analytics Platform

An end-to-end Machine Learning and Data Analytics project that analyzes telecom customer behavior, predicts customer churn, and provides actionable retention insights through an interactive Streamlit dashboard.

## 1. Business Problem

Customer churn is a major challenge for subscription-based businesses. Losing existing customers affects recurring revenue and increases customer acquisition costs.

This project aims to:

- Analyze customer behavior and churn patterns.
- Identify factors associated with customer churn.
- Predict customers who are likely to leave.
- Support data-driven customer retention decisions.

## 2. Technology Stack

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn
- **Dashboard:** Streamlit
- **Version Control:** Git and GitHub

## 3. Project Workflow

1. Dataset collection and inspection.
2. Data cleaning and preprocessing.
3. Exploratory Data Analysis (EDA).
4. Feature encoding and scaling.
5. Training and comparison of ML models.
6. Model evaluation.
7. Customer churn prediction.
8. Interactive dashboard development.

## 4. Dataset

IBM Telco Customer Churn sample dataset.

- Original records: 7,043
- Original columns: 21
- Records after cleaning: 7,032
- Target variable: `Churn`

The target identifies whether a customer stayed or left.

## 5. Exploratory Data Analysis Findings

- Overall churn rate: 26.58%.
- Month-to-month contract churn: 42.71%.
- One-year contract churn: 11.28%.
- Two-year contract churn: 2.85%.
- Average tenure of churned customers: 17.98 months.
- Average tenure of retained customers: 37.65 months.
- Fiber optic customer churn: 41.89%.

These findings represent associations observed in the dataset, not causal relationships.

## 6. Machine Learning Models

Three classification algorithms were compared:

- Logistic Regression
- Decision Tree
- Random Forest

Logistic Regression achieved the strongest overall results among the three models on the selected test split.

### Model Performance

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 80.4% |
| Precision | 64.8% |
| Recall    | 57.2% |
| F1-score  | 60.8% |
| ROC-AUC   | 0.836 |

The final model uses a Scikit-learn preprocessing and classification pipeline.

## 7. Dashboard Features

The Streamlit application contains four sections:

1. **Business Overview:** Customer KPIs and churn distribution.
2. **Churn Analytics:** Contract, tenure, monthly charges and internet service analysis.
3. **Model Performance:** Evaluation metrics, confusion matrix and feature coefficients.
4. **Customer Prediction:** Interactive customer information form and churn probability.

## 8. Project Structure

```text
Customer-Retention-Intelligence/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── inspect_data.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── compare_models.py
│   ├── final_model.py
│   ├── feature_analysis.py
│   └── predict_customer.py
├── models/
│   └── final_churn_pipeline.pkl
├── dashboard/
│   └── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 9. Run Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Customer-Retention-Intelligence
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## 10. Future Improvements

- Improve churn recall through threshold tuning and model optimization.
- Add SHAP-based model explanations.
- Introduce customer segmentation.
- Add retention campaign effectiveness analysis.
- Improve the dashboard with additional business KPIs.

## 11. Limitations

The project uses a sample telecom dataset and historical customer records. Predictions are estimates and should not be interpreted as guarantees of future customer behavior.

## 12. Author

Developed as an end-to-end Data Analytics and Machine Learning project.
