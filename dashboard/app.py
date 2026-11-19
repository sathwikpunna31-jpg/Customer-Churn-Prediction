
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Retention Intelligence",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cleaned_customer_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/final_churn_pipeline.pkl")

df = load_data()
pipeline = load_model()

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    background: #111C32;
    border: 1px solid #263650;
    padding: 18px;
    border-radius: 12px;
}

[data-testid="stSidebar"] {
    background: #0F172A;
}

h1, h2, h3 {
    color: #F1F5F9;
}

.stButton > button {
    border-radius: 8px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# GLOBAL METRICS
# --------------------------------------------------

total_customers = len(df)
churned = (df["Churn"] == "Yes").sum()
retained = total_customers - churned
churn_rate = churned / total_customers * 100

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📊 CRI Platform")
st.sidebar.caption("Customer Retention Intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Business Overview",
        "Churn Analytics",
        "Model Performance",
        "Customer Prediction"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Powered by Python & Machine Learning")

# --------------------------------------------------
# PAGE 1: BUSINESS OVERVIEW
# --------------------------------------------------

if page == "Business Overview":

    st.title("Customer Retention Intelligence")
    st.caption("Business Overview | Customer Churn Analytics")

    st.subheader("Business Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Retained Customers", f"{retained:,}")
    c4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Distribution")

        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Status", "Customers"]

        fig = px.pie(
            churn_counts,
            names="Status",
            values="Customers",
            hole=0.62,
            color="Status",
            color_discrete_map={
                "Yes": "#F87171",
                "No": "#38BDF8"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Churn by Contract")

        contract = pd.crosstab(
            df["Contract"],
            df["Churn"],
            normalize="index"
        ).reset_index()

        contract["Yes"] *= 100
        contract["No"] *= 100

        fig = px.bar(
            contract,
            x="Contract",
            y="Yes",
            color_discrete_sequence=["#38BDF8"],
            labels={"Yes": "Churn Rate (%)"}
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customer Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

# --------------------------------------------------
# PAGE 2: CHURN ANALYTICS
# --------------------------------------------------

elif page == "Churn Analytics":

    st.title("Churn Analytics")
    st.caption("Explore customer behavior and churn patterns")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Churn by Internet Service")

        internet = pd.crosstab(
            df["InternetService"],
            df["Churn"],
            normalize="index"
        ).reset_index()

        internet["Yes"] *= 100

        fig = px.bar(
            internet,
            x="InternetService",
            y="Yes",
            color_discrete_sequence=["#38BDF8"],
            labels={"Yes": "Churn Rate (%)"}
        )

        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Average Tenure")

        tenure = df.groupby("Churn")["tenure"].mean().reset_index()

        fig = px.bar(
            tenure,
            x="Churn",
            y="tenure",
            color="Churn",
            color_discrete_map={
                "Yes": "#F87171",
                "No": "#38BDF8"
            },
            labels={"tenure": "Average Months"}
        )

        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Monthly Charges by Churn")

        fig = px.box(
            df,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            color_discrete_map={
                "Yes": "#F87171",
                "No": "#38BDF8"
            }
        )

        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Churn by Contract")

        contract = pd.crosstab(
            df["Contract"],
            df["Churn"],
            normalize="index"
        ).reset_index()

        contract["Yes"] *= 100

        fig = px.bar(
            contract,
            x="Contract",
            y="Yes",
            color_discrete_sequence=["#F87171"],
            labels={"Yes": "Churn Rate (%)"}
        )

        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Business Recommendations")

    st.info(
        "Prioritize retention outreach for month-to-month customers, "
        "investigate fiber optic churn, and examine early-tenure "
        "customer experiences."
    )

# --------------------------------------------------
# PAGE 3: MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("Machine Learning Performance")
    st.caption("Logistic Regression | Test Dataset Evaluation")

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    }

    cols = st.columns(5)

    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.3f}")

    st.divider()

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, predictions)

    fig = px.imshow(
        cm,
        text_auto=True,
        x=["Predicted Stay", "Predicted Churn"],
        y=["Actual Stay", "Actual Churn"],
        color_continuous_scale="Blues"
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Influence")

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": classifier.coef_[0]
    })

    importance["Absolute Impact"] = importance["Coefficient"].abs()
    importance = importance.nlargest(15, "Absolute Impact")
    importance = importance.sort_values("Coefficient")

    fig = px.bar(
        importance,
        x="Coefficient",
        y="Feature",
        orientation="h",
        color="Coefficient",
        color_continuous_scale="RdBu"
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Feature coefficients show model associations, not causal effects."
    )

# --------------------------------------------------
# PAGE 4: CUSTOMER PREDICTION
# --------------------------------------------------

elif page == "Customer Prediction":

    st.title("Customer Churn Prediction")
    st.caption("Estimate churn probability for a customer")

    with st.form("prediction_form"):

        st.subheader("Customer Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )

        with c2:
            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )
            security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )
            backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )
            protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )
            support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )
            tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )
            movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

        with c3:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox(
                "Paperless Billing", ["Yes", "No"]
            )
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )
            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=200.0,
                value=70.0
            )
            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=10000.0,
                value=840.0
            )

        submitted = st.form_submit_button(
            "Analyze Customer Risk",
            use_container_width=True
        )

    if submitted:

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        probability = pipeline.predict_proba(customer)[0][1]
        prediction = pipeline.predict(customer)[0]

        st.divider()
        st.subheader("Prediction Result")

        c1, c2 = st.columns(2)

        c1.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

        c2.metric(
            "Model Prediction",
            "Churn" if prediction == 1 else "No Churn"
        )

        st.progress(float(probability))

        if probability >= 0.5:
            st.error("Elevated predicted churn risk")
            st.warning(
                "Suggested action: Prioritize customer retention outreach."
            )
        else:
            st.success("Lower predicted churn risk")
            st.info(
                "Suggested action: Continue regular customer engagement."
            )

        st.caption(
            "Probability is a model estimate, not a guarantee of future behavior."
        )

