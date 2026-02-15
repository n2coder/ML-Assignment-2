import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------
# App Title
# ---------------------------------------------------
st.title("Breast Cancer Diagnosis - ML Classification Dashboard")
st.write("This application evaluates 6 Machine Learning models on the WDBC dataset.")

# ---------------------------------------------------
# File Upload
# ---------------------------------------------------
uploaded_file = st.file_uploader("Upload Test Data (CSV)", type="csv")

# ---------------------------------------------------
# Model Selection
# ---------------------------------------------------
model_options = [
    "logistic_regression",
    "decision_tree",
    "knn",
    "naive_bayes",
    "random_forest",
    "xgboost"
]

selected_model = st.selectbox("Select a Model", model_options)

# ---------------------------------------------------
# If File Uploaded
# ---------------------------------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ---------------------------
    # Validate dataset
    # ---------------------------
    if "diagnosis" not in df.columns:
        st.error("CSV must contain 'diagnosis' column.")
        st.stop()

    # Convert diagnosis to numeric
    if df["diagnosis"].dtype == object:
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    # Remove ID column if exists
    if "id" in X.columns:
        X = X.drop("id", axis=1)

    # ---------------------------
    # Load Model
    # ---------------------------
    model_path = f"model/{selected_model}.joblib"

    if not os.path.exists(model_path):
        st.error(f"Model file {model_path} not found.")
        st.stop()

    model = joblib.load(model_path)
    st.success(f"Model '{selected_model}' loaded successfully!")

    # ---------------------------
    # Prediction
    # ---------------------------
    y_pred = model.predict(X)

    # Some models support predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
    else:
        auc = roc_auc_score(y, y_pred)

    # ---------------------------
    # Metrics
    # ---------------------------
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Model Evaluation Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy, 4))
    col2.metric("Precision", round(precision, 4))
    col3.metric("Recall", round(recall, 4))

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", round(f1, 4))
    col5.metric("AUC Score", round(auc, 4))
    col6.metric("MCC Score", round(mcc, 4))

    # ---------------------------
    # Confusion Matrix
    # ---------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # ---------------------------
    # Classification Report
    # ---------------------------
    st.subheader("Classification Report")

    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # ---------------------------
    # Observations (Bonus for Marks)
    # ---------------------------
    st.subheader("Model Observation")

    st.write("""
    - Logistic Regression performs well when data is linearly separable.
    - Decision Tree may overfit if depth is not controlled.
    - KNN performance depends on scaling and value of k.
    - Naive Bayes assumes feature independence.
    - Random Forest improves stability using bagging.
    - XGBoost often achieves best performance using boosting.
    """)

else:
    st.info("Please upload a CSV file to begin.")
