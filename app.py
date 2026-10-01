# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Streamlit Page Configuration
st.set_page_config(page_title="Churn Predictor - Customer Analytics", layout="wide", page_icon="🔮")

# Inject Custom CSS (Clean White Theme + High Contrast Text)
clean_white_css = """
<style>
    /* Main Background & Text Color */
    .stApp {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0;
    }

    /* Titles & Headers */
    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 800 !important;
    }

    /* Sidebar Radio Label Text Contrast */
    div[data-testid="stSidebar"] label, div[data-testid="stSidebar"] span {
        color: #1e293b !important;
        font-weight: 600 !important;
    }

    /* Developer Name Badge Styling */
    .dev-badge {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Buttons Styling */
    .stButton > button {
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
    }
    .stButton > button:hover {
        background-color: #1d4ed8 !important;
    }
</style>
"""
st.markdown(clean_white_css, unsafe_allow_html=True)

st.title("🔮 Churn Predictor - Customer Churn Analytics Platform")

# Initialize Session States
if 'df' not in st.session_state:
    st.session_state.df = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None

# Navigation Sidebar with Your Name
st.sidebar.markdown('<div class="dev-badge">👨‍‍💻 Developed by SOUMAVA SARKAR</div>', unsafe_allow_html=True)
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Select Module", [
    "📂 Data Upload & Quality Analysis",
    "📊 Exploratory Data Analysis (EDA)",
    "🤖 Train Models & AutoML",
    "📈 Model Comparison & Evaluation",
    "🔮 Single-Customer Churn Prediction"
])

# ---------------------------------------------------------
# 📂 1. Data Upload & Data Quality Analysis
# ---------------------------------------------------------
if menu == "📂 Data Upload ":
    st.header("📂 Data Upload)

    uploaded_file = st.file_uploader("Upload CSV or Excel dataset", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        st.session_state.df = df
        st.success("Dataset loaded and preprocessed successfully!")

    if st.session_state.df is not None:
        df = st.session_state.df

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Duplicate Rows", df.duplicated().sum())
        col4.metric("Total Missing Values", df.isnull().sum().sum())

        st.subheader("🔍 Data Quality Breakdown")
        quality_df = pd.DataFrame({
            "Data Type": df.dtypes,
            "Missing Values": df.isnull().sum(),
            "Missing %": (df.isnull().sum() / len(df)) * 100,
            "Unique Values": df.nunique()
        })
        st.dataframe(quality_df)

# ---------------------------------------------------------
# 📊 2. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
elif menu == "📊 Exploratory Data Analysis (EDA)":
    st.header("📊 Exploratory Data Analysis")

    if st.session_state.df is None:
        st.warning("Please upload a dataset in Module 1 first.")
    else:
        df = st.session_state.df

        tab1, tab2, tab3 = st.tabs(["Distributions", "Categorical Analysis", "Correlation Heatmap"])

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        with tab1:
            st.subheader("Numerical Feature Distributions")
            if num_cols:
                selected_num = st.selectbox("Select Numerical Feature", num_cols)
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(df[selected_num], kde=True, ax=ax, color="#2563eb")
                st.pyplot(fig)

        with tab2:
            st.subheader("Categorical Feature Counts & Churn Relationship")
            if cat_cols:
                selected_cat = st.selectbox("Select Categorical Feature", cat_cols)
                fig, ax = plt.subplots(figsize=(8, 4))
                if 'Churn' in df.columns:
                    sns.countplot(data=df, x=selected_cat, hue='Churn', ax=ax, palette="Blues")
                else:
                    sns.countplot(data=df, x=selected_cat, ax=ax, palette="Blues")
                plt.xticks(rotation=45)
                st.pyplot(fig)

        with tab3:
            st.subheader("Correlation Heatmap")
            if len(num_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[num_cols].corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax)
                st.pyplot(fig)

# ---------------------------------------------------------
# 🤖 3. Model Training & AutoML
# ---------------------------------------------------------
elif menu == "🤖 Train Models & AutoML":
    st.header("🤖 Train ML Models & Native AutoML Engine")

    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state.df

        st.subheader("Configuration")
        target_col = st.selectbox("Select Target Variable", df.columns, index=df.columns.get_loc('Churn') if 'Churn' in df.columns else 0)
        ignore_cols = st.multiselect("Select ID/Ignored Columns", df.columns, default=[c for c in df.columns if 'id' in c.lower() or c == 'Unnamed: 0'])

        mode = st.radio("Choose Execution Mode", ["Manual Models & Neural Network (MLP)", "AutoML Engine"])

        if mode == "Manual Models & Neural Network (MLP)":
            if st.button("Train All Manual Models & Neural Net"):
                with st.spinner("Preprocessing data & training models..."):
                    clean_df = df.dropna(subset=[target_col]).copy()

                    X = clean_df.drop(columns=[target_col] + ignore_cols, errors='ignore')
                    y_raw = clean_df[target_col]
                    y = y_raw.map({'Yes': 1, 'No': 0, True: 1, False: 0, 1: 1, 0: 0}) if y_raw.dtype in ['object', 'bool'] else y_raw

                    X_num = X.select_dtypes(include=[np.number]).columns.tolist()
                    X_cat = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

                    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
                    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

                    preprocessor = ColumnTransformer([('num', num_transformer, X_num), ('cat', cat_transformer, X_cat)])

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    X_train_proc = preprocessor.fit_transform(X_train)
                    X_test_proc = preprocessor.transform(X_test)

                    st.session_state.pipeline = preprocessor
                    st.session_state.X_test = X_test_proc
                    st.session_state.y_test = y_test
                    st.session_state.feature_names = X.columns.tolist()

                    models = {
                        "Logistic Regression": LogisticRegression(max_iter=1000),
                        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
                        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
                        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
                    }

                    results = {}
                    trained_models = {}

                    for name, model in models.items():
                        model.fit(X_train_proc, y_train)
                        y_pred = model.predict(X_test_proc)
                        y_prob = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, "predict_proba") else y_pred

                        trained_models[name] = model
                        results[name] = {
                            "Accuracy": accuracy_score(y_test, y_pred),
                            "Precision": precision_score(y_test, y_pred, zero_division=0),
                            "Recall": recall_score(y_test, y_pred, zero_division=0),
                            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
                            "ROC-AUC": roc_auc_score(y_test, y_prob),
                            "y_pred": y_pred,
                            "y_prob": y_prob
                        }
                        if hasattr(model, "loss_curve_"):
                            results[name]["loss_curve"] = model.loss_curve_

                    st.session_state.models = trained_models
                    st.session_state.results = results
                    st.success("All traditional models & Neural Network trained successfully!")

        elif mode == "AutoML Engine":
            if st.button("Run AutoML Engine"):
                with st.spinner("AutoML searching and evaluating models..."):
                    clean_df = df.dropna(subset=[target_col]).copy()
                    X = clean_df.drop(columns=[target_col] + ignore_cols, errors='ignore')
                    y_raw = clean_df[target_col]
                    y = y_raw.map({'Yes': 1, 'No': 0, True: 1, False: 0, 1: 1, 0: 0}) if y_raw.dtype in ['object', 'bool'] else y_raw

                    X_num = X.select_dtypes(include=[np.number]).columns.tolist()
                    X_cat = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

                    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
                    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
                    preprocessor = ColumnTransformer([('num', num_transformer, X_num), ('cat', cat_transformer, X_cat)])

                    X_proc = preprocessor.fit_transform(X)

                    candidate_models = {
                        "Logistic Regression": LogisticRegression(max_iter=1000),
                        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
                        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                    }

                    automl_results = []
                    for name, m in candidate_models.items():
                        scores = cross_val_score(m, X_proc, y, cv=5, scoring='roc_auc')
                        acc_scores = cross_val_score(m, X_proc, y, cv=5, scoring='accuracy')
                        automl_results.append({
                            "Model": name,
                            "Mean ROC-AUC": np.mean(scores),
                            "Mean Accuracy": np.mean(acc_scores)
                        })

                    automl_df = pd.DataFrame(automl_results).sort_values(by="Mean ROC-AUC", ascending=False)
                    st.subheader("🤖 AutoML Model Evaluation Matrix")
                    st.dataframe(automl_df)
                    st.success("AutoML evaluation completed!")

# ---------------------------------------------------------
# 📈 4. Model Comparison & Evaluation
# ---------------------------------------------------------
elif menu == "📈 Model Comparison & Evaluation":
    st.header("📈 Model Evaluation Metrics & Visualizations")

    if "results" not in st.session_state or not st.session_state.results:
        st.warning("Please train models in Module 3 first.")
    else:
        results = st.session_state.results

        st.subheader("Model Performance Summary Table")
        metrics_df = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
        st.dataframe(metrics_df)

        selected_model = st.selectbox("Select Model for Detailed Inspection", list(results.keys()))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Confusion Matrix ({selected_model})")
            cm = confusion_matrix(st.session_state.y_test, results[selected_model]["y_pred"])
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

        with col2:
            st.subheader(f"ROC-AUC Curve ({selected_model})")
            fpr, tpr, _ = roc_curve(st.session_state.y_test, results[selected_model]["y_prob"])
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {results[selected_model]['ROC-AUC']:.3f}", color='#2563eb', linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)

        if "loss_curve" in results[selected_model]:
            st.subheader(f"🧠 {selected_model} Loss Curve")
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(results[selected_model]["loss_curve"], label='Training Loss', color='#2563eb', linewidth=2)
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Loss")
            ax.legend()
            st.pyplot(fig)

# ---------------------------------------------------------
# 🔮 5. Single-Customer Churn Prediction
# ---------------------------------------------------------
elif menu == "🔮 Single-Customer Churn Prediction":
    st.header("🔮 Single-Customer Churn Classification")

    if "pipeline" not in st.session_state or not st.session_state.models:
        st.warning("Please train the models in Module 3 first before running predictions.")
    else:
        df = st.session_state.df
        selected_model_name = st.selectbox("Select Model for Prediction", list(st.session_state.models.keys()))
        model = st.session_state.models[selected_model_name]
        pipeline = st.session_state.pipeline

        st.subheader("Enter Customer Feature Values")

        feature_cols = st.session_state.feature_names
        input_data = {}

        col_a, col_b = st.columns(2)
        for idx, col in enumerate(feature_cols):
            with col_a if idx % 2 == 0 else col_b:
                if df[col].dtype in ['object', 'category', 'bool']:
                    unique_vals = df[col].dropna().unique().tolist()
                    input_data[col] = st.selectbox(f"{col}", unique_vals)
                else:
                    min_v, max_v = float(df[col].min()), float(df[col].max())
                    mean_v = float(df[col].mean())
                    input_data[col] = st.number_input(f"{col}", min_value=min_v, max_value=max_v, value=mean_v)

        if st.button("Classify Customer Churn Risk"):
            input_df = pd.DataFrame([input_data])
            processed_input = pipeline.transform(input_df)

            pred = model.predict(processed_input)[0]
            prob = model.predict_proba(processed_input)[0][1] if hasattr(model, "predict_proba") else float(pred)

            st.subheader("Inference Result")
            if pred == 1:
                st.error(f"⚠️ High Churn Risk! (Probability: {prob:.2%})")
            else:
                st.success(f"✅ Customer Retained (Churn Probability: {prob:.2%})")
