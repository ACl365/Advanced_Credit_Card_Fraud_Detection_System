#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Credit Card Fraud Detection - Streamlit Demo Application
-------------------------------------------------------
This module provides a web interface for the credit card fraud detection model.

Author: Alex
Last Updated: March 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import time

# Add the parent directory to the path for importing the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("🔍 Credit Card Fraud Detection System")
st.markdown("""
    This application demonstrates an advanced ensemble learning approach for credit card fraud detection.
    The system uses a stacked ensemble of machine learning models to identify fraudulent transactions with high accuracy.
""")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Dashboard", "Model Performance", "Sample Prediction", "About"])

# Import the two-stage model
try:
    from two_stage_model import create_two_stage_model
except ImportError:
    st.error("Two-stage model module not found. Please ensure two_stage_model.py exists.")

# Load models and results
@st.cache_resource
def load_models():
    """Load trained models and results data."""
    models = {}
    try:
        models["logistic"] = joblib.load("models/meta_learner_logistic_model.pkl")
        models["gradient_boosting"] = joblib.load("models/meta_learner_gradient_boosting_model.pkl")
        
        # Load some base models too
        models["random_forest"] = joblib.load("models/random_forest_model.pkl")
        models["xgboost"] = joblib.load("models/xgboost_model.pkl")
        
        # Create and add the two-stage model
        try:
            models["two_stage"] = create_two_stage_model(threshold1=0.3, threshold2=0.7)
            st.sidebar.success("Two-stage model loaded successfully!")
        except Exception as e:
            st.sidebar.warning(f"Could not load two-stage model: {e}")
        
        return models
    except FileNotFoundError:
        st.error("Model files not found. Please run the main.py script first to train the models.")
        return None

@st.cache_data
def load_results():
    """Load model evaluation results."""
    try:
        results = pd.read_csv("results/model_evaluation_metrics.csv")
        results.set_index("Unnamed: 0", inplace=True)
        results.index.name = "Model"
        return results
    except FileNotFoundError:
        st.error("Results file not found. Please run the main.py script first to generate evaluation metrics.")
        return None

# Cache data loading
@st.cache_data
def load_sample_data():
    """Load a sample of the data for demonstration."""
    try:
        data = pd.read_csv("credit_card_fraud/creditcard.csv", nrows=10000)  # Limit to first 10k rows
        return data
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'credit_card_fraud/creditcard.csv' exists.")
        return None

# Load data
models = load_models()
results = load_results()
sample_data = load_sample_data()

# Dashboard page
if page == "Dashboard":
    st.header("Fraud Detection Dashboard")
    
    if sample_data is not None:
        # Calculate fraud statistics
        fraud_count = sample_data["Class"].sum()
        total_count = len(sample_data)
        fraud_percentage = (fraud_count / total_count) * 100
        
        # Display KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{total_count:,}")
        with col2:
            st.metric("Fraudulent Transactions", f"{fraud_count:,}")
        with col3:
            st.metric("Fraud Rate", f"{fraud_percentage:.3f}%")
        with col4:
            if results is not None:
                best_model = results["auc"].idxmax()
                best_auc = results.loc[best_model, "auc"]
                st.metric("Best Model AUC", f"{best_auc:.4f}")
        
        # Transaction amount distribution
        st.subheader("Transaction Amount Distribution")
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        # Normal transactions
        normal = sample_data[sample_data["Class"] == 0]["Amount"]
        ax[0].hist(normal, bins=50, color="blue", alpha=0.7)
        ax[0].set_title("Normal Transactions")
        ax[0].set_xlabel("Amount (£)")
        ax[0].set_ylabel("Frequency")
        ax[0].set_yscale("log")
        
        # Fraudulent transactions
        fraud = sample_data[sample_data["Class"] == 1]["Amount"]
        ax[1].hist(fraud, bins=50, color="red", alpha=0.7)
        ax[1].set_title("Fraudulent Transactions")
        ax[1].set_xlabel("Amount (£)")
        ax[1].set_ylabel("Frequency")
        ax[1].set_yscale("log")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Time of day analysis
        st.subheader("Time of Day Analysis")
        
        # Convert time to hours since midnight
        sample_data["Hour"] = sample_data["Time"] / 3600 % 24
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by hour and calculate fraud rate
        hourly_data = sample_data.groupby(sample_data["Hour"].astype(int)).agg(
            {"Class": ["count", "sum"]}
        )
        hourly_data.columns = ["transactions", "frauds"]
        hourly_data["fraud_rate"] = (hourly_data["frauds"] / hourly_data["transactions"]) * 100
        
        ax.bar(hourly_data.index, hourly_data["transactions"], color="blue", alpha=0.5, label="Transactions")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Number of Transactions")
        ax.set_title("Transaction Volume by Hour of Day")
        
        ax2 = ax.twinx()
        ax2.plot(hourly_data.index, hourly_data["fraud_rate"], color="red", marker="o", label="Fraud Rate")
        ax2.set_ylabel("Fraud Rate (%)")
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        
        plt.tight_layout()
        st.pyplot(fig)

# Model Performance page
elif page == "Model Performance":
    st.header("Model Performance Analysis")
    
    if results is not None:
        # Display results table
        st.subheader("Model Evaluation Metrics")
        st.dataframe(results.style.highlight_max(axis=0, color="lightgreen"))
        
        # Create bar charts for key metrics
        metrics = ["auc", "precision", "recall", "f1"]
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            results_sorted = results.sort_values(metric, ascending=False)
            axes[i].bar(results_sorted.index, results_sorted[metric], color="skyblue")
            axes[i].set_title(f"{metric.upper()} Score Comparison")
            axes[i].set_xlabel("Model")
            axes[i].set_ylabel(f"{metric.upper()}")
            axes[i].set_xticklabels(results_sorted.index, rotation=45, ha="right")
            
            # Annotate bars with values
            for j, v in enumerate(results_sorted[metric]):
                axes[i].text(j, v + 0.01, f"{v:.4f}", ha="center")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # ROC Curves
        st.subheader("ROC Curves")
        
        # Use sample data for demonstration
        if sample_data is not None and models is not None:
            # Preprocess the sample data
            X_sample = sample_data.drop("Class", axis=1).iloc[:1000]  # Use a small sample
            y_sample = sample_data["Class"].iloc[:1000]
            
            # Scale Amount and Time
            scaler = StandardScaler()
            X_sample["Amount"] = scaler.fit_transform(X_sample["Amount"].values.reshape(-1, 1))
            X_sample["Time"] = scaler.fit_transform(X_sample["Time"].values.reshape(-1, 1))
            
            # Plot ROC curves for the meta models
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Base rate
            ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")
            
            for model_name in ["random_forest", "xgboost"]:
                if model_name in models:
                    model = models[model_name]
                    if hasattr(model, "predict_proba"):
                        try:
                            y_score = model.predict_proba(X_sample)[:, 1]
                            fpr, tpr, _ = roc_curve(y_sample, y_score)
                            roc_auc = auc(fpr, tpr)
                            ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")
                        except Exception as e:
                            st.warning(f"Could not generate ROC curve for {model_name}: {e}")
            
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Receiver Operating Characteristic (ROC) Curve")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show PR curves too
            st.subheader("Precision-Recall Curves")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Base rate
            ax.axhline(y=y_sample.mean(), color="k", linestyle="--", label=f"Baseline (Precision = {y_sample.mean():.4f})")
            
            for model_name in ["random_forest", "xgboost"]:
                if model_name in models:
                    model = models[model_name]
                    if hasattr(model, "predict_proba"):
                        try:
                            y_score = model.predict_proba(X_sample)[:, 1]
                            precision, recall, _ = precision_recall_curve(y_sample, y_score)
                            avg_precision = np.mean(precision)
                            ax.plot(recall, precision, lw=2, label=f"{model_name} (Avg Precision = {avg_precision:.4f})")
                        except Exception as e:
                            st.warning(f"Could not generate PR curve for {model_name}: {e}")
            
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

# Sample Prediction page
elif page == "Sample Prediction":
    st.header("Transaction Fraud Prediction")
    
    if models is not None and sample_data is not None:
        st.write("Enter transaction details to predict if it's fraudulent:")
        
        # Option to use a random sample or enter manual values
        input_option = st.radio("Input Method", ["Use a random transaction from the dataset", "Enter custom values"])
        
        if input_option == "Use a random transaction from the dataset":
            # Get a random transaction
            random_idx = np.random.randint(0, len(sample_data))
            random_transaction = sample_data.iloc[random_idx]
            
            # Display transaction details
            transaction_df = pd.DataFrame(random_transaction).T
            st.dataframe(transaction_df)
            
            # Prepare features
            X = transaction_df.drop("Class", axis=1)
            
            # Scale Amount and Time
            scaler = StandardScaler()
            X["Amount"] = scaler.fit_transform(X["Amount"].values.reshape(-1, 1))
            X["Time"] = scaler.fit_transform(X["Time"].values.reshape(-1, 1))
            
            # Actual label
            actual_label = "Fraudulent" if random_transaction["Class"] == 1 else "Legitimate"
            
            # Make prediction button
            if st.button("Predict"):
                with st.spinner("Running prediction models..."):
                    # Simulate some computation time
                    time.sleep(1.5)
                    
                    # Container for showing prediction
                    prediction_container = st.container()
                    
                    with prediction_container:
                        st.subheader("Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.info(f"Actual Transaction Label: {actual_label}")
                        
                        # Make predictions with different models
                        results = {}
                        
                        for model_name, model in models.items():
                            if hasattr(model, "predict_proba"):
                                pred_proba = model.predict_proba(X)[0, 1]
                                prediction = "Fraudulent" if pred_proba > 0.5 else "Legitimate"
                                results[model_name] = {"prediction": prediction, "confidence": pred_proba}
                            else:
                                pred = model.predict(X)[0]
                                prediction = "Fraudulent" if pred == 1 else "Legitimate"
                                results[model_name] = {"prediction": prediction, "confidence": None}
                        
                        # Display results table
                        results_df = pd.DataFrame.from_dict(results, orient="index")
                        st.dataframe(results_df)
                        
                        # Highlight the two-stage model if available
                        if "two_stage" in results:
                            st.subheader("Two-Stage Model Analysis")
                            st.info("""
                            The two-stage model combines high recall (Stage 1: Logistic Regression) with high precision
                            (Stage 2: XGBoost) to optimize fraud detection performance.
                            
                            - Stage 1 aims to catch all potential fraud (threshold = 0.3)
                            - Stage 2 reduces false positives (threshold = 0.7)
                            """)
                            
                            two_stage_result = results["two_stage"]
                            st.metric(
                                "Two-Stage Model Prediction",
                                two_stage_result["prediction"],
                                delta="High Confidence" if abs(two_stage_result["confidence"] - 0.5) > 0.3 else "Medium Confidence"
                            )
                        
                        # Visualize predictions
                        with col2:
                            for model_name, result in results.items():
                                if result["confidence"] is not None:
                                    # Create gauge chart
                                    confidence = result["confidence"]
                                    color = "red" if confidence > 0.5 else "green"
                                    
                                    fig, ax = plt.subplots(figsize=(4, 0.5))
                                    ax.barh(model_name, confidence, color=color)
                                    ax.barh(model_name, 1, color="lightgrey", alpha=0.3)
                                    ax.set_xlim(0, 1)
                                    ax.set_xlabel("Fraud Probability")
                                    plt.tight_layout()
                                    st.pyplot(fig)
        else:
            # Custom input form
            st.info("Note: Feature values V1-V28 are PCA-transformed features that have been anonymised for privacy.")
            
            # Create a simpler form for a few features to avoid overwhelming the user
            col1, col2 = st.columns(2)
            
            with col1:
                amount = st.number_input("Transaction Amount (£)", min_value=0.0, value=100.0)
                time = st.number_input("Time (seconds from first transaction)", min_value=0, value=43000)
                
            with col2:
                v1 = st.slider("V1 (Anonymised Feature)", min_value=-10.0, max_value=10.0, value=0.0)
                v2 = st.slider("V2 (Anonymised Feature)", min_value=-10.0, max_value=10.0, value=0.0)
            
            # Create dataframe with entered values and default values for other features
            # Fill in other V's with average values from the sample
            input_data = {"Amount": amount, "Time": time, "V1": v1, "V2": v2}
            
            # Add default values for V3-V28
            v_cols = [f"V{i}" for i in range(3, 29)]
            for col in v_cols:
                input_data[col] = sample_data[col].mean()
            
            input_df = pd.DataFrame([input_data])
            
            # Scale Amount and Time
            scaler = StandardScaler()
            input_df["Amount"] = scaler.fit_transform(input_df["Amount"].values.reshape(-1, 1))
            input_df["Time"] = scaler.fit_transform(input_df["Time"].values.reshape(-1, 1))
            
            # Make prediction button
            if st.button("Predict"):
                with st.spinner("Running prediction models..."):
                    # Simulate some computation time
                    time.sleep(1.5)
                    
                    # Container for showing prediction
                    prediction_container = st.container()
                    
                    with prediction_container:
                        st.subheader("Prediction Results")
                        
                        # Make predictions with different models
                        results = {}
                        
                        for model_name, model in models.items():
                            if hasattr(model, "predict_proba"):
                                pred_proba = model.predict_proba(input_df)[0, 1]
                                prediction = "Fraudulent" if pred_proba > 0.5 else "Legitimate"
                                results[model_name] = {"prediction": prediction, "confidence": pred_proba}
                            else:
                                pred = model.predict(input_df)[0]
                                prediction = "Fraudulent" if pred == 1 else "Legitimate"
                                results[model_name] = {"prediction": prediction, "confidence": None}
                        
                        # Display results table
                        results_df = pd.DataFrame.from_dict(results, orient="index")
                        st.dataframe(results_df)
                        
                        # Highlight the two-stage model if available
                        if "two_stage" in results:
                            st.subheader("Two-Stage Model Analysis")
                            st.info("""
                            The two-stage model combines high recall (Stage 1: Logistic Regression) with high precision
                            (Stage 2: XGBoost) to optimize fraud detection performance.
                            
                            - Stage 1 aims to catch all potential fraud (threshold = 0.3)
                            - Stage 2 reduces false positives (threshold = 0.7)
                            """)
                            
                            two_stage_result = results["two_stage"]
                            st.metric(
                                "Two-Stage Model Prediction",
                                two_stage_result["prediction"],
                                delta="High Confidence" if abs(two_stage_result["confidence"] - 0.5) > 0.3 else "Medium Confidence"
                            )
                        
                        # Visualize predictions
                        for model_name, result in results.items():
                            if result["confidence"] is not None:
                                # Create gauge chart
                                confidence = result["confidence"]
                                color = "red" if confidence > 0.5 else "green"
                                
                                fig, ax = plt.subplots(figsize=(8, 0.5))
                                ax.barh(model_name, confidence, color=color)
                                ax.barh(model_name, 1, color="lightgrey", alpha=0.3)
                                ax.set_xlim(0, 1)
                                ax.set_xlabel("Fraud Probability")
                                plt.tight_layout()
                                st.pyplot(fig)

# About page
elif page == "About":
    st.header("About This Project")
    
    st.markdown("""
    ## Credit Card Fraud Detection
    
    This project demonstrates an advanced approach to credit card fraud detection using machine learning techniques.
    
    ### Methodology
    
    The system uses a stacked ensemble learning approach, combining multiple base classifiers:
    
    - Logistic Regression
    - Random Forest
    - Decision Tree
    - K-Nearest Neighbors
    - Naive Bayes
    - Gradient Boosting
    - XGBoost
    
    These base models are combined using meta-learners (Logistic Regression and Gradient Boosting) that learn from the predictions of the base models.
    
    ### Two-Stage Fraud Detection Approach
    
    The system implements a specialized two-stage approach optimized for credit card fraud detection:
    
    1. **Stage 1 (High Recall)**: Uses Logistic Regression with a lower threshold (0.3) to catch as many potential fraudulent transactions as possible
    2. **Stage 2 (High Precision)**: Transactions flagged by Stage 1 are passed to XGBoost with a higher threshold (0.7) to reduce false positives
    
    This approach prioritizes recall (catching all fraud) while using a second stage to minimize false positives, balancing the operational costs of fraud investigation with the need to catch all fraudulent activity.
    
    ### Key Features
    
    - **Two-stage detection pipeline**: Optimizes the trade-off between catching all fraud and minimizing false positives
    - **Handles class imbalance**: Uses SMOTE (Synthetic Minority Over-sampling Technique) to address the imbalance between fraudulent and legitimate transactions
    - **Optimises for relevant metrics**: Focuses on recall and AUC, which are critical in fraud detection
    - **Ensemble approach**: Leverages the strengths of diverse models to improve overall performance
    - **Feature scaling**: Standardises transaction amount and time features for better model performance
    
    ### Dataset
    
    The dataset used in this project contains anonymised credit card transactions, with features transformed using PCA for privacy protection.
    
    ### Contact
    
    For more information, please contact:
    
    - **Email**: alex@example.com
    - **GitHub**: [github.com/alexuser](https://github.com/alexuser)
    - **LinkedIn**: [linkedin.com/in/alexuser](https://linkedin.com/in/alexuser)
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Credit Card Fraud Detection Project | Created by Alex")
