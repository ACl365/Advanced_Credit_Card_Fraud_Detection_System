#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Credit Card Fraud Detection using Ensemble Learning
---------------------------------------------------
This module implements an ensemble learning approach to detect credit card fraud.
The implementation uses stacking ensemble with multiple base models and meta-learners.

Author: Alex
Last Updated: March 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_score, 
    recall_score, f1_score, accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
import joblib
import logging
import os
import time
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fraud_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


class CreditCardFraudDetection:
    """
    A class for building and evaluating credit card fraud detection models.
    
    This class implements a stacked ensemble learning approach to detect credit card fraud.
    It provides methods for data preprocessing, model training, and evaluation.
    """
    
    def __init__(self, data_path="credit_card_fraud/creditcard.csv", random_state=42, test_size=0.2):
        """
        Initialise the credit card fraud detection model.
        
        Parameters:
        -----------
        data_path : str
            Path to the credit card transaction data CSV file.
        random_state : int, optional (default=42)
            Random seed for reproducibility.
        test_size : float, optional (default=0.2)
            Proportion of the dataset to include in the test split.
        """
        self.data_path = data_path  # Corrected path
        self.random_state = random_state
        self.test_size = test_size
        self.models = {}
        self.model_predictions = {}
        self.meta_learners = {}
        self.scaler = StandardScaler()
        
        # Create output directory for model artifacts
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess the credit card transaction data.
        
        Returns:
        --------
        X_train, X_test, y_train, y_test : tuples
            Preprocessed training and testing data.
        """
        logger.info(f"Loading data from {self.data_path}")
        start_time = time.time()
        
        # Load the dataset
        try:
            data = pd.read_csv(self.data_path)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        
        # Data exploration summary
        logger.info("Data summary statistics:")
        logger.info(f"Total transactions: {data.shape[0]}")
        logger.info(f"Fraudulent transactions: {data['Class'].sum()}")
        logger.info(f"Fraud rate: {data['Class'].mean() * 100:.4f}%")
        
        # Preprocessing
        logger.info("Starting data preprocessing...")
        
        # Scale the 'Amount' and 'Time' features
        data['Amount'] = self.scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
        data['Time'] = self.scaler.fit_transform(data['Time'].values.reshape(-1, 1))
        
        # Drop duplicate rows
        initial_rows = data.shape[0]
        data.drop_duplicates(inplace=True)
        logger.info(f"Removed {initial_rows - data.shape[0]} duplicate transactions")
        
        # Split the data into training and testing sets
        X = data.drop('Class', axis=1)
        y = data['Class']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        logger.info(f"Training set fraud rate: {y_train.mean() * 100:.4f}%")
        logger.info(f"Test set fraud rate: {y_test.mean() * 100:.4f}%")
        
        # Handle class imbalance using SMOTE oversampling
        logger.info("Applying SMOTE to handle class imbalance...")
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info(f"After SMOTE: Training set size: {X_train_resampled.shape[0]}")
        logger.info(f"After SMOTE: Training set fraud rate: {y_train_resampled.mean() * 100:.4f}%")
        
        logger.info(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_resampled = X_train_resampled
        self.y_train_resampled = y_train_resampled
        
        return X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled
    
    def train_base_models(self):
        """
        Train the base models for the stacking ensemble.
        
        Returns:
        --------
        models : dict
            Dictionary of trained base models.
        """
        logger.info("Training base models...")
        
        # Define base models
        base_models = {
            'logistic': LogisticRegression(max_iter=1000, n_jobs=-1),
            'random_forest': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            'decision_tree': DecisionTreeClassifier(random_state=self.random_state),
            'knn': KNeighborsClassifier(n_jobs=-1),
            'naive_bayes': GaussianNB(),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
            'xgboost': xgb.XGBClassifier(random_state=self.random_state, n_jobs=-1)
        }
        
        # Train base models
        for name, model in base_models.items():
            logger.info(f"Training {name}...")
            start_time = time.time()
            try:
                model.fit(self.X_train_resampled, self.y_train_resampled)
                self.models[name] = model
                logger.info(f"{name} trained in {time.time() - start_time:.2f} seconds")
                
                # Save model
                joblib.dump(model, f"models/{name}_model.pkl")
                logger.info(f"Saved {name} model to models/{name}_model.pkl")
                
            except Exception as e:
                logger.error(f"Error training {name} model: {e}")
        
        return self.models
    
    def get_oof_predictions(self, model, X_train, y_train, X_test, n_splits=5):
        """
        Generate out-of-fold predictions for stacking.
        
        Parameters:
        -----------
        model : estimator
            The model to use for generating out-of-fold predictions.
        X_train : array-like
            Training features.
        y_train : array-like
            Training labels.
        X_test : array-like
            Test features.
        n_splits : int, optional (default=5)
            Number of folds for cross-validation.
            
        Returns:
        --------
        oof_predictions, test_predictions : tuple
            Out-of-fold predictions for the training set and predictions for the test set.
        """
        oof_predictions = np.zeros((len(X_train),))
        test_predictions = np.zeros((len(X_test),))
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
            
            model.fit(X_train_fold, y_train_fold)
            oof_predictions[val_index] = model.predict(X_val_fold)
            test_predictions += model.predict(X_test) / n_splits
        
        return oof_predictions, test_predictions
    
    def create_meta_features(self, n_splits=5):
        """
        Create meta-features for stacking using out-of-fold predictions.
        
        Parameters:
        -----------
        n_splits : int, optional (default=5)
            Number of folds for cross-validation.
            
        Returns:
        --------
        meta_train, meta_test : tuple
            Meta-features for training and testing the meta-learners.
        """
        logger.info(f"Creating meta-features using {n_splits}-fold cross-validation...")
        
        oof_train_predictions = {}
        test_predictions = {}
        
        # Get out-of-fold predictions for each base model
        for name, model in self.models.items():
            logger.info(f"Generating OOF predictions for {name}...")
            start_time = time.time()
            
            oof_train, test_pred = self.get_oof_predictions(
                model, 
                self.X_train_resampled, 
                self.y_train_resampled, 
                self.X_test,
                n_splits=n_splits
            )
            
            oof_train_predictions[name] = oof_train
            test_predictions[name] = test_pred
            
            logger.info(f"OOF predictions for {name} completed in {time.time() - start_time:.2f} seconds")
        
        # Store predictions for later analysis
        self.model_predictions = test_predictions
        
        # Create meta-features - select top performing models based on domain knowledge
        selected_models = ['logistic', 'random_forest', 'gradient_boosting', 'xgboost']
        logger.info(f"Selected models for meta-features: {selected_models}")
        
        meta_train = np.column_stack([oof_train_predictions[name] for name in selected_models])
        meta_test = np.column_stack([test_predictions[name] for name in selected_models])
        
        logger.info(f"Meta-features created. Meta-train shape: {meta_train.shape}, Meta-test shape: {meta_test.shape}")
        
        self.meta_train = meta_train
        self.meta_test = meta_test
        
        return meta_train, meta_test
    
    def train_meta_learners(self):
        """
        Train meta-learners using the meta-features.
        
        Returns:
        --------
        meta_learners : dict
            Dictionary of trained meta-learners.
        """
        logger.info("Training meta-learners...")
        
        # Meta-learner 1: Logistic Regression
        logger.info("Training Logistic Regression meta-learner...")
        param_grid_logistic = {'C': [0.01, 0.1, 1, 10, 100]}
        grid_search_logistic = GridSearchCV(
            LogisticRegression(max_iter=1000), 
            param_grid_logistic, 
            scoring='roc_auc', 
            cv=5,
            n_jobs=-1
        )
        
        start_time = time.time()
        grid_search_logistic.fit(self.meta_train, self.y_train_resampled)
        logger.info(f"Logistic Regression meta-learner trained in {time.time() - start_time:.2f} seconds")
        
        logger.info(f"Best Logistic Regression parameters: {grid_search_logistic.best_params_}")
        meta_learner_logistic = grid_search_logistic.best_estimator_
        self.meta_learners['logistic'] = meta_learner_logistic
        
        # Meta-learner 2: Gradient Boosting
        logger.info("Training Gradient Boosting meta-learner...")
        param_grid_gb = {
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'max_depth': [3, 5]
        }
        grid_search_gb = GridSearchCV(
            GradientBoostingClassifier(random_state=self.random_state), 
            param_grid_gb, 
            scoring='roc_auc', 
            cv=5,
            n_jobs=-1
        )
        
        start_time = time.time()
        grid_search_gb.fit(self.meta_train, self.y_train_resampled)
        logger.info(f"Gradient Boosting meta-learner trained in {time.time() - start_time:.2f} seconds")
        
        logger.info(f"Best Gradient Boosting parameters: {grid_search_gb.best_params_}")
        meta_learner_gb = grid_search_gb.best_estimator_
        self.meta_learners['gradient_boosting'] = meta_learner_gb
        
        # Save meta-learners
        for name, model in self.meta_learners.items():
            joblib.dump(model, f"models/meta_learner_{name}_model.pkl")
            logger.info(f"Saved {name} meta-learner model to models/meta_learner_{name}_model.pkl")
        
        return self.meta_learners
    
    def evaluate_models(self):
        """
        Evaluate all models including base models and meta-learners.
        
        Returns:
        --------
        results : dict
            Dictionary of evaluation results for all models.
        """
        logger.info("Evaluating models...")
        
        results = {}
        
        # Evaluate base models
        for name, model in self.models.items():
            logger.info(f"Evaluating {name}...")
            
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            auc = roc_auc_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            results[name] = {
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            }
            
            logger.info(f"{name} - AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
            
            # Save detailed classification report
            with open(f"results/{name}_classification_report.txt", 'w') as f:
                f.write(f"{name} Classification Report:\n")
                f.write(classification_report(self.y_test, y_pred))
                f.write(f"\nAUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        
        # Evaluate meta-learners
        for name, model in self.meta_learners.items():
            logger.info(f"Evaluating meta-learner: {name}...")
            
            y_pred_meta = model.predict(self.meta_test)
            
            # Calculate metrics
            auc_meta = roc_auc_score(self.y_test, y_pred_meta)
            precision_meta = precision_score(self.y_test, y_pred_meta)
            recall_meta = recall_score(self.y_test, y_pred_meta)
            f1_meta = f1_score(self.y_test, y_pred_meta)
            accuracy_meta = accuracy_score(self.y_test, y_pred_meta)
            
            results[f"meta_{name}"] = {
                'auc': auc_meta,
                'precision': precision_meta,
                'recall': recall_meta,
                'f1': f1_meta,
                'accuracy': accuracy_meta
            }
            
            logger.info(f"Meta-Learner ({name}) - AUC: {auc_meta:.4f}, Precision: {precision_meta:.4f}, Recall: {recall_meta:.4f}, F1: {f1_meta:.4f}, Accuracy: {accuracy_meta:.4f}")
            
            # Save detailed classification report
            with open(f"results/meta_learner_{name}_classification_report.txt", 'w') as f:
                f.write(f"Meta-Learner ({name}) Classification Report:\n")
                f.write(classification_report(self.y_test, y_pred_meta))
                f.write(f"\nAUC: {auc_meta:.4f}, Precision: {precision_meta:.4f}, Recall: {recall_meta:.4f}, F1: {f1_meta:.4f}, Accuracy: {accuracy_meta:.4f}")
        
        # Save overall results to CSV
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.to_csv("results/model_evaluation_metrics.csv")
        logger.info("Model evaluation results saved to results/model_evaluation_metrics.csv")
        
        self.results = results
        return results
    
    def generate_visualisations(self):
        """
        Generate visualisations of model performance.
        """
        logger.info("Generating visualisations...")
        
        # Create results directory if it doesn't exist
        os.makedirs('results/figures', exist_ok=True)
        
        # 1. Metrics comparison bar chart
        plt.figure(figsize=(12, 8))
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        results_df.plot(kind='bar', figsize=(15, 8))
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/figures/model_comparison.png', dpi=300)
        plt.close()
        
        # 2. Confusion matrices for best models
        for name in ['meta_gradient_boosting', 'meta_logistic']:
            if name == 'meta_gradient_boosting':
                y_pred = self.meta_learners['gradient_boosting'].predict(self.meta_test)
                title = 'Gradient Boosting Meta-Learner'
            else:
                y_pred = self.meta_learners['logistic'].predict(self.meta_test)
                title = 'Logistic Regression Meta-Learner'
            
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Confusion Matrix - {title}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(f'results/figures/confusion_matrix_{name.lower()}.png', dpi=300)
            plt.close()
        
        # 3. ROC curves
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curves for meta-learners
        for name, model in self.meta_learners.items():
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(self.meta_test)[:, 1]
            else:
                y_score = model.predict(self.meta_test)
                
            fpr, tpr, _ = roc_curve(self.y_test, y_score)
            roc_auc = roc_auc_score(self.y_test, y_score)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        
        # Plot baseline
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/figures/roc_curves.png', dpi=300)
        plt.close()
        
        # 4. Precision-Recall curves
        plt.figure(figsize=(10, 8))
        
        # Plot PR curves for meta-learners
        for name, model in self.meta_learners.items():
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(self.meta_test)[:, 1]
            else:
                y_score = model.predict(self.meta_test)
                
            precision, recall, _ = precision_recall_curve(self.y_test, y_score)
            avg_precision = average_precision_score(self.y_test, y_score)
            plt.plot(recall, precision, lw=2, label=f'{name} (AP = {avg_precision:.4f})')
        
        # Plot baseline
        plt.axhline(y=self.y_test.mean(), color='k', linestyle='--', label='Baseline')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/figures/precision_recall_curves.png', dpi=300)
        plt.close()
        
        logger.info("Visualisations saved to results/figures/")
    
    def run_pipeline(self):
        """
        Run the complete fraud detection pipeline.
        """
        logger.info("Starting credit card fraud detection pipeline...")
        start_time = time.time()
        
        # Step 1: Load and preprocess data
        self.load_and_preprocess_data()
        
        # Step 2: Train base models
        self.train_base_models()
        
        # Step 3: Create meta-features
        self.create_meta_features()
        
        # Step 4: Train meta-learners
        self.train_meta_learners()
        
        # Step 5: Evaluate models
        self.evaluate_models()
        
        # Step 6: Generate visualisations
        self.generate_visualisations()
        
        logger.info(f"Pipeline completed in {time.time() - start_time:.2f} seconds")
        
        # Return best model based on AUC
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        best_model_name = results_df['auc'].idxmax()
        
        if best_model_name.startswith('meta_'):
            meta_name = best_model_name.replace('meta_', '')
            best_model = self.meta_learners[meta_name]
            logger.info(f"Best model: {best_model_name} (AUC: {results_df.loc[best_model_name, 'auc']:.4f})")
        else:
            best_model = self.models[best_model_name]
            logger.info(f"Best model: {best_model_name} (AUC: {results_df.loc[best_model_name, 'auc']:.4f})")
        
        return best_model, results_df


def main():
    """
    Main function to run the credit card fraud detection pipeline.
    """
    data_path = "credit_card_fraud/creditcard.csv"  # Corrected path to the data
    
    # Create an instance of the fraud detection class
    fraud_detector = CreditCardFraudDetection(data_path=data_path)
    
    # Run the pipeline
    best_model, results = fraud_detector.run_pipeline()
    
    # Print summary of results
    print("\nModel Performance Summary:")
    print(results.sort_values('auc', ascending=False))
    
    print(f"\nBest Model: {results['auc'].idxmax()} (AUC: {results['auc'].max():.4f})")


if __name__ == "__main__":
    main()