#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the credit card fraud detection model.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile

# Add the parent directory to the path for importing the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the CreditCardFraudDetection class from the improved-main-py module
from improved_main_py import CreditCardFraudDetection


class TestCreditCardFraudDetection(unittest.TestCase):
    """Test cases for the CreditCardFraudDetection class."""

    def setUp(self):
        """Set up test data."""
        # Create a small synthetic dataset for testing
        np.random.seed(42)
        n_samples = 1000
        n_features = 30
        
        # Generate feature data
        X = np.random.randn(n_samples, n_features)
        
        # Generate target data (imbalanced)
        y = np.zeros(n_samples)
        y[:50] = 1  # 5% fraud rate
        
        # Create feature names
        feature_names = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        
        # Create DataFrame
        self.data = pd.DataFrame(X, columns=feature_names)
        self.data["Class"] = y
        
        # Create a temporary CSV file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.temp_dir.name, "test_data.csv")
        self.data.to_csv(self.data_path, index=False)
        
        # Create a fraud detection instance
        self.fraud_detector = CreditCardFraudDetection(
            data_path=self.data_path,
            random_state=42,
            test_size=0.2
        )

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_load_and_preprocess_data(self):
        """Test data loading and preprocessing."""
        # Load and preprocess data
        X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled = \
            self.fraud_detector.load_and_preprocess_data()
        
        # Check dimensions
        self.assertEqual(X_train.shape[0] + X_test.shape[0], self.data.shape[0])
        self.assertEqual(X_train.shape[1], self.data.shape[1] - 1)  # Minus Class column
        
        # Check resampling
        self.assertGreater(
            y_train_resampled.mean(),
            y_train.mean(),
            "SMOTE should increase the proportion of fraud cases"
        )
        
        # Check class balance after SMOTE
        unique, counts = np.unique(y_train_resampled, return_counts=True)
        self.assertEqual(len(unique), 2, "Should have two classes")
        self.assertEqual(counts[0], counts[1], "Classes should be balanced after SMOTE")

    def test_train_base_models(self):
        """Test training of base models."""
        # Load and preprocess data
        self.fraud_detector.load_and_preprocess_data()
        
        # Train base models
        models = self.fraud_detector.train_base_models()
        
        # Check that models were created
        expected_models = [
            'logistic', 'random_forest', 'decision_tree', 
            'knn', 'naive_bayes', 'gradient_boosting', 'xgboost'
        ]
        
        for model_name in expected_models:
            self.assertIn(model_name, models, f"{model_name} model should be created")

    def test_get_oof_predictions(self):
        """Test generation of out-of-fold predictions."""
        # Load and preprocess data
        self.fraud_detector.load_and_preprocess_data()
        
        # Train a base model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
        model.fit(self.fraud_detector.X_train_resampled, self.fraud_detector.y_train_resampled)
        
        # Get OOF predictions
        oof_train, test_preds = self.fraud_detector.get_oof_predictions(
            model,
            self.fraud_detector.X_train_resampled,
            self.fraud_detector.y_train_resampled,
            self.fraud_detector.X_test,
            n_splits=2
        )
        
        # Check dimensions
        self.assertEqual(len(oof_train), len(self.fraud_detector.X_train_resampled))
        self.assertEqual(len(test_preds), len(self.fraud_detector.X_test))
        
        # Check that predictions are binary
        self.assertTrue(np.all((oof_train == 0) | (oof_train == 1)))
        self.assertTrue(np.all((test_preds == 0) | (test_preds == 1)))

    def test_create_meta_features(self):
        """Test creation of meta-features."""
        # Load and preprocess data
        self.fraud_detector.load_and_preprocess_data()
        
        # Train base models
        self.fraud_detector.train_base_models()
        
        # Create meta-features
        meta_train, meta_test = self.fraud_detector.create_meta_features(n_splits=2)
        
        # Check dimensions
        self.assertEqual(meta_train.shape[0], len(self.fraud_detector.X_train_resampled))
        self.assertEqual(meta_test.shape[0], len(self.fraud_detector.X_test))
        
        # Check that we have the right number of meta-features
        # We expect 4 meta-features (from logistic, random_forest, gradient_boosting, xgboost)
        self.assertEqual(meta_train.shape[1], 4)
        self.assertEqual(meta_test.shape[1], 4)

    def test_train_meta_learners(self):
        """Test training of meta-learners."""
        # Load and preprocess data
        self.fraud_detector.load_and_preprocess_data()
        
        # Train base models
        self.fraud_detector.train_base_models()
        
        # Create meta-features
        self.fraud_detector.create_meta_features(n_splits=2)
        
        # Train meta-learners
        meta_learners = self.fraud_detector.train_meta_learners()
        
        # Check that meta-learners were created
        expected_meta_learners = ['logistic', 'gradient_boosting']
        
        for meta_learner_name in expected_meta_learners:
            self.assertIn(
                meta_learner_name, 
                meta_learners, 
                f"{meta_learner_name} meta-learner should be created"
            )

    def test_evaluate_models(self):
        """Test model evaluation."""
        # Load and preprocess data
        self.fraud_detector.load_and_preprocess_data()
        
        # Train base models
        self.fraud_detector.train_base_models()
        
        # Create meta-features
        self.fraud_detector.create_meta_features(n_splits=2)
        
        # Train meta-learners
        self.fraud_detector.train_meta_learners()
        
        # Evaluate models
        results = self.fraud_detector.evaluate_models()
        
        # Check that results were created for all models
        expected_models = [
            'logistic', 'random_forest', 'decision_tree', 
            'knn', 'naive_bayes', 'gradient_boosting', 'xgboost',
            'meta_logistic', 'meta_gradient_boosting'
        ]
        
        for model_name in expected_models:
            self.assertIn(model_name, results, f"{model_name} results should be created")
        
        # Check that all expected metrics are present
        expected_metrics = ['auc', 'precision', 'recall', 'f1', 'accuracy']
        
        for metric_name in expected_metrics:
            self.assertIn(
                metric_name, 
                results['logistic'], 
                f"{metric_name} metric should be computed"
            )

    def test_run_pipeline(self):
        """Test the complete pipeline."""
        # Run the pipeline
        best_model, results = self.fraud_detector.run_pipeline()
        
        # Check that we have a best model
        self.assertIsNotNone(best_model, "Should have a best model")
        
        # Check that we have results
        self.assertIsNotNone(results, "Should have results")
        self.assertGreater(len(results), 0, "Results should not be empty")


if __name__ == "__main__":
    unittest.main()
