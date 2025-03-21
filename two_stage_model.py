import numpy as np
import joblib
import logging
import os
from sklearn.base import BaseEstimator, ClassifierMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TwoStageClassifier(BaseEstimator, ClassifierMixin):
    """
    A two-stage classifier for credit card fraud detection.
    
    Stage 1: High-recall model (Logistic Regression) to catch as many fraudulent transactions as possible
    Stage 2: High-precision model (XGBoost or Meta-Gradient Boosting) to reduce false positives
    
    Parameters
    ----------
    stage1_model_path : str
        Path to the stage 1 model (high recall)
    stage2_model_path : str
        Path to the stage 2 model (high precision)
    threshold1 : float, default=0.5
        Probability threshold for stage 1 model
    threshold2 : float, default=0.5
        Probability threshold for stage 2 model
    """
    
    def __init__(self, stage1_model_path, stage2_model_path, threshold1=0.5, threshold2=0.5):
        self.stage1_model_path = stage1_model_path
        self.stage2_model_path = stage2_model_path
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        
        # Load models
        if os.path.exists(stage1_model_path) and os.path.exists(stage2_model_path):
            self.stage1_model = joblib.load(stage1_model_path)
            self.stage2_model = joblib.load(stage2_model_path)
            logger.info(f"Loaded stage 1 model from {stage1_model_path}")
            logger.info(f"Loaded stage 2 model from {stage2_model_path}")
        else:
            logger.error(f"Model files not found: {stage1_model_path} or {stage2_model_path}")
            raise FileNotFoundError(f"Model files not found: {stage1_model_path} or {stage2_model_path}")
    
    def fit(self, X, y):
        """
        Fit method (not used as we're loading pre-trained models)
        """
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        array of shape (n_samples, 2)
            The class probabilities of the input samples.
        """
        # Get stage 1 probabilities
        stage1_probs = self.stage1_model.predict_proba(X)
        
        # For samples that pass stage 1 threshold, get stage 2 probabilities
        stage1_decisions = stage1_probs[:, 1] >= self.threshold1
        
        # Initialize final probabilities with stage 1 results
        final_probs = stage1_probs.copy()
        
        # For samples that pass stage 1, update with stage 2 probabilities
        if np.any(stage1_decisions):
            X_stage2 = X[stage1_decisions]
            stage2_probs = self.stage2_model.predict_proba(X_stage2)
            final_probs[stage1_decisions] = stage2_probs
        
        return final_probs
    
    def predict(self, X):
        """
        Predict class labels for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        array of shape (n_samples,)
            The predicted classes.
        """
        # Get stage 1 probabilities and decisions
        stage1_probs = self.stage1_model.predict_proba(X)
        stage1_decisions = stage1_probs[:, 1] >= self.threshold1
        
        # Initialize predictions as all negative (0)
        predictions = np.zeros(X.shape[0], dtype=int)
        
        # For samples that pass stage 1, apply stage 2
        if np.any(stage1_decisions):
            X_stage2 = X[stage1_decisions]
            stage2_probs = self.stage2_model.predict_proba(X_stage2)
            stage2_decisions = stage2_probs[:, 1] >= self.threshold2
            
            # Only mark as positive if both stages agree
            predictions[stage1_decisions] = stage2_decisions.astype(int)
        
        return predictions
    
    def decision_function(self, X):
        """
        Decision function for the model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        array of shape (n_samples,)
            The decision function of the samples.
        """
        probs = self.predict_proba(X)
        return probs[:, 1]

def create_two_stage_model(threshold1=0.3, threshold2=0.7):
    """
    Create a two-stage model for credit card fraud detection.
    
    Parameters
    ----------
    threshold1 : float, default=0.3
        Probability threshold for stage 1 model (Logistic Regression)
        Lower threshold increases recall
    threshold2 : float, default=0.7
        Probability threshold for stage 2 model (XGBoost)
        Higher threshold increases precision
        
    Returns
    -------
    model : TwoStageClassifier
        The two-stage classifier model
    """
    # Paths to the pre-trained models
    stage1_model_path = "models/logistic_model.pkl"  # High recall model
    stage2_model_path = "models/xgboost_model.pkl"   # High precision model
    
    # Create and return the two-stage model
    model = TwoStageClassifier(
        stage1_model_path=stage1_model_path,
        stage2_model_path=stage2_model_path,
        threshold1=threshold1,
        threshold2=threshold2
    )
    
    return model

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    
    # Create a synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, 
                              weights=[0.99, 0.01], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the two-stage model
    model = create_two_stage_model(threshold1=0.3, threshold2=0.7)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    print(classification_report(y_test, y_pred))
    print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")