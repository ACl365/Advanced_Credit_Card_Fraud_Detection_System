#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for credit card fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, 
    accuracy_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging

logger = logging.getLogger(__name__)


def load_data(file_path, sample_size=None):
    """
    Load the credit card transaction data.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file.
    sample_size : int, optional
        Number of rows to sample. If None, all data is loaded.
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data.
    """
    try:
        if sample_size:
            data = pd.read_csv(file_path, nrows=sample_size)
        else:
            data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def compute_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Compute various classification metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_pred_proba : array-like, optional
        Predicted probabilities for the positive class.
        
    Returns:
    --------
    dict
        Dictionary of computed metrics.
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # AUC if probability estimates are available
    if y_pred_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def save_model(model, model_path):
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : estimator
        Trained model to save.
    model_path : str
        Path to save the model to.
        
    Returns:
    --------
    bool
        True if the model was saved successfully.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False


def load_model(model_path):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model.
        
    Returns:
    --------
    estimator
        Loaded model.
    """
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(8, 6), cmap='Blues', normalize=False):
    """
    Plot a confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        List of labels to index the matrix.
    figsize : tuple, optional
        Figure size.
    cmap : str, optional
        Colormap to use.
    normalize : bool, optional
        Whether to normalize the confusion matrix.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, cbar=False, ax=ax)
    
    # Set labels
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    # Return the figure
    return fig


def plot_feature_importance(model, feature_names, top_n=20, figsize=(12, 8)):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : estimator
        Trained model with feature_importances_ attribute.
    feature_names : list
        List of feature names.
    top_n : int, optional
        Number of top features to plot.
    figsize : tuple, optional
        Figure size.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    try:
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame for plotting
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot horizontal bar chart
        sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
        
        # Set labels
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        # Return the figure
        return fig
    except AttributeError:
        logger.warning("Model does not have feature_importances_ attribute.")
        return None


def calculate_threshold_metrics(y_true, y_score, thresholds=None):
    """
    Calculate metrics at different classification thresholds.
    
    Parameters:
    -----------
    y_true : array-like
        True labels.
    y_score : array-like
        Predicted scores or probabilities.
    thresholds : list, optional
        List of thresholds to evaluate. If None, uses np.arange(0, 1.01, 0.05).
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with metrics at each threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0, 1.01, 0.05)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })
    
    return pd.DataFrame(results)


def get_optimal_threshold(y_true, y_score, metric='f1'):
    """
    Find the optimal threshold based on a specified metric.
    
    Parameters:
    -----------
    y_true : array-like
        True labels.
    y_score : array-like
        Predicted scores or probabilities.
    metric : str, optional
        Metric to optimise. Options: 'f1', 'precision', 'recall', 'accuracy'.
        
    Returns:
    --------
    float
        Optimal threshold value.
    """
    threshold_metrics = calculate_threshold_metrics(y_true, y_score)
    optimal_threshold = threshold_metrics.loc[threshold_metrics[metric].idxmax(), 'threshold']
    
    logger.info(f"Optimal threshold for {metric}: {optimal_threshold}")
    return optimal_threshold