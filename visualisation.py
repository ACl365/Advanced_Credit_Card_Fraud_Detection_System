#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualisation functions for credit card fraud detection.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)


def setup_plotting_style():
    """
    Set up the plotting style for consistent visualisations.
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Set colour palette
    sns.set_palette("muted")
    
    # Set context
    sns.set_context("notebook", font_scale=1.2)
    
    # Set default figure size
    plt.rcParams["figure.figsize"] = (10, 6)
    
    # Set font
    plt.rcParams["font.family"] = "sans-serif"
    
    # Set matplotlib defaults
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12


def plot_class_distribution(data, target_col, title=None, figsize=(10, 6), save_path=None):
    """
    Plot the class distribution for a binary classification problem.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data.
    target_col : str
        Name of the target column.
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    # Count classes
    class_counts = data[target_col].value_counts()
    
    # Calculate percentages
    class_percentages = 100 * class_counts / len(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    ax.bar(
        class_counts.index,
        class_counts.values,
        color=["#3498db", "#e74c3c"],
        edgecolor="black",
        alpha=0.8,
    )
    
    # Add count labels
    for i, (count, percentage) in enumerate(zip(class_counts.values, class_percentages)):
        ax.text(
            i,
            count + max(class_counts) * 0.02,
            f"{count:,} ({percentage:.2f}%)",
            ha="center",
            fontweight="bold",
        )
    
    # Set labels and title
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title or "Class Distribution")
    ax.set_xticks(class_counts.index)
    ax.set_xticklabels(["Normal (0)", "Fraud (1)"])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Class distribution plot saved to {save_path}")
    
    return fig


def plot_correlation_matrix(data, figsize=(12, 10), cmap="coolwarm", mask_upper=True, 
                            save_path=None):
    """
    Plot a correlation matrix.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data.
    figsize : tuple, optional
        Figure size.
    cmap : str, optional
        Colormap to use.
    mask_upper : bool, optional
        Whether to mask the upper triangle of the matrix.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    # Calculate correlation matrix
    corr = data.corr()
    
    # Create mask for upper triangle if requested
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        annot=False,
        fmt=".2f",
        ax=ax,
    )
    
    # Set title
    ax.set_title("Correlation Matrix", fontsize=16)
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Correlation matrix plot saved to {save_path}")
    
    return fig


def plot_roc_curves(y_true, model_scores, model_names=None, figsize=(10, 8), save_path=None):
    """
    Plot ROC curves for multiple models.
    
    Parameters:
    -----------
    y_true : array-like
        True labels.
    model_scores : list of array-like
        List of predicted scores or probabilities from different models.
    model_names : list, optional
        List of model names for the legend.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    # Create default model names if not provided
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(model_scores))]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot random classifier
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier (AUC = 0.5)")
    
    # Plot ROC curve for each model
    for scores, name in zip(model_scores, model_names):
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")
    
    # Set labels and title
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves")
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc="lower right")
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"ROC curves plot saved to {save_path}")
    
    return fig


def plot_precision_recall_curves(y_true, model_scores, model_names=None, figsize=(10, 8), 
                                 save_path=None):
    """
    Plot precision-recall curves for multiple models.
    
    Parameters:
    -----------
    y_true : array-like
        True labels.
    model_scores : list of array-like
        List of predicted scores or probabilities from different models.
    model_names : list, optional
        List of model names for the legend.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    # Create default model names if not provided
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(model_scores))]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot baseline
    no_skill = len(y_true[y_true == 1]) / len(y_true)
    ax.plot([0, 1], [no_skill, no_skill], "k--", lw=2, 
            label=f"Baseline (AP = {no_skill:.4f})")
    
    # Plot precision-recall curve for each model
    for scores, name in zip(model_scores, model_names):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        avg_precision = average_precision_score(y_true, scores)
        ax.plot(recall, precision, lw=2, label=f"{name} (AP = {avg_precision:.4f})")
    
    # Set labels and title
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc="upper right")
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Precision-recall curves plot saved to {save_path}")
    
    return fig


def plot_threshold_metrics(y_true, y_score, figsize=(12, 8), save_path=None):
    """
    Plot metrics as a function of classification threshold.
    
    Parameters:
    -----------
    y_true : array-like
        True labels.
    y_score : array-like
        Predicted scores or probabilities.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    # Calculate metrics at different thresholds
    thresholds = np.arange(0, 1.01, 0.01)
    metrics = []
    
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        metrics.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot metrics
    ax.plot(metrics_df["threshold"], metrics_df["precision"], "b-", lw=2, label="Precision")
    ax.plot(metrics_df["threshold"], metrics_df["recall"], "r-", lw=2, label="Recall")
    ax.plot(metrics_df["threshold"], metrics_df["f1"], "g-", lw=2, label="F1 Score")
    
    # Find optimal F1 threshold
    best_f1_idx = metrics_df["f1"].idxmax()
    best_threshold = metrics_df.loc[best_f1_idx, "threshold"]
    best_f1 = metrics_df.loc[best_f1_idx, "f1"]
    
    # Add vertical line at optimal threshold
    ax.axvline(best_threshold, color="black", linestyle="--", alpha=0.7,
               label=f"Optimal Threshold = {best_threshold:.2f}")
    
    # Add point at optimal F1
    ax.plot(best_threshold, best_f1, "ko", ms=8)
    
    # Set labels and title
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics at Different Classification Thresholds")
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc="best")
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Threshold metrics plot saved to {save_path}")
    
    return fig


def plot_amount_distribution_by_class(data, amount_col="Amount", class_col="Class", 
                                      figsize=(12, 8), bins=50, save_path=None):
    """
    Plot the distribution of transaction amounts by class.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data.
    amount_col : str, optional
        Name of the amount column.
    class_col : str, optional
        Name of the class column.
    figsize : tuple, optional
        Figure size.
    bins : int, optional
        Number of bins for the histogram.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    
    # Split data by class
    normal = data[data[class_col] == 0][amount_col]
    fraud = data[data[class_col] == 1][amount_col]
    
    # Plot normal transactions
    ax[0].hist(normal, bins=bins, color="blue", alpha=0.7)
    ax[0].set_title("Normal Transactions")
    ax[0].set_xlabel(f"{amount_col} (£)")
    ax[0].set_ylabel("Frequency")
    ax[0].set_yscale("log")
    
    # Plot fraudulent transactions
    ax[1].hist(fraud, bins=bins, color="red", alpha=0.7)
    ax[1].set_title("Fraudulent Transactions")
    ax[1].set_xlabel(f"{amount_col} (£)")
    ax[1].set_ylabel("Frequency")
    ax[1].set_yscale("log")
    
    # Add overall title
    plt.suptitle("Transaction Amount Distribution by Class", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Amount distribution plot saved to {save_path}")
    
    return fig


def plot_feature_distributions(data, features, class_col="Class", figsize=(15, 10), 
                               n_cols=3, save_path=None):
    """
    Plot the distribution of selected features by class.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data.
    features : list
        List of feature names to plot.
    class_col : str, optional
        Name of the class column.
    figsize : tuple, optional
        Figure size.
    n_cols : int, optional
        Number of columns in the subplot grid.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    # Calculate number of rows needed
    n_rows = int(np.ceil(len(features) / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    axes = axes.flatten()
    
    # Plot each feature
    for i, feature in enumerate(features):
        # Skip if we've run out of features
        if i >= len(features):
            break
        
        # Get data by class
        normal_data = data[data[class_col] == 0][feature]
        fraud_data = data[data[class_col] == 1][feature]
        
        # Create KDE plot
        sns.kdeplot(normal_data, ax=axes[i], label="Normal", color="blue", alpha=0.7)
        sns.kdeplot(fraud_data, ax=axes[i], label="Fraud", color="red", alpha=0.7)
        
        # Set title and labels
        axes[i].set_title(feature)
        axes[i].set_xlabel("Value")
