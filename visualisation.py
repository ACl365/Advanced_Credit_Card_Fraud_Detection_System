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
        axes[i].set_ylabel("Density")
        axes[i].legend()
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Add overall title
    plt.suptitle("Feature Distributions by Class", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Feature distributions plot saved to {save_path}")
    
    return fig


def plot_time_distribution(data, time_col="Time", class_col="Class", figsize=(12, 6), 
                           save_path=None):
    """
    Plot the distribution of transactions over time by class.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data.
    time_col : str, optional
        Name of the time column.
    class_col : str, optional
        Name of the class column.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    # Convert time to hours
    data = data.copy()
    data["Hour"] = data[time_col] / 3600 % 24
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group by hour and calculate fraud rate
    hourly_data = data.groupby(data["Hour"].astype(int)).agg({
        class_col: ["count", "sum"]
    })
    
    hourly_data.columns = ["transactions", "frauds"]
    hourly_data["fraud_rate"] = (hourly_data["frauds"] / hourly_data["transactions"]) * 100
    
    # Plot transaction volume
    ax.bar(hourly_data.index, hourly_data["transactions"], color="blue", alpha=0.5, 
           label="Transactions")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Transactions")
    ax.set_title("Transaction Volume and Fraud Rate by Hour of Day")
    
    # Create secondary y-axis for fraud rate
    ax2 = ax.twinx()
    ax2.plot(hourly_data.index, hourly_data["fraud_rate"], color="red", marker="o", 
             label="Fraud Rate")
    ax2.set_ylabel("Fraud Rate (%)")
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Time distribution plot saved to {save_path}")
    
    return fig


def plot_model_comparison(results_df, metric_cols=None, figsize=(12, 8), save_path=None):
    """
    Plot a comparison of model performance metrics.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing model performance results.
    metric_cols : list, optional
        List of metric column names to plot. If None, plot all numeric columns.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    # Copy the dataframe to avoid modifying the original
    df = results_df.copy()
    
    # If no specific metrics are provided, use all numeric columns
    if metric_cols is None:
        metric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Filter to only include specified metrics
    df = df[metric_cols]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bar chart
    df.plot(kind="bar", ax=ax)
    
    # Set labels and title
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    
    # Add legend
    ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    # Add grid
    ax.grid(True, alpha=0.3, axis="y")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Model comparison plot saved to {save_path}")
    
    return fig


def create_all_visualisations(data, results_df, y_true, y_scores, model_names, output_dir="results/figures"):
    """
    Create and save all visualisations for the credit card fraud detection project.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the transaction data.
    results_df : pandas.DataFrame
        DataFrame containing model performance results.
    y_true : array-like
        True labels for the test set.
    y_scores : list of array-like
        List of predicted scores from different models on the test set.
    model_names : list
        List of model names corresponding to the scores.
    output_dir : str, optional
        Directory to save the visualisations.
    """
    logger.info("Creating all visualisations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plotting style
    setup_plotting_style()
    
    # 1. Class Distribution
    plot_class_distribution(
        data, 
        target_col="Class", 
        title="Class Distribution in Credit Card Transaction Data",
        save_path=os.path.join(output_dir, "class_distribution.png")
    )
    
    # 2. Amount Distribution by Class
    plot_amount_distribution_by_class(
        data,
        save_path=os.path.join(output_dir, "amount_distribution.png")
    )
    
    # 3. Time Distribution
    plot_time_distribution(
        data,
        save_path=os.path.join(output_dir, "time_distribution.png")
    )
    
    # 4. Feature Distributions (for top 6 important features)
    important_features = ["V17", "V14", "V12", "V10", "V16", "V11"]  # Example features
    plot_feature_distributions(
        data,
        features=important_features,
        save_path=os.path.join(output_dir, "feature_distributions.png")
    )
    
    # 5. Correlation Matrix
    plot_correlation_matrix(
        data,
        save_path=os.path.join(output_dir, "correlation_matrix.png")
    )
    
    # 6. ROC Curves
    plot_roc_curves(
        y_true,
        y_scores,
        model_names=model_names,
        save_path=os.path.join(output_dir, "roc_curves.png")
    )
    
    # 7. Precision-Recall Curves
    plot_precision_recall_curves(
        y_true,
        y_scores,
        model_names=model_names,
        save_path=os.path.join(output_dir, "precision_recall_curves.png")
    )
    
    # 8. Model Comparison
    plot_model_comparison(
        results_df,
        save_path=os.path.join(output_dir, "model_comparison.png")
    )
    
    # 9. Threshold Metrics (for best model)
    best_model_idx = results_df["auc"].idxmax()
    best_model_scores = y_scores[model_names.index(best_model_idx)]
    
    plot_threshold_metrics(
        y_true,
        best_model_scores,
        save_path=os.path.join(output_dir, "threshold_metrics.png")
    )
    
    logger.info(f"All visualisations saved to {output_dir}")
