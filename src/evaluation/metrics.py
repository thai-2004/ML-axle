"""
Evaluation metrics and scoring functions.
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import pandas as pd
import numpy as np
from models.train import predict, predict_proba


def calculate_metrics(y_true, y_pred, y_proba=None, average='binary'):
    """
    Calculate comprehensive classification metrics.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_proba : array-like, optional
        Predicted probabilities (for positive class).
    average : str
        Averaging strategy for multi-class (default: 'binary').
    
    Returns
    -------
    dict
        Dictionary of metric names and values.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        except ValueError:
            # Handle case where only one class is present
            metrics['roc_auc'] = np.nan
            metrics['pr_auc'] = np.nan
    
    return metrics


def get_confusion_matrix(y_true, y_pred, labels=None):
    """
    Get confusion matrix.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : array-like, optional
        Label ordering.
    
    Returns
    -------
    np.ndarray
        Confusion matrix.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Print detailed classification report.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    target_names : list, optional
        Names of classes.
    """
    print(classification_report(y_true, y_pred, target_names=target_names))


def evaluate_model(model, X, y, preprocessor=None):
    """
    Evaluate a trained model on a dataset.
    
    Parameters
    ----------
    model : sklearn estimator or Pipeline
        Trained model.
    X : array-like or DataFrame
        Features.
    y : array-like
        True labels.
    preprocessor : ColumnTransformer, optional
        Preprocessing pipeline (if model is not a Pipeline).
    
    Returns
    -------
    dict
        Dictionary of metrics.
    """
    y_pred = predict(model, X, preprocessor)
    
    try:
        y_proba = predict_proba(model, X, preprocessor)
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]  # Get probabilities for positive class
    except:
        y_proba = None
    
    metrics = calculate_metrics(y, y_pred, y_proba)
    return metrics

