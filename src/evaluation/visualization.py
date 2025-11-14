"""
Visualization functions for model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import numpy as np
import pandas as pd
from pathlib import Path


def plot_confusion_matrix(y_true, y_pred, labels=None, 
                         title='Confusion Matrix', 
                         save_path=None):
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : array-like, optional
        Label names.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels or ['False', 'True'],
                yticklabels=labels or ['False', 'True'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_proba, title='ROC Curve', save_path=None):
    """
    Plot ROC curve.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_proba : array-like
        Predicted probabilities for positive class.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(y_true, y_proba, title='Precision-Recall Curve', 
                               save_path=None):
    """
    Plot Precision-Recall curve.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_proba : array-like
        Predicted probabilities for positive class.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_feature_importance(importances, feature_names, top_n=20, 
                           title='Feature Importance', save_path=None):
    """
    Plot feature importance.
    
    Parameters
    ----------
    importances : array-like
        Feature importance values.
    feature_names : list
        Feature names.
    top_n : int
        Number of top features to display.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.
    """
    # Create DataFrame and sort
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(df)), df['importance'])
    plt.yticks(range(len(df)), df['feature'])
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

