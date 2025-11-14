"""
Data preprocessing functions.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing values in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    
    Returns
    -------
    pd.DataFrame
        Summary of missing values.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    return pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_pct
    })


def check_outliers(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Detect outliers using IQR method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        Columns to check. If None, checks all numeric columns.
    
    Returns
    -------
    pd.DataFrame
        Summary of outliers per column.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_summary = []
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary.append({
            'Column': col,
            'Outlier Count': len(outliers),
            'Percentage': (len(outliers) / len(df)) * 100
        })
    
    return pd.DataFrame(outlier_summary)


def encode_categorical(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Encode categorical variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        Categorical columns to encode. If None, auto-detect.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with encoded categorical variables.
    """
    df_encoded = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    label_encoders = {}
    for col in columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    return df_encoded, label_encoders


def split_data(df: pd.DataFrame, target_col: str = 'Revenue', 
               test_size: float = 0.2, val_size: float = 0.1,
               random_state: int = 42, stratify: bool = True) -> tuple:
    """
    Split data into train, validation, and test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Name of target column.
    test_size : float
        Proportion of test set.
    val_size : float
        Proportion of validation set (from remaining after test).
    random_state : int
        Random seed.
    stratify : bool
        Whether to stratify by target variable.
    
    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    stratify_param = y if stratify else None
    
    # First split: train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    
    # Second split: train and val
    stratify_param_val = y_temp if stratify else None
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=stratify_param_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

