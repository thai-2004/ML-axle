"""
Feature engineering functions.
"""

import pandas as pd
import numpy as np


def create_total_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create total duration and total pages features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with new total features.
    """
    df_feat = df.copy()
    
    # Total duration
    duration_cols = [col for col in df.columns if 'Duration' in col]
    if duration_cols:
        df_feat['total_duration'] = df[duration_cols].sum(axis=1)
    
    # Total pages
    page_cols = ['Administrative', 'Informational', 'ProductRelated']
    if all(col in df.columns for col in page_cols):
        df_feat['total_pages'] = df[page_cols].sum(axis=1)
    
    return df_feat


def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ratio features for different page types.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with new ratio features.
    """
    df_feat = df.copy()
    
    # Duration ratios
    if 'Administrative_Duration' in df.columns and 'total_duration' in df_feat.columns:
        df_feat['admin_duration_ratio'] = df['Administrative_Duration'] / (df_feat['total_duration'] + 1e-6)
    
    if 'Informational_Duration' in df.columns and 'total_duration' in df_feat.columns:
        df_feat['informational_duration_ratio'] = df['Informational_Duration'] / (df_feat['total_duration'] + 1e-6)
    
    if 'ProductRelated_Duration' in df.columns and 'total_duration' in df_feat.columns:
        df_feat['product_duration_ratio'] = df['ProductRelated_Duration'] / (df_feat['total_duration'] + 1e-6)
    
    # Page ratios
    if 'total_pages' in df_feat.columns:
        if 'Administrative' in df.columns:
            df_feat['admin_pages_ratio'] = df['Administrative'] / (df_feat['total_pages'] + 1e-6)
        if 'Informational' in df.columns:
            df_feat['informational_pages_ratio'] = df['Informational'] / (df_feat['total_pages'] + 1e-6)
        if 'ProductRelated' in df.columns:
            df_feat['product_pages_ratio'] = df['ProductRelated'] / (df_feat['total_pages'] + 1e-6)
    
    return df_feat


def create_duration_per_page(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create duration per page features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with duration per page features.
    """
    df_feat = df.copy()
    
    if 'Administrative' in df.columns and 'Administrative_Duration' in df.columns:
        df_feat['admin_duration_per_page'] = df['Administrative_Duration'] / (df['Administrative'] + 1e-6)
    
    if 'Informational' in df.columns and 'Informational_Duration' in df.columns:
        df_feat['informational_duration_per_page'] = df['Informational_Duration'] / (df['Informational'] + 1e-6)
    
    if 'ProductRelated' in df.columns and 'ProductRelated_Duration' in df.columns:
        df_feat['product_duration_per_page'] = df['ProductRelated_Duration'] / (df['ProductRelated'] + 1e-6)
    
    if 'total_duration' in df_feat.columns and 'total_pages' in df_feat.columns:
        df_feat['avg_duration_per_page'] = df_feat['total_duration'] / (df_feat['total_pages'] + 1e-6)
    
    return df_feat


def create_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create seasonal features from Month column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with seasonal features.
    """
    df_feat = df.copy()
    
    if 'Month' in df.columns:
        # Q4 indicator (Oct, Nov, Dec - shopping season)
        q4_months = ['Oct', 'Nov', 'Dec']
        df_feat['is_q4'] = df['Month'].isin(q4_months).astype(int)
        
        # Quarter encoding
        month_to_quarter = {
            'Jan': 1, 'Feb': 1, 'Mar': 1,
            'Apr': 2, 'May': 2, 'Jun': 2,
            'Jul': 3, 'Aug': 3, 'Sep': 3,
            'Oct': 4, 'Nov': 4, 'Dec': 4
        }
        df_feat['quarter'] = df['Month'].map(month_to_quarter)
    
    return df_feat


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with interaction features.
    """
    df_feat = df.copy()
    
    # PageValues * ProductRelated (indicates engagement with product pages)
    if 'PageValues' in df.columns and 'ProductRelated' in df.columns:
        df_feat['PageValues_x_ProductRelated'] = df['PageValues'] * df['ProductRelated']
    
    # SpecialDay * PageValues (special day engagement)
    if 'SpecialDay' in df.columns and 'PageValues' in df.columns:
        df_feat['SpecialDay_x_PageValues'] = df['SpecialDay'] * df['PageValues']
    
    # BounceRates * ExitRates (combined exit behavior)
    if 'BounceRates' in df.columns and 'ExitRates' in df.columns:
        df_feat['BounceRates_x_ExitRates'] = df['BounceRates'] * df['ExitRates']
    
    return df_feat


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all engineered features.
    """
    df_feat = df.copy()
    
    # Apply all feature engineering steps
    df_feat = create_total_features(df_feat)
    df_feat = create_ratio_features(df_feat)
    df_feat = create_duration_per_page(df_feat)
    df_feat = create_seasonal_features(df_feat)
    df_feat = create_interaction_features(df_feat)
    
    return df_feat

