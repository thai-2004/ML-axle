"""
Load and inspect data from CSV file.
"""

import pandas as pd
import os
from pathlib import Path


def load_raw_data(data_path: str = None) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    
    Parameters
    ----------
    data_path : str, optional
        Path to the CSV file. If None, uses default path.
    
    Returns
    -------
    pd.DataFrame
        Raw dataset as pandas DataFrame.
    """
    if data_path is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / "data" / "raw" / "online_shoppers_intention.csv"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to inspect.
    """
    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nBasic statistics:\n{df.describe()}")


if __name__ == "__main__":
    # Example usage
    df = load_raw_data()
    inspect_data(df)

