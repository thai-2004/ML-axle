"""
Helper utility functions.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns
    -------
    Path
        Path to project root.
    """
    return Path(__file__).parent.parent.parent


def ensure_dir(dir_path: Path) -> None:
    """
    Ensure a directory exists, create if it doesn't.
    
    Parameters
    ----------
    dir_path : Path
        Directory path to ensure exists.
    """
    dir_path.mkdir(parents=True, exist_ok=True)


def save_results(results: dict, filename: str = 'evaluation_results.json'):
    """
    Save evaluation results to JSON file.
    
    Parameters
    ----------
    results : dict
        Dictionary of results to save.
    filename : str
        Output filename.
    """
    import json
    
    project_root = get_project_root()
    results_dir = project_root / "results" / "metrics"
    ensure_dir(results_dir)
    
    filepath = results_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {filepath}")


def load_results(filename: str = 'evaluation_results.json') -> dict:
    """
    Load evaluation results from JSON file.
    
    Parameters
    ----------
    filename : str
        Input filename.
    
    Returns
    -------
    dict
        Loaded results dictionary.
    """
    import json
    
    project_root = get_project_root()
    filepath = project_root / "results" / "metrics" / filename
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results

