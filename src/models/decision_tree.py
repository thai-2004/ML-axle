"""
Decision Tree model implementation.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import joblib
import os
from pathlib import Path


def create_decision_tree_model(max_depth: int = None, 
                              min_samples_split: int = 2,
                              min_samples_leaf: int = 1,
                              class_weight: str = 'balanced',
                              random_state: int = 42) -> DecisionTreeClassifier:
    """
    Create a Decision Tree classifier.
    
    Parameters
    ----------
    max_depth : int, optional
        Maximum depth of the tree.
    min_samples_split : int
        Minimum samples required to split a node.
    min_samples_leaf : int
        Minimum samples required in a leaf node.
    class_weight : str or dict
        Weights for classes. 'balanced' uses class frequencies.
    random_state : int
        Random seed.
    
    Returns
    -------
    DecisionTreeClassifier
        Configured Decision Tree model.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state
    )
    return model


def tune_decision_tree(X_train, y_train, cv=5, scoring='f1'):
    """
    Tune Decision Tree hyperparameters using GridSearchCV.
    
    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric.
    
    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV object with best parameters.
    """
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'class_weight': ['balanced']
    }
    
    base_model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv, scoring=scoring,
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search


def save_model(model, filename: str = 'decision_tree.pkl'):
    """
    Save trained model to disk.
    
    Parameters
    ----------
    model : sklearn model
        Trained model to save.
    filename : str
        Filename for saved model.
    """
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / filename
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def load_model(filename: str = 'decision_tree.pkl'):
    """
    Load trained model from disk.
    
    Parameters
    ----------
    filename : str
        Filename of saved model.
    
    Returns
    -------
    sklearn model
        Loaded model.
    """
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "models" / filename
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = joblib.load(model_path)
    return model

