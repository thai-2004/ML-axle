"""
Random Forest model implementation.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os
from pathlib import Path


def create_random_forest_model(n_estimators: int = 100,
                              max_depth: int = None,
                              max_features: str = 'sqrt',
                              min_samples_split: int = 2,
                              min_samples_leaf: int = 1,
                              class_weight: str = 'balanced',
                              random_state: int = 42,
                              n_jobs: int = -1) -> RandomForestClassifier:
    """
    Create a Random Forest classifier.
    
    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int, optional
        Maximum depth of trees.
    max_features : str or int
        Number of features to consider for best split.
    min_samples_split : int
        Minimum samples required to split a node.
    min_samples_leaf : int
        Minimum samples required in a leaf node.
    class_weight : str or dict
        Weights for classes. 'balanced' uses class frequencies.
    random_state : int
        Random seed.
    n_jobs : int
        Number of jobs to run in parallel.
    
    Returns
    -------
    RandomForestClassifier
        Configured Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )
    return model


def tune_random_forest(X_train, y_train, cv=5, scoring='f1'):
    """
    Tune Random Forest hyperparameters using GridSearchCV.
    
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
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'max_features': ['sqrt', 'log2', 0.5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced']
    }
    
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv, scoring=scoring,
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search


def get_feature_importance(model, feature_names: list) -> dict:
    """
    Get feature importance from trained Random Forest model.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained Random Forest model.
    feature_names : list
        List of feature names.
    
    Returns
    -------
    dict
        Dictionary mapping feature names to importance scores.
    """
    importances = model.feature_importances_
    feature_importance = dict(zip(feature_names, importances))
    return feature_importance


def save_model(model, filename: str = 'random_forest.pkl'):
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


def load_model(filename: str = 'random_forest.pkl'):
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

