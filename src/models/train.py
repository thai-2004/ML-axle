"""
Training pipeline for models.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd


def create_preprocessing_pipeline(categorical_cols: list = None,
                                 numerical_cols: list = None,
                                 scale: bool = True) -> Pipeline:
    """
    Create preprocessing pipeline.
    
    Note: Tree-based models (Decision Tree, Random Forest) don't require scaling,
    but it's included here for flexibility with other models.
    
    Parameters
    ----------
    categorical_cols : list
        List of categorical column names.
    numerical_cols : list
        List of numerical column names.
    scale : bool
        Whether to apply scaling to numerical features.
    
    Returns
    -------
    ColumnTransformer
        Preprocessing pipeline.
    """
    transformers = []
    
    if numerical_cols and scale:
        transformers.append(('num', StandardScaler(), numerical_cols))
    
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols))
    
    if transformers:
        preprocessor = ColumnTransformer(transformers, remainder='passthrough')
        return preprocessor
    else:
        return None


def train_model(model, X_train, y_train, preprocessor=None):
    """
    Train a model with optional preprocessing.
    
    Parameters
    ----------
    model : sklearn estimator
        Model to train.
    X_train : array-like or DataFrame
        Training features.
    y_train : array-like
        Training labels.
    preprocessor : ColumnTransformer, optional
        Preprocessing pipeline.
    
    Returns
    -------
    Pipeline or fitted model
        Trained model (with preprocessing if provided).
    """
    if preprocessor:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        return pipeline
    else:
        model.fit(X_train, y_train)
        return model


def predict(model, X, preprocessor=None):
    """
    Make predictions using trained model.
    
    Parameters
    ----------
    model : sklearn estimator or Pipeline
        Trained model.
    X : array-like or DataFrame
        Features to predict on.
    preprocessor : ColumnTransformer, optional
        Preprocessing pipeline (if model is not a Pipeline).
    
    Returns
    -------
    array-like
        Predictions.
    """
    if hasattr(model, 'predict'):
        # If it's a Pipeline, it handles preprocessing automatically
        return model.predict(X)
    elif preprocessor:
        X_processed = preprocessor.transform(X)
        return model.predict(X_processed)
    else:
        return model.predict(X)


def predict_proba(model, X, preprocessor=None):
    """
    Get prediction probabilities using trained model.
    
    Parameters
    ----------
    model : sklearn estimator or Pipeline
        Trained model.
    X : array-like or DataFrame
        Features to predict on.
    preprocessor : ColumnTransformer, optional
        Preprocessing pipeline (if model is not a Pipeline).
    
    Returns
    -------
    array-like
        Prediction probabilities.
    """
    if hasattr(model, 'predict_proba'):
        if isinstance(model, Pipeline):
            return model.predict_proba(X)
        elif preprocessor:
            X_processed = preprocessor.transform(X)
            return model.predict_proba(X_processed)
        else:
            return model.predict_proba(X)
    else:
        raise AttributeError("Model does not support predict_proba")

