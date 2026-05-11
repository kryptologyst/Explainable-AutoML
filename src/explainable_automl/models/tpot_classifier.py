"""
Model implementations including TPOT wrapper and baseline models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


class BaseModel(ABC):
    """Abstract base class for models."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        pass


class TPOTClassifierWrapper(BaseModel):
    """Wrapper for TPOT classifier with additional functionality."""
    
    def __init__(
        self,
        generations: int = 5,
        population_size: int = 20,
        random_state: int = 42,
        verbosity: int = 2,
        cv_folds: int = 5,
        scoring: str = "accuracy",
        max_time_mins: int = 10,
        max_eval_time_mins: int = 5,
        **kwargs
    ):
        self.generations = generations
        self.population_size = population_size
        self.random_state = random_state
        self.verbosity = verbosity
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.max_time_mins = max_time_mins
        self.max_eval_time_mins = max_eval_time_mins
        self.kwargs = kwargs
        
        # Initialize TPOT classifier
        from tpot import TPOTClassifier
        self.tpot = TPOTClassifier(
            generations=generations,
            population_size=population_size,
            random_state=random_state,
            verbosity=verbosity,
            cv=cv_folds,
            scoring=scoring,
            max_time_mins=max_time_mins,
            max_eval_time_mins=max_eval_time_mins,
            **kwargs
        )
        
        self.fitted_pipeline_ = None
        self.feature_names_ = None
        self.target_names_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TPOTClassifierWrapper":
        """Fit TPOT classifier to training data."""
        self.feature_names_ = list(X.columns)
        self.target_names_ = list(y.unique()) if hasattr(y, 'unique') else None
        
        print("Starting TPOT optimization...")
        self.tpot.fit(X, y)
        self.fitted_pipeline_ = self.tpot.fitted_pipeline_
        
        print(f"Best pipeline: {self.tpot.fitted_pipeline_}")
        print(f"Best CV score: {self.tpot.score(X, y):.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best pipeline."""
        if self.fitted_pipeline_ is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.fitted_pipeline_.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using the best pipeline."""
        if self.fitted_pipeline_ is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.fitted_pipeline_.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if the best pipeline supports it."""
        if self.fitted_pipeline_ is None:
            return None
        
        # Try to get feature importance from the pipeline
        try:
            if hasattr(self.fitted_pipeline_, 'feature_importances_'):
                return self.fitted_pipeline_.feature_importances_
            elif hasattr(self.fitted_pipeline_, 'coef_'):
                return np.abs(self.fitted_pipeline_.coef_[0])
            else:
                return None
        except AttributeError:
            return None
    
    def export_pipeline(self, filename: str) -> None:
        """Export the best pipeline to a Python file."""
        if self.fitted_pipeline_ is None:
            raise ValueError("Model must be fitted before exporting")
        
        self.tpot.export(filename)
        print(f"Best pipeline exported to {filename}")
    
    def save_model(self, filepath: str) -> None:
        """Save the fitted model."""
        if self.fitted_pipeline_ is None:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.fitted_pipeline_, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a fitted model."""
        self.fitted_pipeline_ = joblib.load(filepath)
        print(f"Model loaded from {filepath}")


class BaselineClassifier(BaseModel):
    """Baseline classifier with multiple algorithm options."""
    
    def __init__(
        self,
        algorithm: str = "logistic_regression",
        random_state: int = 42,
        **kwargs
    ):
        self.algorithm = algorithm
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Initialize the classifier
        self.classifier = self._get_classifier()
        self.fitted_model_ = None
        self.feature_names_ = None
        self.target_names_ = None
    
    def _get_classifier(self) -> BaseEstimator:
        """Get the appropriate classifier based on algorithm name."""
        classifiers = {
            "logistic_regression": LogisticRegression(random_state=self.random_state, **self.kwargs),
            "random_forest": RandomForestClassifier(random_state=self.random_state, **self.kwargs),
            "gradient_boosting": GradientBoostingClassifier(random_state=self.random_state, **self.kwargs),
            "svm": SVC(random_state=self.random_state, probability=True, **self.kwargs),
            "knn": KNeighborsClassifier(**self.kwargs),
            "naive_bayes": GaussianNB(**self.kwargs),
            "decision_tree": DecisionTreeClassifier(random_state=self.random_state, **self.kwargs),
        }
        
        if self.algorithm not in classifiers:
            raise ValueError(f"Algorithm {self.algorithm} not supported. Available: {list(classifiers.keys())}")
        
        return classifiers[self.algorithm]
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineClassifier":
        """Fit the baseline classifier to training data."""
        self.feature_names_ = list(X.columns)
        self.target_names_ = list(y.unique()) if hasattr(y, 'unique') else None
        
        print(f"Training {self.algorithm}...")
        self.classifier.fit(X, y)
        self.fitted_model_ = self.classifier
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.fitted_model_ is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.fitted_model_.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.fitted_model_ is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.fitted_model_.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        if self.fitted_model_ is None:
            return None
        
        try:
            if hasattr(self.fitted_model_, 'feature_importances_'):
                return self.fitted_model_.feature_importances_
            elif hasattr(self.fitted_model_, 'coef_'):
                return np.abs(self.fitted_model_.coef_[0])
            else:
                return None
        except AttributeError:
            return None


def get_model(model_name: str, **kwargs) -> BaseModel:
    """
    Factory function to get appropriate model.
    
    Args:
        model_name: Name of the model ("tpot", "logistic_regression", "random_forest", etc.)
        **kwargs: Additional arguments for the model
        
    Returns:
        BaseModel: Appropriate model instance
        
    Raises:
        ValueError: If model name is not supported
    """
    if model_name == "tpot":
        return TPOTClassifierWrapper(**kwargs)
    else:
        return BaselineClassifier(algorithm=model_name, **kwargs)
