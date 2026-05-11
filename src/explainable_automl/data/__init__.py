"""
Data loading utilities for different datasets.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Any, Dict
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split


class BaseDataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
        """
        Load dataset.
        
        Returns:
            Tuple of (features, targets, feature_names, target_names)
        """
        pass
    
    @abstractmethod
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train/test sets."""
        pass


class IrisDataLoader(BaseDataLoader):
    """Data loader for the Iris dataset."""
    
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names = feature_names or [
            "sepal_length", "sepal_width", "petal_length", "petal_width"
        ]
        self.target_names = target_names or ["setosa", "versicolor", "virginica"]
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
        """Load Iris dataset."""
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=self.feature_names)
        y = pd.Series(iris.target, name="species")
        
        return X, y, self.feature_names, self.target_names
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split Iris data into train/test sets."""
        test_size = test_size or self.test_size
        random_state = random_state or self.random_state
        
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )


class WineDataLoader(BaseDataLoader):
    """Data loader for the Wine dataset."""
    
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names = feature_names
        self.target_names = target_names or [f"wine_class_{i}" for i in range(3)]
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
        """Load Wine dataset."""
        wine = load_wine()
        X = pd.DataFrame(wine.data, columns=wine.feature_names)
        y = pd.Series(wine.target, name="wine_class")
        
        if self.feature_names:
            X.columns = self.feature_names
        
        return X, y, list(X.columns), self.target_names
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split Wine data into train/test sets."""
        test_size = test_size or self.test_size
        random_state = random_state or self.random_state
        
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )


class BreastCancerDataLoader(BaseDataLoader):
    """Data loader for the Breast Cancer dataset."""
    
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names = feature_names
        self.target_names = target_names or ["malignant", "benign"]
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
        """Load Breast Cancer dataset."""
        cancer = load_breast_cancer()
        X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        y = pd.Series(cancer.target, name="diagnosis")
        
        if self.feature_names:
            X.columns = self.feature_names
        
        return X, y, list(X.columns), self.target_names
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split Breast Cancer data into train/test sets."""
        test_size = test_size or self.test_size
        random_state = random_state or self.random_state
        
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )


def get_data_loader(dataset_name: str, **kwargs) -> BaseDataLoader:
    """
    Factory function to get appropriate data loader.
    
    Args:
        dataset_name: Name of the dataset ("iris", "wine", "breast_cancer")
        **kwargs: Additional arguments for the data loader
        
    Returns:
        BaseDataLoader: Appropriate data loader instance
        
    Raises:
        ValueError: If dataset name is not supported
    """
    loaders = {
        "iris": IrisDataLoader,
        "wine": WineDataLoader,
        "breast_cancer": BreastCancerDataLoader,
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(loaders.keys())}")
    
    return loaders[dataset_name](**kwargs)
