"""
Basic tests for the Explainable AutoML framework.
"""

import pytest
import pandas as pd
import numpy as np
from omegaconf import OmegaConf

from explainable_automl.data.loaders import get_data_loader, IrisDataLoader
from explainable_automl.models.tpot_classifier import get_model, BaselineClassifier
from explainable_automl.explainers.shap_explainer import get_explainer, SHAPExplainer
from explainable_automl.metrics import ModelMetrics, ExplainabilityMetrics
from explainable_automl.utils.seed import set_seed
from explainable_automl.utils.device import get_device


class TestDataLoaders:
    """Test data loading functionality."""
    
    def test_iris_data_loader(self):
        """Test Iris data loader."""
        loader = IrisDataLoader()
        X, y, feature_names, target_names = loader.load_data()
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(feature_names) == 4
        assert len(target_names) == 3
        assert X.shape[0] == 150
        assert X.shape[1] == 4
    
    def test_data_split(self):
        """Test data splitting."""
        loader = IrisDataLoader(test_size=0.3, random_state=42)
        X, y, _, _ = loader.load_data()
        X_train, X_test, y_train, y_test = loader.split_data(X, y)
        
        assert len(X_train) == 105  # 70% of 150
        assert len(X_test) == 45    # 30% of 150
        assert len(y_train) == 105
        assert len(y_test) == 45
    
    def test_get_data_loader_factory(self):
        """Test data loader factory function."""
        loader = get_data_loader("iris")
        assert isinstance(loader, IrisDataLoader)
        
        with pytest.raises(ValueError):
            get_data_loader("invalid_dataset")


class TestModels:
    """Test model functionality."""
    
    def test_baseline_classifier(self):
        """Test baseline classifier."""
        model = BaselineClassifier(algorithm="logistic_regression", random_state=42)
        
        # Create dummy data
        X = pd.DataFrame(np.random.randn(100, 4), columns=['a', 'b', 'c', 'd'])
        y = pd.Series(np.random.randint(0, 2, 100))
        
        # Test fitting
        model.fit(X, y)
        
        # Test predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == 100
        assert probabilities.shape == (100, 2)
        assert model.get_feature_importance() is not None
    
    def test_get_model_factory(self):
        """Test model factory function."""
        model = get_model("logistic_regression")
        assert isinstance(model, BaselineClassifier)
        
        model = get_model("random_forest")
        assert isinstance(model, BaselineClassifier)


class TestExplainers:
    """Test explainer functionality."""
    
    def test_shap_explainer_init(self):
        """Test SHAP explainer initialization."""
        explainer = SHAPExplainer(method="kernel", background_samples=50)
        assert explainer.method == "kernel"
        assert explainer.background_samples == 50
    
    def test_get_explainer_factory(self):
        """Test explainer factory function."""
        explainer = get_explainer("shap")
        assert isinstance(explainer, SHAPExplainer)
        
        with pytest.raises(ValueError):
            get_explainer("invalid_explainer")


class TestMetrics:
    """Test metrics functionality."""
    
    def test_model_metrics(self):
        """Test model metrics computation."""
        metrics = ModelMetrics(target_names=["class_0", "class_1"])
        
        # Create dummy data
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        y_proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.7, 0.3]])
        
        result = metrics.compute_classification_metrics(y_true, y_pred, y_proba)
        
        assert "accuracy" in result
        assert "precision_macro" in result
        assert "recall_macro" in result
        assert "f1_macro" in result
        assert result["accuracy"] == 1.0  # Perfect predictions
    
    def test_explainability_metrics(self):
        """Test explainability metrics."""
        xai_metrics = ExplainabilityMetrics()
        
        # Create dummy model and data
        class DummyModel:
            def predict_proba(self, X):
                return np.random.rand(len(X), 2)
        
        model = DummyModel()
        X = pd.DataFrame(np.random.randn(10, 4), columns=['a', 'b', 'c', 'd'])
        explanations = {
            "importance_mean": np.array([0.3, 0.2, 0.1, 0.4]),
            "feature_names": ['a', 'b', 'c', 'd']
        }
        
        # Test faithfulness metrics
        faithfulness = xai_metrics.compute_faithfulness_metrics(
            model, X, explanations, method="deletion"
        )
        
        assert "deletion_auc" in faithfulness
        assert "deletion_score" in faithfulness


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This should not raise an exception
        assert True
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("auto")
        assert device is not None
        
        device = get_device("cpu")
        assert str(device) == "cpu"


class TestIntegration:
    """Integration tests."""
    
    def test_basic_pipeline(self):
        """Test basic pipeline functionality."""
        # Load data
        loader = IrisDataLoader(test_size=0.2, random_state=42)
        X, y, _, _ = loader.load_data()
        X_train, X_test, y_train, y_test = loader.split_data(X, y)
        
        # Train model
        model = BaselineClassifier(algorithm="logistic_regression", random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        metrics = ModelMetrics()
        model_metrics = metrics.compute_classification_metrics(y_test, y_pred)
        
        assert model_metrics["accuracy"] > 0.8  # Should be reasonably accurate
        
        # Generate explanations
        explainer = SHAPExplainer(method="kernel", background_samples=20)
        explanations = explainer.explain(model, X_test.iloc[:5])  # Use subset for speed
        
        assert "method" in explanations
        assert explanations["method"] == "SHAP"


if __name__ == "__main__":
    pytest.main([__file__])
