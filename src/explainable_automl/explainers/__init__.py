"""
Explainability methods including SHAP, LIME, Integrated Gradients, and more.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class BaseExplainer(ABC):
    """Abstract base class for explainers."""
    
    @abstractmethod
    def explain(self, model: Any, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Generate explanations for the model."""
        pass
    
    @abstractmethod
    def plot_explanations(self, explanations: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot explanations."""
        pass


class SHAPExplainer(BaseExplainer):
    """SHAP (SHapley Additive exPlanations) explainer."""
    
    def __init__(
        self,
        method: str = "kernel",
        background_samples: int = 100,
        max_samples: int = 1000,
        feature_names: Optional[List[str]] = None
    ):
        self.method = method
        self.background_samples = background_samples
        self.max_samples = max_samples
        self.feature_names = feature_names
        
        try:
            import shap
            self.shap = shap
        except ImportError:
            raise ImportError("SHAP is required. Install with: pip install shap")
    
    def explain(self, model: Any, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Generate SHAP explanations."""
        print(f"Generating SHAP explanations using {self.method} method...")
        
        # Prepare background data
        background_data = X.sample(min(self.background_samples, len(X)), random_state=42)
        
        # Create explainer based on method
        if self.method == "kernel":
            explainer = self.shap.KernelExplainer(
                model.predict_proba, 
                background_data,
                feature_names=self.feature_names or list(X.columns)
            )
        elif self.method == "tree":
            explainer = self.shap.TreeExplainer(model)
        elif self.method == "linear":
            explainer = self.shap.LinearExplainer(model, background_data)
        else:
            raise ValueError(f"SHAP method {self.method} not supported")
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X.iloc[:self.max_samples])
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            # Multi-class: use first class for summary
            shap_values_summary = shap_values[0]
        else:
            shap_values_summary = shap_values
        
        return {
            "explainer": explainer,
            "shap_values": shap_values,
            "shap_values_summary": shap_values_summary,
            "expected_value": explainer.expected_value,
            "feature_names": self.feature_names or list(X.columns),
            "method": "SHAP"
        }
    
    def plot_explanations(self, explanations: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot SHAP explanations."""
        shap_values = explanations["shap_values"]
        X_explain = explanations.get("X_explain")
        
        if X_explain is None:
            print("Warning: No data to explain provided")
            return
        
        # Summary plot
        plt.figure(figsize=(10, 6))
        self.shap.summary_plot(shap_values, X_explain, show=False)
        plt.title("SHAP Summary Plot")
        if save_path:
            plt.savefig(f"{save_path}_shap_summary.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Waterfall plot for first instance
        if hasattr(self.shap, 'waterfall_plot'):
            plt.figure(figsize=(10, 6))
            self.shap.waterfall_plot(
                explanations["explainer"].expected_value[0], 
                shap_values[0][0], 
                X_explain.iloc[0],
                show=False
            )
            plt.title("SHAP Waterfall Plot (First Instance)")
            if save_path:
                plt.savefig(f"{save_path}_shap_waterfall.png", dpi=300, bbox_inches='tight')
            plt.show()


class LIMEExplainer(BaseExplainer):
    """LIME (Local Interpretable Model-agnostic Explanations) explainer."""
    
    def __init__(
        self,
        mode: str = "classification",
        feature_names: Optional[List[str]] = None,
        random_state: int = 42
    ):
        self.mode = mode
        self.feature_names = feature_names
        self.random_state = random_state
        
        try:
            import lime
            import lime.lime_tabular
            self.lime = lime
            self.lime_tabular = lime.lime_tabular
        except ImportError:
            raise ImportError("LIME is required. Install with: pip install lime")
    
    def explain(self, model: Any, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Generate LIME explanations."""
        print("Generating LIME explanations...")
        
        # Create LIME explainer
        explainer = self.lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=self.feature_names or list(X.columns),
            class_names=y.unique() if y is not None else None,
            mode=self.mode,
            random_state=self.random_state
        )
        
        # Generate explanations for first few instances
        explanations_list = []
        for i in range(min(5, len(X))):
            exp = explainer.explain_instance(
                X.iloc[i].values,
                model.predict_proba,
                num_features=len(X.columns)
            )
            explanations_list.append(exp)
        
        return {
            "explainer": explainer,
            "explanations": explanations_list,
            "feature_names": self.feature_names or list(X.columns),
            "method": "LIME"
        }
    
    def plot_explanations(self, explanations: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot LIME explanations."""
        explanations_list = explanations["explanations"]
        
        # Plot explanations for first instance
        if explanations_list:
            plt.figure(figsize=(10, 6))
            explanations_list[0].show_in_notebook(show_table=True)
            plt.title("LIME Explanation (First Instance)")
            if save_path:
                plt.savefig(f"{save_path}_lime_explanation.png", dpi=300, bbox_inches='tight')
            plt.show()


class PermutationImportanceExplainer(BaseExplainer):
    """Permutation importance explainer."""
    
    def __init__(
        self,
        n_repeats: int = 10,
        random_state: int = 42,
        scoring: str = "accuracy"
    ):
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.scoring = scoring
    
    def explain(self, model: Any, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Generate permutation importance explanations."""
        print("Generating permutation importance explanations...")
        
        if y is None:
            raise ValueError("Target values required for permutation importance")
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X, y,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            scoring=self.scoring
        )
        
        return {
            "importance_mean": perm_importance.importances_mean,
            "importance_std": perm_importance.importances_std,
            "feature_names": list(X.columns),
            "method": "Permutation Importance"
        }
    
    def plot_explanations(self, explanations: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot permutation importance."""
        importance_mean = explanations["importance_mean"]
        importance_std = explanations["importance_std"]
        feature_names = explanations["feature_names"]
        
        # Sort by importance
        indices = np.argsort(importance_mean)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance_mean)), importance_mean[indices], 
                yerr=importance_std[indices], capsize=5)
        plt.xticks(range(len(importance_mean)), 
                  [feature_names[i] for i in indices], rotation=45)
        plt.title("Permutation Importance")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_permutation_importance.png", dpi=300, bbox_inches='tight')
        plt.show()


class IntegratedGradientsExplainer(BaseExplainer):
    """Integrated Gradients explainer for neural networks."""
    
    def __init__(self, baseline: Optional[np.ndarray] = None):
        self.baseline = baseline
        
        try:
            import torch
            import captum
            self.torch = torch
            self.captum = captum
        except ImportError:
            raise ImportError("Captum is required. Install with: pip install captum")
    
    def explain(self, model: Any, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Generate Integrated Gradients explanations."""
        print("Generating Integrated Gradients explanations...")
        
        # Convert to torch tensors
        X_tensor = self.torch.tensor(X.values, dtype=self.torch.float32)
        
        # Set baseline
        if self.baseline is None:
            baseline = self.torch.zeros_like(X_tensor)
        else:
            baseline = self.torch.tensor(self.baseline, dtype=self.torch.float32)
        
        # Create Integrated Gradients explainer
        ig = self.captum.attr.IntegratedGradients(model)
        
        # Generate attributions
        attributions = ig.attribute(X_tensor, baselines=baseline)
        
        return {
            "attributions": attributions.detach().numpy(),
            "feature_names": list(X.columns),
            "method": "Integrated Gradients"
        }
    
    def plot_explanations(self, explanations: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot Integrated Gradients explanations."""
        attributions = explanations["attributions"]
        feature_names = explanations["feature_names"]
        
        # Plot attributions for first instance
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(attributions[0])), attributions[0])
        plt.xticks(range(len(attributions[0])), feature_names, rotation=45)
        plt.title("Integrated Gradients Attribution (First Instance)")
        plt.xlabel("Features")
        plt.ylabel("Attribution")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_integrated_gradients.png", dpi=300, bbox_inches='tight')
        plt.show()


def get_explainer(explainer_name: str, **kwargs) -> BaseExplainer:
    """
    Factory function to get appropriate explainer.
    
    Args:
        explainer_name: Name of the explainer ("shap", "lime", "permutation", "integrated_gradients")
        **kwargs: Additional arguments for the explainer
        
    Returns:
        BaseExplainer: Appropriate explainer instance
        
    Raises:
        ValueError: If explainer name is not supported
    """
    explainers = {
        "shap": SHAPExplainer,
        "lime": LIMEExplainer,
        "permutation": PermutationImportanceExplainer,
        "integrated_gradients": IntegratedGradientsExplainer,
    }
    
    if explainer_name not in explainers:
        raise ValueError(f"Explainer {explainer_name} not supported. Available: {list(explainers.keys())}")
    
    return explainers[explainer_name](**kwargs)
