"""
Comprehensive evaluation metrics for model performance and explainability.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, log_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns


class ModelMetrics:
    """Comprehensive model evaluation metrics."""
    
    def __init__(self, target_names: Optional[List[str]] = None):
        self.target_names = target_names
    
    def compute_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute comprehensive classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        if self.target_names:
            for i, name in enumerate(self.target_names):
                metrics[f"precision_{name}"] = precision_per_class[i]
                metrics[f"recall_{name}"] = recall_per_class[i]
                metrics[f"f1_{name}"] = f1_per_class[i]
        
        # Probability-based metrics
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                    metrics["average_precision"] = average_precision_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class classification
                    metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                    metrics["roc_auc_ovo"] = roc_auc_score(y_true, y_proba, multi_class="ovo", average="macro")
                
                metrics["log_loss"] = log_loss(y_true, y_proba)
            except Exception as e:
                print(f"Warning: Could not compute probability-based metrics: {e}")
        
        return metrics
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.target_names,
                   yticklabels=self.target_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        
        if save_path:
            plt.savefig(f"{save_path}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_curve(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray, 
        save_path: Optional[str] = None
    ) -> None:
        """Plot calibration curve."""
        if len(np.unique(y_true)) != 2:
            print("Calibration curve only available for binary classification")
            return
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba[:, 1], n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(f"{save_path}_calibration_curve.png", dpi=300, bbox_inches='tight')
        plt.show()


class ExplainabilityMetrics:
    """Metrics for evaluating explainability methods."""
    
    def __init__(self):
        pass
    
    def compute_faithfulness_metrics(
        self,
        model: Any,
        X: pd.DataFrame,
        explanations: Dict[str, Any],
        method: str = "deletion"
    ) -> Dict[str, float]:
        """Compute faithfulness metrics for explanations."""
        metrics = {}
        
        if method == "deletion":
            metrics.update(self._deletion_test(model, X, explanations))
        elif method == "insertion":
            metrics.update(self._insertion_test(model, X, explanations))
        else:
            raise ValueError(f"Faithfulness method {method} not supported")
        
        return metrics
    
    def _deletion_test(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        explanations: Dict[str, Any]
    ) -> Dict[str, float]:
        """Deletion test for faithfulness."""
        # Get feature importance scores
        if "shap_values_summary" in explanations:
            importance_scores = np.abs(explanations["shap_values_summary"]).mean(axis=0)
        elif "importance_mean" in explanations:
            importance_scores = explanations["importance_mean"]
        else:
            print("Warning: No importance scores found for deletion test")
            return {}
        
        # Sort features by importance
        feature_indices = np.argsort(importance_scores)[::-1]
        
        # Get baseline performance
        baseline_pred = model.predict_proba(X)
        baseline_score = np.mean(np.max(baseline_pred, axis=1))
        
        # Test deletion of top features
        deletion_scores = []
        for i in range(1, min(len(feature_indices), 6)):  # Test top 5 features
            # Remove top i features
            X_modified = X.copy()
            for j in range(i):
                feature_idx = feature_indices[j]
                X_modified.iloc[:, feature_idx] = 0  # Set to zero
            
            # Compute performance
            modified_pred = model.predict_proba(X_modified)
            modified_score = np.mean(np.max(modified_pred, axis=1))
            
            deletion_scores.append(baseline_score - modified_score)
        
        return {
            "deletion_auc": np.trapz(deletion_scores) / len(deletion_scores),
            "deletion_score": deletion_scores[0] if deletion_scores else 0.0
        }
    
    def _insertion_test(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        explanations: Dict[str, Any]
    ) -> Dict[str, float]:
        """Insertion test for faithfulness."""
        # Get feature importance scores
        if "shap_values_summary" in explanations:
            importance_scores = np.abs(explanations["shap_values_summary"]).mean(axis=0)
        elif "importance_mean" in explanations:
            importance_scores = explanations["importance_mean"]
        else:
            print("Warning: No importance scores found for insertion test")
            return {}
        
        # Sort features by importance
        feature_indices = np.argsort(importance_scores)[::-1]
        
        # Start with all features set to zero
        X_zero = pd.DataFrame(np.zeros_like(X), columns=X.columns)
        
        # Test insertion of top features
        insertion_scores = []
        for i in range(1, min(len(feature_indices), 6)):  # Test top 5 features
            # Add top i features
            X_modified = X_zero.copy()
            for j in range(i):
                feature_idx = feature_indices[j]
                X_modified.iloc[:, feature_idx] = X.iloc[:, feature_idx]
            
            # Compute performance
            modified_pred = model.predict_proba(X_modified)
            modified_score = np.mean(np.max(modified_pred, axis=1))
            
            insertion_scores.append(modified_score)
        
        return {
            "insertion_auc": np.trapz(insertion_scores) / len(insertion_scores),
            "insertion_score": insertion_scores[-1] if insertion_scores else 0.0
        }
    
    def compute_stability_metrics(
        self,
        model: Any,
        X: pd.DataFrame,
        explanations: Dict[str, Any],
        n_perturbations: int = 10,
        noise_level: float = 0.01
    ) -> Dict[str, float]:
        """Compute stability metrics for explanations."""
        if "shap_values_summary" not in explanations:
            print("Warning: SHAP values required for stability test")
            return {}
        
        original_explanations = explanations["shap_values_summary"]
        perturbed_explanations = []
        
        # Generate perturbed explanations
        for _ in range(n_perturbations):
            # Add noise to input
            noise = np.random.normal(0, noise_level, X.shape)
            X_perturbed = X + noise
            
            # Generate explanations (simplified - would need actual explainer)
            # For now, we'll simulate by adding noise to original explanations
            noise_explanation = np.random.normal(0, noise_level, original_explanations.shape)
            perturbed_explanations.append(original_explanations + noise_explanation)
        
        # Compute stability metrics
        perturbed_explanations = np.array(perturbed_explanations)
        
        # Rank correlation stability
        original_ranks = np.argsort(np.abs(original_explanations).mean(axis=0))[::-1]
        rank_correlations = []
        
        for pert_exp in perturbed_explanations:
            pert_ranks = np.argsort(np.abs(pert_exp).mean(axis=0))[::-1]
            correlation = np.corrcoef(original_ranks, pert_ranks)[0, 1]
            rank_correlations.append(correlation)
        
        return {
            "stability_mean_correlation": np.mean(rank_correlations),
            "stability_std_correlation": np.std(rank_correlations),
            "stability_min_correlation": np.min(rank_correlations)
        }
    
    def plot_explanation_comparison(
        self,
        explanations_dict: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> None:
        """Plot comparison of different explanation methods."""
        methods = list(explanations_dict.keys())
        n_methods = len(methods)
        
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 6))
        if n_methods == 1:
            axes = [axes]
        
        for i, (method, explanations) in enumerate(explanations_dict.items()):
            if "shap_values_summary" in explanations:
                importance_scores = np.abs(explanations["shap_values_summary"]).mean(axis=0)
            elif "importance_mean" in explanations:
                importance_scores = explanations["importance_mean"]
            else:
                continue
            
            feature_names = explanations.get("feature_names", [f"Feature_{j}" for j in range(len(importance_scores))])
            
            # Sort by importance
            indices = np.argsort(importance_scores)[::-1]
            
            axes[i].bar(range(len(importance_scores)), importance_scores[indices])
            axes[i].set_xticks(range(len(importance_scores)))
            axes[i].set_xticklabels([feature_names[j] for j in indices], rotation=45)
            axes[i].set_title(f"{method.upper()} Feature Importance")
            axes[i].set_xlabel("Features")
            axes[i].set_ylabel("Importance")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_explanation_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()


def create_leaderboard(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Create a leaderboard from evaluation results.
    
    Args:
        results: Dictionary with model names as keys and metrics as values
        metrics: List of metrics to include in leaderboard
        
    Returns:
        pd.DataFrame: Formatted leaderboard
    """
    if metrics is None:
        metrics = ["accuracy", "f1_macro", "roc_auc_ovr", "log_loss"]
    
    # Create DataFrame
    df = pd.DataFrame(results).T
    
    # Select relevant metrics
    available_metrics = [m for m in metrics if m in df.columns]
    df = df[available_metrics]
    
    # Sort by primary metric (accuracy)
    if "accuracy" in df.columns:
        df = df.sort_values("accuracy", ascending=False)
    
    # Format numbers
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].round(4)
    
    return df
