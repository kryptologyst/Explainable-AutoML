"""
Main training and evaluation pipeline for Explainable AutoML.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from omegaconf import DictConfig
import joblib

from .utils.seed import set_seed
from .utils.device import get_device
from .utils.logging import setup_logging
from .data.loaders import get_data_loader
from .models.tpot_classifier import get_model
from .explainers.shap_explainer import get_explainer
from .metrics import ModelMetrics, ExplainabilityMetrics, create_leaderboard


class ExplainableAutoMLPipeline:
    """Main pipeline for Explainable AutoML."""
    
    def __init__(self, config: DictConfig):
        """Initialize the pipeline with configuration."""
        self.config = config
        
        # Setup logging
        self.logger = setup_logging(
            level=config.logging.level,
            format_string=config.logging.format
        )
        
        # Set random seed
        set_seed(config.experiment.seed)
        
        # Setup device
        self.device = get_device(config.device)
        
        # Initialize components
        self.data_loader = None
        self.model = None
        self.explainer = None
        self.model_metrics = None
        self.xai_metrics = None
        
        # Results storage
        self.results = {}
        
        self.logger.info("Explainable AutoML Pipeline initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Seed: {config.experiment.seed}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load and split data."""
        self.logger.info("Loading data...")
        
        # Get data loader
        self.data_loader = get_data_loader(
            self.config.data.name,
            test_size=self.config.data.test_size,
            random_state=self.config.experiment.seed,
            feature_names=self.config.data.get("feature_names"),
            target_names=self.config.data.get("target_names")
        )
        
        # Load data
        X, y, feature_names, target_names = self.data_loader.load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = self.data_loader.split_data(X, y)
        
        self.logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Train set: {X_train.shape[0]} samples")
        self.logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model."""
        self.logger.info("Training model...")
        
        # Get model
        model_config = self.config.model
        self.model = get_model(
            model_config._target_.split('.')[-1].replace('Wrapper', '').lower(),
            **{k: v for k, v in model_config.items() if k != '_target_'}
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        self.logger.info("Model training completed")
    
    def evaluate_model(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        self.logger.info("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Initialize metrics
        if self.model_metrics is None:
            target_names = getattr(self.data_loader, 'target_names', None)
            self.model_metrics = ModelMetrics(target_names)
        
        # Compute metrics
        metrics = self.model_metrics.compute_classification_metrics(y_test, y_pred, y_proba)
        
        # Plot confusion matrix
        if save_path:
            self.model_metrics.plot_confusion_matrix(y_test, y_pred, save_path)
            if len(np.unique(y_test)) == 2:
                self.model_metrics.plot_calibration_curve(y_test, y_proba, save_path)
        
        self.logger.info(f"Model evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def explain_model(
        self, 
        X_test: pd.DataFrame, 
        y_test: Optional[pd.Series] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate model explanations."""
        self.logger.info("Generating model explanations...")
        
        # Get explainer
        explainer_config = self.config.explainer
        self.explainer = get_explainer(
            explainer_config._target_.split('.')[-1].replace('Explainer', '').lower(),
            **{k: v for k, v in explainer_config.items() if k != '_target_'}
        )
        
        # Generate explanations
        explanations = self.explainer.explain(self.model, X_test, y_test)
        explanations["X_explain"] = X_test
        
        # Plot explanations
        if save_path:
            self.explainer.plot_explanations(explanations, save_path)
        
        self.logger.info(f"Explanations generated using {explanations['method']}")
        
        return explanations
    
    def evaluate_explanations(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        explanations: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate explanation quality."""
        self.logger.info("Evaluating explanation quality...")
        
        # Initialize XAI metrics
        if self.xai_metrics is None:
            self.xai_metrics = ExplainabilityMetrics()
        
        # Compute faithfulness metrics
        faithfulness_metrics = self.xai_metrics.compute_faithfulness_metrics(
            self.model, X_test, explanations, method="deletion"
        )
        
        # Compute stability metrics
        stability_metrics = self.xai_metrics.compute_stability_metrics(
            self.model, X_test, explanations
        )
        
        # Combine metrics
        xai_metrics = {**faithfulness_metrics, **stability_metrics}
        
        self.logger.info("Explanation evaluation completed")
        
        return xai_metrics
    
    def run_full_pipeline(self, save_results: bool = True) -> Dict[str, Any]:
        """Run the complete pipeline."""
        self.logger.info("Starting full Explainable AutoML pipeline...")
        
        try:
            # Load data
            X_train, X_test, y_train, y_test = self.load_data()
            
            # Train model
            self.train_model(X_train, y_train)
            
            # Evaluate model
            model_metrics = self.evaluate_model(X_test, y_test, "assets/model_evaluation")
            
            # Generate explanations
            explanations = self.explain_model(X_test, y_test, "assets/explanations")
            
            # Evaluate explanations
            xai_metrics = self.evaluate_explanations(X_test, y_test, explanations)
            
            # Combine results
            self.results = {
                "model_metrics": model_metrics,
                "xai_metrics": xai_metrics,
                "explanations": explanations,
                "config": self.config
            }
            
            # Save results
            if save_results:
                self.save_results()
            
            self.logger.info("Pipeline completed successfully!")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def save_results(self) -> None:
        """Save results to disk."""
        os.makedirs("assets", exist_ok=True)
        
        # Save model
        if self.model:
            self.model.save_model("assets/best_model.pkl")
        
        # Save results
        joblib.dump(self.results, "assets/results.pkl")
        
        # Save configuration
        with open("assets/config.yaml", "w") as f:
            from omegaconf import OmegaConf
            OmegaConf.save(self.config, f)
        
        self.logger.info("Results saved to assets/")
    
    def create_leaderboard(self, additional_results: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
        """Create a leaderboard of results."""
        if additional_results:
            all_results = {**self.results["model_metrics"], **additional_results}
        else:
            all_results = {"current_model": self.results["model_metrics"]}
        
        return create_leaderboard(all_results)
    
    def print_summary(self) -> None:
        """Print a summary of results."""
        if not self.results:
            print("No results available. Run the pipeline first.")
            return
        
        print("\n" + "="*50)
        print("EXPLAINABLE AUTOML PIPELINE SUMMARY")
        print("="*50)
        
        # Model performance
        model_metrics = self.results["model_metrics"]
        print(f"\nModel Performance:")
        print(f"  Accuracy: {model_metrics['accuracy']:.4f}")
        print(f"  F1-Score (Macro): {model_metrics['f1_macro']:.4f}")
        if 'roc_auc_ovr' in model_metrics:
            print(f"  ROC-AUC (OVR): {model_metrics['roc_auc_ovr']:.4f}")
        
        # Explanation quality
        xai_metrics = self.results["xai_metrics"]
        print(f"\nExplanation Quality:")
        if 'deletion_auc' in xai_metrics:
            print(f"  Faithfulness (Deletion AUC): {xai_metrics['deletion_auc']:.4f}")
        if 'stability_mean_correlation' in xai_metrics:
            print(f"  Stability (Mean Correlation): {xai_metrics['stability_mean_correlation']:.4f}")
        
        # Safety disclaimer
        print(f"\nSafety Disclaimer:")
        print(f"  {self.config.safety.disclaimer}")
        print(f"  {self.config.safety.ethics_note}")
        
        print("\n" + "="*50)
