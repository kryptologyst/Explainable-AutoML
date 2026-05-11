#!/usr/bin/env python3
"""
Main script to run the Explainable AutoML pipeline.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omegaconf import OmegaConf
from explainable_automl.pipeline import ExplainableAutoMLPipeline


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Run Explainable AutoML Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["iris", "wine", "breast_cancer"],
        help="Override dataset choice"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["tpot", "logistic_regression", "random_forest", "gradient_boosting"],
        help="Override model choice"
    )
    parser.add_argument(
        "--explainer",
        type=str,
        choices=["shap", "lime", "permutation"],
        help="Override explainer choice"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = OmegaConf.load(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    # Override config with command line arguments
    if args.dataset:
        config.data.name = args.dataset
    if args.model:
        if args.model == "tpot":
            config.model._target_ = "explainable_automl.models.tpot_classifier.TPOTClassifierWrapper"
        else:
            config.model._target_ = f"explainable_automl.models.tpot_classifier.BaselineClassifier"
            config.model.algorithm = args.model
    if args.explainer:
        config.explainer._target_ = f"explainable_automl.explainers.shap_explainer.{args.explainer.title()}Explainer"
    
    # Set verbose logging
    if args.verbose:
        config.logging.level = "DEBUG"
    
    # Print configuration
    print("="*60)
    print("EXPLAINABLE AUTOML PIPELINE")
    print("="*60)
    print(f"Dataset: {config.data.name}")
    print(f"Model: {config.model._target_.split('.')[-1]}")
    print(f"Explainer: {config.explainer._target_.split('.')[-1]}")
    print(f"Seed: {config.experiment.seed}")
    print("="*60)
    
    # Initialize and run pipeline
    try:
        pipeline = ExplainableAutoMLPipeline(config)
        results = pipeline.run_full_pipeline(save_results=not args.no_save)
        
        # Print summary
        pipeline.print_summary()
        
        # Create leaderboard
        print("\nLEADERBOARD:")
        leaderboard = pipeline.create_leaderboard()
        print(leaderboard.to_string())
        
        print("\nPipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
