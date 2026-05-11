"""
Explainable AutoML: A comprehensive framework for automated machine learning with interpretability.

This package provides tools for:
- Automated model selection using TPOT
- Multiple explainability methods (SHAP, LIME, Integrated Gradients, etc.)
- Comprehensive evaluation of model interpretability
- Interactive demos for exploring explanations

Author: kryptologyst
GitHub: https://github.com/kryptologyst

DISCLAIMER: This is a research/educational tool. Not for production decisions or control.
"""

__version__ = "1.0.0"
__author__ = "kryptologyst"
__email__ = "kryptologyst@example.com"

from .utils.seed import set_seed
from .utils.device import get_device
from .utils.logging import setup_logging

__all__ = [
    "set_seed",
    "get_device", 
    "setup_logging",
]
