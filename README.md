# Explainable AutoML

A comprehensive framework for automated machine learning with interpretability, combining TPOT (AutoML) with multiple explainability methods including SHAP, LIME, and Integrated Gradients.

## Author

**kryptologyst**  
GitHub: https://github.com/kryptologyst

## Safety Disclaimer

⚠️ **This is a research/educational tool. Not for production decisions or control.**

Ensure proper consent and privacy protection when using with real data. This framework is designed for research and educational purposes only.

## Features

- **Automated Model Selection**: Uses TPOT for automated pipeline optimization
- **Multiple Explainability Methods**: SHAP, LIME, Permutation Importance, Integrated Gradients
- **Comprehensive Evaluation**: Model performance metrics and explanation quality assessment
- **Interactive Demo**: Streamlit-based web application for exploring results
- **Reproducible Research**: Deterministic seeding and structured logging
- **Modern Architecture**: Type hints, configuration management, and modular design

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Explainable-AutoML.git
cd Explainable-AutoML

# Install the package in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Optional Dependencies

For specific explainability methods:

```bash
# For SHAP
pip install shap

# For LIME
pip install lime

# For Captum (PyTorch explainability)
pip install captum

# For Streamlit demo
pip install streamlit
```

## Quick Start

### 1. Run the Pipeline

```bash
# Run with default configuration (Iris dataset)
python scripts/run_pipeline.py

# Run with specific dataset
python scripts/run_pipeline.py --dataset wine

# Run with specific model
python scripts/run_pipeline.py --model random_forest

# Run with specific explainer
python scripts/run_pipeline.py --explainer shap

# Run with custom configuration
python scripts/run_pipeline.py --config configs/custom_config.yaml
```

### 2. Launch Interactive Demo

```bash
# Start Streamlit demo
streamlit run demo/streamlit_app.py
```

### 3. Programmatic Usage

```python
from omegaconf import OmegaConf
from explainable_automl.pipeline import ExplainableAutoMLPipeline

# Load configuration
config = OmegaConf.load("configs/config.yaml")

# Initialize pipeline
pipeline = ExplainableAutoMLPipeline(config)

# Run full pipeline
results = pipeline.run_full_pipeline()

# Print summary
pipeline.print_summary()
```

## Configuration

The framework uses YAML configuration files for easy customization:

### Main Configuration (`configs/config.yaml`)

```yaml
# @package _global_
defaults:
  - _self_
  - data: iris
  - model: tpot_classifier
  - explainer: shap_explainer
  - experiment: default

experiment:
  name: "explainable_automl"
  tags: ["xai", "automl", "research"]
  seed: 42

device: "auto"  # auto, cpu, cuda, mps

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Dataset Configuration (`configs/data/iris.yaml`)

```yaml
# @package data
_target_: explainable_automl.data.loaders.IrisDataLoader

name: "iris"
test_size: 0.2
random_state: 42
feature_names:
  - "sepal_length"
  - "sepal_width" 
  - "petal_length"
  - "petal_width"
target_names:
  - "setosa"
  - "versicolor"
  - "virginica"
```

### Model Configuration (`configs/model/tpot_classifier.yaml`)

```yaml
# @package model
_target_: explainable_automl.models.tpot_classifier.TPOTClassifierWrapper

generations: 5
population_size: 20
random_state: 42
verbosity: 2
cv_folds: 5
scoring: "accuracy"
max_time_mins: 10
max_eval_time_mins: 5
```

## Supported Datasets

- **Iris**: Classic 4-feature classification dataset
- **Wine**: 13-feature wine classification dataset  
- **Breast Cancer**: 30-feature medical classification dataset

## Supported Models

### AutoML Models
- **TPOT**: Tree-based Pipeline Optimization Tool

### Baseline Models
- **Logistic Regression**: Linear classification
- **Random Forest**: Ensemble tree-based classifier
- **Gradient Boosting**: Gradient boosting classifier
- **SVM**: Support Vector Machine
- **k-NN**: k-Nearest Neighbors
- **Naive Bayes**: Gaussian Naive Bayes
- **Decision Tree**: Single decision tree

## Supported Explainability Methods

### Global Explanations
- **SHAP**: SHapley Additive exPlanations
- **Permutation Importance**: Feature importance via permutation

### Local Explanations
- **LIME**: Local Interpretable Model-agnostic Explanations
- **Integrated Gradients**: Gradient-based attribution (for neural networks)

## Evaluation Metrics

### Model Performance
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and macro-averaged metrics
- **ROC-AUC**: Area under ROC curve (binary/multi-class)
- **Log Loss**: Cross-entropy loss
- **Calibration**: Reliability of probability estimates

### Explanation Quality
- **Faithfulness**: How well explanations reflect model behavior
  - Deletion Test: Performance drop when removing important features
  - Insertion Test: Performance gain when adding important features
- **Stability**: Consistency of explanations under perturbations
- **Rank Correlation**: Stability of feature importance rankings

## Project Structure

```
explainable-automl/
├── src/
│   └── explainable_automl/
│       ├── data/           # Data loading utilities
│       ├── models/          # Model implementations
│       ├── explainers/      # Explainability methods
│       ├── metrics/         # Evaluation metrics
│       ├── utils/           # Utility functions
│       └── pipeline.py      # Main pipeline
├── configs/                 # Configuration files
├── data/                    # Data storage
├── assets/                  # Output artifacts
├── tests/                   # Unit tests
├── scripts/                 # Execution scripts
├── demo/                    # Demo applications
├── notebooks/               # Jupyter notebooks
└── README.md               # This file
```

## Usage Examples

### Basic Usage

```python
from explainable_automl.pipeline import ExplainableAutoMLPipeline
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("configs/config.yaml")

# Run pipeline
pipeline = ExplainableAutoMLPipeline(config)
results = pipeline.run_full_pipeline()

# Access results
model_metrics = results["model_metrics"]
xai_metrics = results["xai_metrics"]
explanations = results["explanations"]
```

### Custom Configuration

```python
from omegaconf import OmegaConf

# Load base configuration
config = OmegaConf.load("configs/config.yaml")

# Override specific settings
config.data.name = "wine"
config.model.generations = 10
config.experiment.seed = 123

# Run with custom settings
pipeline = ExplainableAutoMLPipeline(config)
results = pipeline.run_full_pipeline()
```

### Multiple Explainers Comparison

```python
from explainable_automl.explainers.shap_explainer import get_explainer

# Compare different explainers
explainers = ["shap", "lime", "permutation"]
results = {}

for explainer_name in explainers:
    explainer = get_explainer(explainer_name)
    explanations = explainer.explain(model, X_test, y_test)
    results[explainer_name] = explanations
```

## Command Line Interface

```bash
# Basic usage
python scripts/run_pipeline.py

# With options
python scripts/run_pipeline.py \
    --dataset wine \
    --model random_forest \
    --explainer shap \
    --verbose

# Custom config
python scripts/run_pipeline.py --config configs/custom.yaml
```

## Interactive Demo

The Streamlit demo provides an interactive interface for exploring results:

```bash
streamlit run demo/streamlit_app.py
```

Features:
- Model performance visualization
- Feature importance analysis
- Explanation quality metrics
- Interactive feature exploration
- Configuration overview

## Development

### Setup Development Environment

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py
```

### Code Formatting

```bash
# Format code
black src/ tests/ scripts/

# Lint code
ruff check src/ tests/ scripts/

# Type checking
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{explainable_automl,
  title={Explainable AutoML: A Framework for Automated Machine Learning with Interpretability},
  author={kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Explainable-AutoML}
}
```

## Acknowledgments

- TPOT team for the AutoML framework
- SHAP authors for explainability methods
- LIME authors for local explanations
- The open-source community for various dependencies
# Explainable-AutoML
