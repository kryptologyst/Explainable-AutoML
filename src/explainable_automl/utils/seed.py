"""
Utility functions for deterministic seeding across different frameworks.
"""

import random
import os
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across different frameworks.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"Random seed set to {seed}")
    print(f"CUDA deterministic: {torch.backends.cudnn.deterministic}")
    print(f"CUDA benchmark: {torch.backends.cudnn.benchmark}")
