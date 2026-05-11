"""
Device management utilities with automatic fallback.
"""

import torch
from typing import Union, Literal


def get_device(device: Union[str, Literal["auto"]] = "auto") -> torch.device:
    """
    Get the best available device with automatic fallback.
    
    Args:
        device: Device specification ("auto", "cpu", "cuda", "mps")
        
    Returns:
        torch.device: The selected device
        
    Raises:
        RuntimeError: If specified device is not available
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA device")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS device (Apple Silicon)")
        else:
            device = "cpu"
            print("Using CPU device")
    
    try:
        torch_device = torch.device(device)
        # Test if device is actually available
        if device == "cuda":
            torch.cuda.get_device_properties(0)
        elif device == "mps":
            torch.zeros(1, device=torch_device)
        return torch_device
    except Exception as e:
        print(f"Device {device} not available, falling back to CPU")
        return torch.device("cpu")
