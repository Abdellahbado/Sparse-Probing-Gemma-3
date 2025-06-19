import os
from dataclasses import dataclass
from typing import List


HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = 'google/gemma-3-1b-it'
TARGET_LAYERS = [0, 1, 2, 3, 4] # Layers to probe



@dataclass
class ProbeConfig:
    """Configuration for sparse probing experiments"""
    k_values: List[int]  # Sparsity levels to test
    l1_reg: float = 1e-4  # L1 regularization strength
    l0_reg: float = 1e-3  # L0 regularization strength (for true sparsity)
    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 10
    cv_folds: int = 5
    random_seed: int = 42
    
PROBE_CONFIG = ProbeConfig(
    k_values=[1, 5, 10, 20, 50],  # Test different sparsity levels
    l1_reg=1e-4,
    l0_reg=1e-3,
    learning_rate=1e-3,
    num_epochs=50,
    batch_size=32,
    early_stopping_patience=5,
    cv_folds=3,
    random_seed=42
)