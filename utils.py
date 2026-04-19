"""
Utility helpers: seeding, logging, checkpoint I/O.
"""

import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a formatted console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dir(path: str) -> str:
    """Create directory if it does not exist, return the path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: str) -> None:
    """Save dictionary as JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> dict:
    """Load JSON file into dictionary."""
    with open(path, "r") as f:
        return json.load(f)


def save_numpy(arr: np.ndarray, path: str) -> None:
    """Save numpy array."""
    ensure_dir(os.path.dirname(path))
    np.save(path, arr)


def load_numpy(path: str) -> np.ndarray:
    """Load numpy array."""
    return np.load(path)
