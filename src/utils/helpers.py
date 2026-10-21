"""
Utility functions for reproducibility, logging, and checkpoint management.
"""

import os
import random
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import torchvision.utils as vutils


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across python, numpy, and torch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Counts total trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    state: Dict[str, Any], is_best: bool, save_dir: Path, filename: str = "last_model.pth"
) -> None:
    """Saves training checkpoint and tracks best model."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / filename
    torch.save(state, filepath)
    if is_best:
        best_path = save_dir / "best_model.pth"
        torch.save(state, best_path)


def save_image_tensor(tensor: torch.Tensor, save_path: Path) -> None:
    """Saves a torch.Tensor in [0, 1] as an image file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    clamped = torch.clamp(tensor, 0.0, 1.0)
    vutils.save_image(clamped, str(save_path))
