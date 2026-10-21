"""
Evaluation metrics for Low-Light Image Enhancement: PSNR and SSIM.
"""

import math
import numpy as np
import torch
from src.losses.composite_loss import ssim_tensor


def calculate_psnr(
    pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0
) -> float:
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Args:
        pred: Predicted tensor of shape (B, C, H, W) in [0, 1].
        target: Ground-truth tensor of shape (B, C, H, W) in [0, 1].
        max_val: Maximum dynamic range (1.0 for normalized tensors).

    Returns:
        float: PSNR in dB.
    """
    with torch.no_grad():
        mse = torch.mean((pred - target) ** 2).item()
        if mse == 0.0:
            return 100.0  # Infinite PSNR clamped
        return 10.0 * math.log10((max_val**2) / mse)


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculates Mean Structural Similarity Index (SSIM).

    Args:
        pred: Predicted tensor of shape (B, C, H, W) in [0, 1].
        target: Ground-truth tensor of shape (B, C, H, W) in [0, 1].

    Returns:
        float: SSIM in [0, 1].
    """
    with torch.no_grad():
        val = ssim_tensor(pred, target, window_size=11, size_average=True)
        return float(val.item())
