"""
Loss functions.
"""

from src.losses.composite_loss import SSIMLoss, CompositeLoss, ssim_tensor

__all__ = ["SSIMLoss", "CompositeLoss", "ssim_tensor"]
