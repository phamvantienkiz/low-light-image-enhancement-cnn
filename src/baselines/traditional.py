"""
Traditional Computer Vision Baselines for Low-Light Image Enhancement.

Implements:
1. Gamma Correction: Non-linear luminance expansion.
2. CLAHE (Contrast-Limited Adaptive Histogram Equalization): Local contrast redistribution.
"""

from typing import Union
import numpy as np
import torch
from PIL import Image


def apply_gamma_correction(
    image: Union[np.ndarray, torch.Tensor], gamma: float = 0.4
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies power-law (Gamma) correction: I_out = I_in ^ gamma.
    For low-light images, gamma < 1.0 brightens shadows non-linearly.

    Args:
        image: Input image as numpy array [0, 1] or torch.Tensor [0, 1].
        gamma: Exponent factor (default: 0.4).

    Returns:
        Enhanced image with same type and range [0, 1].
    """
    if isinstance(image, torch.Tensor):
        return torch.pow(torch.clamp(image, 1e-8, 1.0), gamma)
    elif isinstance(image, np.ndarray):
        return np.power(np.clip(image, 1e-8, 1.0), gamma)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def apply_clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE) on the Luminance channel.
    Uses PIL / numpy pure histogram transformation.

    Args:
        image: RGB numpy float array in [0, 1] with shape (H, W, 3).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Number of tiles in grid (M, N).

    Returns:
        Enhanced RGB numpy float array in [0, 1].
    """
    # Convert RGB to YCbCr / Luminance
    # Y = 0.299 R + 0.587 G + 0.114 B
    img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    
    # Process Luminance per tile
    h, w, c = img_uint8.shape
    r, g, b = img_uint8[:, :, 0], img_uint8[:, :, 1], img_uint8[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b

    # Simple adaptive histogram equalization across tiles
    th, tw = tile_grid_size
    step_y, step_x = h // th, w // tw
    y_eq = np.zeros_like(y, dtype=np.float32)

    for i in range(th):
        for j in range(tw):
            y_start, y_end = i * step_y, (i + 1) * step_y if i < th - 1 else h
            x_start, x_end = j * step_x, (j + 1) * step_x if j < tw - 1 else w
            tile = y[y_start:y_end, x_start:x_end].astype(np.uint8)

            hist, bins = np.histogram(tile.flatten(), 256, [0, 256])
            # Clip histogram
            clip_val = int(clip_limit * (tile.size / 256.0))
            excess = np.sum(np.maximum(hist - clip_val, 0))
            hist = np.minimum(hist, clip_val) + excess // 256

            # Cumulative distribution function (CDF)
            cdf = hist.cumsum()
            cdf_normalized = cdf * 255.0 / (cdf[-1] + 1e-8)
            y_eq[y_start:y_end, x_start:x_end] = np.interp(tile.flatten(), bins[:-1], cdf_normalized).reshape(tile.shape)

    # Reconstruct RGB preserving chromaticity ratios
    ratio = (y_eq + 1e-3) / (y + 1e-3)
    r_out = np.clip(r * ratio, 0, 255)
    g_out = np.clip(g * ratio, 0, 255)
    b_out = np.clip(b * ratio, 0, 255)

    out = np.stack([r_out, g_out, b_out], axis=2) / 255.0
    return out.astype(np.float32)
