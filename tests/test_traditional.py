"""
Unit tests for traditional baselines (Gamma and CLAHE).
"""

import numpy as np
import torch
from src.baselines import apply_gamma_correction, apply_clahe


def test_gamma_correction():
    x = torch.full((1, 3, 32, 32), 0.1)
    gamma_out = apply_gamma_correction(x, gamma=0.5)
    # 0.1^0.5 = ~0.316 (brighter)
    assert gamma_out.mean() > x.mean()
    assert gamma_out.max() <= 1.0


def test_clahe():
    arr = np.random.uniform(0.05, 0.2, (64, 64, 3)).astype(np.float32)
    clahe_out = apply_clahe(arr, clip_limit=2.0)
    assert clahe_out.shape == (64, 64, 3)
    assert clahe_out.min() >= 0.0 and clahe_out.max() <= 1.0
