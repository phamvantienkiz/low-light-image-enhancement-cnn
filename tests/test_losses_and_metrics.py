"""
Unit tests for losses and evaluation metrics (PSNR, SSIM, CompositeLoss).
"""

import pytest
import torch
from src.losses import CompositeLoss, SSIMLoss
from src.metrics import calculate_psnr, calculate_ssim


def test_psnr_and_ssim_identical():
    img1 = torch.rand(1, 3, 64, 64)
    img2 = img1.clone()

    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)

    assert psnr >= 90.0  # Clamped to high value for zero MSE
    assert abs(ssim - 1.0) < 1e-4


def test_psnr_and_ssim_different():
    img1 = torch.zeros(1, 3, 64, 64)
    img2 = torch.ones(1, 3, 64, 64)

    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)

    assert psnr <= 1.0
    assert ssim < 0.1


def test_composite_loss():
    criterion = CompositeLoss(l1_weight=1.0, ssim_weight=0.2)
    pred = torch.rand(2, 3, 64, 64, requires_grad=True)
    target = torch.rand(2, 3, 64, 64)

    loss, metrics = criterion(pred, target)
    assert loss.item() > 0.0
    assert "loss_l1" in metrics
    assert "loss_ssim" in metrics

    loss.backward()
    assert pred.grad is not None
