"""
Unit tests for model architectures: PlainCNN, CompactUNet, LightweightAttnUNet.
"""

import pytest
import torch
from src.models import PlainCNN, CompactUNet, LightweightAttnUNet
from src.utils import count_parameters


def test_plain_cnn_forward():
    model = PlainCNN(in_channels=3, hidden_channels=64, out_channels=3)
    x = torch.rand(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 3, 128, 128)
    assert out.min() >= 0.0 and out.max() <= 1.0
    params = count_parameters(model)
    assert 50_000 < params < 200_000


def test_compact_unet_forward():
    model = CompactUNet(in_channels=3, out_channels=3, base_ch=16)
    x = torch.rand(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 3, 128, 128)
    assert out.min() >= 0.0 and out.max() <= 1.0
    params = count_parameters(model)
    assert 300_000 < params < 700_000


def test_lightweight_attn_unet_forward_and_backward():
    model = LightweightAttnUNet(in_channels=3, out_channels=3, base_ch=16, use_residual=True)
    x = torch.rand(2, 3, 128, 128, requires_grad=True)
    target = torch.rand(2, 3, 128, 128)

    out = model(x)
    assert out.shape == (2, 3, 128, 128)
    assert out.min() >= 0.0 and out.max() <= 1.0

    # Test backward gradient flow
    loss = torch.nn.functional.l1_loss(out, target)
    loss.backward()

    # Verify gradients exist and are not NaN
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN in gradient of {name}"

    params = count_parameters(model)
    assert 500_000 < params < 1_200_000
