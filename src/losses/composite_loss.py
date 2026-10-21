"""
Loss functions for Low-Light Image Enhancement.

Includes:
1. L1 Pixel Loss (Color fidelity and crisp edges).
2. Differentiable SSIM Loss (Structural consistency).
3. CompositeLoss: Combines L1 + lambda * (1 - SSIM).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ],
        dtype=torch.float32,
    )
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_tensor(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """
    Computes differentiable Structural Similarity Index (SSIM) between two image batches.

    Args:
        img1: Tensor of shape (B, C, H, W) in [0, 1].
        img2: Tensor of shape (B, C, H, W) in [0, 1].
        window_size: Gaussian window size (default: 11).
        size_average: If True, average over batch.

    Returns:
        torch.Tensor: SSIM value in [0, 1].
    """
    channel = img1.size(1)
    window = _create_window(window_size, channel).to(img1.device, dtype=img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss: 1 - SSIM."""

    def __init__(self, window_size: int = 11) -> None:
        super().__init__()
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim_tensor(pred, target, window_size=self.window_size)


class CompositeLoss(nn.Module):
    """
    Composite Loss for LLIE training:
    Loss = l1_weight * L1 + ssim_weight * (1 - SSIM)
    """

    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 0.2) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1 = nn.L1Loss()
        self.ssim_loss = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        l1_val = self.l1(pred, target)
        if self.ssim_weight > 0.0:
            ssim_val = self.ssim_loss(pred, target)
            total_loss = self.l1_weight * l1_val + self.ssim_weight * ssim_val
        else:
            ssim_val = torch.tensor(0.0, device=pred.device)
            total_loss = self.l1_weight * l1_val

        metrics = {
            "loss_total": total_loss.item(),
            "loss_l1": l1_val.item(),
            "loss_ssim": ssim_val.item(),
        }
        return total_loss, metrics
