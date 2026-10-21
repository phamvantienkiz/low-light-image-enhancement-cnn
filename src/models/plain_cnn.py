"""
Plain CNN Baseline for Low-Light Image Enhancement.

A sequence of standard 2D convolutional layers without downsampling or skip connections.
Used as an empirical baseline to illustrate the limitation of narrow receptive fields
prior to adopting an encoder-decoder U-Net topology.
"""

import torch
import torch.nn as nn


class PlainCNN(nn.Module):
    """
    Plain 5-layer Convolutional Neural Network baseline.

    Args:
        in_channels (int): Number of input image channels (default: 3 for RGB).
        hidden_channels (int): Base hidden feature channels (default: 64).
        out_channels (int): Number of output image channels (default: 3 for RGB).
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input low-light image tensor of shape (B, 3, H, W) in [0, 1].
        Returns:
            torch.Tensor: Enhanced image tensor of shape (B, 3, H, W) in [0, 1].
        """
        return self.net(x)
