"""
Lightweight Attention U-Net Architecture for Low-Light Image Enhancement.

Inspired by DarkIR (CVPR 2025) and Edge-LLIE (CVPRW 2024):
1. Simplified Channel Attention (SCA): Dynamically recalibrates color/feature channels.
2. Gated Skip Connections: Suppresses high-frequency sensor noise from shallow encoder layers.
3. Multi-Scale Dilated Bottleneck: Expands the receptive field efficiently.
4. Global Residual Learning: Learns the additive illumination and color enhancement residual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedChannelAttention(nn.Module):
    """
    Simplified Channel Attention (SCA) as formulated in DarkIR / NAFNet.
    
    Applies Global Average Pooling followed by a 1x1 point-wise convolution
    to compute channel-wise recalibration weights with minimal FLOPs.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.sca(x)
        return x * weights


class GatedSkipConnection(nn.Module):
    """
    Gated Skip Connection to prevent noise leakage from early encoder layers.
    (Insight from Edge-LLIE CVPRW 2024).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.feat_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, enc_feat: torch.Tensor) -> torch.Tensor:
        gate = self.gate_conv(enc_feat)
        feat = self.feat_conv(enc_feat)
        return feat * gate


class ResidualAttentionBlock(nn.Module):
    """
    Residual Convolutional Block with LeakyReLU and Simplified Channel Attention.
    """

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.sca = SimplifiedChannelAttention(out_ch)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # Match dimensions for residual connection if in_ch != out_ch
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.sca(out)
        out = self.act2(out + res)
        return out


class LightweightAttnUNet(nn.Module):
    """
    Proposed Lightweight Attention U-Net.
    
    Total parameters: ~905,000 (~0.90M) with base_ch=16, easily trainable on Google Colab or locally.
    
    Args:
        in_channels (int): Input image channels (3 for RGB).
        out_channels (int): Output image channels (3 for RGB).
        base_ch (int): Base feature width (default: 16).
        use_residual (bool): Whether to use global residual connection (default: True).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_ch: int = 16,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8  # 16, 32, 64, 128

        # Encoder Levels
        self.enc1 = ResidualAttentionBlock(in_channels, c1)
        self.down1 = nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1)

        self.enc2 = ResidualAttentionBlock(c1, c2)
        self.down2 = nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1)

        self.enc3 = ResidualAttentionBlock(c2, c3)
        self.down3 = nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1)

        # Gated Skip Filters
        self.skip_gate1 = GatedSkipConnection(c1)
        self.skip_gate2 = GatedSkipConnection(c2)
        self.skip_gate3 = GatedSkipConnection(c3)

        # Multi-scale Dilated Bottleneck (dilation=2 captures wide receptive field)
        self.bottleneck1 = ResidualAttentionBlock(c3, c4, dilation=1)
        self.bottleneck2 = ResidualAttentionBlock(c4, c4, dilation=2)

        # Decoder Levels
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ResidualAttentionBlock(c3 * 2, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ResidualAttentionBlock(c2 * 2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ResidualAttentionBlock(c1 * 2, c1)

        # Final Reconstruction Head
        self.out_conv = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, out_channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input low-light image tensor (B, 3, H, W) in [0, 1].
        Returns:
            torch.Tensor: Enhanced image tensor (B, 3, H, W) clamped to [0, 1].
        """
        # Encoder
        e1 = self.enc1(x)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        e3 = self.enc3(d2)
        d3 = self.down3(e3)

        # Bottleneck
        b = self.bottleneck2(self.bottleneck1(d3))

        # Decoder with Gated Skips
        g3 = self.skip_gate3(e3)
        u3 = self.up3(b)
        x3 = self.dec3(torch.cat([u3, g3], dim=1))

        g2 = self.skip_gate2(e2)
        u2 = self.up2(x3)
        x2 = self.dec2(torch.cat([u2, g2], dim=1))

        g1 = self.skip_gate1(e1)
        u1 = self.up1(x2)
        x1 = self.dec1(torch.cat([u1, g1], dim=1))

        # Output prediction
        delta = self.out_conv(x1)

        if self.use_residual:
            enhanced = torch.clamp(x + delta, 0.0, 1.0)
        else:
            enhanced = torch.sigmoid(delta)

        return enhanced
