"""
Kiến trúc Standard U-Net (base=32, 1,928,483 tham số) đối chuẩn trong đồ án.
Sử dụng các khối tích chập kép kết hợp GroupNorm(4), ReLU, và kết nối thặng dư toàn cục.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .darkir_edge import pad8


def _cb(i: int, o: int, g: int = 4) -> nn.Sequential:
    """Khối tích chập kép cơ bản: Conv 3x3 + GroupNorm + ReLU x 2."""
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(g, o), nn.ReLU(True),
        nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(g, o), nn.ReLU(True)
    )


class UNetBaseline(nn.Module):
    """
    Standard U-Net Plain Convolution:
    Cùng quy mô tham số (~1.93M) và 3 tầng downsample/upsample đối xứng với DarkIR-Edge.
    """
    def __init__(self, base: int = 32):
        super().__init__()
        b = base
        self.e1 = _cb(3, b)
        self.e2 = _cb(b, 2 * b)
        self.e3 = _cb(2 * b, 4 * b)
        self.mid = _cb(4 * b, 8 * b)

        self.u3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2)
        self.u2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2)
        self.u1 = nn.ConvTranspose2d(2 * b, b, 2, 2)

        self.d3 = _cb(8 * b, 4 * b)
        self.d2 = _cb(4 * b, 2 * b)
        self.d1 = _cb(2 * b, b)
        self.out = nn.Conv2d(b, 3, 1)

    def forward(self, x: torch.Tensor, return_side: bool = False):
        h, w = x.shape[-2:]
        x_pad = pad8(x)
        e1 = self.e1(x_pad)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        m = self.mid(F.max_pool2d(e3, 2))

        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        out = (self.out(d1) + x_pad)[..., :h, :w]
        return (out, None) if return_side else out
