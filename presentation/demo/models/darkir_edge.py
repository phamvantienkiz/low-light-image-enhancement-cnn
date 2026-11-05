"""
Kiến trúc đề xuất DarkIR-Edge (C=24, ~1.924M tham số) phục vụ suy luận Demo.
Kế thừa cấu trúc bất đối xứng từ DarkIR (CVPR 2025) và 2 cải tiến vệ tinh:
  1. Max-RGB Illumination Prior (I_max = max(R, G, B)) ở đầu vào 4 kênh.
  2. Residual Gated Skip Connection (GatedSkip) tại 3 tầng nối tắt để lọc nhiễu.
Hỗ trợ cả biến thể đầy đủ (use_fft=True) và biến thể cắt bỏ (use_fft=False).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad8(x: torch.Tensor) -> torch.Tensor:
    """Đệm phản xạ (Reflection Padding) lên bội số của 8 để hỗ trợ ảnh kích thước bất kỳ."""
    h, w = x.shape[-2:]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    return F.pad(x, (0, pw, 0, ph), mode='reflect') if (ph or pw) else x


def extract_imax_prior(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Trích xuất bản đồ độ rọi cực đại Max-RGB:
        I_max = max(R, G, B)
    Đầu vào: rgb_tensor (B, 3, H, W) -> Đầu ra: [R, G, B, I_max] (B, 4, H, W).
    """
    i_max = rgb_tensor.amax(dim=1, keepdim=True)
    return torch.cat([rgb_tensor, i_max], dim=1)


class LayerNorm2d(nn.Module):
    """LayerNorm tính trên chiều kênh (Channel-wise LayerNorm) dạng (B, C, H, W)."""
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        mu = x32.mean(1, keepdim=True)
        var = (x32 - mu).pow(2).mean(1, keepdim=True)
        y = (x32 - mu) / torch.sqrt(var + self.eps)
        return (y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)).to(x.dtype)


class SimpleGate(nn.Module):
    """Cổng phi tuyến không tham số: tách đôi kênh và nhân chéo (x1 * x2)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class FreMLP(nn.Module):
    """
    [DarkIR CVPR 2025 - Nhánh Tần Số Fourier]
    Điều biến biên độ phổ Fourier (Amplitude) để nâng sáng toàn cục, giữ nguyên pha (Phase).
    Chạy ở FP32 và nhân trực tiếp với Phasor đơn vị (xf / mag) để ổn định số học.
    """
    def __init__(self, nc: int, expand: int = 2, norm: str = 'backward', eps: float = 1e-6):
        super().__init__()
        self.norm, self.eps = norm, eps
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        # Bắt buộc tính FFT ở FP32
        x_float = x.float()
        xf = torch.fft.rfft2(x_float, norm=self.norm)
        mag = torch.sqrt(xf.real ** 2 + xf.imag ** 2 + self.eps)
        mag_out = self.process1(mag)
        yf = xf * (mag_out / mag)
        y = torch.fft.irfft2(yf, s=(h, w), norm=self.norm)
        return y.to(x.dtype)


class Branch(nn.Module):
    """Nhánh tích chập chiều sâu giãn nở (Dilated Depthwise Convolution)."""
    def __init__(self, c: int, dw_expand: int, dilation: int = 1):
        super().__init__()
        self.dw_channel = dw_expand * c
        self.branch = nn.Sequential(
            nn.Conv2d(
                self.dw_channel, self.dw_channel, kernel_size=3, padding=dilation,
                stride=1, groups=self.dw_channel, bias=True, dilation=dilation
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class EBlock(nn.Module):
    """Khối Encoder chuẩn DarkIR: Tích hợp SCA + SimpleGate + FreMLP."""
    def __init__(self, c: int, dw_expand: int = 2, dilations=(1,), extra_depth_wise: bool = True, use_fft: bool = True):
        super().__init__()
        self.dw_channel = dw_expand * c
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity()
        self.conv1 = nn.Conv2d(c, self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.branches = nn.ModuleList([Branch(c, dw_expand, dilation=d) for d in dilations])
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dw_channel // 2, self.dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(self.dw_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLP(nc=c, expand=2) if use_fft else None
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(self.extra_conv(x))
        z = sum(branch(x) for branch in self.branches)
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x

        # Bước 2: Điều biến đặc trưng tần số (nếu có FFT)
        x_step2 = self.norm2(y)
        if self.freq is not None:
            x_freq = self.freq(x_step2)
            x = y * x_freq
            x = y + x * self.gamma
        else:
            x = y
        return x


class DBlock(nn.Module):
    """Khối Decoder chuẩn DarkIR: Cụm tích chập đa tỷ lệ Di-SpAM d=[1, 4, 9]."""
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, dilations=(1, 4, 9), extra_depth_wise: bool = True):
        super().__init__()
        self.dw_channel = dw_expand * c
        self.conv1 = nn.Conv2d(c, self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity()
        self.branches = nn.ModuleList([Branch(self.dw_channel, dw_expand=1, dilation=d) for d in dilations])
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dw_channel // 2, self.dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        )
        self.sg1 = SimpleGate()
        self.sg2 = SimpleGate()
        self.conv3 = nn.Conv2d(self.dw_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.extra_conv(self.conv1(x))
        z = sum(branch(x) for branch in self.branches)
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x

        # Bước 2: FFN khôi phục chi tiết không gian
        x = self.conv4(self.norm2(y))
        x = self.sg2(x)
        x = self.conv5(x)
        return y + x * self.gamma


class GatedSkip(nn.Module):
    """Cổng lọc nhiễu dư Residual Gated Skip Connection: F_gated = x + Conv(x) * Sigmoid(Conv(x))."""
    def __init__(self, c: int):
        super().__init__()
        self.f = nn.Conv2d(c, c, 1)
        self.g = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.f(x) * torch.sigmoid(self.g(x))


class DarkIREdge(nn.Module):
    """
    Kiến trúc chính DarkIR-Edge:
    - Width C=24 (1,924,038 tham số khi use_fft=True; 1,495,422 tham số khi use_fft=False).
    - Cải tiến 1: use_imax=True -> Intro Conv nhận tensor 4 kênh [R, G, B, I_max].
    - Cải tiến 2: use_gate=True -> Cổng GatedSkip tại 3 tầng nối tắt.
    """
    def __init__(
        self,
        width: int = 24,
        enc_blk_nums=(1, 2, 3),
        dec_blk_nums=(3, 1, 1),
        middle_blk_num_enc: int = 2,
        middle_blk_num_dec: int = 2,
        dilations=(1, 4, 9),
        extra_depth_wise: bool = True,
        use_imax: bool = True,
        use_gate: bool = True,
        use_fft: bool = True
    ):
        super().__init__()
        self.use_imax = use_imax
        in_ch = 4 if use_imax else 3
        self.intro = nn.Conv2d(in_ch, width, kernel_size=3, padding=1, stride=1, bias=True)
        self.ending = nn.Conv2d(width, 3, kernel_size=3, padding=1, stride=1, bias=True)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        self.decoders, self.ups, self.gates = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[EBlock(chan, extra_depth_wise=extra_depth_wise, use_fft=use_fft) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks_enc = nn.Sequential(*[EBlock(chan, extra_depth_wise=extra_depth_wise, use_fft=use_fft) for _ in range(middle_blk_num_enc)])
        self.middle_blks_dec = nn.Sequential(*[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_dec)])
        self.side_out = nn.Conv2d(chan, 3, kernel_size=3, stride=1, padding=1)

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.gates.append(GatedSkip(chan) if use_gate else nn.Identity())
            self.decoders.append(nn.Sequential(*[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(num)]))

    def forward(self, inp: torch.Tensor, return_side: bool = False):
        h, w = inp.shape[-2:]
        x_pad = pad8(inp)
        x_in = extract_imax_prior(x_pad) if self.use_imax else x_pad
        x = self.intro(x_in)

        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x_light = self.middle_blks_enc(x)
        out_side = self.side_out(x_light) if return_side else None
        x = self.middle_blks_dec(x_light)
        x = x + x_light  # Bottleneck residual

        for decoder, up, gate, skip in zip(self.decoders, self.ups, self.gates, skips[::-1]):
            x = up(x)
            x = x + gate(skip)
            x = decoder(x)

        x = self.ending(x) + x_pad
        out = x[..., :h, :w]
        if return_side and out_side is not None:
            out_side = out_side[..., :h // 8, :w // 8]
        return (out, out_side) if return_side else out
