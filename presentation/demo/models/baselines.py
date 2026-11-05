"""
Các thuật toán xử lý ảnh thiếu sáng truyền thống (Zero-param Baselines):
  1. Gamma Correction (g=2.2)
  2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
  3. MSRCR (Multi-Scale Retinex with Color Restoration)
"""

import numpy as np
import cv2


def gamma_fn(img: np.ndarray, g: float = 2.2) -> np.ndarray:
    """Biến đổi Gamma hiệu chỉnh độ sáng phi tuyến: I_out = (I_in / 255)^(1/g) * 255."""
    return (np.power(img.astype(np.float64) / 255.0, 1.0 / g) * 255.0).round().clip(0, 255).astype(np.uint8)


def clahe_fn(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """Cân bằng biểu đồ cục bộ thích ứng có giới hạn tương phản trên kênh L không gian LAB."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2RGB)


def msrcr_fn(
    img: np.ndarray,
    sigmas=(15, 80, 250),
    alpha: float = 125.0,
    beta: float = 46.0,
    G: float = 192.0,
    off: float = -30.0,
    lo_p: float = 1.0,
    hi_p: float = 99.0
) -> np.ndarray:
    """
    Multi-Scale Retinex with Color Restoration (MSRCR):
    Phục hồi màu sắc và phản xạ ánh sáng đa tỷ lệ với các thang lọc Gaussian (15, 80, 250).
    """
    x = img.astype(np.float64) + 1.0
    msr = sum(np.log10(x) - np.log10(cv2.GaussianBlur(x, (0, 0), s)) for s in sigmas) / len(sigmas)
    cr = beta * (np.log10(alpha * x) - np.log10(x.sum(axis=2, keepdims=True)))
    out = G * (msr * cr - off)
    for c in range(3):
        lo, hi = np.percentile(out[..., c], [lo_p, hi_p])
        out[..., c] = np.clip((out[..., c] - lo) / (hi - lo + 1e-6), 0, 1) * 255.0
    return out.round().astype(np.uint8)
