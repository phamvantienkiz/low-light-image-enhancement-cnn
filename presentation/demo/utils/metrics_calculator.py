"""
Tiện ích đo lường chất lượng hình ảnh (PSNR, SSIM, PSNR_gtmean) và đo độ trễ suy luận (FPS / ms).
"""

import math
import time
from typing import Dict, Optional, Tuple
import numpy as np
from skimage.metrics import structural_similarity


def psnr_u8(pred: np.ndarray, gt: np.ndarray) -> float:
    """Tính PSNR thô (dB) trên mảng uint8 [0..255] RGB."""
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        return 99.0
    return 10.0 * math.log10(255.0 ** 2 / mse)


def ssim_u8(pred: np.ndarray, gt: np.ndarray) -> float:
    """Tính chỉ số cấu trúc SSIM chuẩn hóa đa kênh theo tiêu chuẩn đồ án."""
    return float(
        structural_similarity(
            pred,
            gt,
            channel_axis=2,
            data_range=255,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False
        )
    )


def psnr_gtmean(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Chỉ số chẩn đoán bóc tách lỗi độ sáng:
    Chuẩn hóa tỷ lệ độ sáng trung bình của ảnh dự đoán theo ảnh Ground Truth
    để đo chất lượng khôi phục cấu trúc thuần túy (loại trừ lỗi phơi sáng cảm biến).
    """
    scale = gt.mean() / max(pred.mean(), 1e-6)
    pred_aligned = np.clip(pred.astype(np.float64) * scale, 0, 255).round().astype(np.uint8)
    return psnr_u8(pred_aligned, gt)


def calculate_metrics(pred: np.ndarray, gt: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    """Tính toàn bộ các chỉ số khi có ảnh tham chiếu Ground Truth."""
    if gt is None:
        mean_lum_pred = float(pred.mean())
        return {
            "has_gt": False,
            "psnr": None,
            "ssim": None,
            "psnr_gtmean": None,
            "mean_lum": round(mean_lum_pred, 1)
        }

    # Đảm bảo cùng kích thước
    if pred.shape != gt.shape:
        return {
            "has_gt": False,
            "psnr": None,
            "ssim": None,
            "psnr_gtmean": None,
            "mean_lum": round(float(pred.mean()), 1)
        }

    raw_psnr = round(psnr_u8(pred, gt), 2)
    ssim_val = round(ssim_u8(pred, gt), 4)
    gtmean_val = round(psnr_gtmean(pred, gt), 2)

    return {
        "has_gt": True,
        "psnr": raw_psnr,
        "ssim": ssim_val,
        "psnr_gtmean": gtmean_val,
        "mean_lum": round(float(pred.mean()), 1)
    }


class BenchmarkTimer:
    """Đo thời gian suy luận (Latency) và tốc độ khung hình (FPS)."""
    def __init__(self):
        self.start_time = 0.0
        self.latency_ms = 0.0
        self.fps = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        self.latency_ms = round(elapsed * 1000.0, 1)
        self.fps = round(1.0 / max(elapsed, 1e-6), 1)
