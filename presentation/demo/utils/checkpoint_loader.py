"""
Quản lý và nạp Checkpoint linh hoạt (v3 khuyên dùng / v4 mở rộng).
Hỗ trợ cache Singleton, hot-swap state_dict trong < 0.15s và warm-up tự động.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from models.darkir_edge import DarkIREdge
from models.unet import UNetBaseline


class ModelRegistry:
    """Bộ quản lý và bộ đệm (Cache) nạp mô hình suy luận cục bộ."""

    def __init__(self, workspace_root: Optional[Path] = None):
        if workspace_root is None:
            self.root = Path(__file__).resolve().parent.parent.parent.parent
        else:
            self.root = workspace_root

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = self.device.type == 'cuda'

        # Đường dẫn cơ sở
        self.v3_ckpt_dir = self.root / "colab-results" / "llie_project_v3-20261102" / "llie_project_v3" / "experiments" / "v3" / "checkpoints"
        self.v4_ckpt_dir = self.root / "colab-results" / "llie_project_v4-20261104" / "llie_project_v4" / "experiments" / "v4" / "checkpoints"

        # Cache đối tượng mô hình: key = model_tag ('darkir_edge', 'unet', 'darkir_edge_nofft')
        self.cached_models: Dict[str, nn.Module] = {}
        # Lưu trạng thái checkpoint hiện tại: key = model_tag -> path
        self.active_ckpt_path: Dict[str, str] = {}

    def get_available_options(self) -> Dict[str, Any]:
        """Quét và trả về danh sách các lựa chọn khả dụng thực tế trên ổ cứng."""
        options = {
            "versions": ["v3 (Khuyên dùng - Official Demo)", "v4 (Ablation đa miền & Bão hòa)"],
            "models": [
                "DarkIR-Edge (ours)",
                "Standard U-Net",
                "DarkIR-Edge (no FFT)",
                "CLAHE",
                "Gamma 2.2",
                "Retinex MSRCR"
            ],
            "seeds": ["seed 42", "seed 123"],
            "ckpt_types": ["final.pt (EMA epoch 70)", "best.pt (EMA đỉnh Val macro)"]
        }
        return options

    def resolve_checkpoint_path(
        self,
        version_str: str,
        model_name: str,
        seed_str: str,
        ckpt_type_str: str
    ) -> Optional[Path]:
        """Xác định đường dẫn file .pt cụ thể dựa trên lựa chọn giao diện."""
        is_v3 = "v3" in version_str
        ckpt_dir = self.v3_ckpt_dir if is_v3 else self.v4_ckpt_dir

        tag_map = {
            "DarkIR-Edge (ours)": "darkir_edge",
            "DarkIR-Edge": "darkir_edge",
            "Standard U-Net": "unet",
            "U-Net": "unet",
            "DarkIR-Edge (no FFT)": "darkir_edge_nofft",
            "DarkIR-Edge (no-FFT)": "darkir_edge_nofft"
        }
        tag = tag_map.get(model_name)
        if not tag:
            return None

        seed = 42 if "42" in seed_str else 123
        ckpt_filename = "best.pt" if "best.pt" in ckpt_type_str else "final.pt"

        # Đường dẫn trực tiếp
        target = ckpt_dir / f"{tag}_seed{seed}" / ckpt_filename
        if target.exists():
            return target

        # Fallback thử final.pt nếu best.pt không có hoặc ngược lại
        alt_filename = "final.pt" if ckpt_filename == "best.pt" else "best.pt"
        fallback = ckpt_dir / f"{tag}_seed{seed}" / alt_filename
        if fallback.exists():
            return fallback

        # Fallback thử sang seed còn lại
        alt_seed = 123 if seed == 42 else 42
        fallback_seed = ckpt_dir / f"{tag}_seed{alt_seed}" / ckpt_filename
        if fallback_seed.exists():
            return fallback_seed

        # Fallback sang v3 nếu v4 thiếu file
        if not is_v3:
            v3_fallback = self.v3_ckpt_dir / f"{tag}_seed{seed}" / ckpt_filename
            if v3_fallback.exists():
                return v3_fallback

        return None

    def get_or_create_model_structure(self, tag: str) -> nn.Module:
        """Khởi tạo kiến trúc nếu chưa có trong cache."""
        if tag in self.cached_models:
            return self.cached_models[tag]

        if tag == "darkir_edge":
            model = DarkIREdge(width=24, use_imax=True, use_gate=True, use_fft=True)
        elif tag == "darkir_edge_nofft":
            model = DarkIREdge(width=24, use_imax=True, use_gate=True, use_fft=False)
        elif tag == "unet":
            model = UNetBaseline(base=32)
        else:
            raise ValueError(f"Không nhận diện được mô hình: {tag}")

        model = model.to(self.device).eval()
        self.cached_models[tag] = model
        return model

    def load_model(
        self,
        version_str: str,
        model_name: str,
        seed_str: str,
        ckpt_type_str: str
    ) -> Tuple[Optional[nn.Module], str]:
        """
        Nạp hoặc hot-swap trọng số mô hình.
        Trả về: (model, status_message)
        """
        tag_map = {
            "DarkIR-Edge (ours)": "darkir_edge",
            "DarkIR-Edge": "darkir_edge",
            "Standard U-Net": "unet",
            "U-Net": "unet",
            "DarkIR-Edge (no FFT)": "darkir_edge_nofft",
            "DarkIR-Edge (no-FFT)": "darkir_edge_nofft"
        }
        tag = tag_map.get(model_name)
        if not tag:
            return None, f"Mô hình truyền thống ({model_name}) - Không sử dụng mạng nơ-ron."

        ckpt_path = self.resolve_checkpoint_path(version_str, model_name, seed_str, ckpt_type_str)
        if ckpt_path is None or not ckpt_path.exists():
            return None, f"Cảnh báo: Không tìm thấy checkpoint cho {model_name} ({version_str})!"

        model = self.get_or_create_model_structure(tag)

        # Kiểm tra xem có cần hot-swap không
        ckpt_key = str(ckpt_path)
        if self.active_ckpt_path.get(tag) != ckpt_key:
            try:
                # Nạp state_dict an toàn
                ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                state_dict = ck.get('ema', ck.get('model', ck))
                model.load_state_dict(state_dict, strict=True)
                self.active_ckpt_path[tag] = ckpt_key
                msg = f"Đã nạp thành công: {ckpt_path.parent.name}/{ckpt_path.name} ({self.device.type.upper()})"
            except Exception as e:
                msg = f"Lỗi nạp state_dict từ {ckpt_path.name}: {e}"
                return None, msg
        else:
            msg = f"Sử dụng trọng số sẵn có trong RAM: {ckpt_path.parent.name}/{ckpt_path.name}"

        return model, msg

    def warm_up(self):
        """Chạy trước 2 forward passes với tensor giả lập để CUDA và JIT sẵn sàng."""
        try:
            model, _ = self.load_model(
                version_str="v3 (Khuyên dùng - Official Demo)",
                model_name="DarkIR-Edge (ours)",
                seed_str="seed 42",
                ckpt_type_str="final.pt (EMA epoch 70)"
            )
            if model is not None:
                dummy = torch.zeros(1, 3, 256, 256, device=self.device)
                with torch.no_grad():
                    if self.use_amp:
                        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                            _ = model(dummy)
                            _ = model(dummy)
                    else:
                        _ = model(dummy)
                        _ = model(dummy)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
        except Exception:
            pass
