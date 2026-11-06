"""
Ứng Dụng Demo Cục Bộ Trực Tiếp Cho Buổi Thuyết Trình / Bảo Vệ Đồ Án (CS2203 - UIT)
Đề tài: Low-Light Image Enhancement Using a Lightweight CNN (DarkIR-Edge)
Techstack: Gradio >= 4.20.0, PyTorch, OpenCV, scikit-image, Pillow

Đặc điểm thiết kế:
  - 100% Local / Offline (không phụ thuộc Internet).
  - 100% Out-of-Split Isolation (không dùng bất kỳ ảnh nào trong Train, Val, Test).
  - Tích hợp nạp Checkpoint v3 (chính thức) và v4 (nghiên cứu mở rộng).
  - Giao diện Light Mode màu trắng trang nhã, phông chữ tối ưu, tinh giản chữ thừa.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import cv2
import torch
import gradio as gr

# Thêm thư mục hiện tại vào sys.path để import cục bộ
DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

# Đảm bảo console Windows in ký tự UTF-8 an toàn
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from models import DarkIREdge, UNetBaseline, clahe_fn, gamma_fn, msrcr_fn
from utils import (
    to_tensor,
    to_numpy_u8,
    verify_out_of_split,
    calculate_metrics,
    ModelRegistry
)

# Khởi tạo Singleton Model Registry
print("==================================================================")
print("[INFO] Dang khoi dong ung dung Demo LLIE DarkIR-Edge...")
print("==================================================================")
REGISTRY = ModelRegistry(workspace_root=DEMO_DIR.parent.parent)
REGISTRY.warm_up()
print(f"[OK] Thiet bi suy luan: {REGISTRY.device.type.upper()} | AMP: {REGISTRY.use_amp}")

# Nạp danh mục Presets từ preset_manifest.json
PRESETS_FILE = DEMO_DIR / "presets" / "preset_manifest.json"
PRESETS_DICT: Dict[str, Dict[str, Any]] = {}

if PRESETS_FILE.exists():
    try:
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            presets_list = data.get("presets", [])
            for p in presets_list:
                PRESETS_DICT[p["name"]] = p
    except Exception as e:
        print(f"Lỗi đọc preset_manifest.json: {e}")


def load_preset_images(preset_info: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Đọc ảnh Low và Ground Truth từ thư mục presets."""
    in_file = preset_info.get("input_file")
    gt_file = preset_info.get("gt_file")

    in_path = (DEMO_DIR / "presets" / in_file) if in_file else None
    gt_path = (DEMO_DIR / "presets" / gt_file) if gt_file else None

    in_img = None
    gt_img = None

    if in_path and in_path.exists():
        in_img = cv2.cvtColor(cv2.imread(str(in_path)), cv2.COLOR_BGR2RGB)
    if gt_path and gt_path.exists():
        gt_img = cv2.cvtColor(cv2.imread(str(gt_path)), cv2.COLOR_BGR2RGB)

    return in_img, gt_img


def run_inference_single(
    image_input: Optional[np.ndarray],
    preset_choice: str,
    version_choice: str,
    model_choice: str,
    seed_choice: str,
    ckpt_type_choice: str,
    gt_state: Optional[np.ndarray] = None
) -> Tuple[Any, str, str, str]:
    """Xử lý suy luận cho Tab 1 (Thanh trượt Before / After)."""
    if image_input is None:
        return None, "⚠️ Chưa có ảnh đầu vào. Hãy chọn ảnh mẫu hoặc tải ảnh.", "", ""

    # Kiểm tra bảo chứng dữ liệu ngoài tập huấn luyện/kiểm thử
    is_safe, verify_msg = verify_out_of_split(image_input)
    badge_status = (
        "🛡️ **100% Out-of-Split Verified** *(Zero-Leakage)*"
        if is_safe else f"⚠️ **Cảnh báo rò rỉ:** {verify_msg}"
    )

    t_start = time.perf_counter()
    pred_u8: Optional[np.ndarray] = None
    exec_info = ""

    # Nhánh thuật toán truyền thống
    if model_choice == "CLAHE":
        pred_u8 = clahe_fn(image_input)
        exec_info = "CLAHE (Không gian màu Lab)"
    elif model_choice == "Gamma 2.2":
        pred_u8 = gamma_fn(image_input, g=2.2)
        exec_info = "Gamma Correction (g=2.2)"
    elif model_choice == "Retinex MSRCR":
        pred_u8 = msrcr_fn(image_input)
        exec_info = "Retinex MSRCR đa tỷ lệ"
    else:
        # Nhánh Deep Learning
        model, load_msg = REGISTRY.load_model(
            version_str=version_choice,
            model_name=model_choice,
            seed_str=seed_choice,
            ckpt_type_str=ckpt_type_choice
        )
        if model is None:
            return None, f"❌ {load_msg}", "", ""

        x = to_tensor(image_input).unsqueeze(0).to(REGISTRY.device)
        with torch.no_grad():
            if REGISTRY.use_amp:
                with torch.autocast(device_type=REGISTRY.device.type, dtype=torch.float16):
                    out_tensor = model(x)
            else:
                out_tensor = model(x)
        pred_u8 = to_numpy_u8(out_tensor)
        exec_info = load_msg

    elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
    fps_val = round(1000.0 / max(elapsed_ms, 1e-4), 1)

    # Đo lường định lượng
    metrics = calculate_metrics(pred_u8, gt_state)
    if metrics["has_gt"]:
        metrics_md = f"""
| Chỉ Số Đo Lường | Kết Quả | Ý Nghĩa / Mục Tiêu |
| :--- | :---: | :--- |
| **PSNR Thô** | **`{metrics['psnr']} dB`** | Mức độ trung thực màu sắc tổng thể (cao hơn là tốt hơn) |
| **SSIM** | **`{metrics['ssim']}`** | Độ bảo toàn cấu trúc và chi tiết biên cạnh (tối đa 1.0) |
| **PSNR gt-mean** | **`{metrics['psnr_gtmean']} dB`** | **Chẩn đoán cấu trúc:** PSNR sau khi căn chỉnh độ chói trung bình |
| **Độ Trễ Suy Luận** | **`{elapsed_ms} ms`** | Thời gian xử lý trên một khung hình |
| **Tốc Độ Xử Lý** | **`~{fps_val} FPS`** | Khả năng đáp ứng thời gian thực (Real-time) |
"""
    else:
        metrics_md = f"""
| Chỉ Số Đo Lường | Kết Quả | Ghi Chú |
| :--- | :---: | :--- |
| **Độ Sáng Gốc** | **`{round(float(image_input.mean()), 1)} / 255`** | Độ chói trung bình ảnh đầu vào |
| **Độ Sáng Sau Khôi Phục** | **`{metrics['mean_lum']} / 255`** | Độ chói sau khi tăng cường |
| **Độ Trễ Suy Luận** | **`{elapsed_ms} ms`** | Thời gian xử lý mô hình |
| **Tốc Độ Xử Lý** | **`~{fps_val} FPS`** | Suy luận thời gian thực |
"""

    selected_p = PRESETS_DICT.get(preset_choice, {})
    speech_notes = selected_p.get("desc", "Quan sát sự bảo toàn chi tiết cạnh và hạn chế nhiễu hạt màu.")
    notes_md = f"**Nhận xét:** {speech_notes} *(Nạp: {exec_info})*"

    return (image_input, pred_u8), badge_status, metrics_md, notes_md


def run_compare_all(
    image_input: Optional[np.ndarray],
    version_choice: str,
    seed_choice: str,
    ckpt_type_choice: str,
    gt_state: Optional[np.ndarray] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str]:
    """Xử lý so sánh đối đầu 4 mô hình đồng thời (Tab 2)."""
    if image_input is None:
        return None, None, None, None, "⚠️ Chưa có ảnh đầu vào."

    # 1. CLAHE
    t0 = time.perf_counter()
    out_clahe = clahe_fn(image_input)
    t_clahe = (time.perf_counter() - t0) * 1000.0

    # 2. Standard U-Net
    m_unet, _ = REGISTRY.load_model(version_choice, "Standard U-Net", seed_choice, ckpt_type_choice)
    x = to_tensor(image_input).unsqueeze(0).to(REGISTRY.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        if REGISTRY.use_amp:
            with torch.autocast(device_type=REGISTRY.device.type, dtype=torch.float16):
                y_unet = m_unet(x)
        else:
            y_unet = m_unet(x)
    t_unet = (time.perf_counter() - t0) * 1000.0
    out_unet = to_numpy_u8(y_unet)

    # 3. DarkIR-Edge
    m_edge, _ = REGISTRY.load_model(version_choice, "DarkIR-Edge (ours)", seed_choice, ckpt_type_choice)
    t0 = time.perf_counter()
    with torch.no_grad():
        if REGISTRY.use_amp:
            with torch.autocast(device_type=REGISTRY.device.type, dtype=torch.float16):
                y_edge = m_edge(x)
        else:
            y_edge = m_edge(x)
    t_edge = (time.perf_counter() - t0) * 1000.0
    out_edge = to_numpy_u8(y_edge)

    # Đo lường so sánh
    m_clahe_data = calculate_metrics(out_clahe, gt_state)
    m_unet_data = calculate_metrics(out_unet, gt_state)
    m_edge_data = calculate_metrics(out_edge, gt_state)

    if gt_state is not None:
        table_md = f"""
| Phương Pháp | PSNR Thô (dB ↑) | SSIM (↑) | PSNR gt-mean (dB ↑) | Độ Trễ (ms ↓) | Nhận Xét Định Tính |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Ảnh Gốc** | — | — | — | — | Tối sâu, thiếu chi tiết |
| **CLAHE (Truyền thống)** | `{m_clahe_data['psnr']}` | `{m_clahe_data['ssim']}` | `{m_clahe_data['psnr_gtmean']}` | `~{round(t_clahe, 1)}` | Khuếch đại mạnh nhiễu hạt |
| **Standard U-Net (1.93M)** | `{m_unet_data['psnr']}` | `{m_unet_data['ssim']}` | `{m_unet_data['psnr_gtmean']}` | `~{round(t_unet, 1)}` | Khôi phục độ sáng và tương phản tốt |
| **DarkIR-Edge (1.92M)** | **`{m_edge_data['psnr']}`** | **`{m_edge_data['ssim']}`** | **`{m_edge_data['psnr_gtmean']}`** | `~{round(t_edge, 1)}` | Màu sắc trong trẻo, kiểm soát tốt nhiễu |
"""
    else:
        table_md = f"""
| Phương Pháp | Độ Sáng Trung Bình (0..255) | Thời Gian Suy Luận | Tốc Độ Ước Tính |
| :--- | :---: | :---: | :---: |
| **Ảnh Gốc** | `{round(float(image_input.mean()), 1)}` | — | — |
| **CLAHE** | `{round(float(out_clahe.mean()), 1)}` | `{round(t_clahe, 1)} ms` | CPU |
| **Standard U-Net** | `{round(float(out_unet.mean()), 1)}` | `{round(t_unet, 1)} ms` | `~{round(1000.0/max(t_unet, 1e-4), 1)} FPS` |
| **DarkIR-Edge** | `{round(float(out_edge.mean()), 1)}` | `{round(t_edge, 1)} ms` | `~{round(1000.0/max(t_edge, 1e-4), 1)} FPS` |
"""

    return image_input, out_clahe, out_unet, out_edge, table_md


def on_preset_selected(preset_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Khi người dùng chọn preset từ danh sách."""
    if preset_name not in PRESETS_DICT:
        return None, None, "Vui lòng chọn ảnh mẫu."
    p_info = PRESETS_DICT[preset_name]
    low_img, gt_img = load_preset_images(p_info)
    desc = p_info.get("desc", "")
    return low_img, gt_img, f"Đã nạp: **{preset_name}** — {desc}"


# =========================================================================
# GIAO DIỆN CHUẨN GỐC GRADIO (THEME TỰ NHIÊN, KHÔNG XUNG ĐỘT MÀU)
# =========================================================================

custom_css = """
.container { max-width: 1240px; margin: auto; }
.header-box { 
    text-align: center; 
    padding: 18px 12px; 
    background: linear-gradient(135deg, #0f172a, #1e293b); 
    color: white; 
    border-radius: 10px; 
    margin-bottom: 16px; 
}
.header-box h1 { margin: 0 0 6px 0; font-size: 24px; color: #38bdf8; font-weight: 700; }
.header-box p { margin: 0 0 10px 0; font-size: 14px; color: #cbd5e1; }
.badge-verified { 
    background-color: #065f46; 
    color: #a7f3d0; 
    padding: 4px 12px; 
    border-radius: 9999px; 
    font-weight: 600; 
    font-size: 12.5px; 
    display: inline-block; 
}
table { width: 100%; border-collapse: collapse; margin-top: 8px; }
th, td { padding: 8px 12px; }
"""

theme = gr.themes.Soft()

with gr.Blocks(title="Demo LLIE DarkIR-Edge") as demo:
    # State ẩn lưu ảnh Ground Truth của preset hiện tại (nếu có)
    gt_state = gr.State(value=None)

    with gr.Column(elem_classes="container"):
        # Header tinh gọn
        gr.HTML("""
        <div class="header-box">
            <h1>Phục Hồi Ảnh Thiếu Sáng — DarkIR-Edge</h1>
            <p>Nghiên cứu & Thực nghiệm CNN Nhẹ Cho Tăng Cường Ảnh Thiếu Sáng | CS2203 - UIT</p>
            <span class="badge-verified">🛡️ 100% Out-of-Split Verified (Không trùng Train, Val, Test)</span>
        </div>
        """)

        with gr.Tabs():
            # =================================================================
            # TAB 1: THANH TRƯỢT SO SÁNH TRƯỚC / SAU
            # =================================================================
            with gr.Tab("Thanh Trượt Trước / Sau"):
                with gr.Row():
                    # Cột điều khiển bên trái
                    with gr.Column(scale=4):
                        preset_dropdown = gr.Dropdown(
                            choices=list(PRESETS_DICT.keys()),
                            value=list(PRESETS_DICT.keys())[0] if PRESETS_DICT else None,
                            label="Ảnh Mẫu (Tuyển chọn từ dataset)",
                            interactive=True
                        )

                        custom_img_input = gr.Image(
                            label="Tải Ảnh Từ Máy Tính",
                            type="numpy",
                            interactive=True
                        )

                        with gr.Row():
                            version_dd = gr.Dropdown(
                                choices=["v3 (Khuyên dùng)", "v4 (Đa miền)"],
                                value="v3 (Khuyên dùng)",
                                label="Phiên bản",
                                interactive=True
                            )
                            model_dd = gr.Dropdown(
                                choices=[
                                    "DarkIR-Edge (ours)",
                                    "Standard U-Net",
                                    "DarkIR-Edge (no FFT)",
                                    "CLAHE",
                                    "Gamma 2.2",
                                    "Retinex MSRCR"
                                ],
                                value="DarkIR-Edge (ours)",
                                label="Mô hình / Thuật toán",
                                interactive=True
                            )

                        with gr.Row():
                            seed_dd = gr.Dropdown(
                                choices=["seed 42", "seed 123"],
                                value="seed 42",
                                label="Seed",
                                interactive=True
                            )
                            ckpt_dd = gr.Dropdown(
                                choices=["final.pt (EMA epoch 70)", "best.pt (EMA đỉnh Val)"],
                                value="final.pt (EMA epoch 70)",
                                label="Trọng số",
                                interactive=True
                            )

                        btn_enhance = gr.Button("Khôi Phục Ảnh", variant="primary", size="lg")
                        status_box = gr.Markdown("Sẵn sàng.")

                    # Cột kết quả bên phải
                    with gr.Column(scale=8):
                        try:
                            slider_component = gr.ImageSlider(
                                label="Kéo thanh trượt: Trái = Ảnh Gốc | Phải = Ảnh Phục Hồi",
                                interactive=False
                            )
                        except Exception:
                            slider_component = gr.Image(label="Ảnh Kết Quả", interactive=False)

                        hud_badge = gr.Markdown("")

                        with gr.Accordion("Chỉ Số Thực Nghiệm Tức Thời", open=True):
                            metrics_hud = gr.Markdown("*(Chỉ số sẽ hiển thị sau khi bấm Khôi Phục)*")

                        speech_notes_box = gr.Markdown("")

            # =================================================================
            # TAB 2: SO SÁNH ĐA MÔ HÌNH
            # =================================================================
            with gr.Tab("So Sánh Đa Mô Hình"):
                with gr.Row():
                    btn_compare = gr.Button("So Sánh Trực Diện (CLAHE vs U-Net vs DarkIR-Edge)", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column():
                        img_orig = gr.Image(label="1. Ảnh Gốc", interactive=False)
                    with gr.Column():
                        img_clahe = gr.Image(label="2. CLAHE (Cổ điển)", interactive=False)

                with gr.Row():
                    with gr.Column():
                        img_unet = gr.Image(label="3. Standard U-Net (1.93M)", interactive=False)
                    with gr.Column():
                        img_edge = gr.Image(label="4. DarkIR-Edge (1.92M)", interactive=False)

                with gr.Row():
                    compare_table = gr.Markdown("Nhấn nút để chạy so sánh đối đầu.")

    # =========================================================================
    # SỰ KIỆN TƯƠNG TÁC
    # =========================================================================

    preset_dropdown.change(
        fn=on_preset_selected,
        inputs=[preset_dropdown],
        outputs=[custom_img_input, gt_state, status_box]
    )

    btn_enhance.click(
        fn=run_inference_single,
        inputs=[custom_img_input, preset_dropdown, version_dd, model_dd, seed_dd, ckpt_dd, gt_state],
        outputs=[slider_component, hud_badge, metrics_hud, speech_notes_box]
    )

    btn_compare.click(
        fn=run_compare_all,
        inputs=[custom_img_input, version_dd, seed_dd, ckpt_dd, gt_state],
        outputs=[img_orig, img_clahe, img_unet, img_edge, compare_table]
    )

    # Nạp preset đầu tiên khi khởi động
    demo.load(
        fn=lambda: on_preset_selected(list(PRESETS_DICT.keys())[0]) if PRESETS_DICT else (None, None, "Sẵn sàng"),
        inputs=None,
        outputs=[custom_img_input, gt_state, status_box]
    )


if __name__ == "__main__":
    print("------------------------------------------------------------------")
    print("[LAUNCH] May chu Gradio khoi chay tai http://127.0.0.1:7860 ...")
    print("         Nhan Ctrl+C trong terminal de dung ung dung.")
    print("------------------------------------------------------------------")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        inbrowser=True,
        theme=theme,
        css=custom_css
    )
