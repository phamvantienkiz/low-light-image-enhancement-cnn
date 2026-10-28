# 📚 Tổng Hợp Báo Cáo Nghiên Cứu Các Bài Báo Khoa Học (Literature Review Reports)

Thư mục này chứa 4 báo cáo nghiên cứu học thuật chuyên sâu được phân tích độc lập từ 4 công trình khoa học đỉnh cao (CVPR & CVPRW giai đoạn 2024 – 2025) phục vụ cho dự án **"Tăng cường chất lượng ảnh trong điều kiện thiếu sáng bằng mạng nơ-ron tích chập gọn nhẹ (Lightweight CNN / Compact U-Net)"**.

---

## 📑 Danh Mục Các Báo Cáo Độc Lập

1. [**Báo Cáo 01: DarkIR (CVPR 2025)**](01_DarkIR_Feijoo_CVPR2025_Report.md)  
   *DarkIR: Robust Low-Light Image Restoration* (Daniel Feijoo et al. - Cidaut AI, Univ. of Würzburg).  
   👉 *Trọng tâm:* Xử lý đồng thời đa suy thoái (nhiễu cảm biến, thiếu sáng và mờ chuyển động/rung tay do phơi sáng lâu) bằng cấu trúc MetaFormer bất đối xứng kết hợp miền tần số Fourier (Encoder) và tích chập giãn nở không gian Di-SpAM (Decoder).

2. [**Báo Cáo 02: NTIRE 2024 Challenge on Low Light Image Enhancement (CVPRW 2024)**](02_NTIRE2024_Challenge_Liu_CVPRW2024_Report.md)  
   *NTIRE 2024 Challenge on Low Light Image Enhancement: Methods and Results* (Xiaoning Liu et al. - Ban tổ chức NTIRE & 22 đội thi).  
   👉 *Trọng tâm:* Chuẩn đối chuẩn mới cho ảnh cực tối, ngược sáng và cảnh đêm ở độ phân giải siêu nét (4K UHD+); phân tích kiến trúc của các đội vô địch (ESDNet-L, Retinexformer, DWT-FFC, DiffLight, UHDMamba) và thách thức nghẽn bộ nhớ GPU thực tế.

3. [**Báo Cáo 03: Edge-LLIE (CVPRW 2024)**](03_EdgeLLIE_Sharif_CVPRW2024_Report.md)  
   *Learning Optimized Low-Light Image Enhancement for Edge Vision Tasks* (S M A Sharif et al. - Opt-AI Inc., Seoul).  
   👉 *Trọng tâm:* Thiết kế FCN U-Net siêu nhẹ kết hợp cổng chặn nhiễu trên Skip Connection, lượng tử hóa INT8 với Histogram Calibration, đạt tốc độ suy luận **199.67 FPS (~5ms)** trên NVIDIA Jetson Orin và tăng cường vượt bậc cho bài toán phát hiện vật thể (YOLOv3) và khớp ảnh stereo (SuperGlue).

4. [**Báo Cáo 04: HVI Color Space & CIDNet (CVPR 2025)**](04_HVI_CIDNet_Yan_CVPR2025_Report.md)  
   *HVI: A New Color Space for Low-light Image Enhancement* (Qingsen Yan et al. - Northwestern Polytechnical Univ. & SMU).  
   👉 *Trọng tâm:* Đột phá về không gian màu; chỉ ra và giải quyết tận gốc 2 khuyết tật hình học của HSV (Red Discontinuity và Black Plane Noise) bằng không gian màu HVI có tham số co sụp học được $C_k$, kết hợp mạng 2 nhánh CIDNet (1.88M params) đạt kỷ lục SOTA trên 10 bộ dữ liệu benchmark.

5. [**Báo Cáo 05: Chiến Lược Thực Hiện, Tối Ưu Hóa Không Tăng Tham Số & Phân Vùng Dữ Liệu Đa Nguồn Trên Colab**](05_Implementation_Strategy_and_Optimization_Report.md)  
   *Tổng hợp hướng đi thực nghiệm kết hợp DarkIR + 3 bài báo vệ tinh, chiến lược trộn dữ liệu đa thiết bị (DSLR + Smartphone) và tối ưu hóa thời gian huấn luyện trên Google Colab.*  
   👉 *Trọng tâm:* 5 kỹ thuật nâng cấp Zero/Micro-Parameter (< 1M params, nhẹ hơn DarkIR-m 3.6 lần) để chạy thời gian thực trên thiết bị biên/smartphone; đánh giá phản biện cách chia dữ liệu phân tầng `(250 : 300 : 200)` kèm mở rộng ảnh smartphone (LSRW-Huawei / Custom Phone); bài toán định lượng thời gian chạy trên Google Colab (T4/L4) chỉ mất 8–20 phút/100 epochs thay vì 5–10 tiếng.

---

## 📊 Bảng So Sánh Đối Chiếu Tổng Quan Giữa 4 Công Trình

| Tiêu chí | DarkIR (CVPR 2025) | NTIRE 2024 Challenge (CVPRW 2024) | Edge-LLIE (CVPRW 2024) | HVI & CIDNet (CVPR 2025) | Dự án Compact U-Net Của Chúng Ta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Loại công trình** | Mô hình đơn lẻ SOTA | Báo cáo toàn cảnh cuộc thi | Mô hình thiết bị biên & thị giác hạ nguồn | Không gian màu mới & Mạng baseline | Nghiên cứu & hiện thực hóa hệ thống hoàn chỉnh |
| **Trường phái kiến trúc** | Asymmetric CNN (Fourier + Spatial Di-SpAM) | Đa dạng: SAM-CNN, Transformer, Diffusion, Mamba | Fully Convolutional U-Net + Residual Blocks | Dual-branch U-Net + Lighten Cross-Attention | **Compact U-Net (Encoder-Decoder + Skip Connections)** |
| **Không gian màu** | sRGB | sRGB | sRGB | **HVI (Polarized HS + Learnable $C_k$)** | sRGB (Mặc định) + Mở rộng HVI |
| **Số tham số (Params)** | 0.87M - 12.96M (Mặc định: 3.31M) | 1.5M - 30M+ (Đa số > 15M) | **4.46M** | **1.88M** | **Dự kiến < 1.5M - 2.0M** |
| **Độ phức tạp tính toán** | 7.25 GMACs (256px) | Đa phần rất nặng (cần A6000 48GB để chạy 4K) | Siêu nhẹ, đo đạc trực tiếp trên ARM64 Orin | 7.57 GFLOPs (256px) | **Mục tiêu < 10 GFLOPs** |
| **Tốc độ thực tế / Trễ** | ~0.05s GPU | Khó chạy 4K trên GPU 12GB | **199.67 FPS (5.01 ms trên Jetson Orin)** | **0.053s GPU** | **Đạt chuẩn Real-time (> 30-60 FPS)** |
| **Kết quả trên LOL-v1** | N/A (Tập trung LOLBlur) | Dùng làm cơ sở dữ liệu bổ trợ | 16.44 dB PSNR / 0.700 SSIM | **28.20 dB PSNR / 0.889 SSIM** | **Kỳ vọng đối chuẩn ~22 - 25 dB** |
| **Kết quả trên LOL-v2 Real**| 23.87 dB PSNR | N/A | 19.04 dB PSNR / 0.755 SSIM | **24.11 dB PSNR / 0.871 SSIM** | **Kỳ vọng đối chuẩn ~20 - 23 dB** |

---

## 🎯 Tổng Hợp Các Bài Học Đắt Giá Cho Việc Hiện Thực Hóa Dự Án Hiện Tại

Qua việc nghiên cứu chuyên sâu 4 bài báo, chúng ta rút ra 5 nguyên lý cốt lõi áp dụng trực tiếp vào cấu trúc mã nguồn `src/` và pipeline thực nghiệm:

1. **Thiết kế Skip Connections chống rò rỉ nhiễu (từ Edge-LLIE):**
   - Không nên dùng kết nối tắt `torch.cat` thô sơ nguyên bản của U-Net y sinh. Phải đặt một tầng tích chập điểm $1 \times 1$ hoặc cơ chế Gating để tinh lọc nhiễu hạt trước khi đưa vào Decoder.
2. **Chiến lược hàm mất mát kết hợp (từ DarkIR & NTIRE Rank 1):**
   - Tuyệt đối tránh chỉ dùng $L_2$ (MSE) đơn lẻ vì sẽ làm ảnh bị mờ bệt chi tiết khi có nhiễu cảm biến cao.
   - Bộ hàm mất mát tối ưu: **$\mathcal{L}_{total} = \mathcal{L}_1 + \lambda_{ssim}\mathcal{L}_{SSIM} + \lambda_{edge}\mathcal{L}_{edge}$** (với tùy chọn bổ sung Perceptual VGG Loss nếu cần tăng độ trung thực thị giác).
3. **Tiềm năng tích hợp tiền xử lý HVI (từ CIDNet):**
   - Mã hóa module chuyển đổi sRGB $\leftrightarrow$ HVI vào `src/utils/color_space.py` như một tùy chọn plug-and-play. Điều này cho phép người dùng kích hoạt để tăng tức thì từ 1 - 3 dB PSNR mà không làm nặng mô hình CNN.
4. **Quy trình huấn luyện tăng dần kích thước Patch (từ NTIRE Rank 1):**
   - Huấn luyện khởi động với patch nhỏ $128 \times 128$ hoặc $256 \times 256$, sau đó tăng lên $384 \times 384$ hoặc $512 \times 512$ ở các epoch cuối cùng để mô hình vừa hội tụ nhanh, vừa thích ứng tốt với cấu trúc chiếu sáng rộng.
5. **Định hướng tối ưu hóa triển khai thiết bị biên (từ Edge-LLIE):**
   - Tránh các toán tử tùy biến khó xuất sang ONNX/TensorRT. Giữ nguyên kiến trúc chuẩn với `Conv2d`, `BatchNorm2d`/`LayerNorm`, `PReLU`/`LeakyReLU`, `ConvTranspose2d` để hỗ trợ xuất INT8 sau này.
