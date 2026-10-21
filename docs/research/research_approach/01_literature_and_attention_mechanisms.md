# 01. Tổng Quan Văn Liệu, Phân Tích DarkIR (CVPR 2025) & Cơ Chế Attention

> **Môn học:** Xử Lý Ảnh và Thị Giác Máy Tính (CS2203) - Master of Science in Computer Science, UIT - VNUHCM  
> **Đề tài:** Tăng cường chất lượng ảnh trong điều kiện thiếu sáng bằng mạng nơ-ron tích chập (Low-Light Image Enhancement Using a Lightweight CNN)  
> **Tài liệu tham khảo chính:** Daniel Feijoo et al., *"DarkIR: Robust Low-Light Image Restoration"*, IEEE/CVF CVPR 2025.

---

## 1. Bối Cảnh Bài Toán & Mô Hình Suy Thoái Vật Lý

Trong điều kiện ban đêm hoặc môi trường ánh sáng yếu, ảnh chụp từ camera kỹ thuật số và thiết bị di động chịu nhiều dạng suy thoái vật lý kết hợp (Coupled Degradations):
1. **Thiếu sáng & sụp đổ dải động (Dynamic range collapse):** Lượng photon thu nhận quá thấp khiến biểu đồ độ chói (luminance histogram) co cụm hoàn toàn về phía bóng tối (shadows).
2. **Nhiễu cảm biến nặng hạt (High-ISO Sensor Noise):** Bộ khuếch đại tín hiệu analog đẩy ISO lên cao làm bộc lộ cả nhiễu lượng tử Poisson (shot noise) và nhiễu đọc Gaussian (read noise).
3. **Mờ nhòe do phơi sáng dài (Motion/Defocus Blur):** Thuật toán camera tự động kéo dài tốc độ màn trập (slow shutter speed) để gom ánh sáng, dẫn đến rung tay hoặc nhòe chuyển động.

Phương trình hình thành ảnh suy thoái tổng quát (chuẩn hóa theo DarkIR CVPR 2025):
$$\mathbf{y} = \gamma\left(\mathbf{x} \otimes \mathbf{k}\right) + \mathbf{n}$$
Trong đó:
- $\mathbf{y}$: Ảnh quan sát đầu vào (tối, mờ, nhiễu).
- $\mathbf{x}$: Cảnh sắc nét, đủ sáng lý tưởng.
- $\mathbf{k}$: Hạt nhân mờ (Point-Spread Function - PSF) do rung lắc.
- $\mathbf{n}$: Nhiễu cảm biến cộng tính.
- $\gamma$: Hàm phi tuyến điều khiển dải động và bão hòa điểm ảnh.

---

## 2. Phân Tích Chuyên Sâu Bài Báo Tham Khảo Chính: DarkIR (CVPR 2025)

### 2.1. Động lực nghiên cứu & Triết lý thiết kế của DarkIR
Trước DarkIR, cộng đồng nghiên cứu chạy theo các xu hướng rất nặng nề:
- **Transformer-based models (như Restormer, RetinexFormer):** Chi phí tính toán cực lớn (>26M tham số, hàng trăm GFLOPs), cơ chế tự chú ý toàn cục ($O(N^2)$) không thể chạy thời gian thực trên thiết bị hạn chế tài nguyên.
- **Diffusion models:** Quá trình khử nhiễu lặp lại (iterative sampling) tốn hàng chục bước tính toán, độ trễ cao.

Tác giả **DarkIR (CVPR 2025)** kiên quyết **chọn kiến trúc CNN thay vì Transformer**, chứng minh một luận điểm then chốt:  
> **Một mạng tích chập hiệu quả (Efficient CNN) với các cơ chế Attention tinh gọn hoàn toàn có thể đạt và vượt SOTA, đồng thời giảm 55% tham số so với LEDNet và 88% tham số so với Restormer.**

### 2.2. Các khối kỹ thuật cốt lõi trong DarkIR
1. **Khối MetaFormer phi hàm kích hoạt (Activation-free MetaFormer):** Kế thừa từ NAFNet, DarkIR loại bỏ các hàm kích hoạt phi tuyến đắt đỏ (GELU/SiLU) và thay bằng:
   - **Simple Gate (SG):** Chia đôi số kênh $C \to [C/2, C/2]$ và nhân từng phần tử: $\text{SG}(X) = X_1 \odot X_2$.
   - **Simplified Channel Attention (SCA):** Thu gọn không gian bằng Global Average Pooling (GAP) rồi chiếu qua tích chập điểm $1 \times 1$:
     $$\text{SCA}(X) = X \odot \sigma\left(\text{Conv}_{1\times1}(\text{GAP}(X))\right)$$
     Khối này tiêu tốn $< 0.1\%$ tính toán nhưng tự động cân chỉnh trọng số giữa kênh độ chói và kênh màu.
2. **Dilated-Spatial Attention Module (Di-SpAM):** Kết hợp các nhánh tích chập sâu (Depthwise Conv) với các hệ số giãn nở (dilation rates = 1, 4, 9) để mở rộng trường tiếp nhận (receptive field) bao trùm vệt mờ mà không làm tăng số lượng trọng số.
3. **Phân tách tần số Fourier (Fre-MLP):** Tận dụng tính chất: *Độ sáng toàn cục của bức ảnh chủ yếu nằm ở Biên độ (Amplitude) trong miền Fourier, trong khi Pha (Phase) lưu giữ cấu trúc hình học*. DarkIR dùng FFT trong Encoder để kéo sáng biên độ.

### 2.3. Chiến lược kế thừa & Thu gọn phù hợp với đồ án môn học UIT
- **Điểm NÊN học hỏi:**
  - Triết lý **CNN-first kết hợp Attention tinh gọn**: Rất phù hợp với đề tài môn học, chạy nhẹ, dễ giải thích.
  - Khối **Simplified Channel Attention (SCA)**: Tính toán cực nhanh, trực tiếp điều tiết cân bằng màu/độ chói.
  - Sử dụng **$L_1$ Pixel Loss**: Duy trì độ tương phản và biên cạnh trung thực hơn MSE ($L_2$).
- **Điểm NÊN lược bỏ (để tránh quá tải):**
  - **Lược bỏ nhánh Fourier FFT (Fre-MLP):** Việc tính toán số phức trong PyTorch khiến code cồng kềnh, dễ lỗi môi trường và rất khó giải thích bản chất toán học khi thuyết trình môn học.
  - **Lược bỏ đa tác vụ khử mờ (Deblurring):** Đề tài môn học là Low-Light Enhancement, không nên ôm đồm sang deblurring làm phân tán trọng tâm.
  - **Lược bỏ Perceptual Loss VGG19:** Tải mạng VGG19 làm tăng thời gian huấn luyện và phụ thuộc vào pre-trained model ngoài lề.

---

## 3. Các Bài Báo Tham Khảo Vệ Tinh Quan Trọng (CVF 2024–2025)

### 3.1. HVI Color Space (CVPR 2025 - Qingsen Yan et al.)
- **Phát hiện:** Đa số mô hình tăng sáng trực tiếp trong không gian **sRGB**. Nhưng sRGB có độ ghép kênh cao (R, G, B vừa mang màu vừa mang sáng), khi kéo sáng rất dễ bị **lệch màu (Color Cast/Shift)** hoặc bão hòa trắng lóa. Không gian HSV tách sáng tốt nhưng gây đứt gãy màu đỏ ($h=0$ và $h=6$) và nhiễu mặt phẳng đen trong bóng tối.
- **Ý nghĩa với đồ án:** Sinh viên có thể dùng lý thuyết của HVI để giải thích tại sao một CNN phẳng đơn giản dễ làm ảnh bị ám màu, và tại sao việc bổ sung hàm mất mát cấu trúc SSIM cùng khối Channel Attention lại giúp kìm hãm sự phân kỳ màu sắc giữa các kênh.

### 3.2. Edge-LLIE (CVPRW 2024 - Muhammad Hanan Sharif et al.)
- **Phát hiện:** Mạng CNN siêu nhẹ tối ưu cho thiết bị biên (>199 FPS). Nghiên cứu chỉ ra rằng: Khi dùng U-Net cho ảnh thiếu sáng, các đường **Skip Connections thô từ tầng nông Encoder sẽ vô tình bơm trực tiếp nhiễu hạt mè cảm biến sang Decoder**, làm ảnh đầu ra bị lốm đốm hạt nhiễu.
- **Giải pháp:** Sử dụng lớp lọc cổng (Gated Skip Connection qua Conv $1\times1$) để chặn nhiễu trước khi ghép kênh.

### 3.3. NTIRE 2024 Challenge on LLIE (CVPRW 2024)
- Thiết lập chuẩn mực thực nghiệm công bằng: Bộ độ đo chuẩn **PSNR, SSIM, LPIPS** và chiến lược cắt mảng ngẫu nhiên (**Random Crop $256 \times 256$**) giúp sinh ra hàng vạn mảng ảnh đa dạng từ một tập ảnh gốc nhỏ.

---

## 4. Khảo Sát & So Sánh Các Cơ Chế Attention Cho U-Net

Trong cấu trúc U-Net, cơ chế Attention giúp mạng nhận biết đặc trưng nào quan trọng và lọc bỏ nhiễu hạt ở vùng tối:

| Cơ chế Attention | Nguyên lý toán học | Ưu điểm trong LLIE | Nhược điểm | Đánh giá cho đồ án CS2203 |
| :--- | :--- | :--- | :--- | :--- |
| **SE-Net** (Hu et al.) | GAP + 2 tầng FC (giảm chiều $r=16$) + Sigmoid | Tái hiệu chuẩn trọng số kênh màu | Giảm chiều làm mất tương quan trực tiếp giữa các kênh RGB | Kinh điển, dễ giải thích |
| **CBAM** (Woo et al.) | Channel Attention (GAP+GMP) $\to$ Spatial Attention (Conv $7\times7$) | Vừa lọc màu vừa định vị cấu trúc biên cạnh | Tích chập $7\times7$ làm tăng chi phí tính toán cục bộ | Khá nặng cho U-Net nhỏ |
| **Coordinate Attention** (Hou et al.) | GAP độc lập theo 2 trục ngang (X) và dọc (Y) | Giữ tọa độ biên cạnh và nét mảnh trong bóng tối | Công thức phức tạp hơn khi thuyết trình | Trung bình |
| **Attention Gates** (Oktay et al.) | Dùng tín hiệu tầng sâu Decoder làm cổng lọc Skip | Chặn nhiễu hạt cực tốt ở đường Skip Connection | Cần thêm tích chập chiếu ở cả 2 nhánh | Rất tốt cho U-Net |
| **Simplified Channel Attention (SCA - DarkIR)** | **GAP + Conv $1\times1$ điểm + Sigmoid** | **Chỉ thêm $C^2$ tham số (<0.1% FLOPs), bảo toàn tương quan màu, hội tụ nhanh** | Không định vị không gian cục bộ | **LỰA CHỌN TỐI ƯU NHẤT** |

---

## 5. Bảng Đối Chiếu: DarkIR Gốc vs. Mô Hình Đề Xuất Của Đồ Án

| Thành phần | DarkIR (CVPR 2025 Gốc) | Lightweight Attention U-Net (Đề xuất Đồ án) | Lý do điều chỉnh cho đồ án môn học |
| :--- | :--- | :--- | :--- |
| **Mục tiêu chính** | Khử mờ + Bù sáng đa tác vụ | Bù sáng + Phục hồi tương phản & màu sắc | Tập trung đúng yêu cầu cốt lõi đề tài |
| **Kiến trúc nền** | Encoder-Decoder bất đối xứng | Compact U-Net 4 tầng đối xứng ($16 \to 32 \to 64 \to 128$) | Dễ giải thích trực quan cấu trúc kim tự tháp đa độ phân giải |
| **Cơ chế Attention** | Di-SpAM (dilation 1,4,9) + Fre-MLP (FFT) | **Simplified Channel Attention (SCA) + Gated Residual Block** | SCA tính toán cực nhanh, trực tiếp học trọng số cân bằng màu/độ chói |
| **Skip Connections** | Nối trực tiếp đặc trưng | **Gated Skip Connection** (qua Conv $1\times1$) | Ngăn rò rỉ nhiễu thô từ tầng nông sang tầng sâu (theo Edge-LLIE CVPRW'24) |
| **Miền tính toán** | Không gian + Miền tần số Fourier (FFT/IFFT) | Miền không gian sRGB thuần túy | Tránh phụ thuộc hàm phức FFT, code tinh gọn, chạy ổn định |
| **Hàm mất mát** | $L_1$ + LPIPS (VGG19) + Edge Loss + $L_{lol}$ | **Composite Loss: $L_1 + 0.2 \times (1 - \text{SSIM})$** | Không phụ thuộc backbone VGG ngoài lề; $L_1$ giữ màu, SSIM giữ cấu trúc |
| **Số tham số** | ~1.6M - 3.8M params | **~905k params (base_ch=16)** | Siêu nhẹ (<1M), huấn luyện cực nhanh trên Colab (T4), suy luận thời gian thực |
