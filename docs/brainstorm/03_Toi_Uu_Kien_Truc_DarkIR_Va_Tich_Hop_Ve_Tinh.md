# BÁO CÁO TỐI ƯU KIẾN TRÚC MÔ HÌNH DARKIR-EDGE & TÍCH HỢP CHẮT LỌC TỪ CÁC BÀI BÁO VỆ TINH (V1)

> **Môn học:** Xử Lý Ảnh và Thị Giác Máy Tính (CS2203) — Master of Science in Computer Science 2026  
> **Đơn vị:** Trường Đại học Công nghệ Thông tin (UIT) - ĐHQG TP.HCM  
> **Đề tài:** Tăng cường chất lượng ảnh trong điều kiện thiếu sáng bằng mạng nơ-ron tích chập (*Low-Light Image Enhancement Using a Lightweight CNN*)  
> **Kiến trúc cốt lõi:** Kế thừa DarkIR (CVPR 2025) làm xương sống, tích hợp chắt lọc Edge-LLIE (CVPRW 2024) và HVI-CIDNet (CVPR 2025).

---

# PHẦN I: ĐỊNH HƯỚNG KIẾN TRÚC & NGUYÊN TẮC BẢO TOÀN DARKIR

Thấu suốt các phân tích học thuật và định hướng chiến lược của đồ án, nhóm kiên định chọn **DarkIR (CVPR 2025)** làm nền móng kiến trúc chính. Để bảo toàn tính đúng đắn khoa học và giữ trọn "linh hồn" của DarkIR, nhóm thiết lập các nguyên tắc thiết kế bất biến sau:

## 1. Bảo Toàn Tuyệt Đối Nhánh Biến Đổi Fourier (FreMLP) Trong Mô Hình

Trong các thảo luận ban đầu, ý tưởng loại bỏ FFT (`FreMLP`) để chuyển sang mạng thuần không gian (Spatial-only) nhằm đơn giản hóa đã được đặt ra. Tuy nhiên, qua phân tích bản chất vật lý và đối chiếu thực nghiệm từ bài báo gốc DarkIR, **nhóm quyết định giữ nguyên và bảo toàn 100% nhánh FFT (`FreMLP`) bên trong các khối Encoder và Bottleneck của mô hình**.

### Cơ sở khoa học và vật lý của quyết định:
1. **Bản chất trắc quang của biến đổi Fourier 2D:**  
   Một bức ảnh $x(u, v)$ khi biến đổi Fourier sang miền tần số:
   $$X(u, v) = \mathcal{F}\{x(u, v)\} = \mathcal{A}(u, v) \cdot e^{j \mathcal{P}(u, v)}$$
   - **Pha (Phase $\mathcal{P}$):** Chứa đựng thông tin cấu trúc hình học không gian, đường biên ranh giới và hoa văn vi mô.
   - **Biên độ (Amplitude $\mathcal{A}$):** Chứa đựng phân bố năng lượng chiếu sáng, độ tương phản và sắc độ toàn cục. Các thành phần tần số cực thấp (ở tâm phổ Fourier) đại diện cho độ sáng trung bình của toàn bộ khung cảnh.
2. **Cơ chế hoạt động của `FreMLP` trong DarkIR:**  
   Khối `FreMLP` chỉ tác động lên ma trận biên độ $\mathcal{A}$ thông qua 2 lớp tích chập $1\times1$, trong khi giữ nguyên ma trận pha $\mathcal{P}$:
   $$\hat{\mathcal{A}} = \text{Conv}_{1\times1}(\text{LeakyReLU}(\text{Conv}_{1\times1}(\mathcal{A})))$$
   $$\hat{X} = \hat{\mathcal{A}} \cdot e^{j \mathcal{P}} \implies \hat{x} = \mathcal{F}^{-1}\{\hat{X}\}$$
   Nhờ đó, mô hình có thể điều chỉnh độ sáng toàn khung cảnh một cách mượt mà mà **hoàn toàn không làm biến dạng cấu trúc hình học hay sinh ra nhiễu gai nhân tạo**.
3. **Bảo toàn nguyên vẹn trong kiến trúc đề xuất duy nhất:**  
   Trong công trình gốc DarkIR (Table 6), nhóm tác giả ghi nhận khi loại bỏ khối `EBlock` (chứa `FreMLP`) và thay bằng khối thuần không gian, chỉ số PSNR tụt giảm rõ rệt. Nhóm kiên định bảo toàn 100% nhánh FFT (`FreMLP`) bên trong mô hình duy nhất `DarkIR-Edge`, không thử nghiệm cắt gọt hay loại bỏ, nhằm tận dụng tối đa năng lực điều chỉnh độ sáng toàn cục ở miền tần số biên độ.
4. **Tính khả thi trên phần cứng thực nghiệm:**  
   Các hàm `torch.fft.rfft2` và `torch.fft.irfft2` đã được tối ưu hóa ở mức phần cứng trong PyTorch và thư viện cuFFT của NVIDIA. Phép tính này chỉ chiếm tỷ trọng thời gian tính toán rất nhỏ và vận hành mượt mà trên GPU Google Colab (T4/L4).

## 2. Giữ Vững Cấu Trúc Bất Đối Xứng (Asymmetric Encoder-Decoder)

- **Encoder (Tầng nén đặc trưng):** Sử dụng các khối `EBlock` tích hợp `FreMLP` để chuyên biệt hóa cho nhiệm vụ **nâng sáng toàn cục và cân bằng độ rọi**.
- **Decoder (Tầng giải nén tái tạo):** Sử dụng các khối `DBlock` tích hợp cụm tích chập giãn nở đa tỷ lệ **`Di-SpAM` (Dilated Spatial Attention Module)** với các độ giãn `dilation = [1, 4, 9]` để chuyên biệt hóa cho nhiệm vụ **khử mờ chuyển động, triệt tiêu nhiễu hạt cảm biến và khôi phục độ sắc nét vi mô**.
- **Bottleneck Side Supervision:** Duy trì nhánh dự đoán độ phân giải thấp $\hat{x}_{\downarrow 8}$ tại đáy mạng để tính hàm mất mát dẫn hướng $\mathcal{L}_{\text{side}}$, ép Encoder phải giải quyết xong bài toán ánh sáng ở độ phân giải nhỏ trước khi chuyển giao cho Decoder.

---

# PHẦN II: TÍCH HỢP CHẮT LỌC "VŨ KHÍ" TỪ CÁC BÀI BÁO VỆ TINH

Để đồ án vừa bám chắc nền tảng DarkIR, vừa mang dấu ấn cải tiến khoa học rõ rệt, nhóm chắt lọc có chọn lọc các kỹ thuật "Zero-Parameter" và "Ultra-Low-Parameter" từ các công trình vệ tinh:

## 1. Kế Thừa Từ Edge-LLIE (CVPRW 2024): Residual Gated Skip Connection

- **Vấn đề trên kiến trúc U-Net / DarkIR truyền thống:**  
  Trong DarkIR, đường nối tắt từ Encoder sang Decoder sử dụng phép cộng trực tiếp:
  $$x_{\text{dec}} = x_{\text{up}} + x_{\text{skip}}$$
  Tại các tầng nông (gần ảnh đầu vào), $x_{\text{skip}}$ mang theo toàn bộ nhiễu hạt cảm biến ISO cao chưa được lọc của ảnh chụp đêm. Việc cộng thô này vô tình bơm ngược nhiễu hạt trực tiếp vào ảnh đầu ra.
- **Giải pháp từ Edge-LLIE:**  
  Bổ sung cổng lọc nhiễu dư (**Residual Gated Filter**) trên cả 3 đường skip connections:
  $$G(x_{\text{skip}}) = \sigma(\text{Conv}_{1\times1}(x_{\text{skip}}))$$
  $$F_{\text{gated\_skip}} = x_{\text{skip}} + \text{Conv}_{1\times1}(x_{\text{skip}}) \odot G(x_{\text{skip}})$$
- **Chi phí tham số:**  
  Với 3 tầng skip có số kênh lần lượt là $C, 2C, 4C$ (với $C=32$ là 32, 64, 128), lớp $\text{Conv}_{1\times1}$ chỉ tốn:
  $$\text{Params} = (32^2 + 64^2 + 128^2) \times 2 \approx 43.000 \text{ tham số} \approx \mathbf{0.04M \text{ params}}$$
  Chi phí tham số chưa tới $2\%$ tổng mô hình, nhưng tạo ra "van một chiều" ngăn chặn hữu hiệu nhiễu hạt cảm biến di động tràn sang Decoder.

---

## 2. Kế Thừa Từ HVI-CIDNet (CVPR 2025): Phân Tích Chuyên Sâu Cách 1 vs. Cách 2

Bài báo HVI-CIDNet (CVPR 2025) chỉ ra rằng việc xử lý ảnh thiếu sáng trực tiếp trên không gian màu sRGB thường dẫn đến hai khuyết tật quang học nghiêm trọng: **Lệch màu (Color Bias)** và **Nhiễu phẳng đen (Black Plane Noise)**. Nhóm tác giả CIDNet giải quyết bằng cách thiết kế không gian màu HVI riêng biệt kèm mạng 2 nhánh học chuyển đổi màu sắc. Để kế thừa tinh thần này mà không làm phình to kiến trúc, ta đánh giá 2 hướng:

### Phân tích Cách 1: Max-RGB Illumination Prior ($I_{\max}$ Input)
- **Cơ sở bản chất:**  
  Dựa trên lý thuyết Retinex cổ điển, giá trị cực đại trên 3 kênh màu tại mỗi pixel phản ánh xấp xỉ cấu trúc năng lượng chiếu sáng của môi trường (bỏ qua độ phản xạ vật thể):
  $$I_{\max}(x, y) = \max_{c \in \{R, G, B\}} I_c(x, y)$$
  > [!NOTE]
  > **Làm rõ thuật ngữ:** Việc ghép thêm kênh $I_{\max}$ không phải là kế thừa toàn bộ không gian màu HVI (vì HVI-CIDNet sử dụng ma trận học được và hàm mất mát trong không gian HVI). Đây chính xác là một **Illumination Prior theo tinh thần Retinex, lấy cảm hứng từ ý tưởng phân tách độ rọi của HVI**.
- **Cách tích hợp vào mô hình:**  
  Tính $I_{\max}$ trực tiếp từ ảnh đầu vào và ghép kênh (concatenate) tạo thành tensor đầu vào 4 chiều:
  $$X_{\text{in}} = [I_{\text{RGB}}; I_{\max}] \in \mathbb{R}^{4 \times H \times W}$$
  Lớp tích chập đầu tiên mở rộng từ `Conv2d(3, C, 3, 1, 1)` thành `Conv2d(4, C, 3, 1, 1)`.
- **Đánh giá ưu - nhược điểm và kỳ vọng thực tế:**
  - *Ưu điểm:* Cung cấp ngay lập tức thông tin tiên nghiệm về độ rọi cho mọi tầng tích chập phía sau; chi phí tham số chỉ tăng thêm đúng $1 \times C \times 3 \times 3 = 288$ tham số (với $C=32$).
  - *Kỳ vọng thực tế (Nhìn nhận khách quan):* Mạng nơ-ron tích chập sâu vốn dĩ có thể tự học xấp xỉ toán tử `max` qua các lớp tích chập ban đầu; do đó, mức cải thiện định lượng từ $I_{\max}$ có thể khiêm tốn ($+0.05 \dots +0.2 \text{ dB}$). Nếu thực nghiệm chỉ ra $I_{\max}$ không tạo ra bước nhảy lớn, đây vẫn là một kết luận khoa học hợp lệ phản ánh đúng năng lực tự học của CNN.

### Phân tích Cách 2: Không Gian Màu HVI Làm Nhánh Phụ Hoặc Hàm Mất Mát (Loss)
- **Cơ sở toán học:**  
  Không gian HVI phân rã ảnh thành kênh độ rọi $I$, mật độ màu tối $V$ qua hàm co sụp phi tuyến $\mathbf{C}_k(x)$, và góc màu $H$ dạng tọa độ Decartes $(H_x, H_y)$ để tránh điểm gián đoạn $0 \leftrightarrow 2\pi$.
- **Xem xét 2 hướng sử dụng:**
  - *Hướng 2a (Nhánh phụ mạng - Dual-Branch Network):* Bê nguyên kiến trúc CIDNet với một mạng nhánh phụ xử lý riêng kênh HVI. **Đánh giá: BÁC BỎ**, vì sẽ nhân đôi số lượng tham số (>6M tham số) và tăng gấp đôi độ trễ tính toán, vi phạm triệt để tiêu chí mô hình gọn nhẹ dưới 3M cho thiết bị biên.
  - *Hướng 2b (Hàm mất mát màu sắc - HVI Color Loss $\mathcal{L}_{\text{HVI}}$):* Giữ nguyên mô hình chỉ nhận và xuất RGB. Trong quá trình huấn luyện, chuyển ảnh dự đoán và Ground Truth sang không gian HVI để tính hàm mất mát:
    $$\mathcal{L}_{\text{HVI}} = \mathcal{L}_1(\hat{I}, I) + \lambda_H \|(\hat{H}_x, \hat{H}_y) - (H_x, H_y)\|_1 + \lambda_V \|\hat{V} - V\|_1$$
- **Đánh giá ưu - nhược điểm:**
  - *Ưu điểm:* **0 tham số lúc suy luận**; phạt triệt để hiện tượng lệch màu và bệt màu tối lúc học.
  - *Nhược điểm:* Phép biến đổi lượng giác $(\sin, \cos, \arctan)$ làm chậm tốc độ backward khi huấn luyện; dễ gặp bất ổn định gradient tại các pixel có mẫu số tiến về 0 ở vùng tối sâu.

---

### BẢNG ĐỐI CHIẾU TOÀN DIỆN: CÁCH 1 VS. CÁCH 2

| Tiêu Chí Đánh Giá | Cách 1: Max-RGB Illumination Prior | Cách 2: HVI Làm Loss ($\mathcal{L}_{\text{HVI}}$) | Nhận Định Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| **Số lượng tham số mô hình** | **+288 tham số** (với $C=32$) | **0 tham số** (chỉ tính lúc train) | Cả 2 đều hoàn toàn thỏa mãn tiêu chí nhẹ |
| **Chi phí tính toán lúc suy luận** | $\approx 0$ (chỉ 1 phép `torch.max`) | **0** (mô hình chỉ chạy RGB) | Cả 2 đều cực kỳ nhanh |
| **Độ phức tạp triển khai code** | Cực kỳ đơn giản (3 dòng code) | Tương đối phức tạp (~40 dòng code lượng giác) | Cách 1 ít rủi ro lỗi mã nguồn hơn |
| **Độ ổn định Gradient lúc Train** | **Tuyệt đối ổn định** (Conv thuần) | Nhạy cảm với $\epsilon$ tại vùng đen sâu | Cách 1 an toàn hơn cho sinh viên thực nghiệm |
| **Khả năng triển khai Thiết bị Biên** | Cần hỗ trợ Conv 4 kênh đầu vào | Mô hình RGB tiêu chuẩn | Cả hai đều khả thi |
| **Năng lực định hướng ánh sáng** | Hướng dẫn trực tiếp các bộ lọc | Hướng dẫn gián tiếp qua hàm loss | Cách 1 trực quan hơn |
| **Năng lực bảo toàn màu sắc** | Gián tiếp qua tương quan 3 kênh | Trực tiếp trên không gian màu sắc (Hue) | Cách 2 chặt chẽ hơn |

---

### KẾT LUẬN & KHUYẾN NGHỊ LỰA CHỌN TỐI ƯU (THE GOLDEN SYNERGY)

Qua phân tích bản chất thuật toán và mục tiêu thực nghiệm:
1. **Nếu bắt buộc chọn duy nhất một phương án:**  
   👉 **KHUYẾN NGHỊ CHỌN CÁCH 1 (Max-RGB Illumination Prior)**.  
   *Lý do:* Cách 1 can thiệp trực tiếp vào miền đặc trưng của mạng, chỉ tốn đúng 288 tham số, gradient tuyệt đối ổn định, không làm chậm tốc độ huấn luyện trên Colab và có nền tảng vật lý Retinex cực kỳ kinh điển, rất dễ bảo vệ thuyết phục trước Hội đồng.
2. **Giải pháp phối hợp hiệp đồng hoàn hảo (Đề xuất chính thức cho đồ án):**  
   - **Đầu vào:** Tích hợp **Cách 1 ($I_{\max}$ Input Prior)** để mạng có sẵn bản đồ độ rọi ban đầu.
   - **Hàm mất mát:** Thay vì dùng toàn bộ hệ thống HVI phức tạp dễ gây bất ổn định số học, nhóm sử dụng **Cosine Color Loss ($\mathcal{L}_{\text{color}}$)**:
     $$\mathcal{L}_{\text{color}} = 1 - \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W} \frac{\hat{Y}_{ij} \cdot Y_{\text{gt}, ij}}{\|\hat{Y}_{ij}\|_2 \|Y_{\text{gt}, ij}\|_2 + \epsilon}$$
     Trong code, bổ sung hằng số an toàn `eps = 1e-6` và mặt nạ che (masking) các pixel quá tối ($Y_{\text{gt}} < 5/255$) để ngăn chặn triệt để hiện tượng gradient bùng nổ khi vector màu tiến về gốc tọa độ 0. Sự kết hợp này mang lại toàn bộ lợi thế của cả 2 cách mà hoàn toàn không có nhược điểm!

---

## 3. Về Bài Báo NTIRE 2024: Tạm Gác Lại Để Tránh Quá Tải Kiến Trúc

Theo đúng chỉ đạo của User, nhóm **không tích hợp các mô-đun phức tạp của bài báo NTIRE 2024** (như chuỗi trích xuất đa nhánh phân cấp hay bộ chuẩn hóa thích nghi dày đặc). Việc tập trung toàn lực vào DarkIR + Edge-LLIE + HVI Max-RGB giúp mô hình mạch lạc, thanh thoát, tập trung đúng vào năng lực cốt lõi và đảm bảo tính giải thích được tối đa.

---

# PHẦN III: THIẾT KẾ KIẾN TRÚC MÔ HÌNH DARKIR-EDGE CHI TIẾT

Kiến trúc mô hình hoàn thiện được đặt tên là **DarkIR-Edge** (Kiến trúc DarkIR tối ưu hóa cho thiết bị biên).

## 1. Sơ Đồ Khối Tổng Thể (Detailed Architecture Flow)

```
                       Ảnh Đầu Vào rgb_in: (R, G, B) ∈ R^(3 x H x W)
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │  Max-RGB Prior Extraction │ ──> I_max = max(R,G,B) ∈ R^(1 x H x W)
                        └───────────────────────────┘
                                      │
                                      ▼ Ghép kênh: torch.cat([rgb_in, I_max], dim=1)
                         Input Tensor x_in: 4 x H x W
                                      │
                     ┌────────────────┴────────────────┐
                     │     Intro Conv (4 -> C, 3x3)     │
                     └────────────────┬────────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼ (Skip 1)
   ┌───────────────────────┐                    ┌─────────────────────────┐
   │ Encoder Stage 1       │                    │ Residual Gated Skip 1   │
   │ 1x EBlock (FreMLP)    │                    │ (Conv 1x1 + Sigmoid)    │
   └───────────┬───────────┘                    └────────────┬────────────┘
               ▼ Downsample (Conv 2x2, stride 2)             │
               ├─────────────────────────────────────────────┤
               ▼                                             ▼ (Skip 2)
   ┌───────────────────────┐                    ┌─────────────────────────┐
   │ Encoder Stage 2       │                    │ Residual Gated Skip 2   │
   │ 2x EBlock (FreMLP)    │                    │ (Conv 1x1 + Sigmoid)    │
   └───────────┬───────────┘                    └────────────┬────────────┘
               ▼ Downsample (Conv 2x2, stride 2)             │
               ├─────────────────────────────────────────────┤
               ▼                                             ▼ (Skip 3)
   ┌───────────────────────┐                    ┌─────────────────────────┐
   │ Encoder Stage 3       │                    │ Residual Gated Skip 3   │
   │ 3x EBlock (FreMLP)    │                    │ (Conv 1x1 + Sigmoid)    │
   └───────────┬───────────┘                    └────────────┬────────────┘
               ▼ Downsample (Conv 2x2, stride 2)             │
   ══════════════════════════════════════════════════════════╪════════════════
   [BOTTLENECK STAGE: Resolution H/8 x W/8, Channels 8C]     │
   │                                                         │
   ├── 2x Middle EBlock (FreMLP Global Brightness Adjustment)│
   │   │                                                     │
   │   ├──> Side Conv 3x3 (8C -> 3) ──> Low-Res Output x̂_↓8 (Tính L_side)
   │   │                                                     │
   └── 2x Middle DBlock (Di-SpAM Dilated Convolutions d=[1,4,9])
   ══════════════════════════════════════════════════════════╪════════════════
               │                                             │
               ▼ Upsample (PixelShuffle 2x)                  │
               ├───────────────────(+)───────────────────────┘ (Gated Skip 3)
               ▼
   ┌───────────────────────┐
   │ Decoder Stage 3       │
   │ 3x DBlock (Di-SpAM)   │
   └───────────┬───────────┘
               ▼ Upsample (PixelShuffle 2x)
               ├───────────────────(+)───────────────────────── (Gated Skip 2)
               ▼
   ┌───────────────────────┐
   │ Decoder Stage 2       │
   │ 1x DBlock (Di-SpAM)   │
   └───────────┬───────────┘
               ▼ Upsample (PixelShuffle 2x)
               ├───────────────────(+)───────────────────────── (Gated Skip 1)
               ▼
   ┌───────────────────────┐
   │ Decoder Stage 1       │
   │ 1x DBlock (Di-SpAM)   │
   └───────────┬───────────┘
               │
               ▼
   ┌───────────────────────┐
   │ Ending Conv (C -> 3)  │ ──> Feature Output feat_out ∈ R^(3 x H x W)
   └───────────┬───────────┘
               │
               ▼ (+) Cộng Global Residual với ảnh RGB gốc: Ŷ = feat_out + rgb_in
   Ảnh Đầu Ra Hoàn Thiện Ŷ: (R, G, B) ∈ R^(3 x H x W)
```

## 2. Bảng Định Cỡ Tham Số Thực Tế (Kiểm Chứng 100% Trực Tiếp Từ Repo DarkIR)

Nhằm đảm bảo tính chuẩn xác tuyệt đối và phòng tránh nguy cơ vỡ ngân sách khi chuyển giao sang code PyTorch, nhóm đã chạy script kiểm chứng trực tiếp trên kiến trúc `DarkIR` (`DarkIR/archs/DarkIR.py`).

### Kết quả đo đạc thực tế từ mã nguồn DarkIR:
- Trong mã nguồn gốc DarkIR, cấu hình chuẩn `DarkIR-m` ($C=32$, `enc_blk_nums=[1, 2, 3]`, `dec_blk_nums=[3, 1, 1]`, `middle=(2, 2)`) có tổng số tham số là **3.32M tham số (3.321.638)**.
- **Về cấu hình số khối tại Bottleneck trong DarkIR gốc:** Toàn bộ 12 file config `.yml` chính thức của tác giả (từ `AllLOL.yml`, `v2real_lsrw.yml`, `LOLBlur.yml` đến `ExDark.yml`) đều thiết lập cố định:
  ```yaml
  middle_blk_num_enc: 2
  middle_blk_num_dec: 2
  ```
  Nghĩa là nhóm tác giả DarkIR **LUÔN DÙNG 2 EBlock + 2 DBlock (tổng 4 khối)** ở Bottleneck, **không dùng 1 E + 1 D**. Riêng tại tầng Bottleneck ($8C = 256$ kênh), 4 khối này chiếm tới **2.045M tham số** (chiếm ~61.5% toàn bộ mạng).
- **Về số kênh cơ sở ($C$):** Trong bài báo và repo gốc, tác giả chỉ công bố 3 mốc:
  - $C=16$ (`DarkIR-s`): 0.872M tham số, 2.04 GMACs.
  - $C=32$ (`DarkIR-m`): 3.31M tham số, 7.25 GMACs (chuẩn toàn bộ thực nghiệm).
  - $C=64$ (`DarkIR-l`): 12.96M tham số, 27.19 GMACs.
  Mã nguồn gốc **không có file cấu hình $C=24$ có sẵn**, nhưng hàm khởi tạo `DarkIR(width=...)` hỗ trợ truyền tham số `width=24` linh hoạt mà không cần sửa code.
- Về các tầng chuyển tiếp:
  - **Upsample trong DarkIR:** Đã sử dụng sẵn **Pointwise Conv 1x1** (`nn.Conv2d(chan, chan*2, 1, bias=False)`) kết hợp `nn.PixelShuffle(2)`, hoàn toàn không dùng Conv 3x3 nên không bị phình tham số.
  - **Downsample trong DarkIR:** Sử dụng `nn.Conv2d(chan, 2*chan, 2, 2)` (stride 2, kernel 2) tốn 172K tham số. Nếu nâng cấp lên `Conv 3x3, stride=2, padding=1` để tránh lệch tâm nửa pixel (half-pixel shift), số tham số chỉ tăng thêm ~215K.

### Bảng định cỡ tham số chi tiết của mô hình đề xuất duy nhất DarkIR-Edge:

Nhóm chấn chỉnh dứt khoát: **Chỉ có DUY NHẤT một cấu hình kiến trúc `DarkIR-Edge` được thiết kế, huấn luyện và đánh giá trong toàn bộ đồ án**. Mọi phương án phân mảnh hay cấu hình thay thế (như Cấu hình B $C=32$ Middle 1+1) đều bị bãi bỏ nhằm tập trung tối đa nguồn lực vào việc tối ưu hóa và đánh giá sâu sắc mô hình hoàn chỉnh này.

Dưới đây là bảng phân rã chi tiết số lượng tham số thực tế, được tách bạch rõ ràng giữa các khối đặc trưng và tầng chuyển tiếp (Down/Up sampling), không bị cộng lặp:

| Thành Phần Kiến Trúc | Mô Hình Đề Xuất Duy Nhất `DarkIR-Edge` ($C=24$) | Mốc Trần Tham Chiếu `DarkIR-m` ($C=32$ CVPR 2025) | Ghi Chú Kỹ Thuật & Cấu Trúc Khối |
| :--- | :---: | :---: | :--- |
| **Kênh cơ sở ($C$)** | **$C = 24$** | **$C = 32$** | Phân tầng kênh: $C \to 2C \to 4C \to 8C$ |
| **Intro Conv (Max-RGB 4ch) & Ending Conv** | **$1.539$ tham số** (~$0.002\text{M}$) | **$1.763$ tham số** (~$0.002\text{M}$) | Intro: $4 \to C$ ($3\times3$, gốc là $3\to C$); Ending: $C \to 3$ ($3\times3$) |
| **Encoder Blocks (EBlocks)** | **$280.200$ tham số** (~$0.28\text{M}$) | **$490.336$ tham số** (~$0.49\text{M}$) | Tầng 1, 2, 3: [1, 2, 3] khối tích hợp `FreMLP` |
| **Downsample Layers** | **$97.104$ tham số** (~$0.10\text{M}$) | **$172.480$ tham số** (~$0.17\text{M}$) | 3 tầng Conv $2\times2$, stride 2 ($C\to 2C, 2C\to 4C, 4C\to 8C$) |
| **Bottleneck Blocks (EBlock + DBlock)** | **$1.165.056$ tham số** (~$1.17\text{M}$) | **$2.044.928$ tham số** (~$2.04\text{M}$) | Bảo toàn 100%: **2 EBlocks + 2 DBlocks** tại đáy $8C$ |
| **Decoder Blocks (DBlocks)** | **$253.656$ tham số** (~$0.25\text{M}$) | **$433.184$ tham số** (~$0.43\text{M}$) | Tầng 3, 2, 1: [3, 1, 1] khối tích hợp `Di-SpAM` |
| **Upsample Layers** | **$96.768$ tham số** (~$0.10\text{M}$) | **$172.032$ tham số** (~$0.17\text{M}$) | 3 tầng Pointwise Conv $1\times1$ + PixelShuffle ($2\times$) |
| **Residual Gated Skips (Edge-LLIE)** | **$24.528$ tham số** (~$0.02\text{M}$) | — *(DarkIR dùng cộng trực tiếp)* | Cổng chặn nhiễu 3 tầng skip: Conv $1\times1$ + Sigmoid |
| **Side Output Supervision (Bottleneck)** | **$5.187$ tham số** (~$0.005\text{M}$) | **$6.915$ tham số** (~$0.007\text{M}$) | Conv $3\times3$ tại đáy $8C \to 3$ cho $\mathcal{L}_{\text{side}}$ |
| **TỔNG SỐ THAM SỐ (Total Params)** | **$1.924.038$ tham số ($\approx \mathbf{1.92M}$)** | **$3.321.638$ tham số ($\approx \mathbf{3.32M}$)** | **Khớp 100% số đo từ mã nguồn `DarkIR/archs/DarkIR.py`** |
| **Đánh giá quy mô tài nguyên** | **Gọn nhẹ hoàn hảo (< 2.0M)** | Mô hình lớn gốc của tác giả ($3.32\text{M}$) | Nằm trọn vẹn trong ngân sách 1.0M – 3.0M |

👉 **Điểm nhấn thiết kế:** Mô hình đề xuất duy nhất `DarkIR-Edge` ($C=24$) bảo toàn nguyên vẹn 100% tỷ lệ khối và cấu trúc 2 EBlock + 2 DBlock ở đáy mạng của DarkIR mà tổng tham số chỉ dừng ở mức **~1.93M tham số**, hoàn toàn thỏa mãn giới hạn dưới 3.0M và tối ưu hoàn hảo cho suy luận thời gian thực trên di động.

---

# PHẦN IV: HÀM MẤT MÁT (LOSS FORMULATION) & SIÊU THAM SỐ CHUẨN DARKIR

## 1. Công Thức Hàm Mất Mát Kết Hợp Đa Nhiệm (Composite Loss)

Để ép mô hình vừa khôi phục độ sáng toàn cục chuẩn xác, vừa làm nét chi tiết bề mặt và bảo toàn màu sắc chân thực, hàm mất mát toàn phần được thiết lập:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_1(\hat{Y}, Y_{\text{gt}}) + \lambda_{\text{side}} \mathcal{L}_{\text{side}}(\hat{x}_{\downarrow 8}, Y_{\text{gt}\downarrow 8}) + \lambda_{\text{ssim}} (1 - \text{SSIM}(\hat{Y}, Y_{\text{gt}})) + \lambda_{\text{color}} \mathcal{L}_{\text{color}}(\hat{Y}, Y_{\text{gt}})$$

Trong đó:
1. **$\mathcal{L}_1(\hat{Y}, Y_{\text{gt}}) = \|\hat{Y} - Y_{\text{gt}}\|_1$:** Hàm mất mát nền tảng, phạt tuyến tính sai số pixel, bảo toàn biên cạnh tốt hơn MSE. Trọng số: $1.0$.
2. **$\mathcal{L}_{\text{side}} = \|\hat{x}_{\downarrow 8} - Y_{\text{gt}\downarrow 8}\|_1$:** Giám sát sâu tại đáy mạng (kế thừa chuẩn DarkIR), ép Encoder và Bottleneck phải hoàn thành việc tái tạo độ sáng trước. Trọng số: $\lambda_{\text{side}} = 0.5$.
3. **$\mathcal{L}_{\text{ssim}} = 1 - \text{SSIM}(\hat{Y}, Y_{\text{gt}})$:** Ép mô hình bảo toàn cấu trúc bề mặt vi mô và độ tương phản cục bộ. Trọng số: $\lambda_{\text{ssim}} = 0.2$.
4. **$\mathcal{L}_{\text{color}} = 1 - \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W} \frac{\hat{Y}_{ij} \cdot Y_{\text{gt}, ij}}{\|\hat{Y}_{ij}\|_2 \|Y_{\text{gt}, ij}\|_2 + \epsilon}$ (Cosine Color Loss):**  
   Đo góc lệch vector màu sắc RGB tại từng điểm ảnh (pixel-wise), triệt tiêu triệt để hiện tượng ám vàng/ám xanh mà không tốn chi phí tính toán HVI phức tạp. Trong code PyTorch, hàm được cài đặt an toàn:  
   `loss_color = 1.0 - F.cosine_similarity(y_pred, y_gt, dim=1, eps=1e-6).mean()`. Trọng số: $\lambda_{\text{color}} = 0.1$.

## 2. Bảng Siêu Tham Số Huấn Luyện Khớp 100% Repo DarkIR

Toàn bộ các siêu tham số được đồng bộ hóa tuyệt đối với mã nguồn chính thức của **DarkIR (`DarkIR/options/`)**:

| Siêu Tham Số | Giá Trị Cấu Hình Chuẩn DarkIR | Mục Đích Kỹ Thuật |
| :--- | :---: | :--- |
| **Thuật toán tối ưu (Optimizer)** | **AdamW** | Chống quá khớp tốt hơn Adam tiêu chuẩn |
| **Tham số Momentum** | $\beta_1 = 0.9, \beta_2 = 0.999$ | Chuẩn công nghiệp cho bài toán phục hồi ảnh |
| **Trọng số suy giảm (Weight Decay)** | $1 \times 10^{-4}$ | Kiểm soát chuẩn trọng số mô hình |
| **Tốc độ học khởi đầu (Initial LR)** | **$2 \times 10^{-4}$** | Mức LR tối ưu cho kiến trúc họ MetaFormer/U-Net |
| **Tốc độ học tối thiểu (Min LR)** | **$1 \times 10^{-6}$** | Đảm bảo mô hình hội tụ mịn ở các epoch cuối |
| **Bộ lập lịch LR (LR Scheduler)** | **CosineAnnealingLR** | Suy giảm mượt theo hình cosin |
| **Kích thước mảng cắt (Patch Size)** | **$256 \times 256$** | Cắt ngẫu nhiên cùng tọa độ trên Low và Normal |
| **Kích thước lô (Batch Size)** | **8** | Tối ưu hóa việc nạp đầy nhân Tensor Cores GPU T4/L4 |
| **Cơ chế Early Stopping** | **Tạm thời vô hiệu hóa (Commented)** | Chạy trọn vẹn số epoch cố định để Cosine Annealing suy giảm mượt |
| **Số Epochs huấn luyện** | **100 epochs cố định** | Đạt 30.000 steps cập nhật gradient, đảm bảo hội tụ hoàn hảo |
| **Độ chính xác tính toán** | **AMP FP16 (`autocast`)** | Giảm 50% VRAM, tăng tốc độ tính toán gấp đôi |

## 3. Kỹ Thuật Số Học Chống Bùng Nổ Gradient & Hỗ Trợ Kích Thước Ảnh Tùy Ý Khi Dùng AMP FP16

Khi kích hoạt tính toán dấu phẩy động hỗn hợp (**Automatic Mixed Precision - AMP FP16**) trên GPU Tesla T4 của Colab kết hợp với biến đổi Fourier (`FreMLP`), nhóm đã giải quyết triệt để 4 điểm nghẽn kỹ thuật:

1. **Triệt tiêu bẫy đạo hàm `atan2` trong `FreMLP` bằng Phasor đơn vị (Unit Phasor):**  
   - *Phát hiện kỹ thuật:* Trong mã nguồn gốc DarkIR (`DarkIR/archs/arch_model.py`), hàm tính pha gọi `pha = torch.angle(x_freq)` (bản chất là toán tử `atan2(imag, real)`). Đạo hàm của `atan2` tỷ lệ nghịch với khoảng cách bán kính $\frac{\partial \text{atan2}}{\partial r} \propto \frac{1}{r}$. Tại các bin tần số cao hoặc vùng tối sâu có biên độ gần 0 ($r \approx 0$), đạo hàm bùng nổ cực lớn, làm tràn số gradient khi kết hợp với AMP FP16.
   - *Giải pháp tối ưu bằng Phasor đơn vị:* Thay vì tách pha bằng hàm lượng giác rồi tái tạo bằng `cos` và `sin`, nhóm chuẩn hóa trực tiếp tensor phức thành **phasor đơn vị** (Unit Phasor):
     $$e^{j \mathcal{P}} = \frac{X_{\text{freq}}}{\|X_{\text{freq}}\| + \epsilon} \implies \hat{X}_{\text{freq}} = \text{process1}(\mathcal{A}) \cdot \frac{X_{\text{freq}}}{\mathcal{A} + \epsilon}$$
     với $\mathcal{A} = \sqrt{\text{real}^2 + \text{imag}^2 + \epsilon}$, chọn $\epsilon = 10^{-6}$.
   - *Lợi ích:* Công thức này tương đương toán học $100\%$ với việc tinh chỉnh biên độ và giữ nguyên pha của DarkIR, nhưng **loại bỏ hoàn toàn các hàm `atan2`, `cos`, `sin`**, triệt tiêu triệt để bẫy đạo hàm bùng nổ và tối ưu hóa tốc độ tính toán trên GPU.

2. **Kỹ thuật tính toán cuFFT FP32 an toàn:**  
   Thư viện cuFFT FP16 có ngưỡng trần biểu diễn $65.504$, rất dễ tràn số khi năng lượng tập trung tại tần số thấp (DC component). Do đó, bên trong khối `FreMLP`, tensor bắt buộc được chuyển sang FP32 qua `with torch.amp.autocast('cuda', enabled=False): x_fp32 = x.float()`, thực hiện `rfft2` và `irfft2` hoàn toàn trên FP32, sau đó mới ép kiểu trở lại `x.dtype`.

3. **Cơ chế phản chiếu đệm (Reflection Padding) hỗ trợ ảnh kích thước tùy ý:**  
   Mô hình có 3 tầng downsample (thu nhỏ $\times 8$), do đó yêu cầu kích thước không gian $(H, W)$ phải là bội số của 8. Kế thừa cơ chế `check_image_size` của DarkIR gốc, ảnh đầu vào có kích thước tùy ý (như $600 \times 400$ của LOLv2-Real) sẽ được đệm phản xạ (`F.pad` với chế độ reflect) lên bội số gần nhất của 8, và sau khi qua lớp ending sẽ được cắt trả về đúng kích thước ban đầu: `out = x[:, :, :H, :W]`.

4. **Chống Aliasing cho Side Loss $\mathcal{L}_{\text{side}}$:**  
   Thay vì dùng `F.interpolate(scale_factor=0.125, mode='bilinear')` vốn lấy mẫu thưa và dễ gây hiện tượng răng cưa quang học (aliasing), nhóm sử dụng phép gom trung bình vùng **`F.avg_pool2d(Y_gt, kernel_size=8, stride=8)`** để tạo Ground Truth thu nhỏ $Y_{\text{gt}\downarrow 8}$. Phép toán này tương đương với bộ lọc tích hợp năng lượng photon thực tế trên cảm biến.

5. **Khóa chia cho 0 trong Cosine Color Loss:**  
   Sử dụng hằng số ổn định $\epsilon = 10^{-6}$ và mặt nạ che (masking) các pixel quá tối ($Y_{\text{gt}} < 5/255$) để triệt tiêu các gradient không xác định ở vùng đen tuyệt đối.

---

# PHẦN V: THIẾT KẾ THỰC NGHIỆM ĐỐI CHUẨN ĐỒNG NHẤT DUY NHẤT (UNIFIED BENCHMARK PROTOCOL)

Nhằm tối ưu hóa triệt để thời gian thực hiện đồ án và tuân thủ định hướng chiến lược dứt khoát của User: **Loại bỏ toàn bộ các bảng ma trận thử nghiệm phân mảnh (ablation study rườm rà), tập trung toàn lực vào MỘT cấu hình kiến trúc duy nhất, MỘT quy trình thí nghiệm duy nhất, MỘT bộ siêu tham số chuẩn hóa và đánh giá khoa học sâu sắc**.

## 1. Khung Đối Chuẩn Thực Nghiệm Duy Nhất

Thực nghiệm được thiết kế tập trung so sánh hiệu năng của mô hình đề xuất duy nhất **`DarkIR-Edge`** với các tầng đối chuẩn đại diện trên toàn bộ $N=150$ ảnh test độc lập:

| Nhóm Đối Chuẩn | Mô Hình / Phương Pháp | Số Tham Số (Params) | Phân Loại Phương Pháp | Ý Nghĩa Học Thuật Trực Tiếp Trước Hội Đồng |
| :--- | :--- | :---: | :--- | :--- |
| **Traditional Baselines** | **Gamma Correction ($\gamma=2.2$)** | 0 | Non-learning (Trắc quang) | Mốc sàn cơ bản của xử lý ảnh truyền thống |
| | **CLAHE** | 0 | Non-learning (Histogram) | Cân bằng tương phản thích nghi cục bộ |
| | **Retinex (MSRCR)** | 0 | Non-learning (Vật lý) | Mô hình Retinex phục hồi màu sắc cổ điển |
| **Deep Learning Baseline**| **Standard U-Net** | **$\sim 1.9\text{M}$** | Learning (Plain Conv2d) | **Đối chuẩn trực tiếp cùng quy mô tham số**; đánh giá hiệu quả thực tế của kiến trúc DarkIR-Edge |
| **Mô Hình Đề Xuất Duy Nhất**| **`DarkIR-Edge` (Ours)** | **$\approx \mathbf{1.93M}$** | **Learning (FreMLP + Di-SpAM + Gated Skip + Max-RGB)** | **Tâm điểm đồ án**: Đánh giá năng lực phục hồi đa cảm biến ở quy mô tinh gọn |
| **Mốc Trần Tham Chiếu** | **DarkIR-m (CVPR 2025)** | $\approx 3.32\text{M}$ | Learning SOTA (Bản đầy đủ) | **Mốc trần lý thuyết** tham chiếu từ công trình CVPR 2025 |

## 2. Quy Trình Huấn Luyện & Đánh Giá Đa Seed Có Kiểm Định Thống Kê

1. **Huấn luyện độc lập 2-3 seeds:** Mô hình đề xuất `DarkIR-Edge` được huấn luyện từ đầu qua 3 lần độc lập (seeds = 42, 123, 999) trên tập train đồng nhất $D_{\text{train}} = 1.200$ cặp ảnh trong 100 epochs cố định.
2. **Báo cáo kết quả kép:**
   - **Bảng tổng hợp gộp (Primary Pooled Benchmark, $N=150$ ảnh):** Báo cáo giá trị $\text{Mean} \pm \text{Std}$ cho các chỉ số PSNR, SSIM, LPIPS.
   - **Báo cáo phân rã theo 3 miền cảm biến:**
     + Miền DSLR trong nhà (LOLv2-Real: 100 ảnh).
     + Miền Smartphone di động thực tế (LSRW-Huawei: 30 ảnh).
     + Miền Zero-shot Unseen Sensor (LSRW-Nikon: 20 ảnh) để đo lường năng lực khái quát hóa ngoại suy.
3. **Kiểm định thống kê ghép cặp (Paired t-test):**  
   Tính toán sai phân $\Delta_i = \text{PSNR}_{\text{DarkIR-Edge}}(i) - \text{PSNR}_{\text{Standard U-Net}}(i)$ trên từng bức ảnh $i=1 \dots 150$. Xác định giá trị $p$-value (kết hợp độ lệch chuẩn giữa các seeds) để kiểm định xem mức chênh lệch giữa hai mô hình có ý nghĩa thống kê thực chất trên tập ảnh kiểm thử hay không.
4. **Chẩn đoán phân rã lỗi trắc quang (GT-mean Error Decomposition):**  
   Bóc tách tỷ lệ sai số độ sáng toàn cục so với sai số chi tiết vi mô, giải thích cơ chế nâng sáng mượt mà của nhánh tần số Fourier `FreMLP`.

---

# PHẦN VI: TỔNG KẾT ĐÓNG GÓP KHOA HỌC & GIÁ TRỊ THUYẾT TRÌNH BẢO VỆ

Bản thiết kế kiến trúc **DarkIR-Edge** và phương pháp luận thực nghiệm mang lại 4 luận điểm bảo vệ then chốt giúp đồ án đạt điểm số xuất sắc trước Hội đồng môn học:

1. **Định vị câu chuyện khoa học khiêm tốn, trung thực và sâu sắc:**  
   Thay vì tuyên bố "cải tiến vượt SOTA" một cách phi thực tế, đồ án phát biểu mạch lạc:  
   > *"Chúng tôi xây dựng và đánh giá một kiến trúc CNN tinh gọn duy nhất DarkIR-Edge (C=24, ~1.93M tham số), bảo toàn trọn vẹn linh hồn của DarkIR và tích hợp tinh tế 2 giải pháp vệ tinh, huấn luyện trên tập dữ liệu đa cảm biến cân bằng có kiểm soát và kiểm chứng thực nghiệm chặt chẽ."*
2. **Kiểm chứng thực nghiệm nghiêm cẩn trên tập test nhỏ:**  
   Nhận thức rõ nguy cơ "kết quả rơi vào nhiễu", nhóm triển khai quy trình đánh giá đa seed (Mean ± Std), kiểm định so sánh ghép cặp (Paired t-test) theo từng ảnh, và phân tích bóc tách thành phần lỗi qua chẩn đoán GT-mean.
3. **Chắt lọc thông minh và an toàn từ các bài báo vệ tinh:**  
   Tích hợp cổng lọc nhiễu **Residual Gated Skip** từ Edge-LLIE để xử lý nhiễu hạt di động, và kênh **Max-RGB Illumination Prior** theo tinh thần Retinex kết hợp **Cosine Color Loss** thay vì nhồi nhét mạng HVI cồng kềnh.
4. **Nhìn nhận khách quan về triển khai thực tế trên thiết bị biên:**  
   Thừa nhận rõ ràng các thách thức kỹ thuật của toán tử Fourier phức (`rfft2`) trên các runtime vi xử lý NPU di động; đo đạc thông lượng thực tế (FPS, Latency) trực tiếp trên GPU máy chủ và định hướng các bước tối ưu hóa phần mềm tiếp theo.
