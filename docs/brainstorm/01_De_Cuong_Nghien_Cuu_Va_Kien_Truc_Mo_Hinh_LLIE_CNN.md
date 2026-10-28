# Image Processing & Computer Vision (CS2203)

Master of Science in Computer Science 2026

University of Information Technology - VNUHCM

## Đồ án môn học

### 1. Tên đề tài

Tên tiếng Việt:

> **Tăng cường chất lượng ảnh trong điều kiện thiếu sáng bằng mạng nơ-ron tích chập (CNN)**

Tên tiếng Anh:

> **Low-Light Image Enhancement Using a Lightweight Convolutional Neural Network**

Tên ngắn hơn:

> **Low-Light Image Enhancement Using CNN**

---

### 2. Tóm tắt nội dung đề tài

> **Đề tài nghiên cứu bài toán tăng cường chất lượng ảnh trong điều kiện thiếu sáng (Low-Light Image Enhancement -- LLIE), trong đó ảnh đầu vào có độ sáng thấp, tương phản kém, nhiễu và mất chi tiết. Nhóm sẽ xây dựng một mô hình mạng nơ-ron tích chập (CNN) nhẹ theo hướng image-to-image learning để học ánh xạ từ ảnh thiếu sáng sang ảnh có chất lượng/độ sáng tốt hơn. Mô hình dự kiến được xây dựng trên kiến trúc CNN dạng U-Net rút gọn, kết hợp các lớp convolution, activation và skip connection nhằm bảo toàn thông tin không gian và chi tiết ảnh. Nhóm sẽ huấn luyện và đánh giá mô hình trên các bộ dữ liệu low-light có cặp ảnh tương ứng, sử dụng các độ đo như PSNR và SSIM, đồng thời so sánh với một số phương pháp xử lý ảnh truyền thống. Các phương pháp LLIE hiện đại trong các công trình CVF giai đoạn 2024--2026 sẽ được sử dụng làm cơ sở tham khảo và đối chiếu, nhưng đồ án tập trung vào việc xây dựng và phân tích một giải pháp CNN cơ bản, nhẹ và phù hợp với phạm vi môn học.**

---

### 3. Yêu cầu của giảng viên

Điều giảng viên muốn:

- có bài toán thực tế;
- có paper hiện đại;
- có dataset;
- có CNN;
- có thực nghiệm;
- **không tuyên bố reproduce SOTA**;
- không cần phát minh phương pháp mới.
- Sử dụng các phương pháp/model ở mức đọ cơ bản
- Cần hiểu rõ đến mức chi tiết bài toán.
- Phải phát biểu được bài toán, phương pháp, thực nghiệm

#### Nội dung thuyết trình/báo cáo

1. Nội báo cáo đồ án gồm:

- Báo cáo chi tiết (viết dạng paper khoa học)
- Source code
- Slide trình bày

2. Nội duy thuyết trình/báo cáo

- Thông tin các thành viên và đánh giá mức độ hoàn thành
- Những lý do hoặc tầm quan trọng của bài toán
- Phát biểu bài toán
  - Input
  - Output
  - Ví dụ minh họa
- Phương pháp giải quyết
- Kết quả thực nghiệm:
  - Dataset
  - Độ đo đánh giá
  - Môi trường thực nghiệm
  - Bảng kết quả
  - Minh họa kết quả
- Đánh giá, nhận xét những ưu điểm và hạn chế
- Tài liệu tham khảo
- Nguồn tham khảo chính `CVF` - "https://openaccess.thecvf.com/menu"

### 4. Các yêu cầu hiện tại của tôi

Hiện tại trên laptop này không có GPU nên việc thực nghiệm tôi sẽ trực tiếp chạy trên Colab hoặc phần cứng khác có GPU, trên laptop này, repo này chỉ thực hiện nghiêu cứu đến bước code, sau khi có code tôi sẽ đem train, và kiểm thử sau đó đem kết quả thực nghiệm về đây.

---

---

# PHẦN II: PHÂN TÍCH ĐỀ TÀI & ĐỊNH HƯỚNG PHẠM VI NGHIÊN CỨU

## 1. Định Vị Đề Tài & Khóa Chặt Phạm Vi

Nhằm đáp ứng trọn vẹn yêu cầu sư phạm của Giảng viên hướng dẫn môn học CS2203 (Hiểu sâu bản chất, có thực nghiệm đối chuẩn, không phát minh phương pháp quá phức tạp, không tuyên bố tái lập SOTA phi thực tế), đề tài được định vị như sau:

- **Bài toán:** Phục hồi và tăng cường chất lượng ảnh thiếu sáng (Supervised Low-Light Image Enhancement - LLIE).
- **Cách tiếp cận:** Học ánh xạ trực tiếp từ ảnh sang ảnh (Image-to-Image Learning) sử dụng mạng tích chập gọn nhẹ (Lightweight CNN).
- **Phạm vi kỹ thuật:**
  - **KHÔNG sử dụng:** Transformer khổng lồ (Restormer, RetinexFormer), Diffusion Models (hàng chục bước lấy mẫu lặp), Foundation Models, hay Transfer Learning mượn từ mạng phân loại ảnh ImageNet.
  - **TẬP TRUNG VÀO:** Mạng tích chập tự thiết kế từ đầu (Trained from scratch) dựa trên kiến trúc **U-Net rút gọn (Compact U-Net)** có skip connections, kết hợp cơ chế chú ý kênh tinh gọn (**Simplified Channel Attention - SCA**) kế thừa từ các công trình CVPR hàng đầu.

## 2. Vì Sao Chọn Cấu Trúc U-Net Thay Vì CNN Phẳng Tuần Tự?

LLIE là bài toán thị giác mức thấp (Low-Level Computer Vision). Mục tiêu không phải là phân loại ảnh (ra một nhãn vô hướng), mà là tái tạo một ma trận ảnh mới $H \times W \times 3$ với độ sáng cân bằng, giữ nguyên:

- Cạnh sắc nét và đường nét ranh giới vật thể.
- Kết cấu bề mặt (Texture) và hoa văn chi tiết.
- Tính trung thực của màu sắc, loại bỏ ám màu (Color Cast).

Một mạng CNN phẳng tuần tự (Plain CNN) không có downsampling sẽ có trường tiếp nhận (Receptive Field) rất nhỏ, không nắm bắt được bối cảnh chiếu sáng toàn cục; nếu có downsampling mà không có **Skip Connections**, thông tin không gian chi tiết ở các tầng nông sẽ bị mất vĩnh viễn ở bottleneck, khiến ảnh đầu ra bị bệt và nhòe mượt. Cấu trúc U-Net giải quyết triệt để vấn đề này nhờ cơ chế truyền trực tiếp đặc trưng không gian từ Encoder sang Decoder.

---

# PHẦN III: CƠ SỞ VẬT LÝ & TỔNG QUAN VĂN LIỆU (CVF 2024–2026)

## 1. Mô Hình Suy Thoái Quang Học Thực Tế

Trong môi trường ánh sáng yếu hoặc ban đêm, ảnh số bị suy thoái do ba yếu tố vật lý kết hợp (Coupled Degradations):

1. **Thiếu sáng & sụp đổ dải động (Luminance Drop):** Số lượng photon đến cảm biến quá ít, biểu đồ histogram tập trung dày đặc ở vùng tối $0 \dots 30$.
2. **Nhiễu cảm biến nặng hạt (High-ISO Sensor Noise):** Mạch ISP tự động khuếch đại tín hiệu analog làm nổ cả nhiễu lượng tử Poisson (shot noise) và nhiễu đọc Gaussian (read noise).
3. **Vệt mờ chuyển động (Motion/Defocus Blur):** Thuật toán máy ảnh kéo dài thời gian phơi sáng (slow shutter) dẫn đến rung tay hoặc nhòe vật thể chuyển động.

Mô hình toán học suy thoái tổng quát (chuẩn hóa theo DarkIR CVPR 2025):
$$\mathbf{y} = \gamma\left(\mathbf{x} \otimes \mathbf{k}\right) + \mathbf{n}$$
Trong đó:

- $\mathbf{y} \in \mathbb{R}^{H \times W \times 3}$: Ảnh quan sát đầu vào (tối, mờ, nhiễu).
- $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$: Cảnh sắc nét, đủ sáng lý tưởng (Ground Truth).
- $\mathbf{k}$: Hạt nhân mờ (Point-Spread Function - PSF) do chuyển động.
- $\mathbf{n}$: Nhiễu cảm biến cộng tính (Shot noise + Read noise).
- $\gamma$: Hàm phi tuyến điều khiển dải động và độ bão hòa quang học.

## 2. Giải Mã Bài Báo Tham Khảo Cốt Lõi: DarkIR (CVPR 2025)

Công trình _DarkIR: Robust Low-Light Image Restoration_ (Daniel Feijoo, Juan C. Benito, Alvaro Garcia, Marcos V. Conde - CVPR 2025) là nền tảng tham chiếu học thuật trung tâm của đề tài.

### 2.1. Động lực khoa học

Trong khi xu hướng chung lạm dụng Vision Transformers (>26M tham số, hàng trăm GFLOPs), nhóm tác giả DarkIR chứng minh luận điểm: **Một mạng CNN được thiết kế tinh gọn với các cơ chế Attention hợp lý hoàn toàn có thể vượt qua Transformer, đồng thời cắt giảm 55% tham số so với LEDNet và 88% tham số so với Restormer.**

### 2.2. Các thành phần kỹ thuật quan trọng của DarkIR

1. **Khối MetaFormer phi hàm kích hoạt (Activation-free MetaFormer):**
   - Loại bỏ các hàm kích hoạt phi tuyến đắt đỏ (GELU/SiLU).
   - **Simple Gate (SG):** Chia đôi kênh $C \to [C/2, C/2]$ và nhân từng phần tử: $\text{SG}(X) = X_1 \odot X_2$.
   - **Simplified Channel Attention (SCA):** Thu gọn không gian bằng Global Average Pooling (GAP) rồi qua tích chập điểm $1 \times 1$:
     $$\text{SCA}(X) = X \odot \sigma\left(\text{Conv}_{1\times1}(\text{GAP}(X))\right)$$
     Khối này tiêu tốn $< 0.1\%$ tính toán nhưng điều tiết tự động trọng số giữa kênh độ chói và kênh màu sắc.
2. **Dilated-Spatial Attention Module (Di-SpAM):** Kết hợp các nhánh tích chập sâu (Depthwise Conv) với dilation rates = 1, 4, 9 để mở rộng trường tiếp nhận bao trùm vùng tối rộng mà không tăng thêm trọng số.
3. **Phân tách tần số Fourier (Fre-MLP):** Đưa tensor sang miền tần số qua 2D FFT, tăng sáng trên phần biên độ (Amplitude) và giữ nguyên pha (Phase).

### 2.3. Lọc bỏ và điều chỉnh cho phù hợp với đồ án môn học

- **Học hỏi:** Triết lý CNN-first, khối SCA siêu nhẹ, hàm mất mát đo trên miền không gian.
- **Lược bỏ:** Nhánh Fourier FFT số phức (do khó triển khai trên phần cứng biên và phức tạp toán học ngoài phạm vi) và nhánh khử mờ deblurring đa nhiệm (tập trung thuần túy vào LLIE để bảo toàn tính tập trung của đề tài).

## 3. Ba Bài Báo Vệ Tinh Bổ Trợ (CVF 2024–2025)

1. **HVI Color Space (Yan et al., CVPR 2025):**
   - _Phát hiện:_ Tăng sáng trực tiếp trên sRGB dễ gây ám màu (Color Bias) do R, G, B bị ràng buộc phi tuyến. Tách không gian sang HSV thì bị đứt gãy màu đỏ ($h=0 \leftrightarrow h=6$) và nổ nhiễu ở vùng tối đen.
   - _Ứng dụng:_ Bổ sung hàm mất mát cấu trúc SSIM kết hợp Channel Attention để kìm hãm sự phân kỳ màu sắc giữa các kênh.
2. **Edge-LLIE (Sharif et al., CVPRW 2024):**
   - _Phát hiện:_ Trên kiến trúc U-Net, đường Skip Connection thô từ tầng nông vô tình bơm trực tiếp nhiễu hạt cảm biến sang Decoder, làm ảnh sau tăng sáng bị lốm đốm nhiễu.
   - _Ứng dụng:_ Đặt bộ cổng lọc `GatedSkipConnection` ($1\times1$ Conv + Sigmoid) trên đường tắt để lọc sạch nhiễu hạt.
3. **NTIRE 2024 Challenge on Low-Light Enhancement (Liu et al., CVPRW 2024):**
   - _Ứng dụng:_ Thiết lập phương pháp luận đo lường chuẩn xác bằng PSNR, SSIM và kỹ thuật cắt mảng ngẫu nhiên (Random Crop $256 \times 256$) giữ nguyên cấu trúc hạt nhiễu vật lý.

---

> Có thể tối ưu để khi áp dụng lượng tử hóa FB16/INT8 (Post-Training Quantization), kích thước mô hình giảm còn **~0.9 MB**, đạt tốc độ suy luận thời gian thực (>60 FPS) trên các thiết bị nhúng.
