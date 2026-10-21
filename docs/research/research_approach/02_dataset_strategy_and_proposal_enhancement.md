# 02. Chiến Lược Dữ Liệu, Phân Vùng 1.000 Cặp & Đề Xuất Cải Tiến

> **Môn học:** CS2203 - UIT - VNUHCM  
> **Đề tài:** Low-Light Image Enhancement Using a Lightweight CNN  
> **Bộ dữ liệu sử dụng:** LOLv1, LOL-v2 Real (Paired) và ExDark (Unpaired Real-world)  
> **Mục tiêu:** Định hình chiến lược phân vùng dữ liệu (50:30):20, đánh giá phản biện hướng đi của người dùng và đề xuất 3 cải tiến khoa học then chốt.

---

## 1. Tổng Quan Cấu Trúc Dữ Liệu Trong Workspace

Trong thư mục `datasets/` hiện tại của đồ án:
1. **LOLv1 (`datasets/LOLv1Dataset/lol_dataset/`):**
   - Tập `our485`: 485 cặp ảnh (low / high). Độ phân giải gốc ~400×600.
   - Tập `eval15`: 15 cặp ảnh (low / high).
   - Tổng cộng: **500 cặp ảnh**.
2. **LOL-v2 Real (`datasets/LOL-v2/Real_captured/`):**
   - Tập `Train`: 689 cặp ảnh (Low / Normal). Chụp từ máy ảnh thực tế với nhiều điều kiện sáng và tỷ lệ phơi sáng khác nhau.
   - Tập `Test`: 100 cặp ảnh (Low / Normal).
   - Tổng cộng: **789 cặp ảnh**.
3. **ExDark (`datasets/ExDark/`):**
   - Bộ dữ liệu ảnh ban đêm đời thực không có ảnh ground truth (unpaired low-light images).
   - Dùng để kiểm tra khả năng tổng quát hóa thực tế ngoài phân phối (Out-of-Distribution - OOD Testing).

---

## 2. Thiết Kế Phân Vùng Dữ Liệu Theo Tỷ Lệ Đề Xuất Mới (Mặc Định 250 Ảnh Khởi Đầu)

Nhằm giảm tải khối lượng tính toán và tối ưu tốc độ huấn luyện trên Colab, tổng quy mô dữ liệu tuyển chọn được rút gọn xuống **750 cặp ảnh** (hoàn toàn nằm dưới ngưỡng khống chế 1.000 ảnh):
- **250 ảnh (Tập Train Khởi đầu - $D_{\text{train\_0}}$):** Mặc định cho vòng huấn luyện đầu tiên.
- **300 ảnh (Tập Cách ly dự trữ - $D_{\text{quarantine}}$):** Chia thành 6 khối $S_1 \dots S_6$, mỗi khối 50 ảnh để thử nghiệm tăng dần.
- **200 ảnh (Tập Validation cố định - $D_{\text{val}}$):** Giữ cố định làm thước đo khách quan xuyên suốt.

### Phân tầng cân bằng (Stratified Sampling)
Giữ đúng tỷ lệ **$48.5\%$ LOLv1 và $51.5\%$ LOLv2 Real** trên mọi phân vùng:
1. **Tập Validation (200 ảnh):** Gồm đúng **97 ảnh LOLv1 + 103 ảnh LOLv2 Real**.
2. **Tập Train ban đầu (250 ảnh):** Gồm đúng **121 ảnh LOLv1 + 129 ảnh LOLv2 Real**.
3. **Tập Cách ly dự trữ (300 ảnh):** Gồm 6 khối, mỗi khối 50 ảnh gồm **24 ảnh LOLv1 + 26 ảnh LOLv2 Real**.
*(Toàn bộ danh sách file đã được khóa xác định trong `docs/ai/split_manifest.json` và kiểm thử zero-leakage).*

---

## 3. Đánh Giá Phản Biện & Giải Đáp Về Data Augmentation & Loss Function

### 3.1. Phản biện về Data Augmentation: Có vô nghĩa trong LLIE hay không?
Bạn đã đặt một câu hỏi rất sắc bén và hoàn toàn chính xác về mặt trực giác:  
> *"Mục đích của chúng ta là tăng sáng ảnh (Low-Level Vision) chứ không phải phân loại hay phát hiện vật thể (High-Level Vision). Tại sao lại cần Random Crop hay Flip khi ta cần phục hồi mọi pixel ở vùng tối?"*

Dưới góc độ vật lý quang học và kiến trúc học sâu, câu trả lời gồm 3 điểm mấu chốt:

1. **Bạn hoàn toàn ĐÚNG khi nói KHÔNG được lạm dụng Augmentation của bài toán phân loại:**
   - Trong bài toán phân loại (Classification), người ta hay dùng `ColorJitter` (chỉnh ngẫu nhiên độ sáng, tương phản, độ bão hòa), `Cutout` (khoét lỗ đen), `Mixup`.
   - **Trong LLIE, việc dùng ColorJitter hay Cutout là sai lầm nghiêm trọng!** Bởi vì mô hình cần học đúng ánh xạ trắc quang thực tế từ cảm biến máy ảnh: $\mathbf{y} = \gamma(\mathbf{x} \otimes \mathbf{k}) + \mathbf{n}$. Nếu ta tự ý thay đổi màu hoặc độ sáng của ảnh input, ta đang phá hỏng phân bố vật lý của dữ liệu.
2. **Vậy tại sao các paper LLIE (DarkIR, RetinexNet, KinD) bắt buộc phải dùng Crop?**
   - **Lý do phần cứng & VRAM:** Ảnh trong LOLv2 Real có kích thước rất lớn (lên tới $1000 \times 1200$ hoặc $2000 \times 3000$). Nếu nạp nguyên ảnh lớn vào GPU, bộ nhớ VRAM sẽ bị tràn ngay lập tức (CUDA Out of Memory), hoặc bắt buộc phải đặt `batch_size = 1`, khiến gradient dao động dữ dội và mạng rất khó hội tụ.
   - **Tại sao không Resize mà phải Crop?** Nếu ta dùng lệnh `Resize` co nhỏ ảnh $2000 \times 3000$ về $256 \times 256$, thuật toán nội suy (interpolation) sẽ làm nhòe mịn các hạt nhiễu (noise grain) và phá vỡ cấu trúc vi mô của cảm biến. Mạng nơ-ron sẽ không học được cách khử nhiễu thật. Do đó, chuẩn mực bắt buộc trong phục hồi ảnh là **Crop mảng nhỏ $256 \times 256$ giữ nguyên độ phân giải gốc**.
   - **Random Crop vs. Fixed Crop:** Random Crop giúp qua 100 epochs, mạng quét qua được mọi góc tối, góc sáng, vùng bóng đổ khác nhau trên ảnh thay vì chỉ nhìn vào duy nhất vùng trung tâm.
3. **Ý nghĩa của Flip (Lật ngang/dọc):**
   - Trong ảnh chụp đêm, nguồn sáng (bóng đèn, mặt trăng, biển quảng cáo) thường nằm lệch về một bên (trái hoặc phải). Phép lật ảnh giúp mạng học được tính **bất biến với hướng chiếu sáng (Illumination Direction Invariance)**, tránh việc mạng bị thiên vị rằng góc trên bên trái luôn luôn có đèn.
4. **KẾT LUẬN & ĐỒNG THUẬN VỚI BẠN:**
   - Để tránh làm phức tạp hóa đồ án môn học, **chúng ta xem Crop $256 \times 256$ thuần túy là một giải pháp kỹ thuật bắt buộc để nạp vừa GPU và giữ nguyên hạt nhiễu**, chứ **KHÔNG biến Data Augmentation thành một biến số cần tinh chỉnh**. Bạn không cần tốn thời gian thử nghiệm các kỹ thuật augment phức tạp khác!

---

### 3.2. Bản Chất Các Hàm Mất Mát (Loss Functions): Vì Sao Lại Chọn $L_1$ và $L_1 + \text{SSIM}$?

1. **Tại sao KHÔNG dùng hàm bình phương sai số MSE ($L_2$ Loss)?**
   - Công thức: $L_2 = (\hat{y} - y)^2$. 
   - Điểm yếu chí mạng của $L_2$: Do có bình phương, sai số càng lớn thì hình phạt tăng theo cấp số nhân. Trong ảnh tối, các vùng biên cạnh (edge) và hạt nhiễu có độ tương phản rất gắt. Để tránh bị phạt quá nặng ở các pixel này, mạng CNN có xu hướng "chọn giải pháp an toàn" bằng cách dự đoán một giá trị trung bình mờ mờ (average blur) của các pixel xung quanh.
   - **Hậu quả:** Ảnh sau phục hồi bị mờ căm (oversmoothed), mất hết vân chi tiết bề mặt.
2. **Tại sao $L_1$ Loss là nền tảng số 1?**
   - Công thức: $L_1 = |\hat{y} - y|$.
   - Đạo hàm của $L_1$ luôn là $+1$ hoặc $-1$ (hằng số), phạt sai số một cách tuyến tính và công bằng. Nó không ép các pixel biên cạnh phải lấy giá trị trung bình mờ, do đó $L_1$ giữ được biên cạnh sắc nét và màu sắc trung thực hơn $L_2$ rất nhiều.
3. **Tại sao lại cần thử nghiệm thêm SSIM Loss ($\text{Composite Loss} = L_1 + 0.2 \times (1 - \text{SSIM})$)?**
   - $L_1$ so sánh từng pixel độc lập tại tọa độ $(x, y)$ giữa ảnh dự đoán và ảnh ground-truth, nhưng **hoàn toàn không quan tâm đến mối liên hệ giữa các pixel lân cận** (cấu trúc cục bộ).
   - **SSIM (Structural Similarity)** là độ đo mô phỏng thị giác con người dựa trên 3 thành phần:
     1. Độ chói trung bình (Luminance).
     2. Độ tương phản cục bộ (Contrast / Variance).
     3. Cấu trúc bề mặt (Structure / Covariance).
   - Khi chỉ dùng $L_1$, đôi khi ảnh đủ sáng nhưng các vùng họa tiết (vân gỗ, sợi vải, nét chữ) nhìn hơi nhợt nhạt và bệt màu. Thành phần $(1 - \text{SSIM})$ đóng vai trò như một lực đẩy bổ sung, buộc mạng nơ-ron phải tái tạo được **độ tương phản vi mô giữa các pixel liền kề**. Trọng số $0.2$ là tỉ lệ nhỏ vừa đủ để hỗ trợ giữ nét mà không làm loãng mục tiêu chính của $L_1$.

---

## 3. Đánh Giá Phản Biện Hướng Đi Đề Xuất (Critical Review)

### 3.1. Những điểm xuất sắc trong đề xuất
1. **Quy mô 1.000 ảnh là cực kỳ hợp lý và chuẩn mực:**
   - Trong các bài báo LLIE kinh điển: RetinexNet (BMVC 2018), KinD (ACM MM 2019), EnlightenGAN (TIP 2021), Zero-DCE (CVPR 2020), tất cả đều chỉ huấn luyện trên **485 ảnh của LOLv1**. Do đó, mốc 500 ảnh của bạn hoàn toàn đủ về mặt thống kê để một mạng CNN/U-Net nhỏ hội tụ xuất sắc.
   - Quy mô này giúp thời gian huấn luyện trên Google Colab (GPU T4) chỉ mất **khoảng 15–20 phút cho 100 epochs**, cho phép thử nghiệm nhiều lượt mà không lo hết quota GPU miễn phí.
2. **Tư duy thực nghiệm tăng dần (+50 ảnh) có giá trị học thuật rất cao:**
   - Đây là điểm sáng nhất trong phương pháp luận của bạn. Thay vì ném toàn bộ dữ liệu vào một lần, việc cách ly 300 ảnh và tăng từng bước 50 ảnh cho phép xây dựng biểu đồ **Sample Efficiency Curve (Đường cong hiệu quả mẫu / Data-Scaling Law)**.
   - Khi báo cáo đồ án, giảng viên sẽ thấy rõ sinh viên có tư duy nghiên cứu khoa học: biết cô lập biến số, phân tích độ nhạy của mô hình với kích thước dữ liệu, và chứng minh được kiến trúc Attention U-Net có khả năng học tốt ngay cả khi dữ liệu bị hạn chế.

### 3.2. Ba rủi ro tiềm ẩn (Potential Pitfalls) cần phòng ngừa

1. **⚠️ Rủi ro 1: Lệch phân phối (Domain Gap) giữa LOLv1 và LOLv2 Real**
   - *Bản chất kỹ thuật:* LOLv1 được chụp bằng máy Canon với ISO thay đổi, độ phân giải ~400x600; LOLv2 Real chụp từ nhiều sensor khác nhau, tỉ lệ phơi sáng khác nhau. Nếu trộn 1.000 ảnh rồi chia ngẫu nhiên thuần túy, có thể tập Valid 200 ảnh vô tình chứa toàn ảnh LOLv1 (hoặc toàn LOLv2), làm kết quả đánh giá bị thiên lệch (bias).
2. **⚠️ Rủi ro 2: Thách thức đánh giá trên ExDark (Unpaired Data)**
   - *Bản chất kỹ thuật:* ExDark gồm hàng nghìn ảnh chụp đêm thực tế nhưng **hoàn toàn không có ảnh đủ sáng tương ứng (Ground Truth)**. Do đó ta **không thể tính điểm PSNR và SSIM trên ExDark**. Nếu để trống số liệu thì sẽ bị giảng viên chất vấn cơ sở khoa học.
3. **⚠️ Rủi ro 3: Bẫy "Tăng dữ liệu là giải pháp đầu tiên"**
   - Trong bài toán phục hồi ảnh, U-Net được huấn luyện bằng cách cắt patch ngẫu nhiên ($256 \times 256$). Với 500 ảnh, qua 100 epochs, mạng thực chất đã nhìn thấy hàng vạn mảng ảnh ngẫu nhiên khác nhau. Do đó, nếu mô hình học kém trên 500 ảnh, nguyên nhân $90\%$ nằm ở siêu tham số (Learning Rate, Data Augmentation, Loss balance) chứ không phải do thiếu 50 ảnh.

---

## 4. Ba Đề Xuất Cải Tiến Khoa Học Cụ Thể (Enhancements)

### 🌟 Cải Tiến 1: Phân Tầng Cân Bằng Tuyệt Đối (Stratified Dataset Splitting)
Khi chọn 1.000 ảnh từ LOLv1 (485 cặp) và LOLv2 Real (515 cặp trong 689 cặp Train):
- Phân tầng chính xác tỷ lệ **$48.5\%$ LOLv1 và $51.5\%$ LOLv2 Real** xuyên suốt cả 3 tập:
  1. **Tập Validation (200 ảnh):** Gồm đúng **97 ảnh LOLv1 + 103 ảnh LOLv2 Real** (cố định vĩnh viễn).
  2. **Tập Train ban đầu (500 ảnh):** Gồm đúng **243 ảnh LOLv1 + 257 ảnh LOLv2 Real**.
  3. **Tập Cách ly dự trữ (300 ảnh):** Gồm đúng **145 ảnh LOLv1 + 155 ảnh LOLv2 Real** (chia đều vào 6 khối, mỗi khối 50 ảnh gồm ~24 ảnh LOLv1 và ~26 ảnh LOLv2).
- *Lợi ích:* Triệt tiêu hoàn toàn hiện tượng lệch domain giữa Train và Valid. Toàn bộ danh sách file đã được khóa xác định trong `docs/ai/split_manifest.json` (zero-leakage).

### 🌟 Cải Tiến 2: Phương Pháp Đánh Giá 2 Tầng (Two-Tier Evaluation Protocol)
Phân định rõ ràng vai trò của 2 nhóm tập dữ liệu kiểm thử:

```
                              KIỂM THỬ THỰC NGHIỆM
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            │                                                     │
   [TẦNG 1: ĐỊNH LƯỢNG NỘI MIỀN]                        [TẦNG 2: NGOẠI MIỀN THỰC TẾ]
   (In-Distribution Quantitative)                       (Out-of-Distribution Qualitative)
            │                                                     │
    Tập Valid (200 ảnh)                                  Tập ExDark (Ảnh đêm thật)
            │                                                     │
   Có Ground-Truth chuẩn                                  Không có Ground-Truth
            │                                                     │
   Đo bằng: PSNR, SSIM, L1 Loss                          Đo bằng:
   (Vẽ biểu đồ hội tụ & so sánh                          1. Lưới ảnh so sánh trực quan
    giữa các mốc 500 -> 800 ảnh)                         2. Chỉ số No-Reference NIQE / BRISQUE
                                                            (Điểm càng giảm -> ảnh càng tự nhiên)
```

- *Ý nghĩa khi bảo vệ đồ án:* Sinh viên tự tin giải trình trước Hội đồng:
  - *"Trên dữ liệu có cặp chuẩn (LOL), mô hình chúng em đạt PSNR 25+ dB, SSIM 0.85+."*
  - *"Trên dữ liệu thực tế không có nhãn (ExDark), mô hình phục hồi rõ nét chi tiết bóng đêm và giảm chỉ số cảm nhận NIQE, chứng minh mô hình không bị quá khớp vào môi trường phòng thí nghiệm."*

### 🌟 Cải Tiến 3: Lộ Trình 3 Pha Tối Ưu Trước Khi Mở Khóa Dữ Liệu Cách Ly
1. **Pha A: Tối ưu Data Augmentation trên 500 ảnh ban đầu**
   - Áp dụng: `RandomCrop(256)` + `RandomHorizontalFlip(p=0.5)` + `RandomVerticalFlip(p=0.5)`. Giữ nguyên phân bố quang học tự nhiên của sensor.
2. **Pha B: Tối ưu Hàm Mất Mát (Loss Tuning)**
   - So sánh giữa thuần $L_1$ Loss với Composite Loss: $\mathcal{L} = L_1 + 0.2 \times (1 - \text{SSIM})$. Đánh giá xem SSIM loss có giúp triệt tiêu hiện tượng mờ nhòe vân bề mặt hay không.
3. **Pha C: Mở khóa dữ liệu cách ly (+50 ảnh/lần) để vẽ đường cong học tập**
   - Sau khi Pha A và B đã tối ưu, tiến hành nạp thêm Khối $S_1 (+50 \to 550)$, $S_2 (+50 \to 600) \dots$ và vẽ biểu đồ **PSNR vs. Training Size**. Đây là phần thực nghiệm đắt giá nhất của đồ án!
