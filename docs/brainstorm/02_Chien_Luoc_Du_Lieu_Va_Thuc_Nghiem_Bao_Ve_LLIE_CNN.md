# BÁO CÁO CHIẾN LƯỢC DỮ LIỆU, CẨM NANG THỰC NGHIỆM GOOGLE COLAB & BẢNG BẢO VỆ ĐỒ ÁN (V2 - TỐI ƯU)

> **Môn học:** Xử Lý Ảnh và Thị Giác Máy Tính (CS2203) — Master of Science in Computer Science 2026  
> **Đơn vị:** Trường Đại học Công nghệ Thông tin (UIT) - ĐHQG TP.HCM  
> **Đề tài:** Tăng cường chất lượng ảnh trong điều kiện thiếu sáng bằng mạng nơ-ron tích chập (*Low-Light Image Enhancement Using a Lightweight CNN*)  
> **Văn bản gốc tham chiếu:** Báo cáo kỹ thuật đối chuẩn DarkIR (CVPR 2025), Edge-LLIE (CVPRW 2024), HVI-CIDNet (CVPR 2025).

---

# PHẦN I: KHẢO SÁT & HỆ QUY CHIẾU THAM CHIẾU (DATASET AUDIT & REFERENCE BENCHMARK)

Để đảm bảo kết quả thực nghiệm có cơ sở khoa học vững chắc và **thiết lập hệ quy chiếu so sánh tham chiếu (Reference Comparison) với bài báo gốc DarkIR (CVPR 2025)**, đề tài thiết lập hệ thống dữ liệu và độ đo đánh giá tương đồng về mặt phân phối nhằm phục vụ báo cáo khoa học và trình bày trước Hội đồng. 

> [!IMPORTANT]
> **Lưu ý phương pháp luận:** DarkIR gốc được huấn luyện trên toàn bộ tập dữ liệu khổng lồ (hàng nghìn đến chục nghìn cặp ảnh, hàng trăm GPU-hours), trong khi đồ án môn học tập trung huấn luyện từ đầu (trained from scratch) trên tập dữ liệu tuyển chọn đồng nhất ($D_{\text{train}} = 1.350$ cặp ảnh, trong đó 150 cặp là LOL-Blur). Vì vậy, kết quả của DarkIR gốc được sử dụng làm **mốc tham chiếu trần lý thuyết**, không tuyên bố tái lập 1-1 (No SOTA reproduction claim), mà tập trung đo lường đóng góp thực tế của kiến trúc đề xuất duy nhất `DarkIR-Edge` trong điều kiện tài nguyên giới hạn.

## 1. Bảng Tổng Hợp Chi Tiết Các Bộ Dữ Liệu & Vai Trò Trong Đồ Án

| Phân Nhóm | Tên Tập Dữ Liệu | Bản Chất Cảm Biến / Thiết Bị | Tập Train (Cặp) | Tập Val (Cặp) | Tập Test (Cặp) | Tổng Quy Mô | Chỉ Số Đánh Giá (Metrics) | Vai Trò Trong Đồ Án |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Chuẩn LLIE hiện đại** | **LOLv2-Real** | Máy ảnh DSLR chuyên nghiệp (Canon/Sony) | **639** | **50** | **100** | **789** | **PSNR, SSIM, LPIPS** | Nguồn DSLR chính (Chuẩn BMVC, $639+50+100=789$) |
| **Đa cảm biến (Smartphone & DSLM)** | **LSRW-Huawei** | Cảm biến Smartphone thực tế (Huawei P40/Mate30) | **561** | **50** | **30** | **641** | **PSNR, SSIM, LPIPS** | Nguồn smartphone chính (Trích từ kho 2.450 ảnh train) |
| | **LSRW-Nikon** | Máy ảnh Nikon chuyên nghiệp | — | — | **20** | **20** | **PSNR, SSIM, LPIPS** | **Zero-shot Unseen Sensor** (chỉ có trong Test, không qua Val) |
| **Khái quát hóa ngoài đời (In-the-wild Unpaired)** | **DICM / LIME** | Đa dạng máy ảnh ngoài trời / trong nhà | — | — | **69 + 10 = 79** | **79** | **Trực quan hóa định tính (Visual Quality)** | Đánh giá định tính ảnh thực tế không cặp |
| **Khử mờ trong tối (v2)** | **LOL-Blur** (tập con) | Cảnh quay mô phỏng vừa thiếu sáng vừa nhòe chuyển động (GT = `high_sharp_scaled`) | **150** | **10** | **40** | **200** | **PSNR, SSIM, LPIPS** | Kích hoạt Decoder học khử mờ; test lấy từ tập test chính thức của LOL-Blur |
| *(Phạm vi lược bỏ)* | *Real-LOLBlur (không có GT) / ExDark* | *Rung nhòe đêm thực / Nhãn nhận diện 12 lớp* | — | — | — | — | *NIQE, BRISQUE* | *Chỉ xem định tính Real-LOLBlur, không tính chỉ số* |

## 2. Quyết Định Khoa Học: Loại Bỏ Hoàn Toàn LOLv1 Để Triệt Tiêu Rò Rỉ Dữ Liệu

Trong bài báo **DarkIR (CVPR 2025, Section 4.1)**, các tác giả đã nêu rõ:
> *"Note that LOLv2-Real is the extended version of LOL [v1], thus we use the v2 version directly."*

- **Bản chất rủi ro:** Bộ dữ liệu LOLv1 (485 cặp train, 15 cặp eval) thực chất là tập con của LOLv2-Real. Rất nhiều khung cảnh, phòng ốc và góc chụp trong LOLv1 bị trùng lặp hoàn toàn với LOLv2-Real. Nếu gộp cả 2 bộ dữ liệu rồi phân chia ngẫu nhiên, mô hình sẽ gặp hiện tượng **Rò rỉ dữ liệu ngầm (Hidden Data Leakage)**: cùng một căn phòng xuất hiện ở tập Train từ nguồn LOLv1 nhưng lại nằm trong tập Validation/Test từ nguồn LOLv2-Real.
- **Giải pháp dứt khoát:** Loại bỏ hoàn toàn LOLv1 khỏi toàn bộ pipeline dữ liệu. Sử dụng **LOLv2-Real** làm nguồn dữ liệu DSLR chuẩn duy nhất, kết hợp với **LSRW-Huawei** làm nguồn dữ liệu smartphone thực tế.

---

# PHẦN II: CHIẾN LƯỢC DỮ LIỆU ĐỒNG NHẤT ĐA CẢM BIẾN & PHÂN VÙNG THỰC NGHIỆM (v2: 1.650 CẶP ẢNH)

## 1. Quyết Định Gộp Dữ Liệu Huấn Luyện Đồng Nhất Cân Bằng DSLR : Smartphone

> [!IMPORTANT]
> **Định hướng chiến lược của đồ án:**  
> Nhằm tận dụng tối đa dữ liệu đã được gán nhãn cặp và chuẩn hóa, đồng thời tạo ra phân phối mẫu cân bằng tối ưu giữa máy ảnh chuyên nghiệp và camera điện thoại di động, nhóm quyết định **gộp toàn bộ dữ liệu huấn luyện đã chuẩn bị thành MỘT tập train duy nhất $D_{\text{train}} = 1.350$ cặp ảnh (gồm 639 cặp LOLv2-Real DSLR, 561 cặp LSRW-Huawei Smartphone — giữ nguyên tỉ lệ 53:47 — và 150 cặp LOL-Blur để huấn luyện khử mờ)**.

### Cơ sở khoa học và thực tiễn của quyết định:
1. **Cân bằng tối ưu giữa hai miền cảm biến DSLR : Smartphone:**  
   Việc thiết lập tỷ lệ ~53% DSLR (639 ảnh LOLv2-Real) và ~47% Smartphone (561 ảnh LSRW-Huawei trích từ kho 2.450 ảnh train) giúp mạng nơ-ron học đồng đều cả hai đặc tính suy thoái: nhiễu Gauss/Poisson mịn trên cảm biến lớn và nhiễu hạt bệt/loang màu ISP trên camera điện thoại.
2. **Khắc phục triệt để lỗi số học phân vùng LOLv2-Real:**  
   Bộ LOLv2-Real chính thức chỉ có đúng 789 cặp ảnh. Nhóm phân bổ chính xác: 639 Train + 50 Val + 100 Test = 789 cặp, giải quyết dứt điểm lỗi thiếu hụt mẫu val.
3. **Giữ sạch miền kiểm thử Unseen Sensor (Zero-Shot Domain Generalization):**  
   Tập dữ liệu **LSRW-Nikon (20 cặp ảnh)** được rút hoàn toàn khỏi tập Validation và chỉ xuất hiện duy nhất ở tập Test độc lập. Nhờ đó, đề tài có một phép thử nghiệm "sạch 100%" về năng lực tổng quát hóa sang một loại cảm biến hoàn toàn chưa từng thấy trong quá trình huấn luyện và chọn mô hình.

---

## 2. Thiết Kế Phân Vùng Tuyển Chọn Chuẩn Hóa (Tổng Quy Mô v2: 1.650 Cặp Ảnh)

Cấu trúc phân vùng dữ liệu được khóa cứng xuyên suốt đồ án:

```
TỔNG QUY MÔ DỮ LIỆU THỰC NGHIỆM (v2): 1.650 CẶP ẢNH (v1 = 1.450)
│
├── 1. Tập Huấn Luyện Duy Nhất (D_train): 1.350 cặp ảnh
│      ├── 639 cặp LOLv2-Real (DSLR sensor)             -> 47,3%
│      ├── 561 cặp LSRW-Huawei (Smartphone sensor)      -> 41,6%   (LOLv2 : Huawei vẫn = 53,25 : 46,75)
│      └── 150 cặp LOL-Blur (thiếu sáng + nhòe)         -> 11,1%
│      --> Mục tiêu: DarkIR-Edge học đồng thời tăng sáng, khử nhiễu và khử mờ.
│
├── 2. Tập Validation Cố Định (D_val): 110 cặp ảnh (khóa cứng xuyên suốt)
│      ├── 50 cặp LOLv2-Real Validation
│      ├── 50 cặp LSRW-Huawei Validation
│      └── 10 cặp LOL-Blur Validation (sequence KHÔNG trùng với train)
│      --> Theo dõi val PSNR macro (trung bình 3 domain) để chọn checkpoint và Early Stopping.
│
└── 3. Tập Benchmark Test Độc Lập Đối Chuẩn Với DarkIR (Zero Leakage): 190 cặp
       ├── Paired DSLR Test: LOLv2-Real Test (100 cặp chuẩn BMVC)
       ├── Paired Smartphone Test: LSRW-Huawei Test (30 cặp chuẩn R2RNet)
       ├── Paired Zero-shot Multi-Sensor Test: LSRW-Nikon Test (20 cặp unseen sensor)
       ├── Paired Deblur Test: LOL-Blur Test chính thức (40 cặp, lấy mẫu cố định seed 42, mọi sequence test đều có đại diện)
       └── Unpaired Qualitative Test: DICM (69 ảnh), LIME (10 ảnh) đánh giá trực quan.
```

## 3. Nguyên Tắc Bảo Đảm Tuyệt Đối Không Rò Rỉ Dữ Liệu (Zero Data Leakage Protocol)

1. **Phân chia theo cảnh/sequence (Scene & Sequence Grouping):** Các ảnh chụp cùng bối cảnh phòng ốc/góc chụp (LOLv2, LSRW: chung tiền tố định danh cảnh) hoặc cùng một video sequence (LOL-Blur: các frame liên tiếp gần như giống hệt nhau) được phân bổ trọn vẹn vào cùng một phân vùng (Train hoặc Val), tuyệt đối không tách rời làm rò rỉ dữ liệu. Với LOL-Blur, mỗi sequence chỉ lấy tối đa vài frame cách xa nhau để tăng đa dạng.
2. **Cô lập tập Test gốc:** Các tập kiểm thử chính thức của tác giả công bố (**LOLv2-Real Test 100 ảnh**, **LSRW-Huawei Test 30 ảnh**, **LSRW-Nikon Test 20 ảnh**, **LOL-Blur Test 40 ảnh lấy từ tập test chính thức**) được giữ nguyên trạng làm tập Test cuối cùng, tuyệt đối không đưa bất kỳ ảnh nào của các tập này vào $D_{\text{train}}$ hay $D_{\text{val}}$.
3. **Lưu vết bằng Manifest (`split_manifest.json`):** Mọi đường dẫn file của từng phân vùng được lưu cố định trong file JSON kèm mã băm SHA-256 để đảm bảo tính tái lập 100% và minh bạch trước Hội đồng.

## 4. Bản Chất Smartphone vs. DSLR: Vì Sao Cần Cân Bằng Smartphone Trong Tập Train?

- **Cảm biến DSLR (LOLv2-Real):** Kích thước cảm biến lớn (Full-frame/APS-C), ống kính quang học đường kính rộng. Nhiễu trong ảnh thiếu sáng chủ yếu là nhiễu Gauss và Poisson thuần túy, hạt nhiễu mịn và đồng đều.
- **Cảm biến Smartphone (LSRW-Huawei):** Cảm biến kích thước siêu nhỏ ($1/1.28''$ đến $1/2.5''$), khẩu độ nhỏ. Tín hiệu quang học yếu buộc bộ xử lý ISP của điện thoại phải khuếch đại ISO cực cao, kèm theo thuật toán nén JPEG và bộ lọc làm mịn tự động gây hiện tượng **nhiễu hạt bệt, loang lổ màu sắc (Chroma Noise) và mất biên cạnh**.
- **Kết luận:** Việc đưa 561 cặp ảnh smartphone (~47% tập Train) tạo ra sự cân bằng lý tưởng giúp mạng học được đồng đều cả hai phân bố suy thoái vật lý, nâng cao năng lực phục hồi thực tế trên cả thiết bị di động lẫn máy ảnh chuyên nghiệp.

## 5. Chuẩn Hóa Tiền Xử Lý Dữ Liệu (Data Augmentation)

1. **Tuyệt đối KHÔNG dùng ColorJitter hay Cutout:** Tránh làm sai lệch đặc tính trắc quang vật lý của cảm biến.
2. **Bắt buộc dùng Random Crop $256 \times 256$:** Giữ nguyên tỉ lệ 1:1 của hạt nhiễu cảm biến, ngăn chặn hiện tượng làm mượt giả tạo khi dùng Resize.
3. **Phép lật ngẫu nhiên (Horizontal & Vertical Flip):** Tăng cường tính bất biến hướng sáng.
4. **Cơ chế Nhân bản ảo (Virtual Multiplier - Multi-crop):** Với mỗi ảnh gốc trong một epoch, bộ nạp dữ liệu lấy mẫu ngẫu nhiên nhiều patches $256 \times 256$ khác nhau. Mặc định thiết lập `repeat_factor = 2` ($1.350 \times 2 = 2.700$ patches/epoch) cho cấu hình huấn luyện chuẩn, tối ưu hóa mật độ cập nhật gradient mà không tốn thêm bộ nhớ đĩa.

---

# PHẦN III: PHÂN TÍCH TÀI NGUYÊN GOOGLE COLAB & TỐI ƯU HUẤN LUYỆN

## 1. Định Lượng Thời Gian Huấn Luyện & Chính Sách Ngân Sách Bước Cố Định

Mô hình mục tiêu đề xuất duy nhất (**DarkIR-Edge (C=24)**, quy mô ~1.93M tham số) kết hợp tập dữ liệu đồng nhất cân bằng $D_{\text{train}}$ (1.350 cặp ảnh) được định lượng tài nguyên chính xác:

- **Số bước cập nhật (Steps) mỗi Epoch:**  
  $$\text{Steps per Epoch} = \frac{\text{Số ảnh} \times \text{Repeat Factor}}{\text{Batch Size}} = \frac{1.350 \times 2}{8} \approx 337 \text{ steps/epoch}$$
- **Chiến lược Huấn luyện theo Ngân Sách Cố Định (Fixed-Budget Training):**  
  - Ngân sách tối đa (v2): **100 epochs** (~33.700 steps) kèm Early Stopping (xem mục 2 bên dưới; không được dừng trước epoch 40). v1 dùng 100 epochs cố định (30.000 steps) và đã bộc lộ hiện tượng hội tụ sớm.
  - Bộ lập lịch Learning Rate: **CosineAnnealingLR** từ $2\times 10^{-4}$ giảm dần về $1\times 10^{-6}$ qua tối đa 100 epochs (`T_max = 100`, giữ nguyên như v1; nếu Early Stopping dừng sớm thì LR chưa giảm hết — đây là hệ quả đã biết và chấp nhận, áp dụng giống hệt cho mọi mô hình).
  - **Quy tắc chọn Checkpoint (sửa đổi từ v1):** Chọn **checkpoint có val PSNR macro (EMA) cao nhất (`best.pt`)**. Quy tắc "Final Checkpoint" của phiên bản đầu bị rút lại có chủ đích vì dữ liệu v1 cho thấy val PSNR của DarkIR-Edge giảm dần từ ~20.88 dB (epoch 5–35) xuống ~20.59 dB ở epoch 100 trong khi train loss vẫn giảm (0.31 → 0.14) — dấu hiệu quá khớp. Quy tắc mới được áp dụng **giống hệt** cho DarkIR-Edge và Standard U-Net, chỉ dựa trên tập Val, tuyệt đối không nhìn tập Test.
  - **Tần suất Validation:** Tập Validation gồm 110 cặp (50 DSLR : 50 Smartphone : 10 Blur) dùng để theo dõi PSNR và chọn checkpoint; kiểm tra **mỗi 5 epochs (`val_interval = 5`)**.

| Kịch bản Thực Nghiệm | Quy mô dữ liệu | Số Steps / Epoch | Tổng số Steps (tối đa 100 Epochs) | Kế hoạch Seeds | Ước tính thời gian trên Colab GPU L4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DarkIR-Edge (C=24)** | **1.350 cặp ảnh** | **337 steps** | **≤ 33.700 steps** | **2 seeds** (42, 123) | **Đo chính xác bằng `probe_speed` 200 bước đầu trên L4; Early Stopping dự kiến dừng quanh epoch 40–60 (do không được dừng trước 40)** |
| **Standard U-Net (~1.9M)**| **1.350 cặp ảnh** | **337 steps** | **≤ 33.700 steps** | **2 seeds** (42, 123; nâng từ 1 để so sánh công bằng) | **Đo bằng `probe_speed` trên L4** |
| **DarkIR-Edge bỏ FFT/FreMLP (ablation, ~1.50M)** | **1.350 cặp ảnh** | **337 steps** | **≤ 33.700 steps** | **2 seeds** (42, 123) | **Cùng quy tắc Early Stopping; chỉ để kiểm chứng giả thuyết FFT** |

👉 **Kiểm soát thực tế:** Khi bắt đầu chạy trên Colab, tiến hành đo tốc độ xử lý thực tế (it/s) ở 200 bước đầu tiên để tính toán chính xác tổng thời gian và điều chỉnh số seed cho phù hợp với hạn mức phiên làm việc.

## 2. Early Stopping (v2): Bằng Chứng Từ v1, Quy Tắc Và Tính Hợp Lệ Học Thuật

**Bằng chứng từ phiên bản v1 (`history.csv`, val PSNR EMA, mỗi 5 epochs):**
- DarkIR-Edge seed 42: 20.88 (ep5) → ~20.8 (ep10–45) → 20.59 (ep100); seed 123: đỉnh 20.88 ở ep20, rồi giảm về 20.56. Train loss giảm đều 0.31 → 0.14 ⇒ **quá khớp nhẹ sau epoch ~20**, giai đoạn LR nhỏ không cải thiện mà còn làm giảm val PSNR (~0.3 dB).
- Standard U-Net: val PSNR phẳng ~20.8 – 21.0 suốt 100 epochs (không giảm, cải thiện rất ít).
- Hệ quả: ~70% thời gian của run 100 epochs là lãng phí.

**Quy tắc đề xuất (áp dụng giống hệt cho mọi mô hình học):**
1. Đại lượng theo dõi: **val PSNR macro (EMA)** = trung bình PSNR của 3 domain val (LOLv2, Huawei, Blur) để domain Blur (chỉ 10 ảnh) không bị lấn át bởi 2 domain lớn.
2. Kiểm tra mỗi 5 epochs; dừng khi **3 lần kiểm tra liên tiếp không vượt đỉnh quá `min_delta = 0.05 dB`** (tức 15 epochs không cải thiện). Nhiễu val giữa các lần đo ~±0.1 dB nên `min_delta` nhỏ hơn mức này sẽ làm kích hoạt dừng/tiếp tục theo ngẫu nhiên.
3. **Không được dừng trước epoch 40 và chỉ bắt đầu đếm patience sau epoch 40** (`min_epochs = 40`): trong 40 epochs đầu chỉ ghi nhận mốc đỉnh `es_ref`, biến đếm `early_stop_wait` luôn bằng 0; chỉ từ sau epoch 40 (mốc 45, 50, 55...) mới bắt đầu cộng dồn 15 epochs không cải thiện (tức sớm nhất dừng ở epoch 55), tránh dừng khi domain Blur chưa kịp hội tụ ở giai đoạn LR giảm.
4. Epochs tối đa = 100, `T_max = 100` (giữ như v1). Nạp lại `best.pt` (EMA) để đánh giá. Quy tắc này **cố định trước khi chạy** và áp dụng giống hệt cho DarkIR-Edge, U-Net và DarkIR-Edge bỏ FFT.
5. Ghi lại epoch dừng, val PSNR đỉnh và epoch đạt đỉnh của từng run vào báo cáo.

**Nên ưu tiên Loss hay PSNR?** Dùng **val PSNR**, không dùng train loss: (i) train loss là hàm tổng hợp (L1 + SSIM + side + color) tính trên patch ngẫu nhiên trong tập train nên giảm đều ngay cả khi mô hình đã quá khớp (đúng như v1); (ii) PSNR/SSIM/LPIPS mới là chỉ số được báo cáo. Có thể ghi thêm val SSIM để tham khảo, không dùng làm điều kiện dừng.

**Có vi phạm học thuật không?** Không, với điều kiện: (a) tiêu chí chỉ dựa trên tập Validation, tập Test không được dùng để quyết định dừng hay chọn checkpoint; (b) quy tắc được cố định **trước khi** chạy và áp dụng giống nhau cho mọi mô hình (Edge và U-Net); (c) báo cáo minh bạch rằng thiết lập này thay đổi so với v1 (100 epochs cố định, final checkpoint) và lý do (bằng chứng ở trên); (d) kết quả v1 vẫn được giữ làm đối chứng. Early Stopping và chọn checkpoint theo validation là thực hành tiêu chuẩn.

**Lưu ý GPU L4:** L4 (24 GB VRAM) nhanh hơn T4 nên giữ nguyên batch 8, patch 256, AMP FP16 để không thay đổi điều kiện huấn luyện; nếu muốn thử AMP bf16 phải áp dụng cho cả hai mô hình. Đo lại tốc độ bằng `probe_speed`; riêng bảng Latency/FPS phần cứng ở phần báo cáo cần ghi rõ GPU đo (T4 ở v1, L4 ở v2) — không trộn hai GPU trong cùng một bảng.

## 3. Cam Kết Báo Cáo Trung Thực v2 Và Quy Tắc Kết Luận (Khóa Trước Khi Chạy)

Sau khi review kết quả v1, nhóm thừa nhận các điểm chưa tốt và **khóa sẵn cách kết luận trước khi biết kết quả v2**, để dù kết quả thế nào báo cáo vẫn đúng với những gì xảy ra (đồ án môn học không yêu cầu phải vượt nghiên cứu gốc).

| # | Vấn đề ở v1 | Số liệu v1 | Cách v2 xử lý | Quy tắc kết luận (khóa trước) |
| :-: | :--- | :--- | :--- | :--- |
| 1 | Edge **chưa hơn** U-Net về chỉ số | PSNR pooled 20.04 vs 20.50 (Δ=−0.46 dB, p≈0.05); U-Net chỉ 1 seed | U-Net chạy **2 seed** (42, 123) như Edge | Dùng paired test (ΔPSNR, CI 95% bootstrap) trên pooled và từng miền. CI chứa 0 → "không khác biệt có ý nghĩa"; CI < 0 → "U-Net tốt hơn"; chỉ khi CI > 0 trên pooled mới được viết "Edge tốt hơn". **Không viết "Edge > U-Net" khi chưa thỏa.** |
| 2 | Chưa chứng minh "edge-friendly" | FLOPs 8.75 vs 18.45 G (chưa tính FFT) nhưng latency FP16 78.7 ms vs 29.7 ms (T4, 600×400): Edge chậm ≈2.6× | Đo lại Params/FLOPs/Latency/VRAM, **ghi rõ GPU** (cột `GPU` trong `efficiency.csv`); không trộn số đo T4 (v1) với L4 (v2) | Báo cáo "ít FLOPs hơn nhưng không nhanh hơn trên GPU" nếu số đo vẫn vậy. Không dùng từ "edge-friendly"/"real-time trên di động" vì chưa đo trên thiết bị biên thật. |
| 3 | Giả thuyết FFT **chưa kiểm chứng** | `gap_illumination` của Edge trên LOLv2 = 5.90 dB, cao hơn U-Net (4.93 dB): FFT không giúp ánh sáng toàn cục theo chẩn đoán này | **Ablation**: `DarkIR-Edge` bỏ FFT/FreMLP (`use_fft=False`, mọi thứ khác giữ nguyên) chạy 2 seed (42, 123) | Paired test Edge − noFFT. CI > 0 → "FFT có đóng góp đo được ở cấu hình này"; CI chứa 0 → "chưa chứng minh được đóng góp của FFT" (có thể bỏ để giảm chi phí); CI < 0 → "FFT không giúp, thậm chí làm giảm". Lưu ý: bản noFFT ít hơn ~0.43M tham số (1.495M vs 1.924M), nên đây là so sánh "bỏ bớt module", không phải so sánh cùng số tham số — phải nêu rõ. |
| 4 | Chọn checkpoint & dừng sớm | v1 đạt đỉnh val rồi giảm nhẹ ở epoch 100 | Early Stopping v2 (mục 3 Phần III), dùng `best.pt` | Báo cáo epoch dừng và epoch đạt đỉnh của từng run; nêu rõ thay đổi so với v1. |
| 5 | Miền Blur | v1 không có số liệu blur | Thêm LOL-Blur (150/10/40) | Báo cáo PSNR/SSIM/LPIPS miền Blur như các miền khác; DarkIR-m (đã học toàn bộ LOL-Blur) chỉ là mốc trần tham chiếu. |

> [!NOTE]
> Mỗi so sánh dùng 2 seed và 190 ảnh test (Huawei 30, Nikon 20, Blur 40 là các miền nhỏ): khoảng tin cậy rộng, kết luận "không khác biệt" nghĩa là *không đủ bằng chứng*, không phải *chứng minh bằng nhau*.

## 4. Ba Quy Tắc Sống Còn Để Đạt Hiệu Suất Tối Đa Trên Colab

1. **Quy tắc 1: Giải nén cục bộ vào ổ SSD ảo `/content/`:**  
   Copy toàn bộ file `llie_dataset.zip` từ Google Drive sang `/content/` và giải nén tại chỗ. Tuyệt đối không đọc ảnh trực tiếp qua đường dẫn `/content/drive/MyDrive/...` (giao thức FUSE mạng làm chậm tốc độ I/O từ 20 đến 50 lần). Thư mục Drive chỉ dùng để lưu file log và trọng số mô hình `best_model.pth`.
2. **Quy tắc 2: Tự động tính toán độ chính xác hỗn hợp (AMP FP16):**  
   Kích hoạt `torch.amp.autocast('cuda')` và `torch.cuda.amp.GradScaler()`. Giảm 50% dung lượng VRAM và tăng tốc độ xử lý lên **~1.8x** trên GPU T4/L4.
3. **Quy tắc 3: Nạp dữ liệu song song đa luồng (`num_workers = 2`, `pin_memory = True`):**  
   CPU nạp trước các batch dữ liệu vào pinned memory trong khi GPU đang xử lý lan truyền ngược.

---

# PHẦN IV: PHƯƠNG PHÁP LUẬN THỐNG KÊ CHỐNG NHIỄU & THIẾT KẾ ĐỐI CHUẨN

## 1. Thách Thức Thống Kê: Quy Mô Tập Test Nhỏ & Giải Pháp Đo Lường Chặt Chẽ

### Bản chất bài toán:
Tập Test độc lập gồm 100 ảnh LOLv2-Real, 30 ảnh LSRW-Huawei, 20 ảnh LSRW-Nikon và 40 ảnh LOL-Blur (tổng $N=190$ cặp ảnh). Với kích thước mẫu này, độ biến thiên ngẫu nhiên giữa các lần khởi tạo hạt giống (random seeds) có thể tạo ra độ lệch nhất định. Do đó, việc đo đạc và báo cáo kết quả cần tuân thủ các nguyên tắc khoa học chuẩn mực:

### Ba trụ cột xử lý khoa học:

1. **Đánh giá đa hạt giống (Multi-Seed Testing):**
   - Chạy mô hình trên **2 hạt giống độc lập (seeds = 42, 123; mở rộng 999 nếu dư thời gian)**.
   - Báo cáo kết quả dưới dạng:
     $$\text{Metric} = \text{Mean} \pm \text{Std}$$
   - Tuyệt đối không cherry-pick kết quả của một lần chạy đơn lẻ.

2. **So sánh sai phân từng ảnh & Báo cáo riêng theo miền cảm biến:**
   - Tính toán sai phân $\Delta_i = \text{PSNR}_{\text{DarkIR-Edge}}(i) - \text{PSNR}_{\text{Standard U-Net}}(i)$ trên từng bức ảnh $i \in \{1, \dots, 190\}$ để quan sát trực quan độ ổn định. Phép kiểm định ghép cặp (**Paired t-test**) được tích hợp như công cụ bổ trợ tùy chọn.
   - Lập các cột báo cáo riêng biệt cho từng miền cảm biến:
     - Miền DSLR trong nhà (LOLv2-Real: 100 ảnh).
     - Miền Smartphone thực tế (LSRW-Huawei: 30 ảnh).
     - Miền Zero-shot Unseen Sensor (LSRW-Nikon: 20 ảnh) để đánh giá năng lực ngoại suy.
     - Miền Khử mờ trong tối (LOL-Blur: 40 ảnh) để đánh giá khả năng khử mờ của Decoder; đối chiếu với DarkIR-m (đã học trên toàn bộ LOL-Blur) chỉ mang tính tham chiếu trần.

3. **Chấp nhận và khai thác giá trị khoa học của Kết quả thực nghiệm:**
   - Trong nghiên cứu khoa học, mọi kết quả thực nghiệm trung thực đều có giá trị phản ánh quy luật bài toán. Nếu mô hình đạt kết quả vượt trội, ta phân tích nguyên nhân kiến trúc; nếu mức chênh lệch khiêm tốn ở một miền cảm biến cụ thể, ta phân tích sâu sắc các hạn chế vật lý (nhiễu cảm biến, dải động trắc quang) để làm rõ bản chất bài toán.

---

## 2. Hệ Thống Baselines Đối Chuẩn Bắt Buộc

Để trả lời thuyết phục câu hỏi của Hội đồng: *"Liệu kiến trúc kế thừa DarkIR có thực sự vượt trội hơn các mô hình truyền thống khi ở cùng ngân sách tài nguyên?"*, hệ thống baseline được thiết lập đa tầng:

1. **Phương pháp xử lý ảnh truyền thống (Non-learning Baselines):**
   - Gamma Correction ($\gamma = 2.2$).
   - Histogram Equalization / CLAHE (Contrast Limited Adaptive Histogram Equalization).
   - Retinex cổ điển (Single-Scale Retinex - SSR, Multi-Scale Retinex - MSR).
2. **Standard U-Net cùng ngân sách tham số (~1.9M params):**
   - Xây dựng một mạng U-Net tích chập tiêu chuẩn (Plain Conv2d + GroupNorm + ReLU, đúng như `UNetBaseline` trong notebook) có cùng độ sâu 3 tầng và số kênh cơ sở tương đương, không có FFT và không có cơ chế Attention đặc thù.
   - **Đây là mốc đối chuẩn quan trọng nhất**: Cho phép đánh giá khách quan xem các khối chuyên biệt hóa (`FreMLP`, `Di-SpAM`, `SCA`, `Gated Skip`) có mang lại ưu thế thực tế so với kiến trúc CNN thông thường khi ở cùng dung lượng tham số hay không.
   - *Lưu ý trung thực về chi phí tính toán (FLOPs):* Dù được thiết kế cùng quy mô tham số (~1.92M params), kiến trúc tích chập đặc tiêu chuẩn của `Standard U-Net` tiêu tốn nhiều phép tính không gian hơn (~18.5 GFLOPs trên patch $256 \times 256$) so với thiết kế tích chập chiều sâu tách biệt (Depthwise Separable) của `DarkIR-Edge` (8.75 GFLOPs ở patch $256 \times 256$ theo số đo v1, chưa tính FFT). Tuy nhiên ít FLOPs không đồng nghĩa nhanh hơn: ở v1, latency FP16 của Edge chậm hơn U-Net ≈2,6 lần trên T4. Báo cáo ghi nhận minh bạch Params, GFLOPs, Latency và GPU đo khi đối sánh (xem mục 2 Phần III).

---

## 3. Phân Rã Lỗi (Error Decomposition) & Phân Tích Chẩn Đoán Sâu

Để tăng tính thuyết phục học thuật mà không tốn thêm tài nguyên huấn luyện:

1. **Chẩn đoán bằng Căn chỉnh độ sáng trung bình Ground Truth (GT-mean Adjustment):**
   - Nhóm thực hiện phép chuẩn hóa độ sáng trung bình của ảnh dự đoán theo ảnh Ground Truth trước khi tính PSNR:
     $$\hat{Y}_{\text{norm}} = \hat{Y} \times \frac{\text{mean}(Y_{\text{gt}})}{\text{mean}(\hat{Y})}$$
   - > [!CAUTION]
     > Phép tính này **chỉ dùng làm công cụ chẩn đoán nội bộ (Diagnostic Tool)** để bóc tách thành phần lỗi, **tuyệt đối không đưa vào bảng kết quả chính thức**.
   - *Nguyên lý chẩn đoán:*
     - Nếu PSNR sau căn chỉnh tăng vọt ($+3 \dots 5 \text{ dB}$): Lỗi chủ yếu nằm ở việc ước lượng sai độ sáng toàn cục $\implies$ Là *gợi ý* về vai trò của nhánh Fourier FFT (`FreMLP`) và $\mathcal{L}_{\text{side}}$ — chỉ ablation bỏ FFT (mục 3 Phần III) mới cho phép kết luận; ở v1 `gap_illumination` của Edge không thấp hơn U-Net nên giả thuyết FFT chưa được ủng hộ.
     - Nếu PSNR sau căn chỉnh hầu như không đổi: Lỗi nằm ở chi tiết bề mặt, mất biên cạnh và hạt nhiễu $\implies$ Kiểm chứng vai trò của khối tích chập đa tỷ lệ `Di-SpAM` và cổng chặn nhiễu `Residual Gated Skip`.
2. **Cắt mảng cục bộ vùng tối phẳng (Flat Dark Patch Inspection):**
   - Cắt các vùng ảnh phẳng tối (như mặt bàn, bức tường đêm) để so sánh phương sai nhiễu cảm biến giữa ảnh đầu vào, baseline U-Net và mô hình đề xuất.
3. **Phân tích các trường hợp thất bại (Failure Cases Analysis):**
   - Nhóm tổng hợp các bức ảnh mô hình chưa phục hồi hoàn hảo (ví dụ: vùng bóng đèn neon bị cháy sáng cục bộ - Over-exposure, hoặc vùng bóng tối hoàn toàn không có ánh sáng phản xạ) kèm lý giải quang học.

---

# PHẦN V: CẨM NANG THỰC NGHIỆM TỪNG BƯỚC TRÊN GOOGLE COLAB

## 1. Kịch Bản Khởi Tạo Môi Trường & Giải Nén

```python
# ==============================================================
# BƯỚC 1: KẾT NỐI GOOGLE DRIVE & GIẢI NÉN VÀO ĐĨA SSD CỤC BỘ
# ==============================================================
from google.colab import drive
import os

drive.mount('/content/drive')

# Copy dữ liệu nén sang SSD ảo
!cp /content/drive/MyDrive/llie_project_v2/llie_data_clean_v2.zip /content/
!unzip -q /content/llie_data_clean_v2.zip -d /content/datasets/

# Kiểm tra thư mục dữ liệu đã sẵn sàng
print("Dữ liệu sẵn sàng tại:", os.listdir('/content/datasets'))
```

## 2. Kịch Bản Huấn Luyện Đa Seed Với Ngân Sách Bước Cố Định

```python
# ==============================================================
# BƯỚC 2: HUẤN LUYỆN MÔ HÌNH ĐỀ XUẤT DUY NHẤT DARKIR-EDGE (C=24) TRÊN D_train (2 SEEDS)
# ==============================================================
for seed in [42, 123]:
    print(f"--- Bắt đầu huấn luyện Seed {seed} ---")
    !python train.py --model darkir_edge \
                     --batch_size 8 \
                     --patch_size 256 \
                     --epochs 100 \
                     --early_stop_min_epochs 40 --early_stop_patience 15 --early_stop_min_delta 0.05 \
                     --lr 2e-4 \
                     --lr_min 1e-6 \
                     --val_interval 5 \
                     --loss_composite \
                     --use_amp \
                     --seed {seed} \
                     --save_dir /content/drive/MyDrive/llie_checkpoints/darkir_edge_seed{seed}/

# ==============================================================
# BƯỚC 3: HUẤN LUYỆN BASELINE STANDARD U-NET (~1.9M) ĐỐI CHUẨN (2 SEEDS, cùng quy tắc Early Stopping)
# ==============================================================
for seed in [42, 123]:
    !python train.py --model standard_unet \
                     --batch_size 8 \
                     --patch_size 256 \
                     --epochs 100 \
                     --early_stop_min_epochs 40 --early_stop_patience 15 --early_stop_min_delta 0.05 \
                     --lr 2e-4 \
                     --lr_min 1e-6 \
                     --val_interval 5 \
                     --use_amp \
                     --seed {seed} \
                     --save_dir /content/drive/MyDrive/llie_checkpoints/standard_unet_seed{seed}/

# ==============================================================
# BƯỚC 3b: ABLATION BỎ FFT/FreMLP (2 SEEDS) — kiểm chứng giả thuyết FFT
# ==============================================================
for seed in [42, 123]:
    !python train.py --model darkir_edge_nofft --epochs 100 \
                     --early_stop_min_epochs 40 --early_stop_patience 15 --early_stop_min_delta 0.05 \
                     --seed {seed} \
                     --save_dir /content/drive/MyDrive/llie_checkpoints/darkir_edge_nofft_seed{seed}/
```

## 3. Kịch Bản Đánh Giá Ghép Cặp (Paired Evaluation)

```bash
# Đánh giá toàn diện trên toàn bộ 190 ảnh test (gộp và tách domain) kèm Paired T-test:
python evaluate.py --model darkir_edge \
                   --checkpoint_dir /content/drive/MyDrive/llie_checkpoints/ \
                   --seeds 42,123 \
                   --test_datasets lolv2_real,lsrw_huawei,lsrw_nikon,lolblur \
                   --paired_ttest \
                   --metrics psnr,ssim,lpips \
                   --save_csv /content/drive/MyDrive/llie_results/benchmark_summary.csv
```

---

# PHẦN VI: CHIẾN LƯỢC BẢO VỆ ĐỒ ÁN TRƯỚC HỘI ĐỒNG CHẤM THI

Định hướng câu chuyện học thuật xuyên suốt của đồ án:  
> *"Chúng tôi xây dựng và đánh giá một kiến trúc CNN tinh gọn duy nhất DarkIR-Edge (~1.93M tham số), bảo toàn trọn vẹn linh hồn của DarkIR và tích hợp tinh tế 2 giải pháp vệ tinh, huấn luyện trên tập dữ liệu đa cảm biến có kiểm soát và kiểm chứng thống kê chặt chẽ."*

Dưới đây là 5 câu hỏi trọng tâm mà Hội đồng thường chất vấn và câu trả lời chuẩn mực giúp bảo vệ đồ án xuất sắc:

### Câu hỏi 1: "Tại sao nhóm không sử dụng toàn bộ các tập dữ liệu có sẵn (hàng chục nghìn ảnh) mà lại giới hạn ở mức 1.650 cặp ảnh với tập train đồng nhất 1.350 cặp?"
- **Trả lời chuẩn mực:**  
  *"Thưa Thầy/Cô, việc thiết lập tập dữ liệu huấn luyện đồng nhất $D_{\text{train}} = 1.350$ cặp ảnh (gồm 639 LOLv2-Real DSLR, 561 LSRW-Huawei Smartphone và 150 LOL-Blur) xuất phát từ 2 lý do khoa học và thực tiễn:  
  1. **Cân bằng phân phối đa cảm biến:** Đây là sự kết hợp chuẩn mực bao quát cả hai môi trường quang học thực tế (DSLR trong nhà và Smartphone ban đêm), giúp mô hình học được các đặc trưng vật lý bất biến và ngăn ngừa hiện tượng dịch chuyển cảm biến. Đồng thời, nhóm rút toàn bộ 20 ảnh LSRW-Nikon khỏi tập huấn luyện và validation để giữ làm bài kiểm tra tổng quát hóa zero-shot (unseen sensor) hoàn toàn khách quan.  
  2. **Hiệu quả thực nghiệm & độ tin cậy thống kê:** Với mô hình gọn nhẹ ~1.93M tham số, quy mô 1.350 cặp ảnh cho phép huấn luyện tối đa 100 epochs (có Early Stopping theo tập Validation, không dừng trước epoch 40) qua 2 random seeds cho mỗi mô hình trong thời gian ngắn trên GPU L4 của Google Colab, vừa đủ để mô hình hội tụ ổn định vừa bảo đảm tính nghiêm cẩn về mặt phương pháp luận thống kê mà không vượt quá hạn mức tài nguyên nghiên cứu của đồ án."*

### Câu hỏi 2: "Tại sao nhóm loại bỏ hoàn toàn bộ dữ liệu LOLv1 kinh điển mà nhiều bài báo trước đây vẫn dùng?"
- **Trả lời chuẩn mực:**  
  *"Thưa Thầy/Cô, đây là phát hiện quan trọng được chỉ ra trong công trình **DarkIR (CVPR 2025, Sec. 4.1)**: LOLv2-Real là phiên bản mở rộng trực tiếp của LOLv1. Hầu hết các khung cảnh trong LOLv1 đều bị trùng lặp góc chụp với LOLv2-Real. Nếu gộp cả hai bộ dữ liệu, mô hình sẽ mắc lỗi nghiêm trọng về **Rò rỉ dữ liệu ngầm (Hidden Data Leakage)** giữa tập Train và tập Test. Để đảm bảo tính liêm chính học thuật và độ tin cậy khoa học tuyệt đối, nhóm kiên quyết loại bỏ LOLv1 và chỉ sử dụng LOLv2-Real kết hợp LSRW."*

### Câu hỏi 3: "Mô hình của nhóm có thể so sánh trực tiếp với các bài báo hàng đầu CVPR như DarkIR hay Retinexformer không khi dữ liệu huấn luyện có sự khác biệt?"
- **Trả lời chuẩn mực:**  
  *"Thưa Thầy/Cô, nhóm thiết lập hệ quy chiếu so sánh tham chiếu (Reference Comparison) hoàn toàn minh bạch:  
  1. **Khối tham chiếu trần (Reference SOTA từ Paper gốc):** Ghi nhận đầy đủ chỉ số của các mô hình lớn (DarkIR-mt 3.32M, Restormer 26.1M, LLFormer 24.5M) được huấn luyện trên 100% dữ liệu lớn làm mốc trần lý thuyết.  
  2. **Khối thực nghiệm nội bộ có kiểm soát (Internal Controlled Experiments):** Nhóm huấn luyện các baseline trên cùng tập dữ liệu chuẩn 1.350 ảnh (CLAHE vs. Standard U-Net vs. DarkIR-Edge đề xuất) và đánh giá qua 2 random seeds với kiểm định paired t-test.  
  Đúng với tinh thần **không tuyên bố reproduce SOTA**, nhóm tập trung làm rõ hiệu năng của kiến trúc đề xuất duy nhất `DarkIR-Edge`, đánh giá liệu mô hình có mang lại cải thiện có ý nghĩa thống kê so với Standard U-Net cùng số lượng tham số hay không."*

### Câu hỏi 4: "Tại sao nhóm bắt buộc phải đưa ảnh chụp từ điện thoại (LSRW-Huawei) vào tập huấn luyện thay vì chỉ để ở tập kiểm thử?"
- **Trả lời chuẩn mực:**  
  *"Thưa Thầy/Cô, cảm biến máy ảnh DSLR lớn cho ra nhiễu Gauss/Poisson tương đối đồng nhất, trong khi cảm biến smartphone siêu nhỏ phải đẩy ISO cực cao và qua bộ nén ISP của điện thoại, tạo ra nhiễu hạt bệt và loang màu cục bộ. Nếu chỉ huấn luyện trên DSLR, mô hình sẽ bị thiên vị phân phối (Domain Bias) và thất bại khi gặp ảnh chụp đêm từ điện thoại thực tế. Việc cân bằng 561 cặp ảnh smartphone trong tập Train giúp mạng học được đồng thời cả hai phân bố nhiễu vật lý, đảm bảo khả năng ứng dụng thực tế trên thiết bị di động."*

### Câu hỏi 5: "Mô hình giữ lại nhánh biến đổi Fourier FFT (FreMLP) của DarkIR. FFT có thật sự giúp ích không, và mô hình có phù hợp thiết bị biên (Edge Devices) không?"
- **Trả lời chuẩn mực:**  
  *"Thưa Thầy/Cô, nhóm không tuyên bố FFT chắc chắn có ích. Ở v1 chẩn đoán độ sáng toàn cục không ủng hộ giả thuyết này, nên ở v2 nhóm chạy ablation bỏ FFT/FreMLP (2 seed) và báo cáo đúng kết quả: nếu khoảng tin cậy ΔPSNR (Edge − noFFT) nằm hoàn toàn trên 0 thì FFT có đóng góp đo được; nếu chứa 0 thì nhóm kết luận *chưa chứng minh được* đóng góp của FFT. Về thiết bị biên: nhóm **không** tuyên bố \"edge-friendly\". Mô hình có ít FLOPs hơn U-Net cùng số tham số, nhưng đo thực tế latency FP16 trên GPU lại chậm hơn U-Net (≈2,6 lần ở v1, T4), và `rfft2` chưa được hỗ trợ đồng đều trên NPU di động (SNPE, CoreML, TensorRT). Nhóm đo FPS/Latency trên GPU máy chủ (ghi rõ T4 hay L4) và coi việc triển khai trên thiết bị biên thật là hướng mở rộng chưa kiểm chứng."*

### Câu hỏi 6: "DarkIR-Edge có thật sự tốt hơn Standard U-Net cùng số tham số không?"
- **Trả lời chuẩn mực:**  
  *"Thưa Thầy/Cô, ở v1 câu trả lời là chưa: PSNR pooled của Edge 20.04 dB so với U-Net 20.50 dB (U-Net nhỉnh hơn, p≈0.05; U-Net khi đó chỉ chạy 1 seed). Ở v2 nhóm chạy U-Net 2 seed như Edge và kết luận theo quy tắc đã khóa trước (mục 3 Phần III): chỉ khi khoảng tin cậy ΔPSNR trên tập gộp nằm hoàn toàn trên 0 nhóm mới nói Edge tốt hơn; nếu chứa 0, nhóm nói hai mô hình không khác biệt có ý nghĩa thống kê; nếu nằm dưới 0, nhóm nói U-Net tốt hơn. Đây là đồ án môn học nên nhóm ưu tiên báo cáo trung thực hơn là chứng minh mô hình đề xuất vượt trội."*
