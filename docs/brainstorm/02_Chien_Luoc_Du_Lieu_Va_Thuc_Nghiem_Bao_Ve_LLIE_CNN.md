# BÁO CÁO CHIẾN LƯỢC DỮ LIỆU, CẨM NANG THỰC NGHIỆM GOOGLE COLAB & BẢNG BẢO VỆ ĐỒ ÁN (V2 - TỐI ƯU)

> **Môn học:** Xử Lý Ảnh và Thị Giác Máy Tính (CS2203) — Master of Science in Computer Science 2026  
> **Đơn vị:** Trường Đại học Công nghệ Thông tin (UIT) - ĐHQG TP.HCM  
> **Đề tài:** Tăng cường chất lượng ảnh trong điều kiện thiếu sáng bằng mạng nơ-ron tích chập (*Low-Light Image Enhancement Using a Lightweight CNN*)  
> **Văn bản gốc tham chiếu:** Báo cáo kỹ thuật đối chuẩn DarkIR (CVPR 2025), Edge-LLIE (CVPRW 2024), HVI-CIDNet (CVPR 2025).

---

# PHẦN I: KHẢO SÁT & HỆ QUY CHIẾU THAM CHIẾU (DATASET AUDIT & REFERENCE BENCHMARK)

Để đảm bảo kết quả thực nghiệm có cơ sở khoa học vững chắc và **thiết lập hệ quy chiếu so sánh tham chiếu (Reference Comparison) với bài báo gốc DarkIR (CVPR 2025)**, đề tài thiết lập hệ thống dữ liệu và độ đo đánh giá tương đồng về mặt phân phối nhằm phục vụ báo cáo khoa học và trình bày trước Hội đồng. 

> [!IMPORTANT]
> **Lưu ý phương pháp luận:** DarkIR gốc được huấn luyện trên toàn bộ tập dữ liệu khổng lồ (hàng nghìn đến chục nghìn cặp ảnh, hàng trăm GPU-hours), trong khi đồ án môn học tập trung huấn luyện từ đầu (trained from scratch) trên tập dữ liệu tuyển chọn đồng nhất ($D_{\text{train}} = 1.200$ cặp ảnh). Vì vậy, kết quả của DarkIR gốc được sử dụng làm **mốc tham chiếu trần lý thuyết**, không tuyên bố tái lập 1-1 (No SOTA reproduction claim), mà tập trung đo lường đóng góp thực tế của kiến trúc đề xuất duy nhất `DarkIR-Edge` trong điều kiện tài nguyên giới hạn.

## 1. Bảng Tổng Hợp Chi Tiết Các Bộ Dữ Liệu & Vai Trò Trong Đồ Án

| Phân Nhóm | Tên Tập Dữ Liệu | Bản Chất Cảm Biến / Thiết Bị | Tập Train (Cặp) | Tập Val (Cặp) | Tập Test (Cặp) | Tổng Quy Mô | Chỉ Số Đánh Giá (Metrics) | Vai Trò Trong Đồ Án |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Chuẩn LLIE hiện đại** | **LOLv2-Real** | Máy ảnh DSLR chuyên nghiệp (Canon/Sony) | **639** | **50** | **100** | **789** | **PSNR, SSIM, LPIPS** | Nguồn DSLR chính (Chuẩn BMVC, $639+50+100=789$) |
| **Đa cảm biến (Smartphone & DSLM)** | **LSRW-Huawei** | Cảm biến Smartphone thực tế (Huawei P40/Mate30) | **561** | **50** | **30** | **641** | **PSNR, SSIM, LPIPS** | Nguồn smartphone chính (Trích từ kho 2.450 ảnh train) |
| | **LSRW-Nikon** | Máy ảnh Nikon chuyên nghiệp | — | — | **20** | **20** | **PSNR, SSIM, LPIPS** | **Zero-shot Unseen Sensor** (chỉ có trong Test, không qua Val) |
| **Khái quát hóa ngoài đời (In-the-wild Unpaired)** | **DICM / LIME** | Đa dạng máy ảnh ngoài trời / trong nhà | — | — | **69 + 10 = 79** | **79** | **Trực quan hóa định tính (Visual Quality)** | Đánh giá định tính ảnh thực tế không cặp |
| *(Phạm vi lược bỏ)* | *Real-LOLBlur / ExDark* | *Rung nhòe đêm / Nhãn nhận diện 12 lớp* | — | — | — | — | *NIQE, BRISQUE* | *Lược bỏ khỏi phạm vi đồ án* |

## 2. Quyết Định Khoa Học: Loại Bỏ Hoàn Toàn LOLv1 Để Triệt Tiêu Rò Rỉ Dữ Liệu

Trong bài báo **DarkIR (CVPR 2025, Section 4.1)**, các tác giả đã nêu rõ:
> *"Note that LOLv2-Real is the extended version of LOL [v1], thus we use the v2 version directly."*

- **Bản chất rủi ro:** Bộ dữ liệu LOLv1 (485 cặp train, 15 cặp eval) thực chất là tập con của LOLv2-Real. Rất nhiều khung cảnh, phòng ốc và góc chụp trong LOLv1 bị trùng lặp hoàn toàn với LOLv2-Real. Nếu gộp cả 2 bộ dữ liệu rồi phân chia ngẫu nhiên, mô hình sẽ gặp hiện tượng **Rò rỉ dữ liệu ngầm (Hidden Data Leakage)**: cùng một căn phòng xuất hiện ở tập Train từ nguồn LOLv1 nhưng lại nằm trong tập Validation/Test từ nguồn LOLv2-Real.
- **Giải pháp dứt khoát:** Loại bỏ hoàn toàn LOLv1 khỏi toàn bộ pipeline dữ liệu. Sử dụng **LOLv2-Real** làm nguồn dữ liệu DSLR chuẩn duy nhất, kết hợp với **LSRW-Huawei** làm nguồn dữ liệu smartphone thực tế.

---

# PHẦN II: CHIẾN LƯỢC DỮ LIỆU ĐỒNG NHẤT ĐA CẢM BIẾN & PHÂN VÙNG THỰC NGHIỆM ($\le 1.500$ CẶP ẢNH)

## 1. Quyết Định Gộp Dữ Liệu Huấn Luyện Đồng Nhất Cân Bằng DSLR : Smartphone

> [!IMPORTANT]
> **Định hướng chiến lược của đồ án:**  
> Nhằm tận dụng tối đa dữ liệu đã được gán nhãn cặp và chuẩn hóa, đồng thời tạo ra phân phối mẫu cân bằng tối ưu giữa máy ảnh chuyên nghiệp và camera điện thoại di động, nhóm quyết định **gộp toàn bộ dữ liệu huấn luyện đã chuẩn bị thành MỘT tập train duy nhất $D_{\text{train}} = 1.200$ cặp ảnh (gồm 639 cặp LOLv2-Real DSLR và 561 cặp LSRW-Huawei Smartphone, đạt tỷ lệ cân bằng xấp xỉ 1:1)**.

### Cơ sở khoa học và thực tiễn của quyết định:
1. **Cân bằng tối ưu giữa hai miền cảm biến DSLR : Smartphone:**  
   Việc thiết lập tỷ lệ ~53% DSLR (639 ảnh LOLv2-Real) và ~47% Smartphone (561 ảnh LSRW-Huawei trích từ kho 2.450 ảnh train) giúp mạng nơ-ron học đồng đều cả hai đặc tính suy thoái: nhiễu Gauss/Poisson mịn trên cảm biến lớn và nhiễu hạt bệt/loang màu ISP trên camera điện thoại.
2. **Khắc phục triệt để lỗi số học phân vùng LOLv2-Real:**  
   Bộ LOLv2-Real chính thức chỉ có đúng 789 cặp ảnh. Nhóm phân bổ chính xác: 639 Train + 50 Val + 100 Test = 789 cặp, giải quyết dứt điểm lỗi thiếu hụt mẫu val.
3. **Giữ sạch miền kiểm thử Unseen Sensor (Zero-Shot Domain Generalization):**  
   Tập dữ liệu **LSRW-Nikon (20 cặp ảnh)** được rút hoàn toàn khỏi tập Validation và chỉ xuất hiện duy nhất ở tập Test độc lập. Nhờ đó, đề tài có một phép thử nghiệm "sạch 100%" về năng lực tổng quát hóa sang một loại cảm biến hoàn toàn chưa từng thấy trong quá trình huấn luyện và chọn mô hình.

---

## 2. Thiết Kế Phân Vùng Tuyển Chọn Chuẩn Hóa (Tổng Quy Mô: 1.450 Cặp Ảnh)

Cấu trúc phân vùng dữ liệu được khóa cứng xuyên suốt đồ án:

```
TỔNG QUY MÔ DỮ LIỆU THỰC NGHIỆM: 1.450 CẶP ẢNH (≤ 1.500 CẶP)
│
├── 1. Tập Huấn Luyện Duy Nhất (D_train): 1.200 cặp ảnh (Tỷ lệ cân bằng ~1:1)
│      ├── 639 cặp LOLv2-Real (DSLR sensor, đa dạng ánh sáng trong nhà)
│      └── 561 cặp LSRW-Huawei (Smartphone sensor, nhiễu hạt ISP di động thực tế)
│      --> Mục tiêu: Huấn luyện mô hình DarkIR-Edge đạt điểm hội tụ vững chắc nhất.
│
├── 2. Tập Validation Cố Định (D_val): 100 cặp ảnh (Khóa cứng 100% xuyên suốt)
│      ├── 50 cặp LOLv2-Real Validation
│      └── 50 cặp LSRW-Huawei Validation
│      --> Mục tiêu: Theo dõi hàm mất mát và giám sát đường cong học tập của mô hình.
│      --> Cân bằng 1:1 (50 DSLR : 50 Smartphone), rút hoàn toàn Nikon để giữ sạch Zero-shot.
│
└── 3. Tập Benchmark Test Độc Lập Đối Chuẩn Với DarkIR (Zero Leakage):
       ├── Paired DSLR Test: LOLv2-Real Test (100 cặp chuẩn BMVC)
       ├── Paired Smartphone Test: LSRW-Huawei Test (30 cặp chuẩn R2RNet)
       ├── Paired Zero-shot Multi-Sensor Test: LSRW-Nikon Test (20 cặp unseen sensor)
       └── Unpaired Qualitative Test: DICM (69 ảnh), LIME (10 ảnh) đánh giá trực quan.
```

## 3. Nguyên Tắc Bảo Đảm Tuyệt Đối Không Rò Rỉ Dữ Liệu (Zero Data Leakage Protocol)

1. **Phân chia theo tiền tố tên file cảnh (Scene Prefix Grouping):** Các ảnh chụp cùng bối cảnh phòng ốc/góc chụp (mang chung tiền tố định danh cảnh) được phân bổ trọn vẹn vào cùng một phân vùng (Train hoặc Val), tuyệt đối không tách rời làm rò rỉ dữ liệu giữa Train và Val.
2. **Cô lập tập Test gốc:** Các tập kiểm thử chính thức của tác giả công bố (**LOLv2-Real Test 100 ảnh**, **LSRW-Huawei Test 30 ảnh**, **LSRW-Nikon Test 20 ảnh**) được giữ nguyên trạng làm tập Test cuối cùng, tuyệt đối không đưa bất kỳ ảnh nào của các tập này vào $D_{\text{train}}$ hay $D_{\text{val}}$.
3. **Lưu vết bằng Manifest (`split_manifest.json`):** Mọi đường dẫn file của từng phân vùng được lưu cố định trong file JSON kèm mã băm SHA-256 để đảm bảo tính tái lập 100% và minh bạch trước Hội đồng.

## 4. Bản Chất Smartphone vs. DSLR: Vì Sao Cần Cân Bằng Smartphone Trong Tập Train?

- **Cảm biến DSLR (LOLv2-Real):** Kích thước cảm biến lớn (Full-frame/APS-C), ống kính quang học đường kính rộng. Nhiễu trong ảnh thiếu sáng chủ yếu là nhiễu Gauss và Poisson thuần túy, hạt nhiễu mịn và đồng đều.
- **Cảm biến Smartphone (LSRW-Huawei):** Cảm biến kích thước siêu nhỏ ($1/1.28''$ đến $1/2.5''$), khẩu độ nhỏ. Tín hiệu quang học yếu buộc bộ xử lý ISP của điện thoại phải khuếch đại ISO cực cao, kèm theo thuật toán nén JPEG và bộ lọc làm mịn tự động gây hiện tượng **nhiễu hạt bệt, loang lổ màu sắc (Chroma Noise) và mất biên cạnh**.
- **Kết luận:** Việc đưa 561 cặp ảnh smartphone (~47% tập Train) tạo ra sự cân bằng lý tưởng giúp mạng học được đồng đều cả hai phân bố suy thoái vật lý, nâng cao năng lực phục hồi thực tế trên cả thiết bị di động lẫn máy ảnh chuyên nghiệp.

## 5. Chuẩn Hóa Tiền Xử Lý Dữ Liệu (Data Augmentation)

1. **Tuyệt đối KHÔNG dùng ColorJitter hay Cutout:** Tránh làm sai lệch đặc tính trắc quang vật lý của cảm biến.
2. **Bắt buộc dùng Random Crop $256 \times 256$:** Giữ nguyên tỉ lệ 1:1 của hạt nhiễu cảm biến, ngăn chặn hiện tượng làm mượt giả tạo khi dùng Resize.
3. **Phép lật ngẫu nhiên (Horizontal & Vertical Flip):** Tăng cường tính bất biến hướng sáng.
4. **Cơ chế Nhân bản ảo (Virtual Multiplier - Multi-crop):** Với mỗi ảnh gốc trong một epoch, bộ nạp dữ liệu lấy mẫu ngẫu nhiên nhiều patches $256 \times 256$ khác nhau. Mặc định thiết lập `repeat_factor = 2` ($1.200 \times 2 = 2.400$ patches/epoch) cho cấu hình huấn luyện chuẩn, tối ưu hóa mật độ cập nhật gradient mà không tốn thêm bộ nhớ đĩa.

---

# PHẦN III: PHÂN TÍCH TÀI NGUYÊN GOOGLE COLAB & TỐI ƯU HUẤN LUYỆN

## 1. Định Lượng Thời Gian Huấn Luyện & Chính Sách Ngân Sách Bước Cố Định

Mô hình mục tiêu đề xuất duy nhất (**DarkIR-Edge (C=24)**, quy mô ~1.93M tham số) kết hợp tập dữ liệu đồng nhất cân bằng $D_{\text{train}}$ (1.200 cặp ảnh) được định lượng tài nguyên chính xác:

- **Số bước cập nhật (Steps) mỗi Epoch:**  
  $$\text{Steps per Epoch} = \frac{\text{Số ảnh} \times \text{Repeat Factor}}{\text{Batch Size}} = \frac{1.200 \times 2}{8} = 300 \text{ steps/epoch}$$
- **Chiến lược Huấn luyện theo Ngân Sách Cố Định (Fixed-Budget Training):**  
  - Ngân sách cố định: **100 epochs** (tương đương chính xác 30.000 steps cập nhật gradient).
  - Bộ lập lịch Learning Rate: **CosineAnnealingLR** từ $2\times 10^{-4}$ giảm dần trơn tru về $1\times 10^{-6}$ qua toàn bộ chu kỳ 100 epochs.
  - **Quy tắc chọn Checkpoint:** Lấy **Checkpoint cuối cùng (Final Checkpoint)** sau khi lịch trình Cosine Annealing đã hoàn tất trọn vẹn và đưa Learning Rate về sát 0. Cơ chế `EarlyStopping` tạm thời được vô hiệu hóa để đảm bảo toàn bộ các lần chạy đều sử dụng đúng ngân sách bước cố định, giúp kết quả so sánh hoàn toàn khách quan.
  - **Tần suất Validation:** Tập Validation gồm 100 cặp ảnh (50 DSLR : 50 Smartphone) chỉ dùng để theo dõi đường cong suy giảm hàm mất mát (loss curve); chu kỳ kiểm tra được thiết lập **mỗi 5 epochs một lần (`val_interval = 5`)**, giúp tiết kiệm tối đa thời gian đánh giá phụ trợ trên Colab.

| Kịch bản Thực Nghiệm | Quy mô dữ liệu | Số Steps / Epoch | Tổng số Steps (100 Epochs) | Kế hoạch Seeds | Ước tính thời gian trên Colab GPU T4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DarkIR-Edge (C=24)** | **1.200 cặp ảnh** | **300 steps** | **30.000 steps** | **2 seeds** (3 nếu dư thời gian) | **Ước tính sơ bộ ~35 – 55 phút / run (đo chính xác bằng `probe_speed` 200 bước đầu)** |
| **Standard U-Net (~1.9M)**| **1.200 cặp ảnh** | **300 steps** | **30.000 steps** | **1 – 2 seeds** | **Ước tính sơ bộ ~25 – 45 phút / run (đo chính xác bằng `probe_speed` 200 bước đầu)** |

👉 **Kiểm soát thực tế:** Khi bắt đầu chạy trên Colab, tiến hành đo tốc độ xử lý thực tế (it/s) ở 200 bước đầu tiên để tính toán chính xác tổng thời gian và điều chỉnh số seed cho phù hợp với hạn mức phiên làm việc.

## 2. Ba Quy Tắc Sống Còn Để Đạt Hiệu Suất Tối Đa Trên Colab

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
Tập Test độc lập gồm 100 ảnh LOLv2-Real, 30 ảnh LSRW-Huawei và 20 ảnh LSRW-Nikon (tổng $N=150$ cặp ảnh). Với kích thước mẫu này, độ biến thiên ngẫu nhiên giữa các lần khởi tạo hạt giống (random seeds) có thể tạo ra độ lệch nhất định. Do đó, việc đo đạc và báo cáo kết quả cần tuân thủ các nguyên tắc khoa học chuẩn mực:

### Ba trụ cột xử lý khoa học:

1. **Đánh giá đa hạt giống (Multi-Seed Testing):**
   - Chạy mô hình trên **2 hạt giống độc lập (seeds = 42, 123; mở rộng 999 nếu dư thời gian)**.
   - Báo cáo kết quả dưới dạng:
     $$\text{Metric} = \text{Mean} \pm \text{Std}$$
   - Tuyệt đối không cherry-pick kết quả của một lần chạy đơn lẻ.

2. **So sánh sai phân từng ảnh & Báo cáo riêng theo miền cảm biến:**
   - Tính toán sai phân $\Delta_i = \text{PSNR}_{\text{DarkIR-Edge}}(i) - \text{PSNR}_{\text{Standard U-Net}}(i)$ trên từng bức ảnh $i \in \{1, \dots, 150\}$ để quan sát trực quan độ ổn định. Phép kiểm định ghép cặp (**Paired t-test**) được tích hợp như công cụ bổ trợ tùy chọn.
   - Lập các cột báo cáo riêng biệt cho từng miền cảm biến:
     - Miền DSLR trong nhà (LOLv2-Real: 100 ảnh).
     - Miền Smartphone thực tế (LSRW-Huawei: 30 ảnh).
     - Miền Zero-shot Unseen Sensor (LSRW-Nikon: 20 ảnh) để đánh giá năng lực ngoại suy.

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
   - Xây dựng một mạng U-Net tích chập tiêu chuẩn (Plain Conv2d + LayerNorm + LeakyReLU) có cùng độ sâu 3 tầng và số kênh cơ sở tương đương, không có FFT và không có cơ chế Attention đặc thù.
   - **Đây là mốc đối chuẩn quan trọng nhất**: Cho phép đánh giá khách quan xem các khối chuyên biệt hóa (`FreMLP`, `Di-SpAM`, `SCA`, `Gated Skip`) có mang lại ưu thế thực tế so với kiến trúc CNN thông thường khi ở cùng dung lượng tham số hay không.
   - *Lưu ý trung thực về chi phí tính toán (FLOPs):* Dù được thiết kế cùng quy mô tham số (~1.92M params), kiến trúc tích chập đặc tiêu chuẩn của `Standard U-Net` tiêu tốn nhiều phép tính không gian hơn (~18.5 GFLOPs trên patch $256 \times 256$) so với thiết kế tích chập chiều sâu tách biệt (Depthwise Separable) của `DarkIR-Edge` (~4.3 GFLOPs phần không gian, chưa tính FFT). Báo cáo sẽ ghi nhận minh bạch cả hai chỉ số Params và GFLOPs/Latency khi đối sánh.

---

## 3. Phân Rã Lỗi (Error Decomposition) & Phân Tích Chẩn Đoán Sâu

Để tăng tính thuyết phục học thuật mà không tốn thêm tài nguyên huấn luyện:

1. **Chẩn đoán bằng Căn chỉnh độ sáng trung bình Ground Truth (GT-mean Adjustment):**
   - Nhóm thực hiện phép chuẩn hóa độ sáng trung bình của ảnh dự đoán theo ảnh Ground Truth trước khi tính PSNR:
     $$\hat{Y}_{\text{norm}} = \hat{Y} \times \frac{\text{mean}(Y_{\text{gt}})}{\text{mean}(\hat{Y})}$$
   - > [!CAUTION]
     > Phép tính này **chỉ dùng làm công cụ chẩn đoán nội bộ (Diagnostic Tool)** để bóc tách thành phần lỗi, **tuyệt đối không đưa vào bảng kết quả chính thức**.
   - *Nguyên lý chẩn đoán:*
     - Nếu PSNR sau căn chỉnh tăng vọt ($+3 \dots 5 \text{ dB}$): Lỗi chủ yếu nằm ở việc ước lượng sai độ sáng toàn cục $\implies$ Kiểm chứng trực tiếp vai trò của nhánh tần số Fourier FFT (`FreMLP`) và hàm mất mát $\mathcal{L}_{\text{side}}$.
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
!cp /content/drive/MyDrive/llie_data_clean.zip /content/
!unzip -q /content/llie_data_clean.zip -d /content/datasets/

# Kiểm tra thư mục dữ liệu đã sẵn sàng
print("Dữ liệu sẵn sàng tại:", os.listdir('/content/datasets'))
```

## 2. Kịch Bản Huấn Luyện Đa Seed Với Ngân Sách Bước Cố Định

```python
# ==============================================================
# BƯỚC 2: HUẤN LUYỆN MÔ HÌNH ĐỀ XUẤT DUY NHẤT DARKIR-EDGE (C=24) TRÊN D_train (2-3 SEEDS)
# ==============================================================
for seed in [42, 123]:
    print(f"--- Bắt đầu huấn luyện Seed {seed} ---")
    !python train.py --model darkir_edge \
                     --batch_size 8 \
                     --patch_size 256 \
                     --epochs 100 \
                     --lr 2e-4 \
                     --lr_min 1e-6 \
                     --val_interval 5 \
                     --loss_composite \
                     --use_amp \
                     --seed {seed} \
                     --save_dir /content/drive/MyDrive/llie_checkpoints/darkir_edge_seed{seed}/

# ==============================================================
# BƯỚC 3: HUẤN LUYỆN BASELINE STANDARD U-NET (~1.9M) ĐỐI CHUẨN (1 SEED)
# ==============================================================
!python train.py --model standard_unet \
                 --batch_size 8 \
                 --patch_size 256 \
                 --epochs 100 \
                 --lr 2e-4 \
                 --lr_min 1e-6 \
                 --val_interval 5 \
                 --use_amp \
                 --seed 42 \
                 --save_dir /content/drive/MyDrive/llie_checkpoints/standard_unet_seed42/
```

## 3. Kịch Bản Đánh Giá Ghép Cặp (Paired Evaluation)

```bash
# Đánh giá toàn diện trên toàn bộ 150 ảnh test (gộp và tách domain) kèm Paired T-test:
python evaluate.py --model darkir_edge \
                   --checkpoint_dir /content/drive/MyDrive/llie_checkpoints/ \
                   --seeds 42,123,999 \
                   --test_datasets lolv2_real,lsrw_huawei,lsrw_nikon \
                   --paired_ttest \
                   --metrics psnr,ssim,lpips \
                   --save_csv /content/drive/MyDrive/llie_results/benchmark_summary.csv
```

---

# PHẦN VI: CHIẾN LƯỢC BẢO VỆ ĐỒ ÁN TRƯỚC HỘI ĐỒNG CHẤM THI

Định hướng câu chuyện học thuật xuyên suốt của đồ án:  
> *"Chúng tôi xây dựng và đánh giá một kiến trúc CNN tinh gọn duy nhất DarkIR-Edge (~1.93M tham số), bảo toàn trọn vẹn linh hồn của DarkIR và tích hợp tinh tế 2 giải pháp vệ tinh, huấn luyện trên tập dữ liệu đa cảm biến có kiểm soát và kiểm chứng thống kê chặt chẽ."*

Dưới đây là 5 câu hỏi trọng tâm mà Hội đồng thường chất vấn và câu trả lời chuẩn mực giúp bảo vệ đồ án xuất sắc:

### Câu hỏi 1: "Tại sao nhóm không sử dụng toàn bộ các tập dữ liệu có sẵn (hàng chục nghìn ảnh) mà lại giới hạn ở mức 1.450 cặp ảnh với tập train đồng nhất 1.200 cặp?"
- **Trả lời chuẩn mực:**  
  *"Thưa Thầy/Cô, việc thiết lập tập dữ liệu huấn luyện đồng nhất $D_{\text{train}} = 1.200$ cặp ảnh (gồm 639 LOLv2-Real DSLR và 561 LSRW-Huawei Smartphone) xuất phát từ 2 lý do khoa học và thực tiễn:  
  1. **Cân bằng phân phối đa cảm biến:** Đây là sự kết hợp chuẩn mực bao quát cả hai môi trường quang học thực tế (DSLR trong nhà và Smartphone ban đêm), giúp mô hình học được các đặc trưng vật lý bất biến và ngăn ngừa hiện tượng dịch chuyển cảm biến. Đồng thời, nhóm rút toàn bộ 20 ảnh LSRW-Nikon khỏi tập huấn luyện và validation để giữ làm bài kiểm tra tổng quát hóa zero-shot (unseen sensor) hoàn toàn khách quan.  
  2. **Hiệu quả thực nghiệm & độ tin cậy thống kê:** Với mô hình gọn nhẹ ~1.93M tham số, quy mô 1.200 cặp ảnh cho phép huấn luyện trọn vẹn 100 epochs (30.000 steps) qua 2-3 random seeds chỉ trong khoảng 1.5 – 2 giờ trên GPU Google Colab, vừa đủ để mô hình hội tụ ổn định vừa bảo đảm tính nghiêm cẩn về mặt phương pháp luận thống kê mà không vượt quá hạn mức tài nguyên nghiên cứu của đồ án."*

### Câu hỏi 2: "Tại sao nhóm loại bỏ hoàn toàn bộ dữ liệu LOLv1 kinh điển mà nhiều bài báo trước đây vẫn dùng?"
- **Trả lời chuẩn mực:**  
  *"Thưa Thầy/Cô, đây là phát hiện quan trọng được chỉ ra trong công trình **DarkIR (CVPR 2025, Sec. 4.1)**: LOLv2-Real là phiên bản mở rộng trực tiếp của LOLv1. Hầu hết các khung cảnh trong LOLv1 đều bị trùng lặp góc chụp với LOLv2-Real. Nếu gộp cả hai bộ dữ liệu, mô hình sẽ mắc lỗi nghiêm trọng về **Rò rỉ dữ liệu ngầm (Hidden Data Leakage)** giữa tập Train và tập Test. Để đảm bảo tính liêm chính học thuật và độ tin cậy khoa học tuyệt đối, nhóm kiên quyết loại bỏ LOLv1 và chỉ sử dụng LOLv2-Real kết hợp LSRW."*

### Câu hỏi 3: "Mô hình của nhóm có thể so sánh trực tiếp với các bài báo hàng đầu CVPR như DarkIR hay Retinexformer không khi dữ liệu huấn luyện có sự khác biệt?"
- **Trả lời chuẩn mực:**  
  *"Thưa Thầy/Cô, nhóm thiết lập hệ quy chiếu so sánh tham chiếu (Reference Comparison) hoàn toàn minh bạch:  
  1. **Khối tham chiếu trần (Reference SOTA từ Paper gốc):** Ghi nhận đầy đủ chỉ số của các mô hình lớn (DarkIR-mt 3.32M, Restormer 26.1M, LLFormer 24.5M) được huấn luyện trên 100% dữ liệu lớn làm mốc trần lý thuyết.  
  2. **Khối thực nghiệm nội bộ có kiểm soát (Internal Controlled Experiments):** Nhóm huấn luyện các baseline trên cùng tập dữ liệu chuẩn 1.200 ảnh (CLAHE vs. Standard U-Net vs. DarkIR-Edge đề xuất) và đánh giá qua 3 random seeds với kiểm định paired t-test.  
  Đúng với tinh thần **không tuyên bố reproduce SOTA**, nhóm tập trung làm rõ hiệu năng của kiến trúc đề xuất duy nhất `DarkIR-Edge`, đánh giá liệu mô hình có mang lại cải thiện có ý nghĩa thống kê so với Standard U-Net cùng số lượng tham số hay không."*

### Câu hỏi 4: "Tại sao nhóm bắt buộc phải đưa ảnh chụp từ điện thoại (LSRW-Huawei) vào tập huấn luyện thay vì chỉ để ở tập kiểm thử?"
- **Trả lời chuẩn mực:**  
  *"Thưa Thầy/Cô, cảm biến máy ảnh DSLR lớn cho ra nhiễu Gauss/Poisson tương đối đồng nhất, trong khi cảm biến smartphone siêu nhỏ phải đẩy ISO cực cao và qua bộ nén ISP của điện thoại, tạo ra nhiễu hạt bệt và loang màu cục bộ. Nếu chỉ huấn luyện trên DSLR, mô hình sẽ bị thiên vị phân phối (Domain Bias) và thất bại khi gặp ảnh chụp đêm từ điện thoại thực tế. Việc cân bằng 561 cặp ảnh smartphone trong tập Train giúp mạng học được đồng thời cả hai phân bố nhiễu vật lý, đảm bảo khả năng ứng dụng thực tế trên thiết bị di động."*

### Câu hỏi 5: "Mô hình giữ lại nhánh biến đổi Fourier FFT (FreMLP) của DarkIR nhưng tuyên bố hướng tới thiết bị biên (Edge Devices). Liệu điều này có mâu thuẫn khi FFT phức hỗ trợ kém trên NPU?"
- **Trả lời chuẩn mực:**  
  *"Thưa Thầy/Cô, nhóm đã nhận diện rõ giới hạn phần cứng này: phép biến đổi Fourier phức (`rfft2`) hiện có độ hỗ trợ chưa đồng đều trên các runtime NPU di động (như SNPE, CoreML, TensorRT). Do đó, trong đồ án này, nhóm bảo toàn nhánh FFT như một thành phần kiến trúc cốt lõi của DarkIR giúp nâng sáng toàn cục mượt mà, đồng thời đo đạc thông lượng thực tế (FPS, Latency) trực tiếp trên GPU máy chủ (NVIDIA T4). Các phương án tối ưu hóa sâu hơn (như xấp xỉ FFT bằng tích chập không gian rời rạc để hỗ trợ NPU chuyên dụng 100%) được đề xuất như hướng nghiên cứu mở rộng tiếp theo."*
