# TẬP 4: Khảo Sát & Chuẩn Bị Dữ Liệu Thực Nghiệm Theo Chuẩn DarkIR (CVPR 2025)

> **Môn học:** Xử Lý Ảnh và Thị Giác Máy Tính (CS2203) - Master of Science in Computer Science, UIT - VNUHCM  
> **Tham chiếu cốt lõi:**  
> - Paper: *DarkIR: Robust Low-Light Image Restoration* (CVPR 2025) - Daniel Feijoo, Juan C. Benito, Alvaro Garcia, Marcos V. Conde.  
> - Repository tác giả: [DarkIR/](file:///E:/MyProject/low-light-image-enhancement-cnn/DarkIR)  
> - OCR Extraction: [docs/temp/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper/batches/](file:///E:/MyProject/low-light-image-enhancement-cnn/docs/temp/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper/batches)

---

## 1. Mục Đích & Bối Cảnh

Tài liệu này ghi lại chi tiết toàn bộ các bộ dữ liệu được công bố trong bài báo **DarkIR (CVPR 2025)** và hiện diện trong mã nguồn chính thức của tác giả tại [DarkIR/](file:///E:/MyProject/low-light-image-enhancement-cnn/DarkIR). 

Mục tiêu nhằm:
1. Xác định chính xác quy mô số lượng ảnh (Train, Test, Val), bản chất suy thoái (Degradation) và nguồn gốc từng tập dữ liệu.
2. Chuẩn hóa cấu trúc cây thư mục (Directory Hierarchy) khớp 100% với các Data Loaders trong [DarkIR/data/dataset_reader/](file:///E:/MyProject/low-light-image-enhancement-cnn/DarkIR/data/dataset_reader) và cấu hình YAML trong [DarkIR/options/test/](file:///E:/MyProject/low-light-image-enhancement-cnn/DarkIR/options/test).
3. Định hướng phân loại và kế thừa dữ liệu cho thực nghiệm mô hình mạng nơ-ron tích chập gọn nhẹ (Lightweight CNN) của đề tài.

---

## 2. Bảng Đối Chuẩn Toàn Diện Các Bộ Dữ Liệu

Bài báo DarkIR tiếp cận bài toán khôi phục ảnh thiếu sáng đa nhiệm (**All-in-One Multi-Task Low-Light Restoration**: Tăng sáng + Khử mờ chuyển động + Khử nhiễu cảm biến). Do đó, hệ thống dữ liệu được chia thành hai nhóm chính: **Dữ liệu có cặp (Paired)** để huấn luyện/đánh giá có tham chiếu, và **Dữ liệu không cặp (Unpaired)** để đánh giá độ tổng quát hóa ngoài đời thực.

| Phân Nhóm | Tên Tập Dữ Liệu | Loại Dữ Liệu | Tập Train (Cặp/Ảnh) | Tập Test/Val (Cặp/Ảnh) | Tổng Số Lượng | Chỉ Số Đánh Giá (Metrics) | Nguồn Gốc Dữ Liệu |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Trọng tâm (Joint LLIE + Blur + Noise)** | **LOL-Blur** | Cặp (Realistic Synthetic) | **10.200** | **1.800** | **12.000** | PSNR, SSIM, LPIPS | [LEDNet (ECCV 2022)](https://github.com/sczhou/LEDNet) |
| **Chuẩn LLIE hiện đại** | **LOLv2-Real** | Cặp (Real-world) | **689** | **100** | **789** | PSNR, SSIM | [Yang et al. (BMVC 2021)](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view) |
| | **LOLv2-Synthetic** | Cặp (Synthetic) | **900** | **100** | **1.000** | PSNR, SSIM | [Yang et al. (BMVC 2021)](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view) |
| **Đa cảm biến (DSLM & Smartphone)** | **LSRW-Nikon** | Cặp (Real-world) | **3.150** | **20** | **3.170** | PSNR, SSIM | [R2RNet (2023)](https://github.com/JianghaiSCU/R2RNet) |
| | **LSRW-Huawei** | Cặp (Real-world) | **2.450** | **30** | **2.480** | PSNR, SSIM | [R2RNet (2023)](https://github.com/JianghaiSCU/R2RNet) |
| **Chuẩn LLIE gốc** | **LOL (v1)** | Cặp (Real-world) | **485** | **15** | **500** | PSNR, SSIM | [Wei et al. (BMVC 2018)](https://daooshee.github.io/BMVC2018website/) |
| **Thực tế đêm mờ nhòe (Unpaired)** | **Real-LOLBlur** *(RealBlur-Night)* | Không cặp (Real handheld blur) | — | **482** *(Bài báo)* / **1.354** *(LEDNet full)* | **482 / 1.354** | MUSIQ, NRQM, NIQE | Trích từ [RealBlur (ECCV 2020)](https://github.com/sczhou/LEDNet) |
| **Khái quát hóa ngoài đời thực (In-the-wild)** | **DICM** | Không cặp (Real-world) | — | **69** | **69** | BRISQUE, NIQE | [Benchmark Drive Link](https://drive.google.com/drive/folders/0B_FjaR958nw_djVQanJqeEhUM1k?usp=sharing) |
| | **LIME** | Không cặp (Real-world) | — | **10** | **10** | BRISQUE, NIQE | // |
| | **MEF** | Không cặp (Real-world) | — | **17** | **17** | BRISQUE, NIQE | // |
| | **NPE** | Không cặp (Real-world) | — | **84** | **84** | BRISQUE, NIQE | // |
| | **VV** | Không cặp (Real-world) | — | **24** | **24** | BRISQUE, NIQE | // |
| **Thị giác nâng cao (Classification)** | **ExDark** | Không cặp / Gán nhãn | — | Lọc split `Test = 3` | 7.363 ảnh gốc | BRISQUE, NIQE, Accuracy | [ExDark Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) |

---

## 3. Đặc Điểm Chi Tiết & Bản Chất Kỹ Thuật Từng Dataset

### 3.1. LOL-Blur Dataset (Tập Huấn Luyện Cốt Lõi)
- **Mô tả bài báo**: Là tập dữ liệu tổng hợp quy mô lớn được xây dựng bằng quy trình mô phỏng vật lý chân thực từ **LEDNet** (ECCV 2022).
- **Quy trình tổng hợp suy thoái**:
  1. Thu thập video tốc độ khung hình cao (240 fps) ở các bối cảnh ban đêm thực tế.
  2. Lấy tích lũy trung bình nhiều khung hình liên tiếp để mô phỏng vệt mờ chuyển động tự nhiên (*Realistic Motion Blur*).
  3. Làm tối ảnh bằng phương pháp phi tuyến tính **EC-Zero-DCE** (biến thể nâng cấp của Zero-DCE).
  4. Bổ sung **nhiễu cảm biến thực tế (Real Sensor Noise)**.
- **Quy mô**: 10.200 cặp Train và 1.800 cặp Test (tổng cộng 12.000 cặp ảnh).
- **Ý nghĩa**: Giúp mô hình giải quyết đồng thời 3 bài toán: tăng sáng, khử mờ và khử nhiễu trong 1 lần suy luận duy nhất (End-to-End All-in-One).

### 3.2. LOLv2 (Real & Synthetic)
- **LOLv2-Real**: Gồm **689 cặp ảnh Train** và **100 cặp ảnh Test** chụp trực tiếp trong thực tế bằng cách thay đổi độ phơi sáng và ISO của máy ảnh. Đây là phiên bản nâng cấp, loại bỏ các lỗi căn chỉnh khung hình (misalignment) và mở rộng quy mô so với LOLv1 gốc.
- **LOLv2-Synthetic**: Gồm **900 cặp ảnh Train** và **100 cặp ảnh Test**, được tổng hợp bằng cách phân rã ánh sáng và áp dụng mô hình toán học chiếu sáng để tạo ảnh tối nhân tạo từ ảnh sáng chuẩn.
- **Ý nghĩa trong DarkIR**: DarkIR sử dụng LOLv2 để huấn luyện mô hình đa nhiệm **DarkIR-mt** (học chung LOLBlur + LOLv2 + LSRW), đạt PSNR **23.87 dB** trên LOLv2-Real và **25.54 dB** trên LOLv2-Synth (vượt trội RetinexFormer và SNR-Net).

### 3.3. LSRW (Low-Light Scholarly Research Wild Dataset)
- **LSRW-Nikon**: 3.150 cặp Train, 20 cặp Test (chụp bằng máy ảnh kỹ thuật số không gương lật Nikon DSLM).
- **LSRW-Huawei**: 2.450 cặp Train, 30 cặp Test (chụp bằng điện thoại thông minh Huawei).
- **Ý nghĩa**: Đại diện cho 2 dải quang học trái ngược: cảm biến lớn chất lượng cao (Nikon) vs cảm biến nhỏ chịu xử lý ISP nén gắt gao (Huawei). Giúp đánh giá khả năng thích ứng của mạng trên nhiều loại thiết bị phần cứng khác nhau.

### 3.4. Real-LOLBlur / RealBlur-Night (Đánh Giá Không Cặp Thực Tế)
- **Bản chất**: Trích xuất từ tập benchmark khử mờ thế giới thực **RealBlur** (Rim et al., ECCV 2020).
- **Số lượng**: 
  - Trong nội dung bài báo DarkIR (Mục 4): Nhóm tác giả chọn lọc kỹ **482 ảnh** ban đêm mờ nhòe thực tế do rung tay (handheld camera shake).
  - Trong kho lưu trữ LEDNet / README DarkIR: Bản phân phối Real-LOLBlur đầy đủ có **1.354 ảnh**.
- **Đặc trưng**: Hoàn toàn không có ảnh Ground Truth vì chụp tự do ngoài phố đêm.
- **Phương pháp đánh giá**: Sử dụng các độ đo đánh giá chất lượng ảnh không tham chiếu (No-Reference Image Quality Assessment - NR-IQA):
  - **MUSIQ** (chạy trên checkpoint KonIQ-10k tập trung vào độ tương phản màu sắc và độ sắc nét).
  - **NRQM** (Naturalness Image Quality Evaluator).
  - **NIQE** (Natural Image Quality Evaluator).

### 3.5. Nhóm 5 Benchmark LLIE Không Cặp (DICM, LIME, MEF, NPE, VV)
- **Quy mô**: DICM (69 ảnh), LIME (10 ảnh), MEF (17 ảnh), NPE (84 ảnh), VV (24 ảnh) — Tổng cộng **204 ảnh**.
- **Ý nghĩa**: Là các bộ dữ liệu chuẩn mực suốt nhiều năm qua của cộng đồng LLIE để kiểm tra tính tổng quát hóa không giám sát (Zero-shot evaluation), đo bằng **BRISQUE** và **NIQE**.

---

## 4. Đặc Tả Cấu Trúc Thư Mục Chuẩn Bị Cho Thực Nghiệm

Để mã nguồn trong [DarkIR/testing.py](file:///E:/MyProject/low-light-image-enhancement-cnn/DarkIR/testing.py) và [DarkIR/testing_unpaired.py](file:///E:/MyProject/low-light-image-enhancement-cnn/DarkIR/testing_unpaired.py) có thể nạp dữ liệu trơn tru mà không cần sửa code, dữ liệu cần được tổ chức đúng cấu trúc cây thư mục dưới đây tại `DarkIR/data/datasets/`:

```text
DarkIR/
└── data/
    └── datasets/
        │
        ├── LOLBlur/                                  # Dữ liệu LOL-Blur (từ LEDNet)
        │   ├── train/
        │   │   ├── low_blur_noise/                   # 10.200 ảnh mờ tối + nhiễu
        │   │   └── high_sharp_scaled/                # 10.200 ảnh rõ sáng chuẩn
        │   └── test/
        │       ├── low_blur_noise/                   # 1.800 ảnh test đầu vào
        │       └── high_sharp_scaled/                # 1.800 ảnh test đối chứng
        │
        ├── LOL-v2/                                   # Dữ liệu LOLv2
        │   ├── Real_captured/
        │   │   ├── train/ {Low/, Normal/}            # 689 cặp train
        │   │   └── test/  {Low/, Normal/}            # 100 cặp test
        │   └── Synthetic/
        │       ├── train/ {Low/, Normal/}            # 900 cặp train
        │       └── test/  {Low/, Normal/}            # 100 cặp test
        │
        ├── Low_Light_Enhancement_Datasets/
        │   └── LSRW_/                                # Dữ liệu LSRW (từ R2RNet)
        │       ├── Train/
        │       │   ├── Huawei/ {low/, high/}         # 2.450 cặp
        │       │   └── Nikon/  {low/, high/}         # 3.150 cặp
        │       └── Eval/
        │           ├── Huawei/ {low/, high/}         # 30 cặp
        │           └── Nikon/  {low/, high/}         # 20 cặp
        │
        ├── RealBlur-Night/                           # Dữ liệu Real-LOLBlur
        │   └── realblur_dataset_test/                # Chứa 482 hoặc 1.354 ảnh đêm không cặp
        │
        ├── DICM/                                     # 69 ảnh (*.jpg / *.png)
        ├── LIME/                                     # 10 ảnh
        ├── MEF/                                      # 17 ảnh
        ├── NPE/                                      # 84 ảnh
        └── VV/                                       # 24 ảnh
```

---

## 5. Hướng Dẫn Kích Hoạt Thực Nghiệm Trên Codebase DarkIR

### 5.1. File cấu hình tương ứng trong `options/test/`

- **Kiểm thử LOLBlur**: `python testing.py -p ./options/test/LOLBlur.yml`
- **Kiểm thử All-in-One LOLv2**: `python testing.py -p ./options/test/AllLOL.yml`
- **Kiểm thử đa thiết bị LSRW**: `python testing.py -p ./options/test/v2real_lsrw.yml`
- **Kiểm thử Real-LOLBlur (No-Reference)**: `python testing_unpaired.py -p ./options/test/RealBlur_Night.yml`
- **Kiểm thử 5 tập Unpaired**: `python testing_unpaired.py -p ./options/test/DICM.yml` (hoặc `LIME.yml`, `MEF.yml`, `NPE.yml`, `VV.yml`)

### 5.2. Tải Checkpoints trọng số chính thức
Tác giả công bố các checkpoint được huấn luyện sẵn:
- `DarkIR_32width.pt` (DarkIR-m: 3.31M params, 7.25 GMACs).
- `DarkIR_64width.pt` (DarkIR-l: 12.96M params, 27.19 GMACs).
- `DarkIR_1k_cr_mt.pt` (DarkIR-mt: Huấn luyện đa nhiệm kết hợp LOLBlur + LOLv2 + LSRW).

---

## 6. Định Hướng Kế Thừa Cho Đề Tài Nghiên Cứu Cao Học / Đồ Án

Đối với đề tài xây dựng **Mạng nơ-ron tích chập gọn nhẹ (Lightweight CNN)** của chúng ta (ngân sách tham số $\le 1\text{M}$ params, tài nguyên huấn luyện Google Colab T4):

1. **Dataset cốt lõi để huấn luyện (Core Training Set)**:
   - Kế thừa theo chiến lược đã xây dựng tại [Tập 2: 02_dataset_strategy_and_proposal_enhancement.md](file:///E:/MyProject/low-light-image-enhancement-cnn/docs/research/research_approach/02_dataset_strategy_and_proposal_enhancement.md): Sử dụng tập **1.000 cặp ảnh tuyển chọn** (485 cặp LOLv1 + 515 cặp LOLv2-Real) với phân chia `(500 Train : 300 Reserved) : 200 Val`.
   - Quy mô này vừa khít với năng lực phần cứng GPU và đảm bảo tính hội tụ nhanh, kiểm soát chặt chẽ hiện tượng Overfitting.

2. **Benchmark mở rộng để kiểm tra độ bền vững (Robustness Testing)**:
   - Dùng **100 ảnh từ tập test của LOLv2-Real** để so sánh trực tiếp chỉ số PSNR/SSIM với các bảng đối chuẩn của DarkIR.
   - Dùng **tập test của Real-LOLBlur (RealBlur-Night)** và các tập **DICM, LIME** làm tập kiểm thử ngoài miền (Out-of-Distribution - OOD) để đánh giá độ bền của mô hình gọn nhẹ khi gặp nhiễu và ánh sáng phức tạp ngoài đời thực bằng chỉ số NIQE/BRISQUE.
