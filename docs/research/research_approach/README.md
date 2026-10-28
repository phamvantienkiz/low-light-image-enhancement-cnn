# Danh Mục Tài Liệu Nghiên Cứu Đề Tài (Research Approach Index)

> **Môn học:** Xử Lý Ảnh và Thị Giác Máy Tính (CS2203) - Master of Science in Computer Science, UIT - VNUHCM  
> **Đề tài:** Tăng cường chất lượng ảnh trong điều kiện thiếu sáng bằng mạng nơ-ron tích chập (Low-Light Image Enhancement Using a Lightweight CNN)

---

## Cấu Trúc Các Tài Liệu Nghiên Cứu Trọng Tâm

Hệ thống tài liệu nghiên cứu và thực nghiệm của đồ án được cấu trúc tinh gọn thành **4 tài liệu chuẩn mực**, loại bỏ hoàn toàn sự trùng lặp và phân mảnh:

```
docs/research/research_approach/
│
├── README.md                                             # (Tài liệu này) Mục lục & Tóm tắt điều hành
│
├── 01_literature_and_attention_mechanisms.md             # TẬP 1: Bối cảnh, mô hình vật lý, phân tích DarkIR (CVPR 2025),
│                                                         #        HVI, Edge-LLIE và khảo sát các cơ chế Attention U-Net.
│
├── 02_dataset_strategy_and_proposal_enhancement.md       # TẬP 2: Phân vùng 1.000 cặp ảnh (50:30):20, đánh giá phản biện
│                                                         #        hướng đi người dùng và 3 đề xuất cải tiến khoa học.
│
├── 03_technical_solution_and_colab_guide.md              # TẬP 3: Đặc tả 4 cấp độ mô hình, cẩm nang Google Colab, bảng kỳ vọng
│                                                         #        thực nghiệm và 5 câu hỏi bảo vệ trước Hội đồng UIT.
│
└── 04_darkir_cvpr2025_dataset_audit_and_preparation.md   # TẬP 4: Khảo sát đối chuẩn chi tiết bộ dữ liệu DarkIR (CVPR 2025),
                                                          #        thống kê số lượng, đặc tả cây thư mục & chuẩn bị thực nghiệm.
```

---

## Tóm Tắt Nhanh Nội Dung Từng Tập

### [Tập 1: Tổng Quan Văn Liệu, Phân Tích DarkIR (CVPR 2025) & Cơ Chế Attention](file:///E:/MyProject/low-light-image-enhancement-cnn/docs/research/research_approach/01_literature_and_attention_mechanisms.md)
- Mô hình toán học suy thoái: $\mathbf{y} = \gamma(\mathbf{x} \otimes \mathbf{k}) + \mathbf{n}$.
- Giải mã bài báo tham khảo chính **DarkIR (CVPR 2025)**: Tại sao CNN vượt trội hơn Transformer về độ gọn nhẹ và tốc độ; cơ chế Simple Gate (SG), Simplified Channel Attention (SCA) và Dilated Conv.
- Bài học từ các bài báo vệ tinh: **HVI (CVPR 2025)** về lệch màu sRGB/HSV; **Edge-LLIE (CVPRW 2024)** về rò rỉ nhiễu qua Skip Connections; **NTIRE 2024 Challenge** về độ đo PSNR/SSIM/LPIPS.
- Khảo sát các cơ chế Attention: SE-Net, CBAM, Coordinate Attention, Attention Gates và SCA.

### [Tập 2: Chiến Lược Dữ Liệu, Phân Vùng 1.000 Cặp & Đề Xuất Cải Tiến](file:///E:/MyProject/low-light-image-enhancement-cnn/docs/research/research_approach/02_dataset_strategy_and_proposal_enhancement.md)
- Cấu trúc dữ liệu: 485 cặp LOLv1 + 515 cặp LOLv2 Real $\to$ 1.000 cặp paired tuyển chọn.
- Phân chia (50:30):20: 500 ảnh Train ban đầu, 300 ảnh Cách ly dự trữ (6 khối x 50), 200 ảnh Validation cố định.
- Đánh giá phản biện hướng đi của bạn: Khẳng định 500 ảnh là chuẩn mực khoa học (tương đương quy mô của LOLv1 gốc); phương pháp tăng dần +50 ảnh giúp xây dựng biểu đồ **Sample Efficiency Curve** rất giá trị.
- 3 cải tiến then chốt: **Phân tầng cân bằng (Stratified Split)** chống lệch domain, **Đánh giá 2 tầng** (PSNR/SSIM trên Valid + NIQE trên ExDark), và **Lộ trình 3 bước** tối ưu trước khi mở khóa ảnh cách ly.

### [Tập 3: Giải Pháp Kỹ Thuật, Cẩm Nang Huấn Luyện Google Colab & Chiến Lược Báo Cáo](file:///E:/MyProject/low-light-image-enhancement-cnn/docs/research/research_approach/03_technical_solution_and_colab_guide.md)
- Bảng đối chuẩn 4 cấp độ mô hình: Model 0 (Traditional CV), Model 1 (Plain CNN ~120k params), Model 2 (Compact U-Net ~483k params), Model 3 (Lightweight Attention U-Net ~905k params).
- Hàm mất mát Composite Loss: $L_1 + 0.2 \times (1 - \text{SSIM})$.
- Hướng dẫn thực hành từng bước trên Google Colab với GPU Tesla T4 (thời gian chạy chỉ 15–20 phút/100 epochs).
- Bộ tiêu chuẩn kỳ vọng đường cong hội tụ và cẩm nang xử lý sự cố (Overfitting, Underfitting, Color Cast).
- 5 câu hỏi cốt lõi và câu trả lời chuẩn mực giúp sinh viên bảo vệ đồ án tự tin trước Hội đồng chấm thi UIT.

### [Tập 4: Khảo Sát & Chuẩn Bị Dữ Liệu Thực Nghiệm Theo Chuẩn DarkIR (CVPR 2025)](file:///E:/MyProject/low-light-image-enhancement-cnn/docs/research/research_approach/04_darkir_cvpr2025_dataset_audit_and_preparation.md)
- Khảo sát toàn diện các bộ dữ liệu từ bài báo và repository DarkIR (CVPR 2025).
- Bảng thống kê chi tiết số lượng ảnh (Train/Test/Val), phân loại cặp có tham chiếu (LOLBlur: 12k cặp, LOLv2-Real: 789 cặp, LOLv2-Synth: 1.000 cặp, LSRW: 5.650 cặp) và không cặp thế giới thực (Real-LOLBlur: 482 / 1.354 ảnh, DICM, LIME, MEF, NPE, VV: 204 ảnh, ExDark).
- Bản chất vật lý và pipeline tạo suy thoái (nhiễu cảm biến thực tế, vệt mờ chuyển động chân thực, làm tối phi tuyến).
- Đặc tả cây thư mục `DarkIR/data/datasets/` khớp 100% mã nguồn Data Loader và lệnh chạy kiểm thử `testing.py`, `testing_unpaired.py`.
- Định hướng kế thừa dữ liệu cho đề tài Lightweight CNN.
