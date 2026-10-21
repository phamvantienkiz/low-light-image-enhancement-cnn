# 03. Giải Pháp Kỹ Thuật, Cẩm Nang Huấn Luyện Google Colab & Chiến Lược Báo Cáo

> **Môn học:** CS2203 - Xử Lý Ảnh và Thị Giác Máy Tính  
> **Chương trình:** Master of Science in Computer Science, UIT - VNUHCM  
> **Đề tài:** Low-Light Image Enhancement Using a Lightweight CNN  
> **Mục tiêu:** Đặc tả 4 cấp độ mô hình, hướng dẫn chi tiết quy trình chạy Google Colab, bảng kỳ vọng thực nghiệm và 5 câu hỏi bảo vệ đồ án trước Hội đồng UIT.

---

## 1. Đặc Tả 4 Cấp Độ Mô Hình Thực Nghiệm

Để bài báo cáo và slide thuyết trình có chiều sâu học thuật vượt trội, ta xây dựng 4 cấp độ phương pháp từ cơ bản đến nâng cao để so sánh đối chuẩn:

| Cấp độ | Phương pháp | Cơ chế chính | Số tham số | Vai trò trong bài báo cáo |
| :--- | :--- | :--- | :--- | :--- |
| **Model 0** | **Traditional CV (Gamma & CLAHE)** | Hàm mũ phi tuyến $I^\gamma$ và cân bằng lược đồ cục bộ trên kênh độ chói $Y$ | 0 params | Điểm đối chuẩn truyền thống cơ bản nhất |
| **Model 1** | **Plain CNN (CNN phẳng 5 tầng)** | 5 lớp tích chập tuần tự $3\times3$, LeakyReLU, không downsampling, không skip | ~120.000 params | Minh chứng hạn chế của trường tiếp nhận hẹp và hiện tượng lệch màu khi thiếu cấu trúc U-Net |
| **Model 2** | **Compact U-Net (U-Net rút gọn)** | Cấu trúc Encoder-Decoder 4 tầng ($16 \to 32 \to 64 \to 128$) với Skip Connections thô | ~483.000 params | Chứng minh vai trò sống còn của Skip Connections trong việc bảo toàn chi tiết biên cạnh |
| **Model 3** | **Lightweight Attention U-Net (Mô hình chính)** | Kế thừa DarkIR (CVPR 2025) và Edge-LLIE (CVPRW 2024):<br>1. Simplified Channel Attention (SCA)<br>2. Gated Skip Connection lọc nhiễu hạt<br>3. Dilated Bottleneck mở rộng tầm nhìn<br>4. Global Residual Learning | ~905.000 params | **Giải pháp trọng tâm của đồ án**: Tối ưu hiệu quả mẫu, hội tụ nhanh với 500 ảnh, đạt độ sắc nét cao |

---

## 2. Thiết Lập Siêu Tham Số & Hàm Mất Mát (Loss Functions)

### 2.1. Bảng siêu tham số chuẩn (Standard Hyperparameters)
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$, eps=$10^{-8}$).
- **Learning Rate:** Khởi tạo $2 \times 10^{-4}$, kết hợp bộ lập lịch `CosineAnnealingLR` giảm về $\eta_{\min} = 1 \times 10^{-6}$ ở cuối quá trình.
- **Batch Size:** 8 (chạy vừa vặn trong VRAM Colab T4 15GB).
- **Patch Size:** $256 \times 256$ (Random Crop để nạp vừa GPU và giữ nguyên cấu trúc hạt nhiễu cảm biến).
- **Epochs:** 100 epochs (với 250 ảnh mặc định ban đầu, mỗi epoch ~31 steps; 100 epochs ~3.100 steps; thời gian chạy chỉ ~7–10 phút trên Colab T4).

### 2.2. Hàm mất mát (Composite Loss)
$$\mathcal{L}_{\text{composite}} = L_1 + \lambda_{\text{ssim}} \cdot \left(1 - \text{SSIM}(\hat{I}, I_{GT})\right)$$
- **$L_1$ Pixel Loss:** Phạt tuyến tính sai số màu, không gây nhòe mượt (oversmoothing) như $L_2$ (MSE).
- **SSIM Loss ($\lambda_{\text{ssim}} = 0.2$):** Tính vi phân qua cửa sổ Gaussian 11x11, hướng dẫn mô hình tối ưu độ tương phản cục bộ và cấu trúc biên cạnh.

---

## 3. Cẩm Nang Huấn Luyện Trên Google Colab Từng Bước

Toàn bộ quá trình huấn luyện diễn ra trên môi trường **Google Colab miễn phí (Tesla T4 GPU)**:

### Bước 1: Chuẩn bị dữ liệu và đẩy lên Google Drive
1. Nén thư mục mã nguồn `src/`, file manifest `docs/ai/split_manifest.json` và bộ dataset thành file `.zip`.
2. Đẩy file zip lên Google Drive cá nhân (~1GB lưu trữ).

### Bước 2: Chạy các ô lệnh trong Colab Notebook
```python
# 1. Kết nối Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Giải nén dữ liệu vào đĩa ảo cục bộ Colab (để I/O nhanh nhất)
!cp /content/drive/MyDrive/llie_project.zip /content/
!unzip -q /content/llie_project.zip -d /content/llie/
%cd /content/llie

# 3. Huấn luyện Mô hình chính (Attention U-Net) trên 250 ảnh ban đầu
!python train.py --model attn_unet \
                 --train_mode initial_250 \
                 --batch_size 8 \
                 --epochs 100 \
                 --lr 2e-4 \
                 --save_dir /content/drive/MyDrive/llie_checkpoints/

# 4. Huấn luyện các baseline đối chuẩn để lấy số liệu so sánh
!python train.py --model compact_unet --train_mode initial_250 --epochs 100 --save_dir /content/drive/MyDrive/llie_checkpoints/
!python train.py --model plain_cnn --train_mode initial_250 --epochs 100 --save_dir /content/drive/MyDrive/llie_checkpoints/

# 5. Thử nghiệm mở rộng dữ liệu (+50 ảnh từ khối cách ly)
!python train.py --model attn_unet --train_mode extended --quarantine_slices slice_1 --epochs 100 --save_dir /content/drive/MyDrive/llie_checkpoints/
```

### Bước 3: Đánh giá và mang kết quả về Repo
Chạy lệnh đánh giá trên tập Validation và tập ExDark:
```bash
python evaluate.py --method attn_unet --checkpoint checkpoints/attn_unet/best_model.pth --split validation
python evaluate.py --method attn_unet --checkpoint checkpoints/attn_unet/best_model.pth --split exdark --save_visuals
```
Tải file `best_model.pth`, `training_history.json` và các ảnh kết quả về máy tính cá nhân để đưa vào báo cáo và slide.

---

## 4. Bộ Tiêu Chuẩn Kỳ Vọng & Bảng Chẩn Đoán Khắc Phục Sự Cố

### 4.1. Đường cong hội tụ kỳ vọng (Loss & Validation Metrics Trajectory)

| Mốc Epoch | Kỳ vọng Training Loss ($L_1$) | Kỳ vọng Validation PSNR (dB) | Kỳ vọng Validation SSIM | Trạng thái học tập của mô hình |
| :--- | :--- | :--- | :--- | :--- |
| **Epoch 1 - 5** | $0.35 \to 0.12$ | $14.0 \to 18.0$ | $0.50 \to 0.68$ | Khởi động, mô hình bắt đầu học nâng sáng thô toàn cục |
| **Epoch 20** | $0.07 \to 0.09$ | $20.0 \to 22.0$ | $0.75 \to 0.80$ | Đã nâng sáng tốt, bắt đầu cân chỉnh màu sắc và khử nhiễu |
| **Epoch 50** | $0.04 \to 0.05$ | $23.0 \to 24.5$ | $0.82 \to 0.85$ | Tái tạo chi tiết sắc nét, các cạnh biên rõ ràng |
| **Epoch 100** | **$0.030 \to 0.038$** | **$24.5 \to 26.5$** | **$0.84 \to 0.88$** | **Hội tụ tối ưu, đạt kết quả vững chắc cho đồ án** |

### 4.2. Bảng chẩn đoán & Hướng khắc phục nhanh (Troubleshooting Guide)
1. **Hiện tượng: Train Loss giảm liên tục nhưng Validation PSNR bão hòa hoặc tụt dần sau Epoch 40 (Quá khớp - Overfitting)**
   - *Hành động 1:* Tăng trọng số suy giảm trọng số `weight_decay = 1e-4`, bật thêm `RandomVerticalFlip`.
   - *Hành động 2:* Mở khóa Khối $S_1$ (+50 ảnh $\to$ 550 ảnh) từ tập cách ly dự trữ.
2. **Hiện tượng: Cả Train Loss và Val Loss đều giảm rất chậm, PSNR dưới 18 dB sau 20 Epochs (Học kém - Underfitting)**
   - *Hành động:* Tăng learning rate từ $1\times 10^{-4}$ lên $3\times 10^{-4}$; kiểm tra xem giá trị ảnh đầu vào đã chuẩn hóa về dải $[0, 1]$ chưa.
3. **Hiện tượng: Ảnh sau phục hồi có màu sắc bị ám đỏ hoặc ám xanh (Color Shift)**
   - *Hành động:* Kích hoạt thành phần $\lambda_{\text{ssim}} \times (1 - \text{SSIM})$ với trọng số $0.2$.

---

## 5. Bảng Kết Quả Kỳ Vọng Tổng Hợp & 5 Câu Hỏi Bảo Vệ Đồ Án

### 5.1. Bảng đối chuẩn kết quả kỳ vọng

| Phương pháp | Loại mô hình | Số tham số | FLOPs (256x256) | Thời gian suy luận (GPU) | PSNR kỳ vọng (dB) | SSIM kỳ vọng |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gamma Correction** | Truyền thống | -- | -- | ~2 ms | ~14.5 - 16.0 | ~0.60 - 0.68 |
| **CLAHE** | Truyền thống | -- | -- | ~5 ms | ~15.5 - 17.5 | ~0.65 - 0.72 |
| **Plain CNN** | Học sâu phẳng | ~0.12M | ~7.8 G | ~4 ms | ~19.0 - 21.5 | ~0.74 - 0.79 |
| **Compact U-Net** | Encoder-Decoder | ~0.48M | ~14.2 G | ~8 ms | ~22.5 - 24.5 | ~0.80 - 0.84 |
| **Attention U-Net (500 ảnh)** | **Đề xuất (CVPR 2025)** | **~0.90M** | **~18.5 G** | **~11 ms** | **~24.5 - 26.5** | **~0.84 - 0.88** |
| *Attention U-Net (+ Khối cách ly)* | Mở rộng dữ liệu | ~0.90M | ~18.5 G | ~11 ms | ~25.5 - 27.2 | ~0.86 - 0.90 |

### 5.2. Năm câu hỏi phản biện cốt lõi trước Hội đồng chấm đồ án UIT
1. **Câu hỏi: *"Tại sao nhóm lại kết hợp LOLv1 và LOLv2 Real mà không dùng duy nhất một bộ?"***  
   *Trả lời:* LOLv1 chỉ có 485 cặp ảnh chụp từ một dòng máy ảnh Canon cố định, dễ gây quá khớp vào một loại sensor. Nhóm kết hợp thêm 515 cặp từ LOLv2 Real để làm giàu phân bố suy giảm ánh sáng và mức độ nhiễu thực tế, giúp mô hình học được tính chất tổng quát hơn.
2. **Câu hỏi: *"Cơ chế Attention trong mô hình của nhóm hoạt động ra sao?"***  
   *Trả lời:* Mô hình kế thừa cơ chế **Simplified Channel Attention (SCA)** từ bài báo DarkIR (CVPR 2025). Khối này dùng Global Average Pooling để thu thập thống kê không gian toàn cục của từng kênh đặc trưng, sau đó chiếu qua tích chập $1\times1$ điểm để học trọng số $[0, 1]$. Nó giúp mạng tự động tăng cường các kênh mang thông tin độ chói và màu sắc hữu ích, đồng thời triệt tiêu các kênh chứa nhiễu cảm biến hạt mè.
3. **Câu hỏi: *"Tại sao nhóm lại cách ly 300 ảnh và tăng dần từng 50 ảnh?"***  
   *Trả lời:* Đây là phương pháp luận thực nghiệm để kiểm tra **hiệu quả mẫu (Sample Efficiency)**. Chúng em muốn chứng minh rằng kiến trúc Attention U-Net có khả năng học quy luật ánh xạ ánh sáng rất mạnh ngay cả khi dữ liệu bị hạn chế (500 ảnh đã đạt >90% hiệu năng tối đa). Việc tăng dần 50 ảnh cho phép nhóm vẽ được đường cong học tập khách quan.
4. **Câu hỏi: *"Trên ExDark không có ground truth thì nhóm đánh giá bằng cách nào?"***  
   *Trả lời:* Trên ExDark, nhóm đánh giá qua 2 kênh: (1) Đánh giá định tính trực quan bằng cách so sánh chi tiết trước và sau tăng sáng trên các cảnh đêm thực tế; (2) Đánh giá định lượng không tham chiếu (No-Reference Image Quality) thông qua chỉ số **NIQE**—chỉ số đo độ tự nhiên của ảnh mà không cần ảnh mẫu đối chứng.
5. **Câu hỏi: *"Tại sao nhóm không dùng mạng Transformer hay Diffusion hiện đại nhất?"***  
   *Trả lời:* Các mô hình Transformer (Restormer, RetinexFormer) hay Diffusion có kích thước rất lớn (>26M tham số) và chi phí tính toán cực cao, không phù hợp cho các thiết bị di động hay vi xử lý biên. Nhóm hướng tới giải pháp **Lightweight CNN (< 1M tham số)** để vừa đảm bảo độ sắc nét vừa có khả năng chạy thời gian thực, bám sát tinh thần tối ưu hóa kiến trúc của các bài báo CVPR 2024–2025.
