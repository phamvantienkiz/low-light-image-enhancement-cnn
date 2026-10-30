# Báo cáo kết quả thực nghiệm v2 — DarkIR-Edge vs. Standard U-Net vs. DarkIR-Edge (no FFT)

> Nguồn số liệu: `colab-results/llie_project_v2-20261029/llie_project_v2/experiments/v2/` (`results/table_main.csv`, `paired_tests.csv`, `efficiency.csv`, `diagnostic_gtmean.csv`, `results_per_image_with_ref.csv`, `checkpoints/*/history.csv`, `run_config.json`).
> Mọi con số dưới đây lấy trực tiếp từ các file đó hoặc tái tính bằng [analyze_v2_results.py](file:///E:/MyProject/low-light-image-enhancement-cnn/docs/ai/analyze_v2_results.py). Báo cáo viết theo quy tắc kết luận đã đăng ký trước (pre-registered) trong `docs/brainstorm/02_*.md` mục III.3.

---

## 0. Tóm tắt điều hành (TL;DR)

| Câu hỏi | Kết quả v2 | Kết luận trung thực |
|---|---|---|
| DarkIR-Edge có tốt hơn Standard U-Net? | ΔPSNR pooled = **−0.615 dB**, 95% CI **[−0.891, −0.335]**, p(Wilcoxon) = 4.4e-5 | **KHÔNG.** U-Net tốt hơn có ý nghĩa thống kê (đặc biệt LOLv2 −1.05 dB). Hòa ở Huawei/Nikon. |
| FFT/FreMLP có giúp? | ΔPSNR pooled (Edge − noFFT) = **−0.014 dB**, CI **[−0.231, +0.199]**, p = 0.43–0.90 | **Không có bằng chứng** FFT giúp (cũng không chứng minh được nó hại). Chênh lệch nằm hoàn toàn trong nhiễu. |
| Edge có "edge-friendly / real-time"? | FLOPs thấp hơn U-Net (8.75 vs 18.45 G) nhưng latency FP32 **31.5 vs 13.0 ms (≈2.4× chậm hơn)** trên L4 | **Không được tuyên bố** "nhanh hơn". Chỉ được nói: ít FLOPs hơn, ít VRAM hơn một chút (308 vs 342 MB). |
| Deblurring (LOL-Blur, 40 ảnh test)? | Edge 22.93 dB, U-Net 23.37 dB (Δ −0.44, CI [−0.83, −0.08]); DarkIR-m chính thức 27.37 dB | U-Net nhỉnh hơn Edge; cả hai thua xa DarkIR-m (≈ −4 dB, nhưng DarkIR-m được huấn luyện trên **toàn bộ** LOL-Blur train → không công bằng). |
| Cả 3 mô hình học được so với truyền thống? | Học sâu 20.4–21.4 dB macro/pooled vs. Gamma 16.9, MSRCR 12.1, CLAHE 11.6, Input 9.7 | **CÓ**, vượt xa baseline truyền thống (+3.8 dB trở lên so với Gamma 2.2). |

**Thông điệp chính cho báo cáo/bảo vệ:** kiến trúc DarkIR-Edge (width=24) với ~1.92 M tham số **không** vượt Standard U-Net cùng ngân sách tham số (1.93 M) về chất lượng hay tốc độ thực đo trong thiết lập này; giả thuyết FFT/FreMLP **chưa được ủng hộ**. Đây là kết quả âm (negative result) và phải được trình bày như vậy.

---

## 1. Thiết lập thực nghiệm (đã chạy thực tế)

### 1.1 Dữ liệu (`run_config.json`, `split_manifest.json`)

| Miền (domain) | Train | Val | Test | Nguồn |
|---|---:|---:|---:|---|
| lolv2 | 639 | 50 | 100 | LOLv2-Real |
| huawei | 561 | 50 | 30 | LSRW-Huawei (huawei_gap = 10) |
| nikon | — | — | 20 | LSRW-Nikon (**chỉ test**, miền chưa thấy khi huấn luyện) |
| blur | 150 | 10 | 40 | LOL-Blur (`low_blur_noise` → `high_sharp_scaled`), chọn cách ly theo chuỗi video |
| **Tổng** | **1,350** | **110** | **190** | |

Số lượng ảnh trong `results_per_image*.csv` khớp: 190 ảnh/baseline truyền thống, 380 dòng/mô hình học sâu (190 × 2 seed), 190 dòng DarkIR-m.

### 1.2 Huấn luyện

- Patch 256, batch 8, `repeat=2`, AdamW (lr 2e-4 → 1e-6, `CosineAnnealingLR`, **T_max = 100**), weight decay 1e-4, grad-clip 1.0, EMA 0.999, validate mỗi 5 epoch.
- Loss: L1 + 0.2·SSIM + 0.1·color (+ 0.5·side-loss cho họ DarkIR-Edge).
- **Early Stopping:** theo val-PSNR macro (EMA) của 3 miền val (lolv2/huawei/blur), `min_delta = 0.05 dB`, `patience = 15`, `min_epochs = 40` (chỉ bắt đầu đếm sau epoch 40 → sớm nhất dừng ở epoch 55).
- Mô hình: DarkIR-Edge (C=24, 1,924,038 tham số), DarkIR-Edge no-FFT (1,495,422), Standard U-Net (base=32, 1,928,483), 2 seed (42, 123) mỗi loại. DarkIR-m (CVPR'25, 3.32 M) chỉ dùng làm **tham chiếu trần**, không huấn luyện lại.

### 1.3 Đánh giá

Trọng số đánh giá: `best.pt` (EMA) chọn theo val-PSNR macro, không phải epoch cuối (chi tiết mục 6.4). PSNR/SSIM/LPIPS trên 190 ảnh test; số "pooled" = trung bình trên toàn bộ 190 ảnh; "macro" = trung bình của 4 miền (không trọng số theo n). Giá trị `±` là độ lệch giữa **2 seed** (n = 2, chỉ mang tính mô tả, **không** phải khoảng tin cậy). Kiểm định ghép cặp (paired) dùng trung bình 2 seed trên mỗi ảnh, bootstrap 95% CI + t-test + Wilcoxon.

---

## 2. Kết quả chất lượng ảnh

### 2.1 Bảng chính (`table_main.csv`) — PSNR (dB)

| Phương pháp | blur (40) | huawei (30) | lolv2 (100) | nikon (20) | **pooled (190)** | macro |
|---|---:|---:|---:|---:|---:|---:|
| Input (low-light) | 10.55 | 9.06 | 9.72 | 9.30 | 9.75 | 9.66 |
| Gamma 2.2 | 18.09 | 15.08 | 17.27 | 15.42 | 16.90 | 16.46 |
| CLAHE | 12.90 | 10.39 | 11.54 | 11.04 | 11.59 | 11.47 |
| MSRCR (Retinex) | 12.20 | 11.44 | 12.03 | 13.45 | 12.12 | 12.28 |
| DarkIR-Edge (no FFT) | 22.73 ± 0.27 | 20.87 ± 0.09 | 20.64 ± 0.29 | 17.21 ± 0.07 | 20.76 ± 0.23 | 20.36 ± 0.18 |
| **DarkIR-Edge (ours)** | **22.93 ± 0.21** | 20.80 ± 0.03 | 20.55 ± 0.01 | **17.24 ± 0.10** | 20.74 ± 0.06 | 20.38 ± 0.08 |
| **Standard U-Net** | **23.37 ± 0.00** | 20.65 ± 0.01 | **21.60 ± 0.43** | 17.17 ± 0.10 | **21.36 ± 0.22** | **20.70 ± 0.09** |
| *DarkIR-m (CVPR'25, tham chiếu)* | *27.37* | *18.63* | *20.01* | *16.43* | *20.96* | *20.61* |

### 2.2 SSIM và LPIPS (thấp hơn = tốt hơn với LPIPS)

| Phương pháp | SSIM pooled | SSIM macro | LPIPS pooled | LPIPS macro |
|---|---:|---:|---:|---:|
| Input | 0.217 | 0.225 | 0.554 | 0.557 |
| Gamma 2.2 | 0.595 | 0.555 | 0.356 | 0.363 |
| CLAHE | 0.405 | 0.384 | 0.466 | 0.472 |
| MSRCR | 0.410 | 0.401 | 0.471 | 0.450 |
| DarkIR-Edge (no FFT) | 0.730 ± 0.009 | 0.670 ± 0.004 | 0.353 ± 0.060 | 0.367 ± 0.048 |
| DarkIR-Edge (ours) | 0.728 ± 0.005 | 0.668 ± 0.003 | 0.368 ± 0.009 | 0.378 ± 0.006 |
| Standard U-Net | **0.749 ± 0.000** | **0.680 ± 0.000** | **0.277 ± 0.002** | **0.305 ± 0.002** |
| DarkIR-m (ref) | 0.728 | — | **0.233** | — |

**Nhận xét:** U-Net dẫn đầu trên cả 3 chỉ số ở mức pooled. Đáng chú ý LPIPS của U-Net (0.277) tốt hơn rõ rệt Edge (0.368) — chênh lệch lớn hơn nhiều so với độ biến thiên giữa seed (±0.009), tức Edge cho ảnh kém tương đồng cảm quan hơn.

### 2.3 Kết quả theo từng seed (pooled PSNR, dB)

| Mô hình | seed 42 | seed 123 | Ghi chú |
|---|---:|---:|---|
| DarkIR-Edge (ours) | 20.781 | 20.701 | ổn định (Δseed = 0.08) |
| DarkIR-Edge (no FFT) | 20.917 | 20.592 | Δseed = 0.33 |
| Standard U-Net | 21.511 | 21.201 | Δseed = 0.31 |

Theo miền (seed 42 / 123): U-Net lolv2 = 21.91 / 21.30; Edge lolv2 = 20.55 / 20.56; mọi seed của U-Net đều ≥ mọi seed của Edge ở lolv2 và pooled. Ở nikon/huawei các seed đan xen nhau (chênh < 0.2 dB).

> [!WARNING]
> Với **2 seed**, độ lệch chuẩn giữa seed không đáng tin cậy. Biến thiên do seed của một mô hình (≈0.1–0.3 dB) *cùng bậc độ lớn* với nhiều chênh lệch ta đang bàn (ví dụ Edge vs no-FFT). Vì vậy chỉ các chênh lệch có CI loại trừ 0 (Edge vs U-Net) mới được xem là có bằng chứng.

---

## 3. Kiểm định thống kê ghép cặp (`paired_tests.csv`)

ΔPSNR = A − B, trung bình 2 seed theo từng ảnh test; CI = bootstrap 95%.

### 3.1 DarkIR-Edge (ours) − Standard U-Net

| Miền | n | ΔPSNR (dB) | 95% CI | p (t) | p (Wilcoxon) | % ảnh Edge thắng |
|---|---:|---:|---|---:|---:|---:|
| **pooled** | 190 | **−0.615** | [−0.891, −0.335] | 2.9e-5 | 4.4e-5 | 38.4% |
| blur | 40 | −0.442 | [−0.832, −0.075] | 0.030 | 0.042 | 35.0% |
| huawei | 30 | +0.148 | [−0.222, +0.514] | 0.449 | 0.205 | 56.7% |
| **lolv2** | 100 | **−1.050** | [−1.516, −0.558] | 3.8e-5 | 1.2e-5 | 30.0% |
| nikon | 20 | +0.065 | [−0.329, +0.463] | 0.757 | 0.596 | 60.0% |

**Áp dụng quy tắc đăng ký trước:** CI pooled nằm hoàn toàn dưới 0 ⇒ *U-Net tốt hơn có ý nghĩa*. Edge không thắng ở bất kỳ miền nào có ý nghĩa; chênh lệch dương nhỏ ở huawei/nikon đều có CI chứa 0.

Lưu ý hiệu chỉnh đa kiểm định: 5 phép thử (pooled + 4 miền) — nếu hiệu chỉnh Bonferroni (α = 0.01), chỉ **pooled** và **lolv2** còn đứng vững; **blur** (p ≈ 0.03–0.04) trở nên không đủ mạnh.

### 3.2 DarkIR-Edge (ours) − DarkIR-Edge (no FFT)

| Miền | n | ΔPSNR (dB) | 95% CI | p (t) | p (Wilcoxon) | % ảnh Edge thắng |
|---|---:|---:|---|---:|---:|---:|
| pooled | 190 | −0.014 | [−0.231, +0.199] | 0.899 | 0.431 | 52.6% |
| blur | 40 | +0.194 | [−0.088, +0.472] | 0.184 | 0.089 | 62.5% |
| huawei | 30 | −0.072 | [−0.277, +0.112] | 0.486 | 0.919 | 46.7% |
| lolv2 | 100 | −0.089 | [−0.465, +0.293] | 0.650 | 0.986 | 47.0% |
| nikon | 20 | +0.031 | [−0.201, +0.212] | 0.781 | 0.143 | 70.0% |

**Kết luận:** tất cả CI chứa 0. **Không có bằng chứng** nhánh FFT/FreMLP cải thiện PSNR (hay hại nó). Xu hướng dương ở `blur` (+0.19 dB, 62.5% ảnh) khớp trực giác "FFT giúp khử mờ" nhưng **không** đạt ý nghĩa và chỉ có 40 ảnh/2 seed → chỉ được nêu như quan sát gợi ý, không phải phát hiện.

### 3.3 Bổ sung (DarkIR-m tham chiếu, bootstrap tái tính)

| So sánh | Miền | ΔPSNR | 95% CI |
|---|---|---:|---|
| DarkIR-m − U-Net | pooled | −0.39 | [−1.25, +0.43] |
| DarkIR-m − Edge | pooled | +0.22 | [−0.59, +1.02] |
| DarkIR-m − U-Net | blur | +4.00 | [+3.07, +4.90] |
| DarkIR-m − Edge | blur | +4.44 | [+3.41, +5.42] |

(Bootstrap tự viết với seed 0, 5000 lần, trong `analyze_v2_results.py`; đây là phân tích phụ, không nằm trong `paired_tests.csv`.) Ngoài miền blur, DarkIR-m tương đương các mô hình nhỏ (pooled không khác biệt có ý nghĩa), và thấp hơn ở huawei/nikon (18.63/16.43) — dấu hiệu ngoài-phân-phối của mô hình chính thức, không phải dấu hiệu mô hình nhỏ "tốt hơn".

---

## 4. Chẩn đoán độ sáng (`diagnostic_gtmean.csv`)

`psnr_gtmean` = PSNR sau khi chuẩn hóa độ sáng trung bình theo ảnh GT; `gap` = psnr_gtmean − psnr (ước lượng phần lỗi do sai lệch độ sáng toàn cục).

| Mô hình | blur | huawei | lolv2 | nikon |
|---|---:|---:|---:|---:|
| DarkIR-Edge (ours) | 1.96 | 2.03 | 5.97 | 1.41 |
| DarkIR-Edge (no FFT) | 2.12 | 1.99 | 5.71 | 1.36 |
| Gamma 2.2 | 3.80 | 6.47 | 6.48 | 2.84 |
| CLAHE | 6.74 | 8.17 | 9.88 | 4.62 |

(Đơn vị dB.) Gap của lolv2 lớn bất thường (~5.7–6.0 dB) ở cả hai biến thể Edge — **phần lớn sai số trên LOLv2 là sai lệch độ sáng toàn cục**, không phải chi tiết. Gap của Edge và no-FFT gần như bằng nhau (5.97 vs 5.71, 1.96 vs 2.12), nên **không có bằng chứng** "FFT sửa độ sáng toàn cục tốt hơn". (File chẩn đoán không chứa dòng riêng cho U-Net trong phần đã đọc; có thể xem đầy đủ trong CSV nếu cần so sánh.)

---

## 5. Hiệu năng/Hiệu quả (`efficiency.csv`, GPU NVIDIA L4)

| Mô hình | Params | GFLOPs @256² | Latency FP32 @600×400 | FPS FP32 | Latency FP16 | FPS FP16 | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| DarkIR-Edge (C=24) | 1.924 M | 8.75 | 31.52 ms | 31.7 | 26.61 ms | 37.6 | 308 MB |
| DarkIR-Edge no-FFT | 1.495 M | 7.50 | 27.14 ms | 36.8 | 21.41 ms | 46.7 | 293 MB |
| Standard U-Net (base=32) | 1.928 M | 18.45 | **12.98 ms** | **77.1** | **11.38 ms** | **87.9** | 342 MB |

- Edge có **ít hơn 53% FLOPs** nhưng **chậm hơn ≈2.43× (FP32) và ≈2.34× (FP16)** so với U-Net: FLOPs không phản ánh latency (kernel depthwise/FFT/gating bị giới hạn bởi băng thông bộ nhớ và kernel-launch; U-Net toàn conv 3×3 chuẩn tận dụng tốt GPU).
- Bỏ FFT giảm 22% tham số, 14% FLOPs, ~14% latency FP32 (≈19% FP16) mà **không mất chất lượng** → với ràng buộc triển khai, biến thể no-FFT hợp lý hơn bản đầy đủ.
- Cả hai đều ≥ 30 FPS ở 600×400 trên L4 (GPU datacenter), nhưng **chưa đo trên thiết bị biên thực sự** ⇒ không được dùng cụm "edge-friendly" như một kết quả đã kiểm chứng.

---

## 6. Quá trình huấn luyện và Early Stopping

| Run | Epoch thực chạy | Val macro tốt nhất (epoch) | Ghi chú |
|---|---:|---|---|
| darkir_edge s42 | 55 | 21.44 (ep 30) | dừng sớm nhất có thể |
| darkir_edge s123 | 55 | 21.41 (ep 55) | dừng sớm nhất có thể |
| darkir_edge_nofft s42 | 55 | 21.72 (ep 25) | dừng sớm nhất có thể |
| darkir_edge_nofft s123 | **80** | 21.53 (ep 80) | chạy lâu hơn |
| unet s42 | 55 | 21.09 (ep 55) | dừng sớm nhất có thể, val còn **đang tăng** |
| unet s123 | **90** | 22.03 (ep 90) | chạy lâu nhất |

**Vấn đề phương pháp luận quan trọng (confound):**

1. **Độ dài huấn luyện không đồng đều** (55 → 90 epoch). `CosineAnnealingLR` có `T_max = 100`; run dừng ở epoch 55 mới đạt lr ≈ 8.5e-5 (chưa hội tụ qua pha lr thấp), còn run dài (U-Net s123, 90 epoch) lr ≈ 5.9e-6. Pha lr thấp thường cho +0.3–0.7 dB. Điều này **có thể thiên về U-Net**: chính U-Net s123 (90 epoch) có val macro 22.03 so với s42 (55 epoch) 21.09 — chênh gần 0.9 dB val trong khi test pooled chỉ chênh 0.31 dB giữa hai seed, cho thấy tác động của độ dài huấn luyện lên val rõ hơn lên test.
2. Với U-Net seed 42, val còn tăng (20.77 → 21.09) tại lúc dừng — early stopping có thể đã cắt sớm. Với Edge, val gần như phẳng từ epoch ~25–30 (21.2–21.4) → **Edge hội tụ sớm và bão hòa**, trong khi U-Net s123 tiếp tục cải thiện đến ep 90. Đây là xu hướng đáng chú ý nhưng không thể tách bạch khỏi confound (1).
3. Early Stopping và chọn `best.pt` đều tính trên val macro (110 ảnh, trong đó 10 ảnh blur) — val nhỏ và nhiễu; một số quyết định dừng nằm trong nhiễu ±0.1 dB.
4. **Checkpoint dùng để đánh giá test = `best.pt` (trọng số EMA), đã đối chiếu trực tiếp notebook.** Cell 21/23/27 gọi `find_checkpoint(tag, seed)` với mặc định `ckpt_name='best.pt'` (chỉ rơi về `final.pt`/`last.pt` nếu thiếu file), và `nn_fn` nạp `ck['ema']`. `best.pt` được ghi (cell 13) mỗi khi val-PSNR macro (EMA, kiểm tra 5 epoch/lần) đạt đỉnh mới (so sánh `>` thuần, không dùng `min_delta`). Ngưỡng `min_delta` chỉ áp cho bộ đếm Early Stopping. Hệ quả: **không phải mô hình ở epoch cuối**, mà là epoch có val macro cao nhất:

| Run | Epoch dừng | Epoch của `best.pt` được đánh giá | Val macro |
|---|---:|---:|---:|
| darkir_edge s42 | 55 | **30** | 21.44 |
| darkir_edge s123 | 55 | 55 | 21.41 |
| darkir_edge_nofft s42 | 55 | **25** | 21.72 |
| darkir_edge_nofft s123 | 80 | 80 | 21.53 |
| unet s42 | 55 | 55 | 21.09 |
| unet s123 | 90 | 90 | 22.03 |

   Điều này làm rõ hơn confound: các checkpoint test của Edge s42 và no-FFT s42 đến từ epoch 25–30 (lr ≈ 1.7e-4, còn rất cao), còn U-Net s123 từ epoch 90 (lr ≈ 5.9e-6). Việc chọn `best.pt` theo val (110 ảnh, tách biệt với test) không rò rỉ test, nhưng val nhỏ/nhiễu nên epoch được chọn có thể lệch ±0.1 dB. Bộ đếm Early Stopping (`es_ref`, `min_delta=0.05`, `patience=15`, đếm từ sau epoch 40) giống nhau cho mọi mô hình; độ dài khác nhau là do U-Net s123 và no-FFT s123 liên tục cải thiện > 0.05 dB nên chưa kích hoạt dừng — đây là hệ quả của quy tắc, **không phải lỗi mã**. Không có chỗ nào trong notebook đánh giá trên `final.pt` ở bước này (`final.pt` chỉ là đích dự phòng).
> [!IMPORTANT]
> Hệ quả: kết luận "U-Net > Edge" **vẫn có thể đúng**, nhưng độ lớn −0.6 dB có thể bị khuếch đại bởi việc U-Net được huấn luyện lâu hơn ở một seed. Báo cáo cần nêu rõ hạn chế này. Một đối chứng công bằng tuyệt đối sẽ là huấn luyện cố định đủ 100 epoch (hoặc cùng số epoch) cho mọi mô hình — chưa làm trong v2.

---

## 7. Phân tích định tính

Ảnh minh họa nằm trong `results/qualitative.png`, `qualitative_lolblur_5samples.png`, `qualitative_with_darkir_m.png`, `visual_samples/` (8 mẫu × 6 phương pháp) và `lolblur_visual_samples/`. Báo cáo này **chưa kiểm tra bằng mắt từng ảnh** ở bước phân tích số liệu; phần nhận xét định tính cần được người báo cáo bổ sung sau khi xem trực tiếp. Từ số liệu: nhận xét cảm quan có cơ sở duy nhất là LPIPS (U-Net 0.277 < Edge 0.368 ⇒ ảnh U-Net gần GT hơn về đặc trưng sâu) và gap độ sáng lớn ở LOLv2.

---

## 8. Các mối đe dọa đến tính hợp lệ (limitations)

1. **Chỉ 2 seed/mô hình** → ước lượng phương sai thô; CI bootstrap chỉ phản ánh biến thiên giữa *ảnh test*, không bao hàm biến thiên giữa seed/khởi tạo. Do đó "có ý nghĩa" ở §3.1 là ý nghĩa theo *ảnh* trên cặp (2 seed trung bình), không phải đảm bảo tổng quát hóa qua seed.
2. **Độ dài huấn luyện khác nhau giữa các run** (mục 6).
3. **Tập test nhỏ**: 20–100 ảnh/miền; nikon chỉ 20 ảnh, test nikon là miền chưa thấy khi huấn luyện (PSNR ~17 dB cho mọi mô hình học sâu — gần như không khác biệt).
4. **Blur chỉ 150 ảnh train / 40 test từ 30 chuỗi video** (test các frame của cùng 30 chuỗi → các ảnh test trong cùng chuỗi tương quan mạnh, làm giảm n hiệu dụng và CI hẹp quá mức).
5. **DarkIR-m không công bằng**: được huấn luyện chính thức trên LOL-Blur đầy đủ (10.2k cặp) với quy mô lớn hơn nhiều; chỉ dùng làm trần tham chiếu. Ngoài miền blur nó không vượt các mô hình nhỏ.
6. **Độ trễ chỉ đo trên L4**, không phải thiết bị biên.
7. **Chưa thực hiện** huấn luyện đa-miền có/không blur để tách ảnh hưởng của dữ liệu blur; không thể kết luận blur-data cải thiện hay làm giảm các miền khác.
8. Chênh lệch kết quả no-FFT so với bản đầy đủ nằm trong nhiễu — không thể kết luận bất kỳ chiều hướng nào.

---

## 9. Kết luận trung thực (đối chiếu quy tắc đăng ký trước)

1. **Chất lượng:** Standard U-Net (1.93 M) vượt DarkIR-Edge (1.92 M) về PSNR pooled **+0.62 dB** (CI [0.34, 0.89]), đồng thời tốt hơn SSIM và LPIPS. Edge không thắng ở miền nào có ý nghĩa; hòa ở Huawei/Nikon.
2. **Giả thuyết FFT/FreMLP:** *không được hỗ trợ* — Δ pooled = −0.01 dB, mọi CI chứa 0. Không nên khẳng định FFT cải thiện khử mờ hay độ sáng.
3. **Hiệu quả:** Edge ít FLOPs hơn (−53%) nhưng **chậm hơn ≈2.4×** trên L4 → không nên tuyên bố nhanh/edge-friendly/real-time-advantage so với U-Net. Biến thể no-FFT rẻ hơn 22% tham số và nhanh hơn ~14% mà chất lượng tương đương.
4. **Giá trị tích cực thực sự có thể báo cáo:** (a) mô hình CNN nhỏ ~1.9 M vượt xa phương pháp truyền thống (+3.8 dB so với Gamma tốt nhất, +9 dB so với CLAHE/MSRCR); (b) việc thêm dữ liệu LOL-Blur giúp mô hình khử mờ-tăng sáng đồng thời đạt ~22.9–23.4 dB trên blur; (c) pipeline đánh giá chặt chẽ (cách ly chuỗi, paired CI, pre-registered rules) và kết quả âm được nêu trung thực.
5. **Khuyến nghị đóng gói câu chữ cho luận văn/slide:** "DarkIR-Edge đạt chất lượng cạnh tranh với baseline nhưng chưa vượt U-Net cùng số tham số; đóng góp chính là phân tích có kiểm soát cho thấy nhánh FFT/FreMLP không mang lại cải thiện đo được ở quy mô này." Tránh các câu: "Edge tốt hơn U-Net", "FFT cải thiện độ sáng/khử mờ", "thời gian thực trên thiết bị biên".

### Việc nên làm tiếp (nếu còn thời gian)
- (Đã xác minh: đánh giá dùng `best.pt` EMA theo val macro — xem mục 6.4.)
- (Tùy chọn) chạy lại U-Net s42 / các run dừng ở 55 đến đủ 100 epoch để loại confound ở mục 6.
- Xem trực tiếp ảnh định tính và bổ sung mục 7.

---

## Phụ lục — Danh sách file nguồn

- `results/table_main.csv`, `paired_tests.csv`, `efficiency.csv`, `diagnostic_gtmean.csv`
- `results/results_per_image_with_ref.csv` (gồm DarkIR-m ref), `results/results_per_image.csv`
- `checkpoints/<run>/history.csv` (6 run), `run_config.json`
- Script tái tính: [analyze_v2_results.py](file:///E:/MyProject/low-light-image-enhancement-cnn/docs/ai/analyze_v2_results.py)


