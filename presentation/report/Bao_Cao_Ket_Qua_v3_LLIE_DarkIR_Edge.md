# Báo cáo kết quả thực nghiệm v3 — DarkIR-Edge vs. Standard U-Net vs. DarkIR-Edge (no FFT)

> Nguồn số liệu: `colab-results/llie_project_v3-20261102/llie_project_v3/experiments/v3/` (`results/table_main.csv`, `table_main_best.csv`, `paired_tests.csv`, `paired_tests_extra.csv`, `table_extra.csv`, `results_extra.csv`, `efficiency.csv`, `diagnostic_gtmean.csv`, `results_per_image_with_ref.csv`, `checkpoints/*/history.csv`, `run_config.json`) và notebook `colab-results/LLIE_DarkIR_Edge_Colab_v3.ipynb`.
> Kế hoạch: [ke_hoach_v3.md](file:///E:/MyProject/low-light-image-enhancement-cnn/presentation/report/ke_hoach_v3.md). Kết luận viết theo quy tắc đăng ký trước (mục 3 của kế hoạch): dựa vào 95% CI bootstrap của ΔPSNR ghép cặp (190 ảnh, trung bình 2 seed); checkpoint chính thức = `final.pt` EMA ở epoch 70.
> Các con số "so với v2" lấy từ [Bao_Cao_Ket_Qua_v2_LLIE_DarkIR_Edge.md](file:///E:/MyProject/low-light-image-enhancement-cnn/presentation/report/Bao_Cao_Ket_Qua_v2_LLIE_DarkIR_Edge.md). Các số per-seed, DarkIR-m theo miền và epoch tốt nhất được tính lại trực tiếp từ CSV.

---

## 0. Tóm tắt điều hành (TL;DR)

| Câu hỏi | Kết quả v3 | Kết luận trung thực |
|---|---|---|
| DarkIR-Edge có tốt hơn Standard U-Net? | ΔPSNR pooled = **−0.214 dB**, 95% CI **[−0.530, +0.132]**, p(t) = 0.22, p(Wilcoxon) = 0.028 | **Ngang nhau** theo quy tắc đăng ký trước (CI chứa 0). Điểm trung bình vẫn nghiêng về U-Net (21.31 vs 21.10), nhưng không đủ bằng chứng. Không miền nào có CI loại trừ 0. |
| So với v2 | Khoảng cách Edge − U-Net thu hẹp từ −0.615 dB (CI loại 0) còn −0.214 dB (CI chứa 0) | Kết luận "U-Net mạnh hơn có ý nghĩa" của v2 **không còn được giữ**. Không tách được nguyên nhân (đổi độ dài huấn luyện **và** tăng dữ liệu blur cùng lúc — mục 8). |
| FFT/FreMLP có giúp? | ΔPSNR pooled (Edge − noFFT) = **+0.096 dB**, CI **[−0.122, +0.310]** | **Không có bằng chứng** FFT giúp PSNR (cũng không hại). Mọi CI theo miền đều chứa 0. |
| "Edge-friendly / real-time"? | FLOPs 8.75 vs 18.45 G (−53%), nhưng latency FP32 **31.71 vs 13.23 ms (≈2.40× chậm hơn)** trên L4 | **Không được tuyên bố** nhanh hơn U-Net. Peak VRAM của Edge thấp hơn (237 vs 270 MB). |
| Deblurring (LOL-Blur, 40 ảnh) | Edge 23.57 dB, U-Net 23.80 dB (Δ −0.23, CI [−0.70, +0.24]); DarkIR-m chính thức 27.37 dB | Edge ≈ U-Net. Độ nét (Laplacian) của đầu ra chỉ bằng 18–23% GT ⇒ chưa khử mờ tốt (mục 5). |
| So với phương pháp truyền thống | Học sâu 20.6–20.7 dB macro, 21.0–21.3 dB pooled vs. Gamma 16.90, MSRCR 12.12, CLAHE 11.59, Input 9.75 | **CÓ**, vượt Gamma +4.1 đến +4.4 dB pooled. |

**Thông điệp chính:** với cùng ngân sách tham số (~1.92 M) và lịch huấn luyện cố định 70 epoch, DarkIR-Edge **ngang** Standard U-Net về PSNR pooled (không vượt, không kém có ý nghĩa), nhưng **kém U-Net về độ nét, LPIPS** và **chậm hơn ≈2.4×** trên GPU. Giả thuyết FFT/FreMLP vẫn **chưa được ủng hộ**. Đây là kết quả âm/trung tính và phải được trình bày như vậy.

---

## 1. Thiết lập thực nghiệm (đã chạy thực tế)

### 1.1 Dữ liệu (`run_config.json`, `split_manifest.json`)

| Miền | Train | Val | Test | Nguồn |
|---|---:|---:|---:|---|
| lolv2 | 639 | 50 | 100 | LOLv2-Real |
| huawei | 561 | 50 | 30 | LSRW-Huawei (`huawei_gap` = 10) |
| nikon | — | — | 20 | LSRW-Nikon (**chỉ test**, miền chưa thấy) |
| blur | **400** (v2: 150) | **50** (v2: 10) | 40 | LOL-Blur, cách ly theo chuỗi video |
| **Tổng** | **1,600** | **150** | **190** | |

Tập test **giữ nguyên so với v2** (190 ảnh): kết quả test v2/v3 so sánh trực tiếp được trên cùng ảnh. Val thì **khác** v2 (150 ảnh, blur 50) nên val-PSNR v2/v3 không so sánh được.

### 1.2 Huấn luyện

- Patch 256, batch 8, `repeat = 2` (400 bước/epoch, 28,000 bước/run), AdamW (lr 2e-4 → 1e-6, `CosineAnnealingLR`, **T_max = 70**), weight decay 1e-4, grad-clip 1.0, EMA 0.999, validate mỗi 5 epoch.
- **Không Early Stopping** (`use_early_stopping = false`); **mọi run đều chạy đủ 70 epoch** (`history.csv` có đúng 70 dòng/run), lr về đúng 1e-6 ở epoch 70 → loại confound độ dài huấn luyện của v2.
- Loss: L1 + 0.2·SSIM + 0.1·color (+ 0.5·side-loss cho họ DarkIR-Edge; U-Net không có side-loss nên tổng loss của hai họ không so sánh trực tiếp được).
- Mô hình: DarkIR-Edge (C=24, 1,924,038 tham số), no-FFT (1,495,422), Standard U-Net (base=32, 1,928,483); 2 seed (42, 123) mỗi loại = 6 run. DarkIR-m (CVPR'25) chỉ là **tham chiếu trần**, không huấn luyện lại.
- Huấn luyện theo kế hoạch PA-A (chia 2 phiên theo seed); `run_config.json` đã ghi đủ cả 2 seed.
- **Chi tiết đã đối chiếu trong notebook v3** (`LLIE_DarkIR_Edge_Colab_v3.ipynb`):
  - Scheduler `CosineAnnealingLR` bước **theo từng step** với `T_max = total_steps = 70 × 400 = 28,000`, `eta_min = 1e-6`; AMP FP16 (riêng FFT/FreMLP và LayerNorm chạy FP32), grad-clip 1.0, EMA 0.999 cập nhật mỗi step; seed cố định (`set_seed`) trước mỗi run.
  - Val được tính trên trọng số **EMA** mỗi 5 epoch **và bắt buộc ở epoch 70**; `best.pt` lưu khi val macro đạt đỉnh mới; `last.pt` (kèm optimizer/scheduler/EMA/history) lưu mỗi lần val để resume. Khối Early Stopping vẫn còn trong mã nhưng bị chặn bởi `use_early_stopping = False`.
  - `find_checkpoint` mặc định `final.pt`; `nn_fn` nạp `ck['ema']` ⇒ bảng chính là **EMA epoch 70**. Một vòng đánh giá riêng dùng `best.pt` → `table_main_best.csv`.
  - **"DarkIR-Edge (ours)" không chỉ là DarkIR + FFT:** `DarkIREdge(width=24)` mặc định bật `use_imax=True` (kênh tiên nghiệm Max-RGB, đầu vào 4 kênh) và `use_gate=True` (3 cổng Gated Skip), cộng side-supervision 8C→3 ở bottleneck. Biến thể "no-FFT" chỉ đặt `use_fft=False` (giữ nguyên imax/gate/side) nên so sánh Edge − no-FFT cô lập đúng nhánh FFT; còn so sánh Edge − U-Net là so sánh **toàn bộ thiết kế**, không cô lập riêng FFT/imax/gate.
  - Thống kê: `±` là std giữa seed của pandas (ddof = 1, n = 2). Kiểm định ghép cặp: PSNR trung bình 2 seed theo `(domain, image)`, bootstrap **2,000** lần (seed 42), t-test ghép cặp, Wilcoxon.

### 1.3 Đánh giá

- **Checkpoint chính thức: `final.pt` (EMA, epoch 70)** → `table_main.csv`, `paired_tests.csv`. Phân tích phụ dùng `best.pt` (EMA, theo val macro) → `table_main_best.csv`.
- PSNR/SSIM/LPIPS trên 190 ảnh; "pooled" = trung bình 190 ảnh; "macro" = trung bình 4 miền. Giá trị `±` = độ lệch giữa **2 seed** (n = 2, chỉ mô tả, **không** phải CI). Kiểm định ghép cặp: trung bình 2 seed theo ảnh, bootstrap 95% CI + t-test + Wilcoxon.
- Phân tích bổ sung (`results_extra.csv`, `paired_tests_extra.csv`): sai lệch độ sáng có dấu, EMD histogram độ chói, CIEDE2000, tỉ lệ độ nét Laplacian out/GT, MAE trên 10% pixel tối/sáng nhất của GT.

---

## 2. Kết quả chất lượng ảnh

### 2.1 Bảng chính (`table_main.csv`, `final.pt`) — PSNR (dB)

| Phương pháp | blur (40) | huawei (30) | lolv2 (100) | nikon (20) | **pooled (190)** | macro |
|---|---:|---:|---:|---:|---:|---:|
| Input (low-light) | 10.55 | 9.06 | 9.72 | 9.30 | 9.75 | 9.66 |
| Gamma 2.2 | 18.09 | 15.08 | 17.27 | 15.42 | 16.90 | 16.46 |
| CLAHE | 12.90 | 10.39 | 11.54 | 11.04 | 11.59 | 11.47 |
| MSRCR (Retinex) | 12.20 | 11.44 | 12.03 | 13.45 | 12.12 | 12.28 |
| DarkIR-Edge (no FFT) | 23.47 ± 0.13 | **20.85** ± 0.23 | 20.84 ± 0.00 | 17.12 ± 0.29 | 21.00 ± 0.02 | 20.57 ± 0.05 |
| **DarkIR-Edge (ours)** | 23.57 ± 0.17 | 20.66 ± 0.05 | 21.07 ± 0.08 | 16.96 ± 0.04 | 21.10 ± 0.07 | 20.57 ± 0.04 |
| **Standard U-Net** | **23.80** ± 0.12 | 20.51 ± 0.18 | **21.39** ± 0.64 | **17.15** ± 0.02 | **21.31** ± 0.34 | **20.71** ± 0.17 |
| *DarkIR-m (CVPR'25, tham chiếu)* | *27.37* | *18.63* | *20.01* | *16.42* | *20.96* | *20.61* |

(DarkIR-m tính lại từ `results_per_image_with_ref.csv`.)

> [!WARNING]
> **Độ tin cậy của dòng "DarkIR-m":** notebook (cell 22) **không dùng mã DarkIR gốc**: nó nạp checkpoint HuggingFace `Cidaut/DarkIR` vào lớp `DarkIREdge(width=32, use_imax=False, use_gate=False)` tự cài đặt, bằng `load_state_dict(..., strict=False)` sau khi đổi tên khóa và bỏ khóa `adapter`. Số khóa `missing/unexpected` chỉ được in ra màn hình, **không lưu lại** trong thư mục kết quả, nên không kiểm chứng được checkpoint đã nạp đủ hay chưa. Vì vậy dòng này chỉ nên coi là mốc tham khảo thô, không phải tái hiện chính xác DarkIR-m chính thức (PSNR blur 27.37 cho thấy trọng số có tác dụng, nhưng chưa chứng minh nạp đầy đủ).

### 2.2 SSIM và LPIPS (LPIPS thấp hơn = tốt hơn)

| Phương pháp | SSIM pooled | SSIM macro | LPIPS pooled | LPIPS macro |
|---|---:|---:|---:|---:|
| Input | 0.217 | 0.225 | 0.554 | 0.557 |
| Gamma 2.2 | 0.595 | 0.555 | 0.356 | 0.363 |
| CLAHE | 0.405 | 0.384 | 0.466 | 0.472 |
| MSRCR | 0.410 | 0.401 | 0.471 | 0.450 |
| DarkIR-Edge (no FFT) | 0.732 ± 0.001 | 0.672 ± 0.001 | 0.350 ± 0.019 | 0.361 ± 0.016 |
| DarkIR-Edge (ours) | 0.733 ± 0.003 | 0.672 ± 0.002 | 0.341 ± 0.022 | 0.357 ± 0.015 |
| Standard U-Net | **0.747** ± 0.004 | **0.678** ± 0.002 | **0.279** ± 0.002 | **0.307** ± 0.003 |
| DarkIR-m (ref) | 0.728 | — | **0.233** | — |

**Nhận xét:** U-Net dẫn đầu SSIM và LPIPS ở mức pooled (LPIPS 0.279 vs 0.341; chênh lớn hơn nhiều so với biến thiên giữa seed ±0.002–0.022). Đây là phân tích mô tả (chưa có kiểm định ghép cặp cho SSIM/LPIPS). Tuy PSNR Edge ≈ U-Net, ảnh U-Net gần GT hơn về đặc trưng cảm quan. Lưu ý LPIPS của Edge dao động mạnh giữa 2 seed (± 0.022).

### 2.3 Kết quả theo từng seed (pooled PSNR, dB, `final.pt`)

| Mô hình | seed 42 | seed 123 | Δseed |
|---|---:|---:|---:|
| DarkIR-Edge (ours) | 21.147 | 21.052 | 0.10 |
| DarkIR-Edge (no FFT) | 21.017 | 20.989 | 0.03 |
| Standard U-Net | 21.552 | 21.075 | 0.48 |

Seed 123 của U-Net (21.075) thấp hơn seed 42 của Edge (21.147): hai họ **đan xen nhau** theo seed. Độ biến thiên seed của U-Net (0.48 dB) lớn hơn cả chênh lệch trung bình Edge–U-Net (0.21 dB) — thêm một lý do để không kết luận hơn/kém.

> [!WARNING]
> Với **2 seed**, độ lệch giữa seed không đáng tin cậy và CI bootstrap chỉ phản ánh biến thiên giữa *ảnh test*, không bao hàm biến thiên giữa seed/khởi tạo.

### 2.4 Phân tích phụ: `best.pt` (`table_main_best.csv`)

| Phương pháp | PSNR pooled (final.pt) | PSNR pooled (best.pt) | PSNR macro (final) | PSNR macro (best) |
|---|---:|---:|---:|---:|
| DarkIR-Edge (no FFT) | 21.003 | 21.005 | 20.569 | 20.571 |
| DarkIR-Edge (ours) | 21.100 | 21.043 | 20.565 | 20.564 |
| Standard U-Net | 21.313 | 21.357 | 20.713 | 20.747 |

Chọn checkpoint theo `final.pt` hay `best.pt` thay đổi pooled PSNR ≤ 0.06 dB và **không đổi thứ tự/kết luận**. Xác nhận kết luận không nhạy với việc chọn checkpoint.

---

## 3. Kiểm định thống kê ghép cặp (`paired_tests.csv`)

ΔPSNR = A − B, trung bình 2 seed theo ảnh; CI = bootstrap 95%.

### 3.1 DarkIR-Edge (ours) − Standard U-Net

| Miền | n | ΔPSNR (dB) | 95% CI | p (t) | p (Wilcoxon) | % ảnh Edge thắng |
|---|---:|---:|---|---:|---:|---:|
| **pooled** | 190 | **−0.214** | [−0.530, +0.132] | 0.222 | 0.028 | 41.6% |
| blur | 40 | −0.234 | [−0.697, +0.243] | 0.343 | 0.227 | 47.5% |
| huawei | 30 | +0.152 | [−0.435, +0.814] | 0.644 | 0.612 | 46.7% |
| lolv2 | 100 | −0.320 | [−0.901, +0.263] | 0.287 | 0.127 | 36.0% |
| nikon | 20 | −0.191 | [−0.614, +0.222] | 0.418 | 0.522 | 50.0% |

**Áp dụng quy tắc đăng ký trước:** CI pooled chứa 0 ⇒ "ngang nhau" (kế hoạch: nếu Edge ≥ U-Net chỉ báo cáo khi CI loại 0; ở đây U-Net nhỉnh hơn về trung bình nhưng cũng không đạt ý nghĩa). Wilcoxon pooled (p = 0.028) là tín hiệu yếu nhất quán với hướng U-Net nhỉnh hơn; nó **không** đạt ngưỡng sau hiệu chỉnh đa kiểm định (5 phép thử, Bonferroni α = 0.01) và mâu thuẫn với t-test/CI, nên không dùng làm kết luận. Edge chỉ thắng 41.6% số ảnh pooled, lolv2 chỉ 36%.

### 3.2 DarkIR-Edge (ours) − DarkIR-Edge (no FFT)

| Miền | n | ΔPSNR (dB) | 95% CI | p (t) | p (Wilcoxon) | % ảnh Edge thắng |
|---|---:|---:|---|---:|---:|---:|
| pooled | 190 | +0.096 | [−0.122, +0.310] | 0.386 | 0.613 | 51.1% |
| blur | 40 | +0.098 | [−0.274, +0.528] | 0.628 | 0.995 | 55.0% |
| huawei | 30 | −0.185 | [−0.407, +0.074] | 0.155 | 0.061 | 33.3% |
| lolv2 | 100 | +0.231 | [−0.133, +0.602] | 0.222 | 0.163 | 55.0% |
| nikon | 20 | −0.162 | [−0.433, +0.131] | 0.272 | 0.368 | 50.0% |

**Kết luận:** mọi CI chứa 0 ⇒ **không có bằng chứng** nhánh FFT/FreMLP cải thiện (hay làm hại) PSNR. Dấu của chênh lệch đổi theo miền (dương ở blur/lolv2, âm ở huawei/nikon) → nhiễu, không phải xu hướng. Khác v2, xu hướng "FFT giúp blur" chỉ còn +0.10 dB (v2: +0.19) và vẫn không có ý nghĩa.

### 3.3 DarkIR-m tham chiếu

DarkIR-m (pooled 20.96) thấp hơn U-Net (21.31) và Edge (21.10) nhẹ ở pooled, nhưng **vượt xa ở blur** (27.37 vs 23.80/23.57, ≈ +3.6–3.8 dB) và thấp hơn ở huawei/nikon/lolv2 (18.63/16.42/20.01). Đây là so sánh **không công bằng** (DarkIR-m được huấn luyện chính thức trên toàn bộ LOL-Blur, 3.32 M tham số) và chưa chạy bootstrap riêng cho v3; chỉ dùng làm trần tham chiếu, không suy diễn tốt/xấu về mô hình nhỏ.

---

## 4. Chẩn đoán độ sáng (`diagnostic_gtmean.csv`)

`gap` = psnr_gtmean − psnr (dB): phần lỗi ước lượng do sai lệch độ sáng toàn cục.

| Mô hình | blur | huawei | lolv2 | nikon |
|---|---:|---:|---:|---:|
| DarkIR-Edge (ours) | 1.89 | 1.87 | 5.30 | 1.53 |
| DarkIR-Edge (no FFT) | 1.96 | 1.90 | 5.52 | 1.50 |
| Standard U-Net | 1.65 | 2.02 | 5.70 | 1.44 |
| Gamma 2.2 | 3.80 | 6.47 | 6.48 | 2.84 |
| CLAHE | 6.74 | 8.17 | 9.88 | 4.62 |

Gap của lolv2 vẫn lớn (5.3–5.7 dB) ở cả 3 mô hình học sâu: **phần lớn sai số trên LOLv2 là sai lệch độ sáng toàn cục**, không phải chi tiết, và nó **không phụ thuộc kiến trúc** (Edge, no-FFT, U-Net gần như bằng nhau) → không có bằng chứng "FFT sửa độ sáng toàn cục tốt hơn". Trên LOLv2, U-Net đạt PSNR cao hơn (21.39) dù gap lớn hơn (5.70), tức sau khi bù độ sáng nó cũng tốt hơn (PSNR_gtmean 27.09 vs 26.37).

---

## 5. Phân tích bổ sung (`results_extra.csv`, `paired_tests_extra.csv`)

Giá trị trung bình theo miền (`table_extra.csv`). `lum_bias` = mean Y(out) − mean Y(GT) (Y theo BT.601, thang 0–255; dương = quá sáng); `lum_abs_err` = |lum_bias| **tính trên mean toàn ảnh** (không phải sai số theo pixel); `lum_emd` = Wasserstein-1 giữa histogram Y; `ciede2000` = ΔE₀₀ trung bình theo pixel; `sharp_ratio` = Var(Laplacian Y_out) / Var(Laplacian Y_gt); `dark10/bright10_mae` = MAE trên 10% pixel tối/sáng nhất của GT. (Định nghĩa đã đối chiếu với cell 29 của notebook.)

| Miền | Mô hình | lum_bias | lum_abs_err | CIEDE2000 | sharp_ratio | dark10_mae | bright10_mae |
|---|---|---:|---:|---:|---:|---:|---:|
| blur | Edge | −0.35 | 9.95 | 6.35 | 0.181 | 11.51 | 13.12 |
| blur | no-FFT | +1.27 | 9.92 | 6.36 | 0.177 | 12.15 | 12.94 |
| blur | U-Net | +1.49 | 8.74 | 6.29 | 0.232 | 11.77 | 11.33 |
| huawei | Edge | −11.60 | 15.67 | 8.68 | 0.067 | 10.55 | 25.70 |
| huawei | no-FFT | −8.83 | 14.01 | 8.39 | 0.066 | 10.48 | 24.83 |
| huawei | U-Net | −6.47 | 14.15 | 8.55 | 0.121 | 11.44 | 24.63 |
| lolv2 | Edge | +12.85 | 19.15 | 10.10 | 0.204 | 13.30 | 27.13 |
| lolv2 | no-FFT | +13.29 | 20.11 | 10.35 | 0.176 | 13.67 | 28.19 |
| lolv2 | U-Net | +12.24 | 18.80 | 9.70 | 0.365 | 12.14 | 26.26 |
| nikon | Edge | −4.76 | 20.58 | 10.69 | 0.129 | 21.38 | 25.20 |
| nikon | no-FFT | +0.07 | 19.47 | 10.40 | 0.113 | 22.94 | 23.48 |
| nikon | U-Net | −4.98 | 19.65 | 10.75 | 0.240 | 19.87 | 26.01 |

Kết quả có ý nghĩa (CI bootstrap loại 0, pooled):

| Chỉ số | Edge − U-Net | 95% CI | Edge − noFFT | 95% CI |
|---|---:|---|---:|---|
| sharp_ratio | **−0.116** (0.170 vs 0.285) | [−0.126, −0.105] | **+0.017** | [+0.014, +0.021] |
| dark10_mae (thấp hơn tốt hơn) | **+0.57** (Edge kém hơn) | [+0.04, +1.10] | **−0.49** (Edge tốt hơn) | [−0.91, −0.04] |
| lum_abs_err | +0.78 | [−0.39, +1.83] (không ý nghĩa) | −0.12 | [−0.89, +0.65] (không ý nghĩa) |
| lum_emd | +0.86 | [−0.18, +1.85] (không ý nghĩa) | −0.12 | [−0.83, +0.59] (không ý nghĩa) |
| CIEDE2000 | +0.24 | [−0.05, +0.52] (không ý nghĩa) | −0.06 | [−0.26, +0.14] (không ý nghĩa) |
| bright10_mae | +0.92 | [−0.40, +2.11] (không ý nghĩa) | −0.20 | [−0.98, +0.55] (không ý nghĩa) |

**Diễn giải (có thận trọng):**
1. **Độ nét:** U-Net giữ chi tiết cao tần tốt hơn Edge rõ rệt (sharp_ratio 0.285 vs 0.170, CI rất hẹp, mọi miền đều có ý nghĩa). Cả ba mô hình đều **làm mờ hơn GT nhiều** (tỉ lệ < 0.4); ở miền blur chỉ đạt 0.18–0.23 ⇒ **chưa chứng minh được khả năng khử mờ**; PSNR blur cao chủ yếu đến từ tăng sáng. Phương sai Laplacian nhạy với nhiễu/kết cấu, chỉ là proxy.
2. **Nhánh FFT** cho độ nét cao hơn no-FFT một chút (+0.017, rất nhỏ nhưng nhất quán ở lolv2/nikon) và lỗi vùng tối nhỏ hơn (−0.49) — đây là *quan sát thăm dò* đáng ghi nhận nhưng **không đồng nghĩa PSNR tốt hơn** (PSNR không khác biệt, mục 3.2) và chưa hiệu chỉnh đa kiểm định (nhiều chỉ số × 5 miền).
3. **Độ sáng:** lolv2 bị quá sáng ~+12–13 (thang 0–255) ở cả 3 mô hình; huawei thiếu sáng (−6 đến −12), Edge thiếu nhiều nhất. Không có chênh lệch có ý nghĩa giữa mô hình về `lum_abs_err`/EMD/CIEDE2000 → giả thuyết "Edge tông tự nhiên hơn" **không được số liệu ủng hộ**.
4. U-Net tốt hơn Edge ở vùng tối (dark10_mae, đặc biệt lolv2: +1.16, CI [0.30, 1.99]) và vùng sáng ở blur (bright10_mae +1.78, CI [0.25, 3.37]).

---

## 6. Hiệu năng/Hiệu quả (`efficiency.csv`, GPU NVIDIA L4)

| Mô hình | Params | GFLOPs @256² | Latency FP32 @600×400 | FPS FP32 | Latency FP16 | FPS FP16 | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| DarkIR-Edge (C=24) | 1.924 M | 8.75 | 31.71 ms | 31.5 | 26.23 ms | 38.1 | 236.7 MB |
| DarkIR-Edge no-FFT | 1.495 M | 7.50 | 27.31 ms | 36.6 | 21.57 ms | 46.4 | 220.6 MB |
| Standard U-Net (base=32) | 1.928 M | 18.45 | **13.23 ms** | **75.6** | **11.55 ms** | **86.5** | 269.7 MB |

- **Cách đo (cell 20 của notebook):** FLOPs dùng `FlopCounterMode` ở 256×256 và theo chính ghi chú trong notebook **chưa tính phần FFT** (nên FLOPs của Edge bị đếm thiếu; con số 8.75 G là cận dưới). Latency: batch 1, đầu vào 400×600 (cao×rộng), 10 lần warm-up + 50 lần đo trung bình, `cuda.synchronize`, **một lần đo duy nhất** (không có độ lệch chuẩn). Peak VRAM là `max_memory_allocated` đọc sau lượt FP16 (đã reset trước mỗi lượt) nên là đỉnh của lượt FP16, chưa gồm bộ nhớ cache allocator.
- Edge có **ít hơn 53% FLOPs (chưa tính FFT)** nhưng **chậm hơn ≈2.40× (FP32) và ≈2.27× (FP16)** so với U-Net: FLOPs không phản ánh latency (depthwise/FFT/gating bị giới hạn bởi băng thông bộ nhớ; U-Net chỉ conv 3×3 chuẩn).
- Bỏ FFT giảm 22% tham số, 14% FLOPs, ≈14% latency FP32 (≈18% FP16) **mà không mất PSNR có ý nghĩa** (mục 3.2) → no-FFT hợp lý hơn bản đầy đủ nếu xét triển khai, dù bản FFT hơi nhỉnh về độ nét/vùng tối (mục 5).
- Peak VRAM v3 (237/221/270 MB) khác v2 (308/293/342 MB); hiệu năng đo trong phiên khác nên chỉ so sánh **trong cùng v3**. Chưa đo trên thiết bị biên thực sự ⇒ không dùng cụm "edge-friendly/real-time" như kết quả đã kiểm chứng.

---

## 7. Quá trình huấn luyện (`history.csv`)

Val macro (EMA) trên 150 ảnh (lolv2/huawei/blur):

| Run | Val epoch 70 (final) | Val tốt nhất (epoch) | Chênh final − best |
|---|---:|---|---:|
| darkir_edge s42 | 21.74 | 21.89 (ep 50) | −0.15 |
| darkir_edge s123 | 21.38 | 21.61 (ep 50) | −0.23 |
| darkir_edge_nofft s42 | 21.45 | 21.54 (ep 55) | −0.09 |
| darkir_edge_nofft s123 | 21.36 | 21.36 (ep 65/70) | 0.00 |
| unet s42 | 21.42 | 21.42 (ep 70) | 0.00 |
| unet s123 | 21.69 | 21.70 (ep 50) | −0.01 |

Quan sát:
1. **Lịch cố định đã loại confound độ dài:** cả 6 run đều qua pha lr thấp (cosine → 1e-6), cùng 28,000 bước.
2. Họ Edge đạt đỉnh val ở epoch ~50 rồi **giảm nhẹ** (−0.1 đến −0.2) → dấu hiệu quá khớp nhẹ/nhiễu val; U-Net tăng đều và ổn định đến cuối (s42 tăng tới epoch 70). Với 150 ảnh val (50 blur) và thay đổi ≤ 0.2 dB, đây nằm trong mức nhiễu — chỉ nêu như xu hướng.
3. Val của U-Net seed 42 (21.42) và seed 123 (21.69) cũng chênh 0.27 dB → biến thiên seed cùng bậc với chênh lệch kiến trúc.
4. Notebook lưu `last.pt` mỗi 5 epoch để resume; khi resume, RNG không được khôi phục nên run đó không tái lập từng bit. Thư mục kết quả không có log cho biết run nào đã resume, nên không khẳng định có hay không.
5. `history.csv` có đủ 70 dòng cho cả 6 run và epoch 70 luôn có val ⇒ khớp với quy tắc "dừng khi đủ 70 epoch" trong mã (không còn đường thoát sớm).

---

## 8. So sánh v2 → v3 (cùng tập test 190 ảnh)

| Chỉ số (pooled PSNR) | v2 | v3 | Δ |
|---|---:|---:|---:|
| DarkIR-Edge | 20.74 | 21.10 | +0.36 |
| DarkIR-Edge (no FFT) | 20.76 | 21.00 | +0.24 |
| Standard U-Net | 21.36 | 21.31 | −0.05 |
| Edge − U-Net (CI) | −0.615 [−0.891, −0.335] | −0.214 [−0.530, +0.132] | thu hẹp |
| blur: Edge / U-Net | 22.93 / 23.37 | 23.57 / 23.80 | +0.64 / +0.43 |

Edge cải thiện nhiều hơn U-Net từ v2 sang v3; nikon (miền chưa thấy) lại giảm nhẹ ở Edge (17.24 → 16.96). **Không thể quy nguyên nhân** cho một yếu tố duy nhất: v3 thay đồng thời (a) độ dài/lịch huấn luyện (bỏ ES, 70 epoch cố định, T_max 70), (b) dữ liệu blur 150 → 400 ảnh train, (c) tập val và checkpoint chính thức (`final.pt` thay `best.pt`). Chênh lệch giữa các run v2/v3 (0.2–0.4 dB) cùng bậc độ lớn với biến thiên seed (0.1–0.5 dB).

---

## 9. Phân tích định tính

Ảnh minh họa: `results/qualitative.png`, `qualitative_lolblur_5samples.png`, `qualitative_with_darkir_m.png`, `curves.png`, `visual_samples/`, `lolblur_visual_samples/`. Báo cáo này đã đối chiếu mã notebook và CSV nhưng **chưa xem trực tiếp từng ảnh**; phần nhận xét cảm quan cần người báo cáo bổ sung. Cơ sở định lượng cho nhận xét cảm quan: LPIPS (U-Net 0.279 < Edge 0.341), độ nét (U-Net 0.285 vs Edge 0.170), thiên lệch độ sáng (lolv2 quá sáng, huawei thiếu sáng). Khảo sát mù theo kế hoạch (mục 4) chưa có số liệu trong thư mục kết quả.

---

## 10. Các mối đe dọa đến tính hợp lệ (limitations)

1. **Chỉ 2 seed/mô hình**; CI chỉ phản ánh biến thiên giữa ảnh test, không giữa seed. Biến thiên seed (U-Net 0.48 dB) lớn hơn chênh lệch Edge–U-Net (0.21 dB).
2. **Thay đổi đồng thời nhiều yếu tố giữa v2 và v3** (mục 8) → không quy nguyên nhân được.
3. **Tập test nhỏ**: 20–100 ảnh/miền; nikon chỉ 20 ảnh và là miền chưa thấy (PSNR ~17 dB cho mọi mô hình).
4. **Blur: 40 ảnh test từ 30 chuỗi video**, ảnh cùng chuỗi tương quan mạnh → CI có thể hẹp quá mức; train 400 ảnh vẫn nhỏ so với 10.2k cặp của DarkIR-m.
5. **DarkIR-m không công bằng và có thể nạp không đầy đủ** (quy mô dữ liệu/huấn luyện chính thức; nạp `strict=False` vào lớp tự cài, không lưu số khóa thiếu/thừa — xem cảnh báo mục 2.1) — chỉ là mốc tham khảo thô.
5b. **So sánh Edge − U-Net là so sánh cả thiết kế** (Max-RGB prior, gated skip, side-loss, FFT) với một U-Net thuần; chỉ Edge − no-FFT cô lập riêng nhánh FFT.
6. **Tổng loss không so sánh được** giữa họ Edge (có side-loss) và U-Net; so sánh chỉ dựa trên chỉ số test.
7. **Latency chỉ đo trên L4**, không phải thiết bị biên; VRAM khác v2.
8. **Phân tích bổ sung (mục 5)** gồm nhiều chỉ số × 5 miền, chưa hiệu chỉnh đa kiểm định → chỉ là thăm dò. Phương sai Laplacian chỉ là proxy của độ nét.
9. **Không có kiểm định ghép cặp cho SSIM/LPIPS**; **chưa có khảo sát mù**; chưa kiểm tra định tính bằng mắt.
10. Chưa có thí nghiệm tách ảnh hưởng của dữ liệu blur (có/không blur) lên các miền khác.

---

## 11. Kết luận trung thực (đối chiếu quy tắc đăng ký trước)

1. **Chất lượng:** Edge **ngang** U-Net về PSNR pooled (Δ = −0.21 dB, CI [−0.53, +0.13] chứa 0); không miền nào có ý nghĩa. U-Net vẫn nhỉnh hơn về SSIM (0.747 vs 0.733), LPIPS (0.279 vs 0.341) và độ nét (CI loại 0), nhưng các chỉ số này chưa kiểm định/ hiệu chỉnh như PSNR (độ nét có CI riêng).
2. **Giả thuyết FFT/FreMLP:** *không được hỗ trợ* ở PSNR (Δ = +0.10 dB, mọi CI chứa 0). Chỉ có quan sát thăm dò rất nhỏ về độ nét (+0.017) và lỗi vùng tối (−0.49) cần kiểm chứng thêm trước khi nêu.
3. **Hiệu quả:** Edge ít FLOPs hơn 53% (chưa tính FFT) nhưng chậm hơn ≈2.4× trên L4 → không tuyên bố nhanh/edge-friendly/real-time-advantage. no-FFT rẻ hơn 22% tham số, nhanh hơn ≈14% với PSNR tương đương.
4. **So với v2:** kết luận "U-Net tốt hơn Edge có ý nghĩa" **bị suy yếu** thành "ngang nhau" khi dùng lịch cố định 70 epoch và nhiều dữ liệu blur hơn; kết luận về FFT không đổi. Không tách được nguyên nhân.
5. **Giá trị tích cực thực sự:** (a) CNN nhỏ ~1.9 M vượt Gamma +4.1 dB, vượt CLAHE/MSRCR ~+9 dB; (b) pipeline đánh giá chặt chẽ, cách ly chuỗi, lịch huấn luyện đồng đều, quy tắc đăng ký trước; (c) kết quả âm/trung tính được nêu trung thực.
6. **Câu chữ khuyến nghị cho luận văn/slide:** "Với cùng số tham số và lịch huấn luyện cố định, DarkIR-Edge đạt PSNR tương đương Standard U-Net (chưa phân biệt được thống kê) nhưng kém về độ nét và chậm hơn trên GPU; nhánh FFT/FreMLP không cho cải thiện PSNR đo được ở quy mô này." Tránh: "Edge tốt hơn U-Net", "FFT cải thiện độ sáng/khử mờ", "real-time trên thiết bị biên", "Edge cho tông màu tự nhiên hơn".

### Việc nên làm tiếp
- Xem trực tiếp ảnh định tính và bổ sung mục 9; làm khảo sát mù nếu muốn có bằng chứng cảm quan.
- Tăng số seed (≥ 3–5) cho cả 3 họ để giảm nhiễu giữa seed.
- Thí nghiệm kiểm soát (v3 dữ liệu × v2 lịch) nếu cần quy nguyên nhân của thay đổi so với v2.

---

## Phụ lục — Danh sách file nguồn

- `experiments/v3/results/`: `table_main.csv`, `table_main_best.csv`, `paired_tests.csv`, `paired_tests_extra.csv`, `table_extra.csv`, `results_extra.csv`, `efficiency.csv`, `diagnostic_gtmean.csv`, `results_per_image.csv`, `results_per_image_best.csv`, `results_per_image_with_ref.csv`
- `experiments/v3/checkpoints/<run>/history.csv` (6 run), `experiments/v3/run_config.json`, `split_manifest.json`
- Notebook: `colab-results/LLIE_DarkIR_Edge_Colab_v3.ipynb`
