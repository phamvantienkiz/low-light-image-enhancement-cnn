# Báo Cáo Kết Quả Thực Nghiệm — LLIE DarkIR-Edge (v4)

**Ngày tạo báo cáo:** 2026-11-04
**Nguồn dữ liệu:** `colab-results/llie_project_v4-20261104/llie_project_v4/`
**Notebook thực thi thực tế:** `colab-results/LLIE_DarkIR_Edge_Colab_v4.ipynb`
**Tham chiếu kế hoạch & đăng ký trước:** `presentation/report/ke_hoach_v4.md`, `presentation/report/ke_hoach_v4_du_lieu.md`, `presentation/report/ke_hoach_v4_notebook.md`, `docs/ai/preregistration_v4.md`
**Báo cáo đối chiếu:** `presentation/report/Bao_Cao_Ket_Qua_v3_LLIE_DarkIR_Edge.md` (v3), `presentation/report/Bao_Cao_Ket_Qua_v2_LLIE_DarkIR_Edge.md` (v2)

> Báo cáo này trình bày kết quả thực nghiệm v4 một cách trung lập, dựa hoàn toàn trên số liệu xuất ra từ các file `.csv`, `.json` và nhật ký chạy thực tế trên Google Colab (GPU NVIDIA L4, High-RAM 53.0 GB). Mọi con số đều có thể kiểm chứng trực tiếp từ thư mục kết quả và tất cả các nhận định đều đối chiếu nghiêm ngặt với **6 quy tắc kết luận đã đăng ký trước** (`preregistration_v4.md`).

---

## 0. Tóm tắt điều hành (TL;DR)

1. **Tập huấn luyện mở rộng gấp 4.88 lần (`6,777` cặp từ 5 miền) với `QuotaSampler` cố định `4,800` cặp/epoch (`42,000` steps/run):** So với v3 (`1,389` cặp từ 3 miền, `28,000` steps), v4 mở rộng `blur` từ `450` lên `1,858` cặp (giữ nguyên `150` video train tách biệt hoàn toàn với `20` video val và `20` video test), mở rộng `huawei` từ `300` lên `2,380` cặp (loại trừ 50 ảnh val và vùng đệm `gap = 10` ở hai đầu), giữ `639` cặp `lolv2`, đồng thời bổ sung hai miền mới: `1,000` cặp `nikon` và `900` cặp `lolv2syn`. Mỗi epoch lấy mẫu cố định `4,800` cặp (`600` steps/epoch × `70` epochs = `42,000` steps) theo hạn ngạch: `1,278 lolv2` (26.6%) + `1,122 huawei` (23.4%) + `1,600 blur` (33.3%) + `400 nikon` (8.3%) + `400 lolv2syn` (8.3%).
2. **Đã huấn luyện đầy đủ cả 3 kiến trúc học sâu × 2 seeds (`42, 123`), bao gồm cả biến thể cắt bỏ `DarkIR-Edge (no FFT)`:** Mặc dù kế hoạch ban đầu dự kiến có thể lược bỏ nhánh `no-FFT` để tiết kiệm thời gian GPU, trong thực tế chạy Colab (Cell Ablation), cấu hình `RUN_NOFFT_ABLATION = True` với `seeds = [42, 123]` đã được bật và chạy trọn vẹn 70 epochs cho cả 2 seeds.
3. **So sánh nội bộ v4 (`final.pt` EMA) — Ba mô hình học sâu tiếp tục hòa nhau về mặt thống kê trên `pooled` (190 ảnh test):**
   - `DarkIR-Edge (no FFT)`: PSNR pooled = **21.10 ± 0.32 dB** (macro = 20.62 ± 0.20 dB, SSIM = 0.743, LPIPS = 0.303).
   - `Standard U-Net`: PSNR pooled = **21.03 ± 0.15 dB** (macro = 20.52 ± 0.10 dB, SSIM = **0.745**, LPIPS = **0.277**).
   - `DarkIR-Edge (ours)`: PSNR pooled = **20.97 ± 0.06 dB** (macro = 20.55 ± 0.10 dB, SSIM = 0.743, LPIPS = 0.307), nhưng có độ ổn định giữa các seed rất cao (chênh lệch giữa 2 seed chỉ `0.086 dB` so với `0.216 dB` của `U-Net` và `0.447 dB` của `no-FFT`).
   - Kiểm định ghép cặp `DarkIR-Edge (ours)` vs `Standard U-Net` trên `pooled`: $\Delta\text{PSNR} = -0.063\text{ dB}$, 95% CI $[-0.360, +0.229]$, $p(t) = 0.680$, $p(\text{Wilcoxon}) = 0.725$, tỷ lệ thắng `50.5%` (96/190 ảnh). Trên từng miền riêng lẻ (`blur`, `huawei`, `lolv2`, `nikon`), mọi khoảng tin cậy 95% đều chứa `0`.
4. **Kiểm định trực tiếp giả thuyết nhánh tần số (`DarkIR-Edge` vs `DarkIR-Edge no-FFT` trong v4):** $\Delta\text{PSNR}_{\text{pooled}} = -0.131\text{ dB}$, 95% CI $[-0.371, +0.105]$, $p(t) = 0.290$. Ngay cả khi đã tăng dữ liệu `blur` lên gấp ~4.1 lần (`1,858` cặp) và tăng số bước cập nhật lên `42,000` steps, việc thêm nhánh `FreMLP` + `FFT loss` vẫn **không** tạo ra cải thiện có ý nghĩa thống kê so với biến thể chỉ dùng nhánh không gian (`no-FFT`), trong khi làm tăng thời gian suy luận từ `27.79 ms` lên `32.29 ms` (FP32) và tăng thời gian huấn luyện từ `153.9 phút` lên `207.9 phút`/run.
5. **Đối chứng trực tiếp `v4 − v3` theo các quy tắc đăng ký trước (`paired_tests_v3_v4.csv`) — Kết quả phủ định (Negative / Null Result) rất rõ ràng:**
   - **Quy tắc 1 (Bảo vệ miền tĩnh `lolv2 + huawei`, 130 ảnh, yêu cầu $\text{CI}_{\text{lo}} > -0.3\text{ dB}$):** **Vi phạm ở cả 3 mô hình.** Cụ thể, `DarkIR-Edge (ours)` giảm `-0.185 dB` với 95% CI $[-0.519, +0.101]$ ($\text{CI}_{\text{lo}} = -0.519 < -0.3$); `Standard U-Net` giảm có ý nghĩa thống kê `-0.407 dB` với 95% CI $[-0.765, -0.067]$ ($p(t)=0.023$); `DarkIR-Edge (no FFT)` tăng nhẹ `+0.060 dB` nhưng phương sai lớn nên 95% CI $[-0.369, +0.491]$ vẫn vượt ngưỡng `-0.3 dB`. Nguyên nhân gốc rễ nằm ở sự dịch chuyển phân phối độ sáng: khi giảm tỷ trọng `lolv2` từ `37.5%` (v3) xuống `26.6%` (v4) và thêm các miền có GT sáng hơn (`nikon`, `lolv2syn`, mở rộng `huawei`), các mô hình dự đoán sáng hơn trên `lolv2` (`lum_bias` tăng từ `+11.5..+12.2` ở v3 lên `+14.9..+16.8` ở v4), kéo PSNR thô trên `lolv2` giảm `-0.33 dB` (`DarkIR-Edge`) và `-0.53 dB` (`U-Net`), dù trên `huawei` `DarkIR-Edge` lại tăng `+0.304 dB` (95% CI $[+0.004, +0.611]$).
   - **Quy tắc 2 (Đưa `nikon` vào tập huấn luyện có thu hẹp khoảng cách miền hay không?):** `nikon` chuyển từ miền chưa từng thấy (unseen ở v2/v3) sang miền đã thấy trong huấn luyện (in-domain ở v4 với `1,000` ảnh train, `400` ảnh/epoch). Tuy nhiên, PSNR trên `nikon` (20 ảnh test) gần như **không đổi** ở `DarkIR-Edge` ($16.961$ vs $16.955\text{ dB}$, $\Delta = +0.006\text{ dB}$, 95% CI $[-0.625, +0.657]$, $p = 0.987$) và `no-FFT` ($\Delta = +0.015\text{ dB}$), thậm chí giảm nhẹ ở `U-Net` ($16.686$ vs $17.147\text{ dB}$, $\Delta = -0.461\text{ dB}$, $p = 0.298$). Điều này chứng minh nút thắt trên LSRW-Nikon không nằm ở việc thiếu mẫu huấn luyện cùng cảm biến, mà nằm ở độ bất định phơi sáng nội tại giữa cặp ảnh Low/GT (`psnr_gtmean` đã đạt tới `23.09–23.54 dB`).
   - **Quy tắc 3 (Tăng dữ liệu `blur` từ 450 lên 1,858 cặp có cải thiện khử mờ hay không?):** Trên `blur` (40 ảnh test), `DarkIR-Edge (ours)` gần như đứng yên ($23.537$ vs $23.569\text{ dB}$, $\Delta = -0.032\text{ dB}$, 95% CI $[-0.534, +0.433]$, $p = 0.896$); `Standard U-Net` nhích nhẹ $+0.207\text{ dB}$ ($24.009$ vs $23.802\text{ dB}$, 95% CI $[-0.237, +0.665]$, $p = 0.369$) và `no-FFT` nhích $+0.247\text{ dB}$ ($p = 0.427$). Không có sự dịch chuyển nào đạt ý nghĩa thống kê, và cả 3 mô hình nhẹ (~1.5–1.9M tham số) vẫn cách rất xa `DarkIR-m` gốc (`27.72 dB` trên `blur`, lớn gấp ~2.9 lần số tham số).
6. **Quyết định theo Quy tắc 5 đăng ký trước:** Do v4 không vượt qua Quy tắc 1 (ngưỡng bảo vệ `lolv2 + huawei`) và không tạo ra cải thiện có ý nghĩa thống kê trên `blur` hay `pooled` so với v3, **bộ trọng số của v3 được giữ lại làm bộ trọng số chính thức cho ứng dụng Demo**; toàn bộ kết quả v4 được báo cáo nguyên vẹn như một **Ablation quy mô lớn chứng minh hiện tượng bão hòa dung lượng (capacity saturation) và đánh đổi phân phối đa miền (multi-domain luminance trade-off)**.

---

## 1. Thiết lập thực nghiệm (đã chạy thực tế)

### 1.1. Dữ liệu (`split_manifest.json` & Cell 4–5 trong Notebook)

Tập `val` (150 cặp) và tập `test` (190 cặp) được giữ **nguyên vẹn 100%** như ở v3 để đảm bảo mọi phép kiểm định ghép cặp (`paired_tests_v3_v4.csv`) là đối chứng trực tiếp trên từng ảnh. Tập `train` được mở rộng từ `1,389` cặp (3 miền ở v3) lên **`6,777` cặp (5 miền ở v4)**:

| Tập | Tổng số cặp | `lolv2` (LOL-v2 Real) | `huawei` (LSRW-Huawei) | `blur` (LOL-Blur) | `nikon` (LSRW-Nikon) | `lolv2syn` (LOL-v2 Synthetic) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (v4)** | **6,777** | 639 | 2,380 | 1,858 (150 video) | 1,000 | 900 |
| *(Đối chiếu Train v3)* | *(1,389)* | *(639)* | *(300)* | *(450 — 150 video)* | *(0)* | *(0)* |
| **Val (giữ nguyên v3)** | **150** | 50 | 50 | 50 (20 video) | 0 | 0 |
| **Test (giữ nguyên v3)** | **190** | 100 | 30 | 40 (20 video) | 20 *(in-domain ở v4)* | 0 |

Chi tiết kỹ thuật tách dữ liệu và chống rò rỉ (data leakage prevention) được xác nhận từ `split_manifest.json` và log thực thi Cell 4–5:
- **`lolv2` (`639` train / `50` val / `100` test):** Giữ nguyên hoàn toàn từ v2/v3 (`low00001..00639.png` cho train, loại trừ 50 ảnh val `low00640..00689.png`).
- **`huawei` (`2,380` train / `50` val / `30` test):** Tải toàn bộ `2,450` cặp Huawei Train từ HuggingFace (`Kbsriharsha/LSRW_Dataset`), loại bỏ 50 ảnh val (`00951..01000.jpg`) cùng hai vùng đệm thời gian `gap = 10` ở hai phía (`00941..00950.jpg` và `01001..01010.jpg`, tổng cộng loại 70 ảnh), thu được chính xác `2,380` cặp hợp lệ.
- **`blur` (`1,858` train / `50` val / `40` test):** Giữ nguyên danh sách `150` chuỗi video train của v3 (tách biệt hoàn toàn cấp độ video với `20` chuỗi val và `20` chuỗi test; kiểm tra giao cắt tập hợp `assert len(train & val) == 0` và `len(train & test) == 0` trong Cell 5 đạt tuyệt đối). Thay vì chỉ lấy 3 frame/chuỗi như v3 (`450` cặp), v4 lấy mẫu thưa với khoảng cách tối thiểu `MIN_FRAME_GAP = 4` và tối đa `15` frames/chuỗi: trong 150 chuỗi video train, có 123 chuỗi cung cấp 13 frames, 16 chuỗi có 12 frames, 4 chuỗi có 8 frames và 7 chuỗi có 5 frames, tổng cộng `1,858` cặp.
- **`nikon` (`1,000` train / `0` val / `20` test):** Lấy mẫu đều (`stride ≈ 3.13`) `1,000` cặp từ `3,150` ảnh Nikon Train trên HuggingFace sau khi đã loại trừ toàn bộ dải tên file trùng với 20 ảnh Nikon Test (`3001..3020.jpg`). **Lưu ý phương pháp luận quan trọng:** Trong `split_manifest.json`, cờ `"nikon_test_in_domain": true` được bật rõ ràng — tức ở v4, tập test `nikon` (20 ảnh) đánh giá khả năng **khái quát hóa nội miền (in-domain generalization)** chứ không còn là zero-shot cross-sensor như ở v2 và v3.
- **`lolv2syn` (`900` train / `0` val / `0` test):** Bổ sung `900` cặp ảnh tổng hợp (`384×384`) từ phân nhánh `Synthetic/Train` của LOL-v2 để tăng đa dạng cấu trúc cảnh chụp mà không làm thay đổi tập `val` hay `test`.

### 1.2. Cấu hình huấn luyện thực tế (kiểm chứng từ `LLIE_DarkIR_Edge_Colab_v4.ipynb`)

1. **Cơ chế nạp dữ liệu lai (Hybrid RAM Cache + Lazy SSD Disk I/O — Cell 6):**
   - Để tránh tràn RAM hệ thống khi tập train tăng lên `6,777` cặp ảnh nhưng vẫn giữ tốc độ đọc nhanh trên Google Colab (High-RAM runtime `53.0 GB` System RAM, GPU NVIDIA L4), lớp `MultiDomainTrainDataset` tự động phân loại theo ngưỡng `> 1,000` ảnh hoặc miền `blur`:
   - **Nạp sẵn vào RAM (`lazy = False`):** `lolv2` (639 cặp, `1.89 GB`), `nikon` (1,000 cặp, `2.74 GB`), `lolv2syn` (900 cặp, `0.78 GB`) → tổng bộ nhớ đệm RAM = **`5.40 GB`** (RAM hệ thống thực tế chỉ dùng `~9.91 / 53.0 GB`, tương đương `20.2%`).
   - **Đọc trực tiếp từ ổ đĩa cục bộ SSD (`lazy = True`):** `blur` (`1,858` cặp ảnh PNG độ phân giải `1120×640`) và `huawei` (`2,380` cặp ảnh JPG). Toàn bộ dữ liệu đã được giải nén ra ổ đĩa cục bộ `/content/llie_v4_local_data` trước khi train nên không phát sinh nghẽn I/O qua Google Drive.

2. **Bộ lấy mẫu hạn ngạch cố định `QuotaSampler` (Cell 7):**
   - Mỗi epoch cố định chính xác **`4,800` cặp ảnh** (`batch_size = 8` → **`600` steps/epoch**, tăng `+50%` số bước mỗi epoch so với `400` steps/epoch của v3):
     - `lolv2`: **`1,278` cặp/epoch** (`26.625%`) — lấy mẫu có hoàn lại (~2.0× từ 639 ảnh gốc) bằng `torch.Generator` gắn seed riêng cho từng epoch (`seed + epoch * 100 + d_idx`).
     - `huawei`: **`1,122` cặp/epoch** (`23.375%`) — lấy mẫu ngẫu nhiên không hoàn lại từ `2,380` ảnh.
     - `blur`: **`1,600` cặp/epoch** (`33.333%`) — lấy mẫu ngẫu nhiên không hoàn lại từ `1,858` ảnh.
     - `nikon`: **`400` cặp/epoch** (`8.333%`) — lấy mẫu ngẫu nhiên không hoàn lại từ `1,000` ảnh.
     - `lolv2syn`: **`400` cặp/epoch** (`8.333%`) — lấy mẫu ngẫu nhiên không hoàn lại từ `900` ảnh.
   - Sau khi gộp đủ `4,800` chỉ số từ 5 miền, `QuotaSampler` xáo trộn toàn cục (`global permutation`) để mỗi mini-batch 8 ảnh chứa hỗn hợp ngẫu nhiên của cả 5 miền.

3. **Siêu tham số tối ưu hóa & lịch trình học (Cell 3, Cell 11–12):**
   - **Số epochs cố định:** `70` epochs × `600` steps/epoch = **`42,000` gradient steps/run** (tăng đúng `+50%` so với `28,000` steps của v3). `use_early_stopping = False` để bộ điều lịch `CosineAnnealingLR(T_max = 42,000, eta_min = 1e-6)` đi hết chu kỳ hội tụ tự nhiên.
   - **Tối ưu hóa:** `AdamW(lr = 2e-4, weight_decay = 1e-4, betas = (0.9, 0.999))`, `patch_size = 256`, `batch_size = 8`, cắt chuẩn gradient `clip_grad_norm_(max_norm = 1.0)`, huấn luyện hỗn hợp `AMP FP16` (`torch.amp.autocast('cuda', enabled=True)`), và trung bình trượt hàm mũ `EMA(decay = 0.999)`.
   - **Hàm mất mát `CompositeLoss` (Cell 11):** Giữ nguyên công thức của v2/v3:
     $$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{Charb}} + 0.1 \cdot \mathcal{L}_{\text{SSIM}} + 0.1 \cdot \mathcal{L}_{\text{Edge}} + 0.1 \cdot \mathcal{L}_{\text{Color}} + 0.05 \cdot \mathcal{L}_{\text{FFT}} + 0.2 \cdot \mathcal{L}_{\text{Side}}$$
     trong đó $\mathcal{L}_{\text{Side}}$ là mất mát Charbonnier đa tỉ lệ ở độ phân giải $1/4$ từ đầu ra phụ `side_out` của decoder (đối với `Standard U-Net` không có `side_out` thì thành phần này bằng `0`, và đối với `DarkIR-Edge (no FFT)` thì $\mathcal{L}_{\text{FFT}} = 0$).

4. **Các mô hình và seeds thực tế đã huấn luyện (Cell 14 & Cell Ablation):**
   - Quá trình huấn luyện được chia làm 2 phiên Colab nối tiếp (`SEEDS_TO_RUN = [42]` ở phiên 1 và `SEEDS_TO_RUN = [123]` ở phiên 2) thông qua cơ chế lưu/khôi phục trạng thái đầy đủ (`last.pt` chứa `model`, `ema`, `opt`, `sched`, `scaler`, `history`).
   - Cả 3 kiến trúc học sâu đều đã hoàn tất 100% chặng đường `70/70` epochs trên cả 2 seeds `[42, 123]` (tổng cộng **6 runs × 42,000 steps = 252,000 steps**):
     - `DarkIR-Edge (ours)` (`C=24`, `enc_blk_nums=[2,2,2]`, `middle_blk_num=4`, `dec_blk_nums=[2,2,2]`, `dilations=[1,4,9]`, `use_fremlp=True`): seeds `42, 123`.
     - `Standard U-Net` (`base=32`, 4 tầng mã hóa/giải mã): seeds `42, 123`.
     - `DarkIR-Edge (no FFT)` (`use_fremlp=False`, `w_fft=0.0`): seeds `42, 123` (được người dùng bật tường minh `RUN_NOFFT_ABLATION = True` trong Cell Ablation của notebook).
   - **Ghi chú minh bạch về lưu trữ checkpoint cục bộ:** Khi tải thư mục `experiments/v4/checkpoints/` từ Google Drive về máy cục bộ (`colab-results/llie_project_v4-20261104`), một số file `.pt` dung lượng lớn không được tải về hết (ví dụ thư mục cục bộ thiếu `darkir_edge_seed42/` và một số file `final.pt` / `history.csv` riêng lẻ). Tuy nhiên, nhật ký chạy thực tế của **Cell 17** trong `LLIE_DarkIR_Edge_Colab_v4.ipynb` chứng minh tại thời điểm đánh giá trên Colab, **cả 6/6 runs đều có mặt đầy đủ cả `final.pt` và `best.pt` trên Google Drive** (không có bất kỳ dòng cảnh báo `[WARN] Missing` nào), và hai bảng dữ liệu gốc `results_per_image.csv` (`final.pt`, 1,900 dòng) cùng `results_per_image_best.csv` (`best.pt`, 1,900 dòng) đều hoàn chỉnh 100% và độc lập nhau.

### 1.3. Giao thức đánh giá (`Cell 16–22`)

- **Phân tích chính (`table_main.csv`, `paired_tests.csv`, `paired_tests_v3_v4.csv`):** Đánh giá trên bộ trọng số EMA cuối quá trình huấn luyện (`final.pt`, epoch 70) để hoàn toàn nhất quán với lịch `CosineAnnealingLR` không dừng sớm và khớp trực tiếp với giao thức của v3.
- **Phân tích phụ (`table_main_best.csv`, `results_per_image_best.csv`):** Đánh giá độc lập trên bộ trọng số `best.pt` (epoch đạt PSNR trung bình cao nhất trên tập `val` 150 ảnh).
- **Trọng số tham chiếu (`DarkIR-m`, CVPR 2025):** Nạp trực tiếp từ `DarkIR_384.pt` (`C=32`, `enc=[1,2,3]`, `mid=3`, `dec=[3,1,1]`, `5,536,548` tham số, `missing=0, unexpected=0`) và suy luận trên cùng 190 ảnh test ở độ phân giải gốc (`pad_factor = 8`).
- **Kiểm định thống kê ghép cặp:** Với các mô hình nhiều seed, điểm số của mỗi ảnh test $i \in \{1,\dots,190\}$ được lấy trung bình qua 2 seeds (`42, 123`) trước khi tính hiệu ghép cặp $\Delta_i = x_{A,i} - x_{B,i}$. Sử dụng đồng thời 3 phương pháp:
  1. Bootstrap 95% Confidence Interval (`n_boot = 10,000`, `seed = 42`).
  2. Paired Student's $t$-test hai phía (`p_ttest`).
  3. Wilcoxon signed-rank test hai phía (`p_wilcoxon`).

---

## 2. Kết quả chất lượng ảnh

### 2.1. Bảng kết quả chính — `final.pt` EMA (`table_main.csv`)

Giá trị báo cáo dưới dạng `Mean ± Std` qua 2 seeds (`42, 123`) đối với 3 mô hình học sâu, và giá trị tất định đối với 4 baseline không học. Cột `pooled` là trung bình theo tổng 190 ảnh test; cột `macro` là trung bình đều qua 4 miền (`blur`, `huawei`, `lolv2`, `nikon`).

#### Bảng 2.1a — PSNR (dB) ↑ và `psnr_gtmean` (dB) ↑

| Phương pháp | `blur` (40) | `huawei` (30) | `lolv2` (100) | `nikon` (20) | **Pooled (190)** | **Macro (4 miền)** | `psnr_gtmean` (Pooled) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DarkIR-Edge (no FFT)** | 23.72 ± 0.26 | 20.66 ± 0.03 | **20.98 ± 0.49** | **17.13 ± 0.01** | **21.10 ± 0.32** | **20.62 ± 0.20** | 25.04 ± 0.01 |
| **Standard U-Net** | **24.01 ± 0.09** | 20.52 ± 0.28 | 20.86 ± 0.25 | 16.69 ± 0.02 | 21.03 ± 0.15 | 20.52 ± 0.10 | **25.08 ± 0.15** |
| **DarkIR-Edge (ours)** | 23.54 ± 0.01 | **20.97 ± 0.20** | 20.74 ± 0.02 | 16.96 ± 0.17 | 20.97 ± 0.06 | 20.55 ± 0.10 | 24.87 ± 0.21 |
| Gamma 2.2 | 18.09 | 15.08 | 17.27 | 15.42 | 16.90 | 16.46 | 21.97 |
| MSRCR (Retinex) | 12.20 | 11.44 | 12.03 | 13.45 | 12.12 | 12.28 | 21.39 |
| CLAHE | 12.90 | 10.39 | 11.54 | 11.04 | 11.59 | 11.47 | 20.72 |
| Input (Low-Light) | 10.55 | 9.06 | 9.72 | 9.30 | 9.75 | 9.66 | 19.65 |

### 2.2. SSIM ↑ và LPIPS (AlexNet) ↓ (`table_main.csv`)

#### Bảng 2.2a — SSIM ↑

| Phương pháp | `blur` (40) | `huawei` (30) | `lolv2` (100) | `nikon` (20) | **Pooled (190)** | **Macro (4 miền)** | `ssim_gtmean` (Pooled) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard U-Net** | 0.765 ± 0.004 | 0.621 ± 0.004 | **0.826 ± 0.000** | 0.487 ± 0.007 | **0.745 ± 0.002** | 0.674 ± 0.004 | **0.765 ± 0.000** |
| **DarkIR-Edge (ours)** | 0.766 ± 0.002 | **0.628 ± 0.001** | 0.816 ± 0.002 | **0.503 ± 0.001** | 0.743 ± 0.001 | **0.678 ± 0.000** | 0.761 ± 0.001 |
| **DarkIR-Edge (no FFT)** | **0.767 ± 0.002** | 0.627 ± 0.001 | 0.816 ± 0.009 | **0.503 ± 0.003** | 0.743 ± 0.005 | **0.678 ± 0.002** | 0.762 ± 0.002 |
| Gamma 2.2 | 0.630 | 0.461 | 0.669 | 0.459 | 0.595 | 0.555 | 0.676 |
| MSRCR (Retinex) | 0.453 | 0.341 | 0.436 | 0.375 | 0.410 | 0.401 | 0.579 |
| CLAHE | 0.427 | 0.338 | 0.429 | 0.344 | 0.405 | 0.384 | 0.579 |
| Input (Low-Light) | 0.258 | 0.214 | 0.197 | 0.233 | 0.217 | 0.225 | 0.463 |

#### Bảng 2.2b — LPIPS (AlexNet, thấp hơn là tốt hơn ↓)

| Phương pháp | `blur` (40) | `huawei` (30) | `lolv2` (100) | `nikon` (20) | **Pooled (190)** | **Macro (4 miền)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard U-Net** | 0.347 ± 0.007 | **0.399 ± 0.018** | **0.217 ± 0.004** | **0.257 ± 0.009** | **0.277 ± 0.000** | **0.305 ± 0.004** |
| **DarkIR-Edge (no FFT)** | **0.345 ± 0.005** | 0.423 ± 0.007 | 0.256 ± 0.014 | 0.271 ± 0.003 | 0.303 ± 0.010 | 0.324 ± 0.007 |
| **DarkIR-Edge (ours)** | 0.347 ± 0.002 | 0.437 ± 0.003 | 0.257 ± 0.015 | 0.282 ± 0.005 | 0.307 ± 0.008 | 0.331 ± 0.005 |
| Gamma 2.2 | 0.460 | 0.458 | 0.287 | 0.246 | 0.356 | 0.363 |
| CLAHE | 0.556 | 0.562 | 0.415 | 0.354 | 0.466 | 0.472 |
| MSRCR (Retinex) | 0.531 | 0.506 | 0.463 | 0.301 | 0.471 | 0.450 |
| Input (Low-Light) | 0.625 | 0.608 | 0.534 | 0.462 | 0.554 | 0.557 |

**Nhận xét từ Bảng 2.1 và 2.2:**
- Cả 3 kiến trúc học sâu đều vượt xa các phương pháp truyền thống (`Gamma 2.2`, `MSRCR`, `CLAHE`) hơn `+4.0 dB` PSNR và `+0.148` SSIM trên tập gộp `pooled`.
- Giữa 3 mô hình học sâu, khoảng cách PSNR pooled cực kỳ nhỏ: `DarkIR-Edge (no FFT)` (`21.10 dB`), `Standard U-Net` (`21.03 dB`) và `DarkIR-Edge (ours)` (`20.97 dB`) chỉ chênh nhau tối đa `0.13 dB`.
- Xét theo từng miền cụ thể, quy luật phân hóa đặc trưng từ v3 vẫn lặp lại ở v4:
  - `Standard U-Net` dẫn đầu trên `blur` (`24.01 dB` vs `23.54 dB` của `DarkIR-Edge`) và đạt chất lượng cảm nhận LPIPS tốt nhất trên mọi miền (LPIPS pooled `0.277` so với `0.303` của `no-FFT` và `0.307` của `DarkIR-Edge`).
  - Ngược lại, họ `DarkIR-Edge` (`ours` và `no-FFT`) nhỉnh hơn `Standard U-Net` trên hai miền cảm biến LSRW là `huawei` (`20.97 dB` vs `20.52 dB`) và `nikon` (`16.96–17.13 dB` vs `16.69 dB`), nhờ đó đạt SSIM macro cao hơn (`0.678` vs `0.674`) và PSNR macro nhỉnh hơn nhẹ (`20.55–20.62 dB` vs `20.52 dB`).

### 2.3. Kết quả chi tiết theo từng seed (nhật ký Cell 17, `final.pt`)

| Mô hình | Seed | PSNR (Pooled) | SSIM (Pooled) | `psnr_gtmean` (Pooled) | LPIPS (Pooled) | Độ lệch giữa 2 seeds ($\|\text{Seed}_{42} - \text{Seed}_{123}\|$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DarkIR-Edge (ours)** | `42` | 20.924 | 0.743 | 24.726 | 0.301 | **0.086 dB** (rất ổn định) |
| **DarkIR-Edge (ours)** | `123` | 21.010 | 0.742 | 25.017 | 0.313 | — |
| **Standard U-Net** | `42` | 20.922 | 0.746 | 25.183 | 0.277 | 0.216 dB |
| **Standard U-Net** | `123` | 21.138 | 0.743 | 24.976 | 0.277 | — |
| **DarkIR-Edge (no FFT)** | `42` | 21.321 | 0.746 | 25.031 | 0.296 | 0.447 dB (dao động lớn trên `lolv2`) |
| **DarkIR-Edge (no FFT)** | `123` | 20.874 | 0.739 | 25.045 | 0.310 | — |

- Điểm đáng chú ý là tại `seed = 42`, `DarkIR-Edge (ours)` (`20.924 dB`) và `Standard U-Net` (`20.922 dB`) đạt kết quả gần như trùng khít nhau. Khoảng cách trung bình giữa các mô hình chủ yếu xuất phát từ biến thiên ngẫu nhiên giữa các seed trên miền `lolv2` (ví dụ `DarkIR-Edge (no FFT)` có độ lệch chuẩn tới `± 0.49 dB` trên `lolv2`, trong khi `psnr_gtmean` của cả 2 seed lại đứng yên ở `25.031` và `25.045 dB`).

### 2.4. Phân tích phụ: Đánh giá trên `best.pt` EMA (`table_main_best.csv`)

Để kiểm tra xem việc lấy checkpoint ở bước cuối cùng (`final.pt`, epoch 70) có làm thiệt thòi mô hình nào so với việc chọn đỉnh validation (`best.pt`) hay không, Bảng 2.4 đối chiếu trực tiếp hai giao thức:

| Phương pháp (`best.pt` EMA) | `blur` (40) | `huawei` (30) | `lolv2` (100) | `nikon` (20) | **Pooled `best.pt`** | **Macro `best.pt`** | Chênh lệch so với `final.pt` (Pooled) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard U-Net** | **24.14 ± 0.07** | 20.59 ± 0.35 | **20.89 ± 0.27** | 16.67 ± 0.09 | **21.08 ± 0.20** | 20.57 ± 0.15 | $+0.054\text{ dB}$ |
| **DarkIR-Edge (no FFT)** | 23.79 ± 0.18 | 20.73 ± 0.10 | 20.79 ± 0.37 | **17.17 ± 0.10** | 21.03 ± 0.23 | **20.62 ± 0.14** | $-0.068\text{ dB}$ |
| **DarkIR-Edge (ours)** | 23.49 ± 0.20 | **20.97 ± 0.20** | 20.75 ± 0.17 | 16.98 ± 0.01 | 20.96 ± 0.16 | 20.55 ± 0.14 | $-0.004\text{ dB}$ |

- Chênh lệch giữa `best.pt` và `final.pt` trên tập `pooled` chỉ nằm trong khoảng `[-0.068, +0.054] dB` (và đối với `DarkIR-Edge (ours)` chỉ là `-0.004 dB`). Điều này xác nhận lịch trình `CosineAnnealingLR` 70 epochs kết hợp `EMA(0.999)` hội tụ rất mượt ở cuối quá trình huấn luyện, không xảy ra hiện tượng quá khớp (overfitting) về cuối kỳ.

---

## 3. Kiểm định thống kê ghép cặp (`paired_tests.csv` & `paired_tests_v3_v4.csv`)

### 3.1. Đối sánh nội bộ v4: `DarkIR-Edge (ours)` vs `Standard U-Net` (`paired_tests.csv`)

Hiệu số được định nghĩa là $\Delta = \text{PSNR}_{\text{DarkIR-Edge}} - \text{PSNR}_{\text{U-Net}}$ ($\Delta > 0$ nghĩa là `DarkIR-Edge` tốt hơn):

| Miền dữ liệu | $N$ | `DarkIR-Edge` (dB) | `Standard U-Net` (dB) | $\Delta\text{PSNR}$ (dB) | Bootstrap 95% CI | $p$-value ($t$-test) | $p$-value (Wilcoxon) | Tỷ lệ thắng của `DarkIR-Edge` | Kết luận thống kê ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`pooled`** | **190** | 20.967 | 21.030 | **-0.063** | `[-0.360, +0.229]` | 0.6804 | 0.7245 | 50.53% (96/190) | **Hòa** (CI chứa 0) |
| `blur` | 40 | 23.537 | 24.009 | **-0.473** | `[-1.029, +0.110]` | 0.1191 | 0.1058 | 40.00% (16/40) | **Hòa** (xu hướng U-Net nhỉnh hơn nhưng CI chứa 0) |
| `huawei` | 30 | 20.968 | 20.517 | **+0.451** | `[-0.026, +0.911]` | 0.0835 | **0.0234** | **80.00%** (24/30) | Xu hướng `DarkIR-Edge` tốt hơn ($p_W = 0.023$, nhưng CI bootstrap chạm nhẹ 0) |
| `lolv2` | 100 | 20.740 | 20.861 | **-0.121** | `[-0.587, +0.389]` | 0.6243 | 0.6599 | 45.00% (45/100) | **Hòa** (CI chứa 0) |
| `nikon` | 20 | 16.961 | 16.686 | **+0.275** | `[-0.202, +0.801]` | 0.2954 | 0.3488 | 55.00% (11/20) | **Hòa** (CI chứa 0) |

### 3.2. Đối sánh nội bộ v4: `DarkIR-Edge (ours)` vs `DarkIR-Edge (no FFT)` (`paired_tests.csv`)

Hiệu số $\Delta = \text{PSNR}_{\text{DarkIR-Edge (ours)}} - \text{PSNR}_{\text{DarkIR-Edge (no FFT)}}$ kiểm định trực tiếp đóng góp của nhánh tần số (`FreMLP` + `FFT loss`):

| Miền dữ liệu | $N$ | `DarkIR-Edge` (dB) | `no-FFT` (dB) | $\Delta\text{PSNR}$ (dB) | Bootstrap 95% CI | $p$-value ($t$-test) | $p$-value (Wilcoxon) | Tỷ lệ thắng của `DarkIR-Edge` | Kết luận thống kê ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`pooled`** | **190** | 20.967 | 21.098 | **-0.131** | `[-0.371, +0.105]` | 0.2903 | 0.4618 | 48.42% (92/190) | **Hòa** (nhánh FFT không mang lại lợi thế) |
| `blur` | 40 | 23.537 | 23.717 | **-0.180** | `[-0.563, +0.265]` | 0.4012 | 0.1831 | 37.50% (15/40) | **Hòa** (CI chứa 0) |
| `huawei` | 30 | 20.968 | 20.658 | **+0.311** | `[+0.020, +0.623]` | 0.0635 | 0.1191 | 66.67% (20/30) | Biên giới cận ý nghĩa (CI > 0 nhẹ nhưng $p_t = 0.064, p_W = 0.119 > 0.05$) |
| `lolv2` | 100 | 20.740 | 20.975 | **-0.235** | `[-0.672, +0.180]` | 0.2675 | 0.5405 | 48.00% (48/100) | **Hòa** (CI chứa 0) |
| `nikon` | 20 | 16.961 | 17.132 | **-0.171** | `[-0.395, +0.055]` | 0.1476 | 0.2943 | 45.00% (9/20) | **Hòa** (CI chứa 0) |

### 3.3. Kiểm định ghép cặp xuyên phiên bản `v4 − v3` theo Quy tắc đăng ký trước (`paired_tests_v3_v4.csv`)

Đây là bảng kiểm định quan trọng nhất của báo cáo v4, đo trực tiếp hiệu số $\Delta = \text{PSNR}_{\text{v4}} - \text{PSNR}_{\text{v3}}$ trên cùng từng bức ảnh test để đối chiếu với các quy tắc đã đăng ký trước trong `docs/ai/preregistration_v4.md`:

| Mô hình | Tổ hợp miền kiểm định | $N$ | Mean v4 (dB) | Mean v3 (dB) | $\Delta(\text{v4}-\text{v3})$ (dB) | Bootstrap 95% CI | $p(t)$ | $p(W)$ | Tỷ lệ v4 > v3 | Đạt ngưỡng bảo vệ ($\text{CI}_{\text{lo}} > -0.3$)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DarkIR-Edge (ours)** | **`lolv2+huawei` (Quy tắc 1)** | **130** | 20.793 | 20.978 | **-0.185** | `[-0.519, +0.101]` | 0.2307 | 0.3127 | 47.69% | **KHÔNG** ($\text{CI}_{\text{lo}} = -0.519$) |
| | **`blur` (Quy tắc 3)** | **40** | 23.537 | 23.569 | **-0.032** | `[-0.534, +0.433]` | 0.8962 | 0.9096 | 47.50% | Không cải thiện |
| | **`nikon` (Quy tắc 2)** | **20** | 16.961 | 16.955 | **+0.006** | `[-0.625, +0.657]` | 0.9866 | 0.8408 | 50.00% | Không cải thiện |
| | `lolv2` | 100 | 20.740 | 21.072 | -0.332 | `[-0.743, +0.027]` | 0.0853 | 0.0884 | 42.00% | — |
| | `huawei` | 30 | 20.968 | 20.665 | +0.304 | `[+0.004, +0.611]` | 0.0628 | 0.1020 | 66.67% | — |
| | **`pooled`** | **190** | 20.967 | 21.100 | **-0.133** | `[-0.383, +0.116]` | 0.2966 | 0.4539 | 47.89% | Hòa (CI chứa 0) |
| **Standard U-Net** | **`lolv2+huawei` (Quy tắc 1)** | **130** | 20.782 | 21.189 | **-0.407** | `[-0.765, -0.067]` | **0.0230** | 0.0664 | 47.69% | **KHÔNG** (Suy giảm có ý nghĩa) |
| | **`blur` (Quy tắc 3)** | **40** | 24.009 | 23.802 | **+0.207** | `[-0.237, +0.665]` | 0.3694 | 0.3203 | 55.00% | Không có ý nghĩa TK |
| | **`nikon` (Quy tắc 2)** | **20** | 16.686 | 17.147 | **-0.461** | `[-1.319, +0.327]` | 0.2980 | 0.4980 | 55.00% | Không cải thiện |
| | `lolv2` | 100 | 20.861 | 21.391 | -0.530 | `[-0.964, -0.106]` | **0.0156** | **0.0369** | 44.00% | Suy giảm có ý nghĩa |
| | `huawei` | 30 | 20.517 | 20.515 | +0.002 | `[-0.265, +0.290]` | 0.9882 | 0.8448 | 60.00% | — |
| | **`pooled`** | **190** | 21.030 | 21.314 | **-0.284** | `[-0.557, -0.006]` | **0.0497** | 0.1235 | 50.00% | Suy giảm nhẹ cận biên |
| **DarkIR-Edge (no FFT)** | **`lolv2+huawei` (Quy tắc 1)** | **130** | 20.902 | 20.842 | **+0.060** | `[-0.369, +0.491]` | 0.7859 | 0.3127 | 50.00% | **KHÔNG** ($\text{CI}_{\text{lo}} = -0.369$) |
| | **`blur` (Quy tắc 3)** | **40** | 23.717 | 23.470 | **+0.247** | `[-0.344, +0.852]` | 0.4272 | 0.1788 | 65.00% | Không có ý nghĩa TK |
| | **`nikon` (Quy tắc 2)** | **20** | 17.132 | 17.117 | **+0.015** | `[-0.512, +0.637]` | 0.9613 | 0.9563 | 50.00% | Không cải thiện |
| | `lolv2` | 100 | 20.975 | 20.931 | +0.044 | `[-0.511, +0.601]` | 0.8758 | 0.4275 | 49.00% | — |
| | `huawei` | 30 | 20.658 | 20.546 | +0.112 | `[-0.088, +0.304]` | 0.2848 | 0.4161 | 53.33% | — |
| | **`pooled`** | **190** | 21.098 | 21.003 | **+0.095** | `[-0.237, +0.437]` | 0.5750 | 0.1816 | 53.16% | Hòa (CI chứa 0) |

**Phân tích khoa học từ Bảng 3.3:**
1. **Sự đánh đổi rõ rệt giữa `lolv2` và `huawei` khi thay đổi hạn ngạch lấy mẫu:** Ở v3, `lolv2` chiếm `37.5%` số bước huấn luyện (`600 / 1,600` cặp/epoch) và `huawei` chỉ có `300` ảnh gốc. Ở v4, `lolv2` giảm tỷ trọng xuống `26.6%` (`1,278 / 4,800` cặp/epoch), trong khi `huawei` được mở rộng lên `2,380` ảnh gốc và bổ sung thêm `nikon` + `lolv2syn`. Hệ quả trực tiếp là `DarkIR-Edge (ours)` tăng `+0.304 dB` trên `huawei` (95% CI `[+0.004, +0.611]`, thắng 20/30 ảnh), nhưng lại mất `-0.332 dB` trên `lolv2` (95% CI `[-0.743, +0.027]`). Do tập test `lolv2` có 100 ảnh (gấp 3.3 lần số ảnh `huawei`), trung bình gộp `lolv2+huawei` (130 ảnh) bị kéo xuống `-0.185 dB` với cận dưới `CI_lo = -0.519 dB < -0.3 dB`, vi phạm Quy tắc bảo vệ số 1. Đối với `Standard U-Net`, sự sụt giảm trên `lolv2` còn mạnh hơn (`-0.530 dB`, $p = 0.0156$) trong khi không thu được lợi ích trên `huawei` (`+0.002 dB`).
2. **Không có bước nhảy vọt trên `blur` dù số lượng frame huấn luyện tăng gấp ~4.1 lần (`450 → 1,858` cặp):** Vì 1,858 cặp ảnh ở v4 vẫn được trích xuất từ **cùng 150 chuỗi video cảnh quay** như v3 (chỉ lấy dày hơn từ 3 frames/video lên 12–13 frames/video), lượng thông tin độc lập về cấu trúc cảnh và quỹ đạo rung (motion blur trajectories) tăng thêm không nhiều. Do đó, PSNR trên `blur` gần như giữ nguyên ở mức `~23.5–24.0 dB`.
3. **Không có bước nhảy vọt trên `nikon` dù đã đưa `1,000` ảnh Nikon vào tập huấn luyện:** Đây là phát hiện thực nghiệm rất giá trị (sẽ được giải thích sâu hơn ở Mục 4 thông qua chẩn đoán `psnr_gtmean`).

### 3.4. Đối chiếu với mô hình tham chiếu `DarkIR-m` (CVPR 2025, `results_per_image_with_ref.csv`)

Vì tập test 190 ảnh và file trọng số công bố `DarkIR_384.pt` (`5.54M` tham số) không thay đổi giữa v3 và v4, điểm số của `DarkIR-m` giữ nguyên tuyệt đối và đóng vai trò mốc chuẩn cố định:

| Phương pháp | Tham số | `blur` (40) | `huawei` (30) | `lolv2` (100) | `nikon` (20) | **Pooled (190)** | **SSIM (Pooled)** | **LPIPS (Pooled)** | `psnr_gtmean` (Pooled) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DarkIR-m` (CVPR 2025, pretrained)** | 5.54M | **27.72** | 18.51 | 20.04 | 16.43 | 20.96 | 0.728 | **0.219** | 23.63 |
| **`DarkIR-Edge (no FFT)` (v4)** | **1.50M** | 23.72 | 20.66 | **20.98** | **17.13** | **21.10** | 0.743 | 0.303 | 25.04 |
| **`Standard U-Net` (v4)** | 1.93M | 24.01 | 20.52 | 20.86 | 16.69 | 21.03 | **0.745** | 0.277 | **25.08** |
| **`DarkIR-Edge (ours)` (v4)** | 1.92M | 23.54 | **20.97** | 20.74 | 16.96 | 20.97 | 0.743 | 0.307 | 24.87 |

- Trên miền `blur` (nơi `DarkIR-m` được huấn luyện chuyên biệt với dung lượng `5.54M` tham số), `DarkIR-m` vẫn vượt trội hoàn toàn (`27.72 dB` vs `23.54–24.01 dB`, LPIPS `0.135` vs `0.345–0.347`).
- Ngược lại, trên 3 miền còn lại (`lolv2`, `huawei`, `nikon`), cả 3 mô hình huấn luyện đa miền của v4 đều cao hơn `DarkIR-m` từ `+0.53 dB` đến `+2.46 dB` PSNR và cao hơn `+1.24 dB` đến `+1.45 dB` về `psnr_gtmean` trên tập `pooled`.

---

## 4. Chẩn đoán độ sáng (`diagnostic_gtmean.csv`)

Bảng dưới đây tách biệt chất lượng tái tạo cấu trúc/màu sắc nội tại (`psnr_gtmean` — PSNR sau khi nhân một hệ số vô hướng đưa độ sáng trung bình của ảnh dự đoán khớp với ảnh GT) và khoảng trống do lệch mức phơi sáng toàn cục ($\text{Gap} = \text{PSNR}_{\text{gt-mean}} - \text{PSNR}_{\text{raw}}$):

| Miền | Phương pháp | PSNR thô (dB) | `psnr_gtmean` (dB) | **Illumination Gap (dB)** | *(Đối chiếu Gap v3)* | SSIM thô | `ssim_gtmean` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`lolv2`** (100) | `DarkIR-Edge (ours)` | 20.74 | 26.45 | **+5.71** | *(+5.45)* | 0.816 | 0.847 |
| | `DarkIR-Edge (no FFT)` | 20.98 | 26.74 | **+5.77** | *(+5.41)* | 0.816 | 0.849 |
| | `Standard U-Net` | 20.86 | **27.01** | **+6.15** | *(+5.04)* | **0.826** | **0.855** |
| **`nikon`** (20) | `DarkIR-Edge (ours)` | 16.96 | 23.38 | **+6.42** | *(+6.36)* | **0.503** | **0.556** |
| | `DarkIR-Edge (no FFT)` | **17.13** | **23.54** | **+6.41** | *(+6.29)* | **0.503** | **0.556** |
| | `Standard U-Net` | 16.69 | 23.09 | **+6.40** | *(+6.22)* | 0.487 | 0.544 |
| **`huawei`** (30) | `DarkIR-Edge (ours)` | **20.97** | **23.95** | **+2.98** | *(+3.31)* | **0.628** | **0.660** |
| | `DarkIR-Edge (no FFT)` | 20.66 | 23.94 | **+3.28** | *(+3.47)* | 0.627 | 0.660 |
| | `Standard U-Net` | 20.52 | 23.70 | **+3.18** | *(+3.41)* | 0.621 | 0.659 |
| **`blur`** (40) | `DarkIR-Edge (ours)` | 23.54 | 22.12 | **-1.42** | *(-1.26)* | 0.766 | 0.723 |
| | `DarkIR-Edge (no FFT)` | 23.72 | 22.36 | **-1.36** | *(-1.21)* | **0.767** | 0.725 |
| | `Standard U-Net` | **24.01** | **22.86** | **-1.15** | *(-1.05)* | 0.765 | **0.733** |

**Hai phát hiện cốt lõi từ chẩn đoán `gt-mean`:**
1. **Giải thích nguyên nhân sụt giảm PSNR thô trên `lolv2` ở v4:** Trên `lolv2`, chất lượng cấu trúc sau khi chuẩn hóa độ sáng (`psnr_gtmean`) của cả 3 mô hình ở v4 thực chất vẫn rất cao (`26.45–27.01 dB`, xấp xỉ mức `26.35–26.52 dB` của v3). Tuy nhiên, `Illumination Gap` trên `lolv2` đã phình to từ `5.04–5.45 dB` (v3) lên **`5.71–6.15 dB`** (v4) — đặc biệt ở `Standard U-Net` tăng hơn `+1.11 dB` gap. Như sẽ thấy ở Bảng 5.1 (`lum_bias`), việc nạp thêm hàng nghìn ảnh từ các miền có GT sáng hơn (`huawei`, `nikon`, `lolv2syn`) và giảm tỷ trọng `lolv2` xuống `26.6%` đã đẩy mức độ sáng dự đoán trung bình của mạng tăng vọt trên `lolv2` (dự đoán thừa sáng hơn so với GT của `lolv2`), làm giảm PSNR thô nhưng lại giúp giảm thiếu sáng trên `huawei` (Gap trên `huawei` thu hẹp từ `3.31 dB` xuống `2.98 dB` ở `DarkIR-Edge`).
2. **Giải thích tại sao thêm `1,000` ảnh `nikon` vào tập train không làm tăng PSNR thô trên `nikon`:** Trên `nikon`, ngay cả khi đã học trực tiếp `1,000` cặp ảnh cùng cảm biến, `Illumination Gap` vẫn giữ nguyên ở mức **`~6.40–6.42 dB`** (v3 là `6.22–6.36 dB`), trong khi `psnr_gtmean` đã đạt tới `23.09–23.54 dB`. Điều này chứng tỏ trong bộ dữ liệu LSRW-Nikon, mối quan hệ giữa độ sáng đầu vào (Low) và độ sáng tham chiếu (GT) có phương sai ngẫu nhiên rất lớn giữa từng bức ảnh (do tốc độ màn trập/ISO khi chụp cặp ảnh thực tế thay đổi theo cảnh). Một mạng tất định chỉ nhìn vào ảnh Low không thể "đoán" chính xác hệ số phơi sáng tùy ý của ảnh GT trên từng bức ảnh đơn lẻ, bất kể có đưa bao nhiêu ảnh Nikon vào tập train.

---

## 5. Phân tích bổ sung (`table_extra.csv` & `paired_tests_extra.csv`)

### 5.1. Các chỉ số chẩn đoán chuyên sâu (`table_extra.csv`, Mean ± Std qua 2 seeds)

Các chỉ số gồm:
- `lum_bias`: Độ lệch độ sáng trung bình trên thang `[0, 255]` ($\text{mean}(Y_{\text{pred}} - Y_{\text{gt}})$, gần `0` là tốt).
- `color_ang_deg`: Sai số góc màu trung bình (độ ↓).
- `sharp_ratio`: Tỷ số phương sai Laplacian $\text{Var}(\nabla^2 Y_{\text{pred}}) / \text{Var}(\nabla^2 Y_{\text{gt}})$ (`1.0` là sắc nét ngang GT; `< 1.0` là mờ hơn GT).
- `dark10_mae`: Sai số tuyệt đối trung bình (trên thang `[0, 255]` ↓) tại 10% điểm ảnh tối nhất của GT.

| Miền | Phương pháp | `lum_bias` (`[0,255]`) | *(v3 `lum_bias`)* | `color_ang_deg` (° ↓) | `sharp_ratio` ($\approx 1.0$) | `dark10_mae` (`[0,255]` ↓) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`pooled`** (190) | `DarkIR-Edge (ours)` | +5.33 ± 0.72 | *(+4.23)* | 4.89 ± 0.03° | 0.221 ± 0.000 | 12.13 ± 0.02 |
| | `DarkIR-Edge (no FFT)` | +5.09 ± 1.72 | *(+5.12)* | 4.86 ± 0.06° | 0.224 ± 0.008 | **12.01 ± 0.56** |
| | `Standard U-Net` | +7.26 ± 1.48 | *(+4.61)* | **4.81 ± 0.04°** | **0.304 ± 0.004** | 12.69 ± 0.72 |
| **`blur`** (40) | `DarkIR-Edge (ours)` | **-2.54 ± 0.16** | *(-2.45)* | 3.74 ± 0.01° | 0.193 ± 0.003 | **5.97 ± 0.06** |
| | `DarkIR-Edge (no FFT)` | -2.97 ± 0.90 | *(-1.90)* | 3.70 ± 0.02° | 0.200 ± 0.005 | 6.08 ± 0.34 |
| | `Standard U-Net` | -2.90 ± 0.23 | *(-2.22)* | **3.60 ± 0.02°** | **0.224 ± 0.004** | 6.01 ± 0.27 |
| **`huawei`** (30) | `DarkIR-Edge (ours)` | -8.23 ± 0.91 | *(-10.23)* | 5.21 ± 0.04° | 0.109 ± 0.005 | **10.46 ± 0.16** |
| | `DarkIR-Edge (no FFT)` | -7.08 ± 0.19 | *(-9.60)* | 5.19 ± 0.01° | 0.114 ± 0.001 | 10.67 ± 0.05 |
| | `Standard U-Net` | **-4.16 ± 1.38** | *(-7.05)* | **5.17 ± 0.06°** | **0.165 ± 0.000** | 11.51 ± 0.51 |
| **`lolv2`** (100) | `DarkIR-Edge (ours)` | +15.27 ± 0.79 | *(+12.22)* | 5.16 ± 0.08° | 0.281 ± 0.004 | 12.82 ± 0.14 |
| | `DarkIR-Edge (no FFT)` | **+14.85 ± 2.66** | *(+12.89)* | 5.13 ± 0.11° | 0.281 ± 0.014 | **12.57 ± 1.00** |
| | `Standard U-Net` | +16.80 ± 2.34 | *(+11.46)* | **5.09 ± 0.06°** | **0.397 ± 0.007** | 12.59 ± 1.13 |
| **`nikon`** (20) | `DarkIR-Edge (ours)` | -8.32 ± 1.10 | *(-10.67)* | 5.38 ± 0.15° | 0.147 ± 0.004 | **23.53 ± 0.17** |
| | `DarkIR-Edge (no FFT)` | -9.37 ± 0.87 | *(-8.67)* | 5.37 ± 0.08° | 0.152 ± 0.000 | 23.62 ± 0.26 |
| | `Standard U-Net` | **-3.01 ± 0.01** | *(-8.52)* | **5.29 ± 0.03°** | **0.197 ± 0.012** | 27.67 ± 0.56 |

### 5.2. Kiểm định ghép cặp các chỉ số bổ sung (`paired_tests_extra.csv`)

1. **Độ sắc nét (`sharp_ratio` — `DarkIR-Edge` vs `Standard U-Net`):**
   - Trên tập `pooled` (190 ảnh), `Standard U-Net` tiếp tục sắc nét hơn `DarkIR-Edge (ours)` một cách có ý nghĩa thống kê rất cao: `0.3036` vs `0.2212`, $\Delta = -0.0824$, 95% CI $[-0.0924, -0.0722]$, $p(t) = 3.4 \times 10^{-37}$, `DarkIR-Edge` chỉ cao hơn ở `4.21%` số ảnh (8/190).
   - Trên miền `blur` (40 ảnh): `0.2237` (`U-Net`) vs `0.1925` (`DarkIR-Edge`), $\Delta = -0.0312$, 95% CI $[-0.0507, -0.0121]$, $p(t) = 0.0028$.
   - So với v3 (`sharp_ratio` trên `blur` là `0.181` ở `DarkIR-Edge` và `0.225` ở `U-Net`), việc tăng dữ liệu `blur` ở v4 giúp `DarkIR-Edge` nhích nhẹ từ `0.181` lên `0.193`, nhưng ảnh khôi phục vẫn chỉ đạt `~19–22%` phương sai Laplacian của ảnh GT sắc nét.
   - Giữa `DarkIR-Edge (ours)` và `DarkIR-Edge (no FFT)`, `sharp_ratio` trên `pooled` gần như bằng nhau (`0.2212` vs `0.2241`, $\Delta = -0.0029$, $p(t) = 0.058$, 95% CI $[-0.0058, +0.0001]$).

2. **Sai số vùng tối sâu (`dark10_mae` — `DarkIR-Edge` vs `Standard U-Net`):**
   - `DarkIR-Edge (ours)` khôi phục 10% vùng tối nhất ổn định hơn và ít nhiễu/lệch sáng hơn `Standard U-Net` một cách có ý nghĩa thống kê trên cả hai miền cảm biến LSRW:
     - Trên `huawei` (30 ảnh): `10.460` vs `11.505`, $\Delta = -1.045$, 95% CI $[-1.925, -0.369]$, $p(t) = 0.0160$, $p(W) = 0.0026$.
     - Trên `nikon` (20 ảnh): `23.534` vs `27.666`, $\Delta = -4.132$, 95% CI $[-6.413, -2.087]$, $p(t) = 0.0016$, $p(W) = 0.0007$.

3. **Sai số góc màu (`color_ang_deg`):**
   - Cả 3 mô hình đều duy trì sai số góc màu rất thấp (`~3.60°–3.74°` trên `blur` và `~4.81°–4.89°` trên `pooled`, so với `9.73°` của ảnh Input gốc). `Standard U-Net` thấp hơn nhẹ so với `DarkIR-Edge` trên `pooled` ($\Delta = +0.080^\circ$, 95% CI $[+0.025^\circ, +0.135^\circ]$, $p(t) = 0.0053$).

---

## 6. Hiệu năng tính toán & Tài nguyên huấn luyện (`efficiency.csv` & Cell 13 `probe_speed`, GPU NVIDIA L4)

### 6.1. Hiệu năng suy luận (`efficiency.csv`)

Đo trên GPU **NVIDIA L4** (Google Colab), có `torch.cuda.synchronize()`, 10 lần chạy làm nóng (warmup) và 50 lần chạy lấy trung bình ở độ phân giải `600×400` (cùng cấu hình đo với v3; GFLOPs tính cho các phép toán không gian ở `256×256`, chưa tính chi phí `rfft2/irfft2`):

| Mô hình | Tham số | Tỷ lệ vs U-Net | GFLOPs (`256×256`) | Latency FP32 (`600×400`) | FPS (FP32) | Latency FP16 (`600×400`) | FPS (FP16) | Peak Inference VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DarkIR-Edge (ours, `C=24`)** | 1,924,038 (`1.924M`) | 99.8% | 8.75 *(chưa tính FFT)* | 32.29 ms | 31.0 | 26.63 ms | 37.5 | 236.7 MB |
| **DarkIR-Edge (no FFT, `C=24`)** | **1,495,422 (`1.495M`)** | **77.5%** | **7.50** | 27.79 ms | 36.0 | 21.98 ms | 45.5 | **220.6 MB** |
| **Standard U-Net (`base=32`)** | 1,928,483 (`1.928M`) | 100.0% | 18.45 | **13.37 ms** | **74.8** | **11.81 ms** | **84.6** | 269.7 MB |

### 6.2. Tốc độ huấn luyện, bộ nhớ RAM và VRAM (`probe_speed` Cell 13 & nhật ký huấn luyện)

Do v4 áp dụng cơ chế nạp dữ liệu lai (Hybrid RAM Cache `5.40 GB` + Lazy SSD) và tăng số bước lên `600` steps/epoch (`42,000` steps/run), dưới đây là thông số thực tế ghi nhận từ Cell 13 (`probe_speed`) và nhật ký hoàn tất 70 epochs:

| Mô hình | Tốc độ train (`batch=8, patch=256`) | Thời gian / step | Thời gian thực tế 1 run (`42,000` steps) | Peak Training VRAM (`batch=8`) | System RAM sử dụng (trên `53.0 GB`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DarkIR-Edge (ours)** | 3.61 it/s | 0.277 s | **207.9 phút** (~3.47 giờ) | 4,815.9 MB | 9.91 GB (20.2%) |
| **DarkIR-Edge (no FFT)** | ~4.55 it/s | ~0.220 s | **153.9 phút** (~2.57 giờ) | 3,263.8 MB | 9.93 GB (20.2%) |
| **Standard U-Net** | **13.75 it/s** | **0.073 s** | **55.4 phút** (~0.92 giờ) | **1,471.2 MB** | 9.94 GB (20.2%) |

- **Đánh giá kỹ thuật hạ tầng v4:** Cơ chế Hybrid RAM/SSD trong Cell 6 hoạt động hoàn hảo: mức tiêu thụ System RAM giữ cố định ở `~9.91–9.94 GB` trong suốt nhiều giờ chạy liên tục, loại bỏ hoàn toàn nguy cơ tràn RAM (`OOM`) khi mở rộng dữ liệu lên `6,777` cặp ảnh mà vẫn đạt tốc độ `13.75 it/s` cho `U-Net` và `3.61 it/s` cho `DarkIR-Edge`.
- **Chi phí thực tế của nhánh FFT:** Khi bỏ nhánh `FreMLP` + `FFT loss` (`DarkIR-Edge no-FFT`), Peak Training VRAM giảm mạnh từ `4,815.9 MB` xuống `3,263.8 MB` (`-32.2%`), thời gian huấn luyện giảm từ `207.9 phút` xuống `153.9 phút`/run (`-26.0%`), và tốc độ suy luận FP16 tăng từ `37.5 FPS` lên `45.5 FPS` (`+21.3%`) mà không làm giảm chất lượng ảnh đầu ra.

---

## 7. Quá trình huấn luyện (`history.csv` & nhật ký huấn luyện trong Notebook)

Mặc dù chỉ có file `experiments/v4/checkpoints/darkir_edge_nofft_seed123/history.csv` được tải đầy đủ về thư mục cục bộ, nhật ký in trực tiếp trong `LLIE_DarkIR_Edge_Colab_v4.ipynb` (Cell 14 và Cell Ablation) đã lưu lại toàn bộ tiến trình 70 epochs của các mô hình:

### 7.1. Bảng tổng hợp hội tụ trên tập Validation (150 ảnh: `50 lolv2 + 50 huawei + 50 blur`)

| Mô hình (Seed) | Epoch 1 (`val_psnr` / `val_ssim`) | Epoch 10 | Epoch 30 | Epoch 50 | Epoch 70 (`final.pt`) | Đỉnh `best_val_psnr` | Thời gian chạy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DarkIR-Edge` (seed 123)** | 20.69 dB / 0.692 | 21.30 dB / 0.727 | 22.02 dB / 0.742 | 22.25 dB / 0.746 | 22.24 dB / 0.748 | **22.39 dB** (ep 62) | 207.9 phút |
| **`Standard U-Net` (seed 123)** | 21.17 dB / 0.707 | 21.56 dB / 0.730 | 21.84 dB / 0.743 | 22.08 dB / 0.748 | 22.14 dB / 0.749 | **22.25 dB** (ep 52) | 55.4 phút |
| **`DarkIR-Edge (no FFT)` (seed 42)** | 20.47 dB / 0.694 | 21.66 dB / 0.729 | 21.80 dB / 0.742 | 21.72 dB / 0.746 | 22.09 dB / 0.748 | **22.26 dB** (ep 31) | 86.2 phút *(resume ep 31→70)* |
| **`DarkIR-Edge (no FFT)` (seed 123)** | 20.82 dB / 0.696 | 21.46 dB / 0.729 | 21.82 dB / 0.742 | 21.96 dB / 0.746 | 22.12 dB / 0.748 | **22.29 dB** (ep 46) | 153.9 phút |

### 7.2. Diễn biến chi tiết từng thành phần mất mát (`darkir_edge_nofft_seed123/history.csv`)

Trích xuất chính xác từ `history.csv` (70 dòng, từ `lr = 2.00e-4` ở epoch 1 giảm theo hình cosin xuống `lr = 1.00e-6` ở epoch 70):
- **Tổng mất mát huấn luyện (`train_loss`):** Giảm đều và đơn điệu từ `0.1518` (epoch 1) → `0.1190` (epoch 10) → `0.1102` (epoch 30) → `0.1059` (epoch 50) → **`0.1036`** (epoch 70).
- **Thành phần tái tạo chính (`l_charb`):** Giảm từ `0.0956` (epoch 1) → `0.0757` (epoch 10) → `0.0708` (epoch 30) → `0.0682` (epoch 50) → **`0.0668`** (epoch 70).
- **Thành phần cấu trúc (`l_ssim` = $1 - \text{SSIM}$):** Giảm từ `0.3052` (epoch 1) → `0.2330` (epoch 10) → `0.2094` (epoch 30) → `0.1994` (epoch 50) → **`0.1948`** (epoch 70).
- **Thành phần biên (`l_edge`):** Giảm từ `0.0442` (epoch 1) → `0.0365` (epoch 10) → `0.0341` (epoch 30) → `0.0332` (epoch 50) → **`0.0329`** (epoch 70).
- **Thành phần màu (`l_color`):** Giảm từ `0.0309` (epoch 1) → `0.0254` (epoch 10) → `0.0241` (epoch 30) → `0.0235` (epoch 50) → **`0.0233`** (epoch 70).
- **Thành phần giám sát sâu đa tỉ lệ (`l_side`):** Giảm từ `0.0910` (epoch 1) → `0.0692` (epoch 10) → `0.0632` (epoch 30) → `0.0604` (epoch 50) → **`0.0590`** (epoch 70).
- **Diễn biến theo từng miền trên tập Validation (`val_psnr_lolv2`, `val_psnr_huawei`, `val_psnr_blur`):**
  - `val_psnr_blur` tăng rất ổn định và đạt đỉnh ở cuối quá trình huấn luyện: từ `21.22 dB` (epoch 1) → `22.71 dB` (epoch 10) → `23.51 dB` (epoch 30) → `23.91 dB` (epoch 50) → **`24.08 dB`** (epoch 70).
  - `val_psnr_huawei` giữ ổn định quanh `21.14–21.48 dB` trong suốt nửa sau quá trình huấn luyện (`21.27 dB` ở epoch 70).
  - `val_psnr_lolv2` dao động mạnh hơn giữa các epoch (đạt `21.84 dB` ở epoch 46 nhưng lùi về `21.02 dB` ở epoch 70), phù hợp với quan sát về độ nhạy độ sáng toàn cục trên `lolv2`.

---

## 8. Tổng hợp chuỗi thực nghiệm v2 → v3 → v4 (trên cùng tập Test 190 ảnh)

Bảng dưới đây tổng hợp toàn bộ hành trình nghiên cứu qua 3 phiên bản (`v2 → v3 → v4`) trên cùng 190 ảnh test (`100 lolv2 + 30 huawei + 20 nikon + 40 blur`) để cung cấp bức tranh toàn cảnh:

| Phiên bản & Thiết lập huấn luyện | Mô hình | `lolv2` (100) | `huawei` (30) | `nikon` (20) | `blur` (40) | **Pooled (190)** | **Macro (4 miền)** | `psnr_gtmean` (Pooled) | SSIM (Pooled) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v2** (`689 lolv2` đơn miền, Early Stopping ~8k steps, 3 seeds) | `DarkIR-Edge (ours)` | 20.47 | 19.29 *(ZS)* | **17.44** *(ZS)* | 19.63 *(ZS)* | 19.79 | 19.21 | 23.62 | 0.688 |
| | `DarkIR-Edge (no FFT)` | 20.99 | 19.48 *(ZS)* | 17.27 *(ZS)* | 19.71 *(ZS)* | 20.09 | 19.36 | 23.71 | 0.689 |
| | `Standard U-Net` | 21.17 | 18.88 *(ZS)* | 17.23 *(ZS)* | 19.04 *(ZS)* | 19.95 | 19.08 | 23.30 | 0.671 |
| **v3** (`1,389` cặp 3 miền `lolv2+huawei+blur`, `28,000` steps, 2 seeds) | `DarkIR-Edge (ours)` | 21.07 | 20.66 | 16.96 *(ZS)* | 23.57 | 21.10 | 20.57 | 24.85 | 0.742 |
| | `DarkIR-Edge (no FFT)` | 20.93 | 20.55 | 17.12 *(ZS)* | 23.47 | 21.00 | 20.52 | 24.84 | 0.740 |
| | `Standard U-Net` | **21.39** | 20.52 | 17.15 *(ZS)* | 23.80 | **21.31** | **20.71** | 24.96 | **0.748** |
| **v4** (`6,777` cặp 5 miền + `QuotaSampler`, `42,000` steps, 2 seeds) | `DarkIR-Edge (ours)` | 20.74 | **20.97** | 16.96 *(ID)* | 23.54 | 20.97 | 20.55 | 24.87 | 0.743 |
| | `DarkIR-Edge (no FFT)` | 20.98 | 20.66 | 17.13 *(ID)* | 23.72 | 21.10 | 20.62 | 25.04 | 0.743 |
| | `Standard U-Net` | 20.86 | 20.52 | 16.69 *(ID)* | **24.01** | 21.03 | 20.52 | **25.08** | 0.745 |

*(Chú thích: `ZS` = Zero-Shot / chưa thấy trong tập train; `ID` = In-Domain / đã có mặt trong tập train).*

**Quy luật khoa học rút ra từ chuỗi `v2 → v3 → v4`:**
1. **Bước nhảy lớn nhất nằm ở chuyển dịch từ đơn miền sang đa miền (`v2 → v3`):** Khi chuyển từ việc chỉ học trên `lolv2` (v2) sang học đa miền có mặt `huawei` và `blur` (v3), PSNR trên `blur` tăng vọt **`+3.94 dB` đến `+4.76 dB`**, trên `huawei` tăng **`+1.07 dB` đến `+1.64 dB`**, kéo PSNR pooled tăng **`+1.31 dB`** và SSIM pooled tăng **`+0.054` đến `+0.077`**.
2. **Hiện tượng bão hòa hoàn toàn khi tiếp tục nhân rộng dữ liệu từ `v3 → v4` (`1,389 → 6,777` cặp):**
   - Chất lượng cấu trúc nội tại sau chuẩn hóa độ sáng (`psnr_gtmean` pooled) thực chất tăng rất nhẹ từ `24.84–24.96 dB` (v3) lên `24.87–25.08 dB` (v4), và SSIM pooled giữ nguyên ở `0.743–0.745`.
   - Tuy nhiên, điểm số PSNR thô hoàn toàn đi ngang quanh ngưỡng `~21.0 dB` (`20.97–21.10 dB` ở v4 so với `21.00–21.31 dB` ở v3). Với một kiến trúc cực nhẹ (`~1.5M–1.9M` tham số, không có cơ chế chuẩn hóa/điều kiện hóa theo miền cảm biến), việc ép một mạng duy nhất học đồng thời 5 phân phối phơi sáng khác nhau (`lolv2`, `huawei`, `blur`, `nikon`, `lolv2syn`) dẫn đến hiện tượng kéo co về mức độ sáng đầu ra (luminance tug-of-war): cải thiện trên `huawei` (`+0.30 dB`) bị đánh đổi trực tiếp bằng suy giảm trên `lolv2` (`-0.33 dB` đến `-0.53 dB`).

---

## 9. Phân tích định tính (`experiments/v4/figures/`) & Trạng thái tập phụ Real-LOLBlur

### 9.1. Hình ảnh trực quan đã trích xuất (`figures/`)

Thư mục `experiments/v4/figures/` lưu trữ 5 ấn phẩm đồ họa chất lượng cao phục vụ báo cáo và thuyết trình:
- `fig_visual_grid.png` (và `.pdf`): Lưới so sánh trực quan 8 cột (`Input`, `CLAHE`, `MSRCR`, `Gamma 2.2`, `Standard U-Net`, `DarkIR-Edge (ours)`, `DarkIR-m (ref)`, `Ground Truth`) trên 4 hàng đại diện cho 4 miền (`lolv2`, `huawei`, `nikon`, `blur`), kèm vùng phóng대 (zoom-in crop) đóng khung đỏ/vàng để soi chi tiết biên và nhiễu.
- `fig_fft_spectrum.png`: Minh họa phổ biên độ tần số 2D FFT ($\log(1 + |\mathcal{F}(Y)|)$) đối chiếu giữa ảnh đầu vào thiếu sáng, ảnh khôi phục bởi `Standard U-Net`, `DarkIR-Edge (ours)` và ảnh `Ground Truth` trên miền `blur`.
- `fig_psnr_by_domain.png`: Biểu đồ cột nhóm so sánh PSNR (`Mean ± Std`) của các phương pháp qua 4 miền và tập gộp.
- `fig_gtmean_gap.png`: Biểu đồ phân rã khoảng cách độ sáng (`Raw PSNR` vs `GT-Mean Aligned PSNR`) chứng minh trực quan sự khác biệt giữa lỗi tái tạo cấu trúc và lỗi lệch mức phơi sáng trên `lolv2`/`nikon` so với `blur`.

### 9.2. Trạng thái kiểm tra định tính không tham chiếu trên `Real-LOLBlur` (Cell 21)

- Trong kế hoạch v4 (`ke_hoach_v4_notebook.md`), Cell 21 được thiết kế tùy chọn để chạy kiểm tra định tính không tham chiếu (no-reference qualitative check) trên 16 ảnh đêm thực tế của bộ `Real-LOLBlur` nếu thư mục này được chuẩn bị sẵn trong `data/Real-LOLBlur`.
- Nhật ký thực thi thực tế của Cell 21 trong `LLIE_DarkIR_Edge_Colab_v4.ipynb` ghi nhận:
  `[SKIP] Không tìm thấy thư mục Real-LOLBlur tại /content/drive/MyDrive/llie_project_v4/data/Real-LOLBlur — bỏ qua kiểm tra định tính phụ.`
- Do đó, báo cáo tuân thủ nguyên tắc trung thực tuyệt đối: không đưa bất kỳ nhận định định tính nào về `Real-LOLBlur`.

---

## 10. Các mối đe dọa đến tính hợp lệ (Threats to Validity & Limitations)

1. **Xung đột phân phối độ sáng đa miền khi dùng chung một bộ trọng số duy nhất (Multi-domain Luminance Conflict):**
   - Trong bài toán LLIE, mỗi bộ dữ liệu (`LOL-v2 Real`, `LOL-v2 Synthetic`, `LSRW-Huawei`, `LSRW-Nikon`, `LOL-Blur`) được thu thập bằng quy trình phơi sáng tham chiếu (Ground Truth exposure) rất khác nhau. Khi gộp cả 5 miền vào huấn luyện chung trên một mạng hoàn toàn bất biến theo miền (domain-agnostic CNN) mà không có nhãn miền hay cơ chế thích nghi độ sáng ở bước suy luận, mạng buộc phải hội tụ về một hàm ánh xạ độ sáng "trung bình thỏa hiệp". Sự dịch chuyển `lum_bias` trên `lolv2` từ `+12.22` (v3) lên `+15.27` (v4) ở `DarkIR-Edge` và từ `+11.46` lên `+16.80` ở `Standard U-Net` là bằng chứng định lượng trực tiếp của hiện tượng này.
2. **Giới hạn dung lượng biểu diễn và thiếu cơ chế khử mờ chuyên biệt trên `LOL-Blur`:**
   - Mặc dù `psnr_gtmean` của các mô hình nhẹ (`~1.5–1.9M` tham số) vượt `DarkIR-m` (`5.54M` tham số) trên cả `lolv2`, `huawei` và `nikon`, chúng vẫn thua `DarkIR-m` khoảng `3.7–4.2 dB` trên `blur` và có tỷ số sắc nét `sharp_ratio` chỉ đạt `~0.19–0.22`. Việc tăng số lượng frame từ cùng 150 video (`450 → 1,858` cặp) không thể bù đắp cho giới hạn dung lượng kênh (`C=24` vs `C=32`, tổng số khối NAF/DarkIR ít hơn) khi phải giải quyết đồng thời bài toán khử mờ chuyển động lớn (large motion blur) và nâng sáng.
3. **Số lượng hạt giống ngẫu nhiên (`2 seeds: 42, 123`) và kích thước mẫu nhỏ ở các tập con:**
   - Tập test `nikon` chỉ có `20` ảnh và `huawei` chỉ có `30` ảnh, dẫn đến khoảng tin cậy bootstrap 95% trên từng miền đơn lẻ khá rộng (độ rộng CI khoảng `~0.6–1.6 dB`). Ngoài ra, với 2 seeds, các dao động ngẫu nhiên về mức độ sáng hội tụ cuối kỳ trên `lolv2` (như độ lệch `0.447 dB` giữa 2 seed của `DarkIR-Edge no-FFT`) vẫn ảnh hưởng đến thứ hạng trung bình điểm số thô.
4. **Độ lệch chuẩn hóa công bằng kiến trúc giữa `DarkIR-Edge` và `Standard U-Net`:**
   - Cả hai mô hình được khớp chặt về số lượng tham số (`1.924M` vs `1.928M`), nhưng `DarkIR-Edge` dùng các tích chập tách chiều sâu (depthwise separable convolutions) kết hợp giãn nở đa tỉ lệ và biến đổi `rfft2/irfft2` nên có GFLOPs không gian thấp hơn (`8.75` vs `18.45 GFLOPs`) nhưng độ trễ thực thi trên GPU lại cao hơn (`32.29 ms` vs `13.37 ms` ở FP32) do chi phí gọi kernel FFT và băng thông bộ nhớ của các nhánh tách chiều sâu.

---

## 11. Kết luận trung thực (đối chiếu nghiêm ngặt 6 quy tắc đăng ký trước)

Đối chiếu từng con số thực nghiệm với **6 quy tắc kết luận đã khóa trước khi huấn luyện** trong `docs/ai/preregistration_v4.md` và `presentation/report/ke_hoach_v4.md`:

1. **Đối chiếu Quy tắc 1 (Kiểm tra không suy giảm trên `lolv2 + huawei`, 130 ảnh, ngưỡng $\text{CI}_{\text{lo}} > -0.3\text{ dB}$):**
   - **Kết quả:** **Không đạt (`pass_protection_rule = False`) ở cả 3 mô hình.** Hiệu số ghép cặp $\Delta(\text{v4}-\text{v3})$ trên `lolv2 + huawei` là `-0.185 dB` (95% CI `[-0.519, +0.101]`) đối với `DarkIR-Edge (ours)`, `-0.407 dB` (95% CI `[-0.765, -0.067]`, $p = 0.023$) đối với `Standard U-Net`, và `+0.060 dB` (95% CI `[-0.369, +0.491]`) đối với `DarkIR-Edge (no FFT)`.
   - **Kết luận theo Quy tắc 1:** Việc giảm tỷ trọng lấy mẫu của `lolv2` xuống `26.6%` để nhường chỗ cho 4 miền còn lại đã gây ra hiện tượng dịch chuyển độ sáng dương (`over-brightening bias`) trên `lolv2`, làm suy giảm PSNR thô trên miền tĩnh vượt quá biên độ bảo vệ `-0.3 dB`.

2. **Đối chiếu Quy tắc 2 (Tác động của việc đưa `nikon` từ Unseen sang In-Domain, 20 ảnh):**
   - **Kết quả:** $\Delta(\text{v4}-\text{v3})$ trên `nikon` = `+0.006 dB` (95% CI `[-0.625, +0.657]`, $p = 0.987$) đối với `DarkIR-Edge (ours)` và `-0.461 dB` (95% CI `[-1.319, +0.327]`, $p = 0.298$) đối với `Standard U-Net`.
   - **Kết luận theo Quy tắc 2:** Không có sự cải thiện có ý nghĩa thống kê khi bổ sung `1,000` ảnh Nikon vào tập huấn luyện. Khoảng trống `~6.41 dB` giữa PSNR thô (`~16.96 dB`) và `psnr_gtmean` (`~23.38 dB`) khẳng định rằng thách thức trên LSRW-Nikon là bài toán **bất định mức phơi sáng (illumination scale ambiguity)** chứ không phải do mô hình chưa nhìn thấy đặc trưng cảm biến Nikon.

3. **Đối chiếu Quy tắc 3 (Tác động của việc mở rộng `blur` từ 450 lên 1,858 cặp và huấn luyện 42,000 steps, 40 ảnh):**
   - **Kết quả:** $\Delta(\text{v4}-\text{v3})$ trên `blur` = `-0.032 dB` (95% CI `[-0.534, +0.433]`, $p = 0.896$) đối với `DarkIR-Edge (ours)` và `+0.207 dB` (95% CI `[-0.237, +0.665]`, $p = 0.369$) đối với `Standard U-Net`.
   - **Kết luận theo Quy tắc 3:** Tăng mật độ lấy mẫu frame trên cùng 150 chuỗi video của `LOL-Blur` và kéo dài thời gian huấn luyện thêm `+50%` bước cập nhật **không** tạo ra cải thiện có ý nghĩa thống kê. Mô hình đã chạm ngưỡng bão hòa dung lượng ở mức `~23.5–24.0 dB` PSNR và `~0.19–0.22` `sharp_ratio`.

4. **Đối chiếu Quy tắc 4 (So sánh kiến trúc nội bộ v4: `DarkIR-Edge` vs `Standard U-Net` và `DarkIR-Edge no-FFT`):**
   - **Kết quả:** Trên tập `pooled` (190 ảnh), `DarkIR-Edge (ours)` (`20.97 dB`), `Standard U-Net` (`21.03 dB`) và `DarkIR-Edge (no FFT)` (`21.10 dB`) hoàn toàn tương đương nhau về mặt thống kê (mọi 95% CI của hiệu ghép cặp đều chứa `0`, $p > 0.29$).
   - **Kết luận theo Quy tắc 4:** Ở quy mô `~1.9M` tham số, `DarkIR-Edge` không vượt trội hơn `Standard U-Net` về PSNR/SSIM tổng thể; mỗi kiến trúc có một vùng ưu thế nhỏ riêng (`U-Net` sắc nét hơn, LPIPS thấp hơn và chạy nhanh gấp ~2.4 lần; `DarkIR-Edge` ổn định hơn giữa các seed và có sai số vùng tối `dark10_mae` thấp hơn trên `huawei`/`nikon`). Đặc biệt, nhánh tần số `FreMLP` + `FFT loss` được chứng minh là dư thừa trong thiết lập này: biến thể `DarkIR-Edge (no FFT)` chỉ dùng `1.495M` tham số (`77.5%` của `U-Net`) vẫn đạt chất lượng tương đương hoàn toàn với bản đầy đủ.

5. **Đối chiếu Quy tắc 5 (Quyết định lựa chọn bộ trọng số cho ứng dụng Demo):**
   - **Quyết định chính thức:** Tuân thủ đúng Quy tắc 5 đã đăng ký trước, vì v4 không vượt qua Quy tắc 1 và không cải thiện có ý nghĩa thống kê trên `blur` hay `pooled` so với v3, **dự án giữ nguyên bộ trọng số của v3 làm checkpoint chính thức cho ứng dụng Demo**, và sử dụng toàn bộ số liệu v4 trong báo cáo khoa học như một nghiên cứu thực nghiệm quy mô lớn chứng minh giới hạn của việc mở rộng dữ liệu đa miền trên kiến trúc CNN siêu nhẹ.

---

## Phụ lục — Danh sách file nguồn kiểm chứng

Toàn bộ dữ liệu trong báo cáo được trích xuất trực tiếp từ các đường dẫn sau:
1. **Notebook đã thực thi trên Colab:** `colab-results/LLIE_DarkIR_Edge_Colab_v4.ipynb`
2. **Manifest phân tách dữ liệu:** `colab-results/llie_project_v4-20261104/llie_project_v4/experiments/v4/split_manifest.json`
3. **Bảng kết quả chính (`final.pt` EMA):** `colab-results/llie_project_v4-20261104/llie_project_v4/experiments/v4/results/table_main.csv`
4. **Bảng kết quả phụ (`best.pt` EMA):** `colab-results/llie_project_v4-20261104/llie_project_v4/experiments/v4/results/table_main_best.csv`
5. **Kiểm định thống kê nội bộ v4:** `colab-results/llie_project_v4-20261104/llie_project_v4/experiments/v4/results/paired_tests.csv`
6. **Kiểm định thống kê đối chứng `v4 − v3`:** `colab-results/llie_project_v4-20261104/llie_project_v4/experiments/v4/results/paired_tests_v3_v4.csv`
7. **Bảng hiệu năng suy luận:** `colab-results/llie_project_v4-20261104/llie_project_v4/experiments/v4/results/efficiency.csv`
8. **Chẩn đoán độ sáng `gt-mean`:** `colab-results/llie_project_v4-20261104/llie_project_v4/results_latest/diagnostic_gtmean.csv`
9. **Các chỉ số bổ sung & kiểm định phụ:** `colab-results/llie_project_v4-20261104/llie_project_v4/results_latest/table_extra.csv`, `colab-results/llie_project_v4-20261104/llie_project_v4/results_latest/paired_tests_extra.csv`
10. **Điểm số chi tiết từng ảnh (1,900 dòng & 2,090 dòng kèm `DarkIR-m`):** `colab-results/llie_project_v4-20261104/llie_project_v4/results_latest/results_per_image.csv`, `results_per_image_best.csv`, `results_per_image_with_ref.csv`
11. **Lịch sử huấn luyện theo epoch:** `colab-results/llie_project_v4-20261104/llie_project_v4/experiments/v4/checkpoints/darkir_edge_nofft_seed123/history.csv`
12. **Biểu đồ & ảnh so sánh định tính:** `colab-results/llie_project_v4-20261104/llie_project_v4/experiments/v4/figures/` (`fig_visual_grid.png`, `fig_fft_spectrum.png`, `fig_psnr_by_domain.png`, `fig_gtmean_gap.png`)
