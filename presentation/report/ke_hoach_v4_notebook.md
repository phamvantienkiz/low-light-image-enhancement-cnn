# Kế hoạch v4 — Notebook Colab (giao bên implement)

> Tổng quan: [ke_hoach_v4.md](file:///E:/MyProject/low-light-image-enhancement-cnn/presentation/report/ke_hoach_v4.md); dữ liệu: [ke_hoach_v4_du_lieu.md](file:///E:/MyProject/low-light-image-enhancement-cnn/presentation/report/ke_hoach_v4_du_lieu.md).
> Notebook mới `LLIE_DarkIR_Edge_Colab_v4.ipynb` **dẫn xuất từ v3 bằng script** `docs/ai/build_notebook_v4.py` (mẫu: `build_notebook_v3.py`), không viết lại; kiểm bằng `docs/ai/verify_notebook_v4.py` (mẫu: `verify_notebook_v3.py`).
> Số cell lấy theo notebook v3 (đã đối chiếu mã nguồn).

## 1. Thay đổi bắt buộc

| Cell v3 | Nội dung | Thay đổi v4 |
|---|---|---|
| 1 (`CFG`) | cấu hình | `EXP_VERSION='v4'`; Drive `MyDrive/llie_project_v4/`; `epochs=70`; `use_early_stopping=False`; **bỏ `repeat`, thêm `quota`** (mẫu/epoch theo miền); `n_train` theo manifest; `darkir_edge_nofft_seeds=[]` |
| 5 | ghép cặp + chia split | nhận thêm `nikon`, `lolv2syn` từ manifest (khóa `train`); cập nhật `EXPECTED_V4` (số lượng theo manifest thực); kiểm tra val/test giống v3; kiểm tra không rò chuỗi blur |
| 6 | `PairDataset`, DataLoader | Nạp lười cho miền blur (§3); sampler theo quota (§2); crop/flip giữ nguyên |
| 13 | `train_model` | `total_steps = epochs × (tổng quota // batch)` = 70 × 600 = 42,000; `T_max = 42,000`; phần còn lại giữ nguyên (EMA, `last.pt` resume mỗi 5 epoch, `best.pt`, `final.pt`) |
| 14 | `probe_speed` | thay `len(train_ds) // batch_size` bằng số bước/epoch theo quota (600); in RAM cache, RAM hệ thống, `max_memory_allocated` |
| mô hình | | Chạy Edge và U-Net; **không** thay đổi kiến trúc, loss, optimizer |
| 20 | `find_checkpoint` | mặc định `final.pt` như v3 |
| 21–24, 29 | đánh giá, paired tests, efficiency, `eval_extra` | giữ nguyên; thêm `paired_tests_v3_v4` (§5) |

Bỏ chạy no-FFT; kiểm tra các cell đánh giá/`ALL_CANDIDATE_RUNS` và `paired_report` chịu được danh sách seed rỗng (so sánh Edge–noFFT bị bỏ qua có chủ đích).

## 2. Sampler theo quota (điểm then chốt)

Mục tiêu: mỗi epoch có đúng **4,800 mẫu** với quota cố định theo miền.

| Miền | Quota/epoch | Pool |
|---|---:|---:|
| lolv2 | 1,278 | 639 (đúng 2 lượt, như v3) |
| huawei | 1,122 | 940+ |
| blur | 1,600 | ≈ 2,000 |
| nikon | 400 | ≈ 1,000 |
| lolv2syn | 400 | 900 |

Quy tắc rút mẫu mỗi epoch cho từng miền: nối các hoán vị ngẫu nhiên (`torch.randperm`) của pool cho đến khi đủ `quota` (không lặp trong một lượt), rồi **xáo trộn toàn bộ 4,800 chỉ số**. Hoán vị dùng `torch.Generator` seed theo `seed` (và theo epoch) để tái lập; sau khi resume từ `last.pt` thứ tự mẫu có thể khác (đã chấp nhận ở v3, ghi lại trong báo cáo).

Cài đặt: lớp `Sampler` tuỳ chỉnh (`__iter__` sinh chỉ số vào `ConcatDataset` hoặc danh sách `(domain, idx)`), `DataLoader(..., sampler=sampler, drop_last=True)`; `len(sampler) = 4,800` ⇒ `len(loader) = 600`. Không dùng `shuffle=True` cùng sampler.

Kiểm tra bắt buộc (test nhanh, `LLIE_TEST=1`): đếm số mẫu theo miền trong một epoch bằng đúng quota; tổng chỉ số = 4,800; lolv2 mỗi ảnh đúng 2 lần.

## 3. Ngân sách RAM/VRAM và chiến lược nạp ảnh

Giả định Colab Pro, L4, High RAM: VRAM dùng được ≈ 22.5 GB; RAM ≈ 25 GB; đĩa ≈ 150 GB (do người dùng cung cấp; **không xác nhận độc lập**).

Ước tính bộ nhớ ảnh train nạp RAM (uint8, độ phân giải gốc, cặp low+high):

| Miền | Ảnh/cặp | Kích thước/cặp | Tổng |
|---|---:|---:|---:|
| lolv2 | 639 | 600×400×3×2 = 1.44 MB | 0.92 GB |
| lolv2syn | 900 | 384×384×3×2 = 0.89 MB | 0.80 GB |
| huawei | 940 | 960×720×3×2 = 4.15 MB | 3.9 GB (2,350 ảnh ⇒ 9.8 GB) |
| nikon | ≈ 1,000 | 960×640×3×2 = 3.69 MB | ≈ 3.7 GB |
| blur | ≈ 2,000 | 640×1120×3×2 = 4.30 MB | ≈ 8.6 GB |
| **Nạp hết** | | | **≈ 18 GB** (≈ 24 GB nếu Huawei 2,350) ⇒ quá sát 25 GB |

**Quyết định:** nạp RAM `lolv2`, `lolv2syn`, `nikon`, `huawei` (≈ 9–10 GB nếu Huawei 940; nếu Huawei lớn hơn thì cũng nạp lười), nạp **lười** `blur` (đọc PNG từ `/content/datasets` theo từng mẫu bằng `cv2.imread`). Ước tính tải: 1,600 mẫu blur/epoch × decode PNG 640×1120 (≈ 10–20 ms, ước lượng chưa đo) trên `num_workers` ≥ 4 ⇒ không nghẽn so với 0.15–0.3 s/bước. **Phải đo** bằng `probe_speed`: giây/bước ở v4 không được lớn hơn v3 quá 20%; nếu nghẽn, nạp thêm miền blur vào RAM tới ngưỡng an toàn (RAM cache ≤ 14 GB, ghi log `psutil.virtual_memory()`).

VRAM: giữ patch 256, batch 8, AMP như v3 ⇒ không đổi; ghi `torch.cuda.max_memory_allocated` đầu run để lưu vào log.

## 4. Chạy huấn luyện

- 2 phiên theo seed (như v3): phiên 1 `seed=42` (Edge → U-Net), phiên 2 `seed=123`; chỉnh danh sách seed trong `CFG` từng phiên, sau cùng đặt lại `[42, 123]` để chạy các cell đánh giá. `run_config.json` ghi lại đủ cả hai seed.
- Ước tính (cần đo): Edge ≈ 210 phút/run, U-Net ≈ 105 phút/run (chưa đo) ⇒ ≈ 6 giờ/phiên gồm val và giải nén, dưới 12 giờ.
- Resume: giữ cơ chế `last.pt` (mất ≤ 5 epoch khi đứt kết nối); không chuyển seed khi seed hiện tại chưa đủ 2 `final.pt`.
- Thứ tự cell giữ như v3; khi xong từng run phải có `final.pt`, `best.pt`, `history.csv` trên Drive.

## 5. Cell đánh giá mới

**5.1 So sánh v3 ↔ v4 (`paired_tests_v3_v4.csv`)** — đọc `results_per_image.csv` của v3 (tải từ `colab-results/` lên Drive, hoặc copy vào thư mục v4) và v4; với mỗi mô hình (Edge, U-Net) tính ΔPSNR = v4 − v3 theo ảnh test (trung bình 2 seed), bootstrap 2,000 lần seed 42, t-test, Wilcoxon, theo từng miền và các nhóm:
- `lolv2+huawei` (130 ảnh): **nhóm bảo vệ tăng sáng**;
- `blur` (40 ảnh);
- `nikon` (20 ảnh): đánh nhãn "miền đã thấy ở v4";
- `pooled` (190 ảnh).
Lược đồ cột giống `paired_tests.csv` (`a, b, domain, n, mean_delta_psnr, ci_lo, ci_hi, t_p, wilcoxon_p, frac_a_better`), thêm cột `group_note`.

**5.2 Phân tích bổ sung:** chạy lại `eval_extra` (cell 29) cho Edge và U-Net v4 để so với v3 (`lum_bias`, `lum_abs_err`, `lum_emd`, `ciede2000`, `sharp_ratio`, `dark10/bright10_mae`). Dùng cùng checkpoint `final.pt`.

**5.3 Real-LOLBlur (không tham chiếu):**
- Giải nén `real_lolblur_eval.zip` vào `/content/datasets/RealLOLBlur`; chạy Edge và U-Net của v3 và v4 (cần `final.pt` v3 còn trên Drive; xác nhận trước) và (tuỳ chọn) DarkIR-m tham chiếu theo cell 22.
- Chỉ số: **NIQE, NRQM, MUSIQ** (như Bảng 5 của paper) qua thư viện `pyiqa` (cài đặt chỉ trong phiên Colab; xác nhận pyiqa nạp được trọng số), cộng độ chói trung bình đầu ra/đầu vào, tỉ lệ pixel bão hòa. Xuất `realblur_noref.csv` (mỗi ảnh × mô hình × version) và bảng tóm tắt trung bình ± độ lệch giữa seed.
- Lưới ảnh định tính: 8–12 ảnh cố định theo seed 42 (không chọn lọc sau khi xem kết quả) → `realblur_visual_samples/`.
- Không có ảnh đích ⇒ **không** báo PSNR/SSIM/LPIPS trên tập này.

**5.4 Efficiency:** không cần chạy lại (kiến trúc không đổi) — dùng số v3, ghi chú GPU.

## 6. Đầu ra cần có trên Drive/`colab-results/`

`table_main.csv` (+`_best`), `paired_tests.csv` (Edge vs U-Net trong v4), `paired_tests_v3_v4.csv`, `results_extra.csv`, `paired_tests_extra.csv`, `realblur_noref.csv`, `history.csv` từng run, `run_config.json`, `curves.png`, ảnh định tính.

## 7. Kiểm thử trước khi giao chạy

1. `ast.parse` mọi cell; chạy toàn bộ với `LLIE_TEST=1` trên dữ liệu mẫu nhỏ.
2. Test sampler (mục 2): đếm theo miền đúng quota.
3. Test nạp lười: so sánh một ảnh blur nạp lười và nạp RAM giống hệt từng pixel.
4. Kiểm tra val/test trong manifest v4 trùng v3.
5. Chạy `probe_speed` trên Colab; ghi giây/bước, RAM và VRAM; so với ước tính trong tài liệu tổng quan trước khi chạy 2 phiên dài.
6. `preregistration_v4.md` đã commit trước khi chạy.
