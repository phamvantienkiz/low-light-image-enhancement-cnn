# Hướng dẫn Chia Phiên Chạy (Run Sessions) và Chống Mất Dữ Liệu trên Google Colab cho Thử Nghiệm `v4`

> **Tài liệu áp dụng cho:** [LLIE_DarkIR_Edge_Colab_v4.ipynb](file:///E:/MyProject/low-light-image-enhancement-cnn/notebook/LLIE_DarkIR_Edge_Colab_v4.ipynb)  
> **Thư mục lưu trữ trên Google Drive:** `/content/drive/MyDrive/llie_project_v4/`  
> **Giới hạn phần cứng Google Colab:** Tối đa **12 giờ/phiên kết nối (session)**; ổ đĩa cục bộ `/content/datasets` bị xóa khi mất kết nối, nhưng mọi dữ liệu trong `/content/drive/MyDrive/llie_project_v4/` được lưu vĩnh viễn.

---

## 1. Bảng Ánh Xạ Toàn Bộ Các Cell trong Notebook `v4`

Notebook `LLIE_DarkIR_Edge_Colab_v4.ipynb` gồm **32 cells** (9 Markdown cells + 23 Code cells). Để tránh nhầm lẫn giữa **chỉ số thứ tự cell trong Jupyter/Colab** và **nhãn `# ===== CELL X =====`** ghi ở dòng đầu của mỗi khối code, bảng dưới đây liệt kê chính xác từng cell:

| Thứ tự Code Cell | Nhãn trong Code (`# ===== CELL ... =====`) | Vai trò của Cell | Có bắt buộc chạy khi khởi động lại Colab không? |
| :---: | :--- | :--- | :---: |
| **Code 1** | `CELL 1` | Imports, khóa `MASTER_SEED = 42`, khai báo `CFG` (`EXP_VERSION = 'v4'`) | **BẮT BUỘC** (mọi phiên) |
| **Code 2** | `CELL 2` | Mount Google Drive, tạo thư mục `experiments/v4/`, giải nén `llie_data_clean_v4.zip` vào `/content/datasets` | **BẮT BUỘC** (mọi phiên) |
| **Code 3** | `CELL 3` | Kiểm tra cây thư mục `/content/datasets` sau giải nén | Khuyến nghị (1 giây) |
| **Code 4** | `CELL 5` | Nạp và kiểm định `split_manifest.json` (`6,777 Train : 150 Val : 190 Test`, `nikon_test_in_domain=True`) | **BẮT BUỘC** (mọi phiên) |
| **Code 5** | `CELL 6` | Khởi tạo `MultiDomainTrainDataset` (nạp RAM `lolv2`, `nikon`, `lolv2syn`; nạp lười `huawei`, `blur`), `val_by_dom` và `QuotaSampler` (`4,800 mẫu/epoch = 600 steps/epoch`) | **BẮT BUỘC** khi Train / Probe |
| **Code 6** | `CELL 8` | Định nghĩa kiến trúc đề xuất `DarkIREdge (C=24)` (`1,924,038 params`) | **BẮT BUỘC** (mọi phiên) |
| **Code 7** | `CELL 9` | Định nghĩa kiến trúc đối chuẩn `UNetBaseline (base=32)` (`1,928,483 params`) | **BẮT BUỘC** (mọi phiên) |
| **Code 8** | `CELL 10` | Kiểm chứng số tham số và smoke-test kích thước ảnh lẻ | Tùy chọn (2 giây) |
| **Code 9** | `CELL 11` | Định nghĩa `CompositeLoss`, `EMA`, `psnr_u8`, `predict_u8`, `val_psnr_macro` | **BẮT BUỘC** (mọi phiên) |
| **Code 10** | `CELL 12` | Định nghĩa hàm `build(tag)` và `train_model(tag, seed)` (có tích hợp **Auto-Resume** từ `last.pt` và **Auto-Skip** khi đã có `final.pt`) | **BẮT BUỘC** (mọi phiên) |
| **Code 11** | `CELL 13` | `probe_speed`: Đo tốc độ thực tế (`it/s`, `s/step`), ước tính số phút/run, kiểm tra RAM/VRAM | Chạy ở Phiên 1 (30 giây) |
| **Code 12** | `CELL 14` | **Huấn luyện `DarkIR-Edge (C=24)`** theo danh sách `SEEDS_TO_RUN_DARKIR_EDGE` | Chạy ở Phiên Train |
| **Code 13** | `CELL 15` | **Huấn luyện `Standard U-Net (base=32)`** theo danh sách `SEEDS_TO_RUN_UNET` | Chạy ở Phiên Train |
| **Code 14** | `CELL ABLATION (v2)` | Huấn luyện `darkir_edge_nofft` (Ở `v4`, `darkir_edge_nofft_seeds = []` nên cell này tự động in `[SKIP]`) | Không cần chạy |
| **Code 15** | `CELL 16` | Định nghĩa Baselines cổ điển (`Gamma 2.2`, `CLAHE`, `MSRCR`), LPIPS, `eval_method`, `find_checkpoint`, `nn_fn` | **BẮT BUỘC** ở Phiên Đánh giá |
| **Code 16** | `CELL 17` | Chạy đánh giá trên 190 ảnh Test cho cả `final.pt` và `best.pt` $\rightarrow$ xuất `results_per_image.csv` & `results_per_image_best.csv` | Chạy ở Phiên Đánh giá |
| **Code 17** | `CELL 18` | Tổng hợp `table_main.csv`, `table_main_best.csv`, `diagnostic_gtmean.csv` và kiểm định ghép cặp nội bộ `paired_tests.csv` | Chạy ở Phiên Đánh giá |
| **Code 18** | `CELL 19` | Xuất lưới ảnh định tính `qualitative.png`, thư mục `visual_samples/` và biểu đồ đường cong học tập `curves.png` | Chạy ở Phiên Đánh giá |
| **Code 19** | `CELL 20` | Đo hiệu năng phần cứng (`Params`, `GFLOPs`, `Latency FP32/FP16`, `FPS`, `Peak VRAM`) $\rightarrow$ xuất `efficiency.csv` | Chạy ở Phiên Đánh giá |
| **Code 20** | `CELL 22` | Tải và đánh giá mốc trần tham chiếu `DarkIR-m (C=32, CVPR 2025)` từ Hugging Face $\rightarrow$ `results_per_image_with_ref.csv` | Tùy chọn ở Phiên Đánh giá |
| **Code 21** | `CELL 23` | Xuất lưới ảnh đối sánh với `DarkIR-m` và 5 ảnh mẫu LOL-Blur từ Hugging Face Space | Tùy chọn ở Phiên Đánh giá |
| **Code 22** | `CELL 24` | Đo các chỉ số bổ sung (`lum_bias`, `lum_emd`, `ciede2000`, `sharp_ratio`, `dark10_mae`, `bright10_mae`) $\rightarrow$ `results_extra.csv`, `table_extra.csv`, `paired_tests_extra.csv` | Chạy ở Phiên Đánh giá |
| **Code 23** | `CELL 25` | **Kiểm định đối chứng ghép cặp `v3 ↔ v4`** (`paired_tests_v3_v4.csv`) và **Đánh giá không tham chiếu `Real-LOLBlur` (482 ảnh)** (`realblur_noref.csv`) | Chạy ở Phiên Đánh giá |

---

## 2. Cơ Chế Bảo Vệ Chống Mất Dữ Liệu Khi Ngắt Kết Nối Đột Ngột

Trong `CELL 12` (`train_model`), notebook đã được lập trình sẵn **2 tầng bảo vệ tự động** lưu trực tiếp lên Google Drive (`/content/drive/MyDrive/llie_project_v4/experiments/v4/checkpoints/{tag}_seed{seed}/`):

1. **Lưu trạng thái định kỳ mỗi 5 Epoch (`last.pt` & `best.pt`):**
   - Cứ mỗi `val_interval = 5` epochs (Epoch 5, 10, 15, ..., 70), hàm `train_model` tự động ghi đè file `last.pt` trực tiếp lên Google Drive.
   - File `last.pt` chứa **toàn bộ trạng thái tại thời điểm đó**: trọng số `model`, trọng số `ema`, trạng thái bộ tối ưu `opt` (`AdamW`), lịch trình giảm tốc độ học `sched` (`CosineAnnealingLR`), bộ khuếch đại đạo hàm `scaler` (`AMP GradScaler`), lịch sử `history`, chỉ số `epoch` đã hoàn thành, `ema_n`, và `best_val_psnr`.
   - **Hệ quả:** Nếu mạng rớt, trình duyệt đóng, hoặc Colab thu hồi GPU ở giữa chừng (ví dụ đang chạy ở Epoch 43), khi kết nối lại và chạy lại đúng cell huấn luyện đó, hàm `train_model` sẽ tự động phát hiện `last.pt`, in ra:
     ```text
     [RESUME] Khôi phục darkir_edge (seed=42) từ epoch 40/70
     ```
     và **tiếp tục chạy từ Epoch 41 đến 70** (tối đa chỉ mất công tính toán của $\le 4$ epochs vừa chạy dở, tuyệt đối không bao giờ phải train lại từ Epoch 1).
2. **Khóa hoàn tất từng Run (`final.pt` & `history.csv`):**
   - Ngay khi một mô hình chạy xong Epoch 70/70, hệ thống lập tức xuất `final.pt` và `history.csv` lên Google Drive.
   - Trong các lần chạy sau, nếu hàm `train_model` thấy `final.pt` đã tồn tại (và `FORCE_RETRAIN = False`), nó sẽ lập tức bỏ qua mô hình/seed đó trong 0.1 giây:
     ```text
     [SKIP] darkir_edge (seed=42) trong phiên bản "v4" đã có final.pt (epoch 70/70) -> bỏ qua.
     ```

> **LƯU Ý QUAN TRỌNG VỀ CỜ `FORCE_RETRAIN` TRONG `CELL 1`:**  
> Luôn giữ `FORCE_RETRAIN = False` (giá trị mặc định) trong suốt quá trình chạy và resume. Nếu đặt `FORCE_RETRAIN = True`, hệ thống sẽ bỏ qua `last.pt` và `final.pt` để huấn luyện lại từ đầu.

---

## 3. Chiến Lược Chia Phiên Chạy (Run Plan) Đảm Bảo An Toàn Dưới Ngưỡng 12 Giờ

Trong thử nghiệm `v4`, mỗi mô hình chạy cố định **70 epochs × 600 steps/epoch = 42,000 steps** (tăng 1.5× số bước so với 28,000 steps của `v3`). Thời gian dự kiến trên GPU Colab (L4 hoặc T4):
- **`DarkIR-Edge (C=24)` (1 seed, 42,000 steps):** ~2.5 – 3.5 giờ (trên L4) hoặc ~3.5 – 4.5 giờ (trên T4).
- **`Standard U-Net (base=32)` (1 seed, 42,000 steps):** ~1.2 – 1.8 giờ (trên L4) hoặc ~1.8 – 2.5 giờ (trên T4).
- **Tổng cộng cả 4 runs** (`DarkIR-Edge` × 2 seeds + `Standard U-Net` × 2 seeds): khoảng **8 – 13 giờ**.

Nếu gom cả 4 runs vào một phiên duy nhất, phiên chạy có nguy cơ chạm trần **12 giờ** của Colab ở run cuối cùng. Do đó, quy trình chuẩn được chia thành **2 Phiên Huấn Luyện (mỗi phiên ~4.5 – 6.5 giờ, an toàn tuyệt đối dưới 12 giờ)** hoặc **3 Phiên (nếu tách riêng phần Đánh giá)**:

### Chuẩn bị trước trên Google Drive (chỉ làm 1 lần trước Phiên 1)
Tải lên thư mục `/MyDrive/llie_project_v4/` trên Google Drive:
1. `llie_data_clean_v4.zip` (4.94 GB — đã chứa sẵn `split_manifest.json` ở gốc).
2. *(Để chạy đối chứng tự động `v3 ↔ v4` ở Cell 25):* Tạo thư mục `/MyDrive/llie_project_v4/v3_reference_results/` và tải file `results_per_image.csv` (và `results_extra.csv`) của `v3` vào đó.
3. *(Tùy chọn, nếu có gói Real-LOLBlur):* Đặt `real_lolblur_eval.zip` vào `/MyDrive/llie_project_v4/real_lolblur_eval.zip`.

---

### 🔹 PHIÊN 1 (Session 1 — Khoảng 4.5 đến 6.5 giờ): Huấn luyện trọn vẹn `seed = 42` (`DarkIR-Edge` + `Standard U-Net`)

**Mục tiêu:** Hoàn tất 100% cả 2 mô hình cho `seed = 42` (`darkir_edge_seed42/final.pt` và `unet_seed42/final.pt`).

#### Bước 1.1 — Chỉnh trực tiếp danh sách seed tại `CELL 14` và `CELL 15` *(Không cần đụng vào `CELL 1`)*:
- Tại **`CELL 14`** (Huấn luyện `DarkIR-Edge`), sửa dòng:
  ```python
  SEEDS_TO_RUN_DARKIR_EDGE = [42]
  ```
- Tại **`CELL 15`** (Huấn luyện `Standard U-Net`), sửa dòng:
  ```python
  SEEDS_TO_RUN_UNET = [42]
  ```

#### Bước 1.2 — Thứ tự các Cell cần chạy trong Phiên 1:
Chạy lần lượt từ trên xuống dưới các cell sau:
1. **`CELL 1`** *(Thiết lập cấu hình, seed, biến môi trường — giữ nguyên không sửa)*
2. **`CELL 2`** *(Mount Google Drive & giải nén `llie_data_clean_v4.zip` vào `/content/datasets` — mất khoảng 2–4 phút)*
3. **`CELL 3`** *(Kiểm tra cây thư mục giải nén)*
4. **`CELL 5`** *(Nạp `split_manifest.json` và kiểm tra 6,777 Train : 150 Val : 190 Test)*
5. **`CELL 6`** *(Khởi tạo Dataset nạp lười + `QuotaSampler` 4,800 mẫu/epoch — mất khoảng 1–2 phút để nạp RAM các miền nhỏ)*
6. **`CELL 8`** *(Định nghĩa `DarkIREdge`)*
7. **`CELL 9`** *(Định nghĩa `UNetBaseline`)*
8. **`CELL 10`** *(Kiểm tra số tham số & smoke test)*
9. **`CELL 11`** *(Định nghĩa `CompositeLoss`, `EMA`, hàm tính PSNR)*
10. **`CELL 12`** *(Định nghĩa `train_model`)*
11. **`CELL 13`** *(Chạy `probe_speed` để xem chính xác trên GPU hiện tại 1 run 42,000 steps mất bao nhiêu phút và kiểm tra RAM/VRAM)*
12. **`CELL 14`** *(Huấn luyện `darkir_edge` với `SEEDS_TO_RUN_DARKIR_EDGE = [42]` $\rightarrow$ chờ tới khi in ra `✔ Hoàn tất darkir_edge (seed=42) sau 70/70 epoch`)*
13. **`CELL 15`** *(Huấn luyện `unet` với `SEEDS_TO_RUN_UNET = [42]` $\rightarrow$ chờ tới khi in ra `✔ Hoàn tất unet (seed=42) sau 70/70 epoch`)*

#### Bước 1.3 — Kiểm tra trước khi đóng Phiên 1:
Đảm bảo trên Google Drive trong thư mục `/MyDrive/llie_project_v4/experiments/v4/checkpoints/` đã có đủ 2 thư mục con:
- `darkir_edge_seed42/` (chứa `final.pt`, `best.pt`, `last.pt`, `history.csv`)
- `unet_seed42/` (chứa `final.pt`, `best.pt`, `last.pt`, `history.csv`)

Sau khi xác nhận có đủ `final.pt` cho cả 2 mô hình ở `seed = 42`, chủ động **Disconnect and delete runtime** để làm mới hoàn toàn bộ đếm 12 giờ cho Phiên 2.

---

### 🔹 PHIÊN 2 (Session 2 — Khoảng 5.0 đến 7.0 giờ): Huấn luyện trọn vẹn `seed = 123` + Chạy Toàn Bộ Đánh Giá (`CELL 16 → CELL 25`)

**Mục tiêu:** Hoàn tất 2 mô hình cho `seed = 123` (`darkir_edge_seed123/final.pt` và `unet_seed123/final.pt`), sau đó chạy thẳng các cell đánh giá trên cả 2 seeds `[42, 123]`.

#### Bước 2.1 — Chỉnh trực tiếp danh sách seed tại `CELL 14` và `CELL 15` sang `[123]`:
- Tại **`CELL 14`**, sửa dòng:
  ```python
  SEEDS_TO_RUN_DARKIR_EDGE = [123]
  ```
- Tại **`CELL 15`**, sửa dòng:
  ```python
  SEEDS_TO_RUN_UNET = [123]
  ```
  *(Lưu ý: kể cả khi để `[42, 123]` ở `CELL 14` và `CELL 15`, hệ thống vẫn tự động `[SKIP]` `seed=42` đã có `final.pt` trên Drive và chỉ train `seed=123`).*

#### Bước 2.2 — Thứ tự các Cell Huấn luyện trong Phiên 2:
1. **`CELL 1` $\rightarrow$ `CELL 2` $\rightarrow$ `CELL 3` $\rightarrow$ `CELL 5` $\rightarrow$ `CELL 6`** *(Khởi tạo lại môi trường và giải nén lại zip vào SSD cục bộ của máy ảo mới)*
2. **`CELL 8` $\rightarrow$ `CELL 9` $\rightarrow$ `CELL 11` $\rightarrow$ `CELL 12`** *(Định nghĩa lại các lớp mô hình, hàm mất mát và hàm huấn luyện — mất 2 giây)*
3. **`CELL 14`**: Huấn luyện `darkir_edge (seed=123)` cho tới hết 70/70 epochs.
4. **`CELL 15`**: Huấn luyện `unet (seed=123)` cho tới hết 70/70 epochs.

#### Bước 2.3 — Chạy Toàn Bộ Các Cell Đánh Giá & Xuất Báo Cáo (`CELL 16 → CELL 25`):
Vì `CELL 1` đã khai báo cố định `darkir_edge_seeds = [42, 123]` và `unet_seeds = [42, 123]`, **sau khi `CELL 15` chạy xong, không cần chỉnh sửa lại bất kỳ cấu hình nào**, chỉ cần chạy tuần tự từ **`CELL 16` đến `CELL 25`** (tổng thời gian đánh giá chỉ mất khoảng **10 – 15 phút**):
1. **`CELL 16`**: Khởi tạo hàm đánh giá và nạp mô hình từ `final.pt` / `best.pt`.
2. **`CELL 17`**: Đánh giá 190 ảnh Test qua cả 2 seeds `[42, 123]` $\rightarrow$ xuất `results_per_image.csv` và `results_per_image_best.csv`.
3. **`CELL 18`**: Xuất `table_main.csv`, `table_main_best.csv`, `diagnostic_gtmean.csv`, `paired_tests.csv`.
4. **`CELL 19`**: Xuất `qualitative.png`, `visual_samples/` và `curves.png`.
5. **`CELL 20`**: Đo tốc độ suy luận, GFLOPs, VRAM $\rightarrow$ xuất `efficiency.csv`.
6. **`CELL 22` & `CELL 23`** *(Tùy chọn)*: Đối sánh với mốc trần `DarkIR-m (CVPR 2025)` từ Hugging Face.
7. **`CELL 24`**: Tính các chỉ số bổ sung (`lum_bias`, `lum_emd`, `ciede2000`, `sharp_ratio`, `dark10_mae`, `bright10_mae`) $\rightarrow$ xuất `results_extra.csv`, `table_extra.csv`, `paired_tests_extra.csv`.
8. **`CELL 25`**:
   - Tự động đọc kết quả `v3` và xuất bảng đối chứng ghép cặp **`paired_tests_v3_v4.csv`** (`lolv2+huawei`, `blur`, `nikon`, `pooled`).
   - Nếu có `real_lolblur_eval.zip` trên Drive: tự động đánh giá không tham chiếu 482 ảnh Real-LOLBlur (`NIQE`, `NRQM`, `MUSIQ`) $\rightarrow$ xuất `realblur_noref.csv` và `realblur_visual_samples/`.

---

## 4. Quy Trình Khôi Phục Khẩn Cấp Khi Bị Ngắt Kết Nối Đột Ngột Giữa Chừng

Nếu đang chạy `CELL 14` hoặc `CELL 15` mà gặp sự cố (mất mạng, Colab báo *"Runtime disconnected"*, tắt nhầm tab trình duyệt):

1. **Kết nối lại Runtime (Reconnect)** trên Google Colab.
2. **Không sửa đổi `EXP_VERSION = 'v4'` và giữ nguyên `FORCE_RETRAIN = False`** ở `CELL 1`.
3. Chạy lại nhóm cell khởi tạo bắt buộc (mất khoảng 3–4 phút nếu máy ảo bị cấp mới, hoặc chỉ 5 giây nếu máy ảo cũ vẫn còn giữ ổ đĩa `/content/datasets`):
   - **`CELL 1` $\rightarrow$ `CELL 2` $\rightarrow$ `CELL 5` $\rightarrow$ `CELL 6` $\rightarrow$ `CELL 8` $\rightarrow$ `CELL 9` $\rightarrow$ `CELL 11` $\rightarrow$ `CELL 12`**
   *(Lưu ý: ở `CELL 2`, câu lệnh `if not os.listdir(CFG['data_root']):` sẽ tự kiểm tra xem `/content/datasets` còn dữ liệu không; nếu còn thì bỏ qua giải nén, nếu mất thì tự động giải nén lại).*
4. Chạy lại **`CELL 14`** và **`CELL 15`**:
   - Các run đã xong 70/70 epochs sẽ tự động `[SKIP]`.
   - Run đang chạy dở sẽ tự động nạp `last.pt` từ Google Drive (`[RESUME] Khôi phục ... từ epoch X/70`) và chạy tiếp phần còn lại cho đến khi hoàn thành Epoch 70 và lưu `final.pt`.

---

## 5. Bảng Kiểm Tra Đầu Ra Cuối Cùng Trên Google Drive (`Checklist`)

Sau khi hoàn tất Phiên 2, kiểm tra thư mục `/MyDrive/llie_project_v4/experiments/v4/` trên Google Drive phải có đầy đủ các file sau trước khi tải về máy local:

- `run_config.json`
- `checkpoints/`
  - `darkir_edge_seed42/` (`final.pt`, `best.pt`, `last.pt`, `history.csv`)
  - `darkir_edge_seed123/` (`final.pt`, `best.pt`, `last.pt`, `history.csv`)
  - `unet_seed42/` (`final.pt`, `best.pt`, `last.pt`, `history.csv`)
  - `unet_seed123/` (`final.pt`, `best.pt`, `last.pt`, `history.csv`)
- `results/` (đồng thời có bản sao tại `/MyDrive/llie_project_v4/results_latest/`)
  - `results_per_image.csv` & `results_per_image_best.csv`
  - `table_main.csv` & `table_main_best.csv`
  - `paired_tests.csv` *(DarkIR-Edge vs Standard U-Net trong v4)*
  - `paired_tests_v3_v4.csv` *(Kiểm định đối chứng ghép cặp v4 vs v3)*
  - `diagnostic_gtmean.csv`
  - `results_extra.csv`, `table_extra.csv`, `paired_tests_extra.csv`
  - `efficiency.csv`
  - `qualitative.png`, `curves.png`, thư mục `visual_samples/`
  - *(Nếu có chạy Cell 22–23 & Real-LOLBlur)*: `results_per_image_with_ref.csv`, `qualitative_with_darkir_m.png`, `qualitative_lolblur_5samples.png`, `realblur_noref.csv`, `realblur_visual_samples/`
