# Kế hoạch v4 — Dữ liệu và script (giao bên implement)

> Tổng quan: [ke_hoach_v4.md](file:///E:/MyProject/low-light-image-enhancement-cnn/presentation/report/ke_hoach_v4.md). Script đặt trong `docs/ai/`, mẫu: `docs/ai/prepare_llie_data_v3.py` (lệnh `select`, `package`).
> Mọi số "hiện có" dưới đây đo trực tiếp trên máy; mọi số "ước tính" cần xác nhận khi chạy.

## 1. Hiện trạng đã đo

| Mục | Giá trị |
|---|---|
| LOL-Blur zip thô trên `D:\Temp_LOL-Blur` | `low_blur_noise.zip` 7.31 GB, `high_sharp_scaled.zip` 7.64 GB, `test.zip` 5.53 GB (đọc trực tiếp từ zip, **không cần giải nén hết**) |
| Blur đã giải nén (v3) | 490 cặp / 0.72 GB ⇒ **≈ 1.47 MB/cặp**, ảnh 640×1120 PNG |
| LOL-Blur train | 10,200 cặp = 170 chuỗi × 60 frame; v3 dùng 150 chuỗi cho train, 20 cho val |
| LOLv2-Real | 689 train + 100 test, ảnh 600×400 (đã trong zip v3) |
| **LOLv2-Synthetic (đã có)** | `datasets/LOL-v2/Synthetic/{Train,Test}/{Low,Normal}` = **900 / 100 cặp**, ảnh 384×384, 0.44 GB |
| LSRW-Huawei Train (có) | **1,000 ảnh** trong `datasets/LSRW/Train/Huawei/{low,high}`, tên `00001.jpg…01000.jpg`, 960×720 JPEG ≈ 107 KB |
| LSRW Eval (có) | Huawei 30 cặp, Nikon 20 cặp (960×640) |
| LSRW-Nikon Train | **chưa có** (gốc 3,150 cặp) |
| Ổ trống | E: 15.9 GB; D: 30.4 GB |
| `llie_data_clean_v3.zip` | 1.69 GB (phần không-blur ≈ 0.97 GB) |

> [!WARNING]
> `docs/ai/dataset_download_links_and_guide.md` nói LSRW-Huawei Train có 2,450 cặp, nhưng thư mục chỉ có 1,000. Nhiều khả năng `list_hf_folder()` trong `download_missing_datasets.py` bị cắt ở trang đầu (API Hugging Face phân trang, script không đi theo trang tiếp). **Chưa kiểm chứng.**

## 2. Dữ liệu v4 (train)

Test (190) và val (150) **giữ nguyên danh sách file của v3** (đọc từ `docs/ai/split_manifest_v3.json`), không thay đổi và không được rò sang train.

| Miền (khóa manifest) | Nguồn | Pool train | Quy tắc chọn |
|---|---|---:|---|
| `lolv2` | LOLv2-Real Train | 639 | Giữ nguyên danh sách v3 |
| `huawei` | LSRW-Huawei Train | **940** (nếu chỉ có 1,000 ảnh) | Dùng toàn bộ pool: bỏ 50 ảnh cuối (val) và 10 ảnh đệm trước chúng, đúng quy tắc `huawei_gap = 10` của v3 ⇒ `hw[:len − 50 − 10]` |
| `blur` | LOL-Blur train | **≈ 2,000** | Xem §2.1 |
| `nikon` | LSRW-Nikon Train (tải thêm) | **≈ 1,000** | Xem §2.2 |
| `lolv2syn` | LOLv2-Synthetic Train | 900 | Dùng toàn bộ; crop 256 từ ảnh 384×384 |

Lưu ý quan trọng cho Huawei nếu tải đủ 2,450 ảnh: **val v3 là 50 ảnh cuối của danh sách 1,000 (`00951–01000`)**. Nếu danh sách dài ra, "50 ảnh cuối" sẽ đổi ⇒ phải **cố định val bằng tên file** (`00951–01000`), loại `00941–00950` (đệm trước) và **`01001–01010` (đệm sau)** khỏi train, rồi mới thêm `01011–02450` vào pool. Test Huawei lấy từ thư mục Eval nên không bị ảnh hưởng.

### 2.1 Blur (≈ 2,000 cặp)

1. **Bao hàm v3:** giữ nguyên 400 train, 50 val, 40 test của v3 (đọc `docs/ai/lolblur_v3_selection.json`). Val/test không đổi.
2. Thêm vào **chỉ trong 150 chuỗi train của v3** (không đụng 20 chuỗi val, không đụng 30 chuỗi test). Số cảnh train **không tăng** (vẫn 150).
3. Thêm bằng `add_frame_with_gap` với `MIN_FRAME_GAP = 4` cho frame mới (400 cặp cũ giữ gap ≥ 10). Mục tiêu ≈ 13 frame/chuỗi; tối đa 15 khi gap = 4. Script phải in số thực tế và **assert ≥ 1,800**; nếu thiếu, hạ gap xuống 3 và ghi rõ vào báo cáo chọn dữ liệu.
4. Bắt buộc kiểm tra như v3: không trùng đường dẫn/chuỗi train–val–test, kích thước (H, W), `mean_gt > mean_in`, độ nét Laplacian, SHA-256, preview ảnh thật.
5. Dung lượng: ≈ +1,600 cặp × 1.47 MB ≈ **+2.4 GB** giải nén (trên D:).

### 2.2 Nikon (≈ 1,000 cặp)

- Tải `LSRW/Train/Nikon/{low,high}` (3,150 cặp, ≈ 0.8 GB ước tính) về `D:\Temp_LSRW\`; chọn đều theo chỉ số (stride ≈ 3, seed 42) để giảm ảnh cùng cảnh liền kề; cố định bằng tên file.
- Kiểm tra: không trùng với 20 ảnh `LSRW/Eval/Nikon` (so SHA-256 và tên). Không đưa Nikon vào val để val giữ nguyên.
- Hệ quả: test Nikon từ v4 là **miền đã thấy**; ghi chú trong manifest (`"nikon_test_in_domain": true`) và trong mọi bảng kết quả.

### 2.3 LOLv2-Synthetic

- Dùng cả 900 cặp train. Kiểm tra ghép cặp tên file giữa `Low` và `Normal` (không giả định quy tắc đổi `low→normal` như LOLv2-Real; ghi lại quy tắc thực tế). Ảnh 384×384 ≥ patch 256 nên không cần pad.
- `Synthetic/Test` (100 cặp) **không** vào bộ test chính; có thể giữ làm tập phụ chỉ-v4 (tùy chọn, không dùng cho kết luận).

### 2.4 Real-LOLBlur (không train)

- 482 ảnh thật chọn từ RealBlur-J, **không có ảnh đích**. Tìm nguồn trong repo LEDNet (Google Drive/Baidu `fh32`; kiểm tra link còn sống và cấu trúc thư mục). Nguồn tìm kiếm còn nhắc tới thêm 872 ảnh Sony RX10 IV; **chỉ dùng 482 ảnh** để khớp Bảng 5 của paper.
- Đặt tại `D:\Temp_RealLOLBlur\`; đóng thành zip riêng `real_lolblur_eval.zip` (không nằm trong zip train); upload Drive cùng thư mục v4. Dung lượng chưa biết, kiểm tra trước khi tải.
- Cách ly: không bao giờ xuất hiện trong manifest train/val.

## 3. Kiểm chứng giả thuyết Huawei bị cắt (làm đầu tiên)

1. Liệt kê `data1/Training data/Huawei/low` và `…/high` bằng `huggingface_hub` (`HfApi().list_repo_tree(..., recursive=True)` có phân trang tự động) hoặc theo header `Link` của API; so với 1,000 file hiện có.
2. Nếu số file > 1,000: sửa/viết lại script tải dùng `snapshot_download(..., allow_patterns=[...])` hoặc đi theo phân trang; tải Huawei đầy đủ và **Nikon Train**; đo dung lượng thực.
3. Nếu thực sự chỉ có 1,000 ảnh: chuyển sang pool Huawei = 940; Nikon Train phải lấy từ mirror/nguồn khác (OpenDataLab hoặc Baidu `wmrr`, xem hướng dẫn cũ); nếu không tải được ⇒ bỏ Nikon khỏi v4 và chuyển quota Nikon (400 mẫu/epoch) sang LOLv2-Synthetic+Huawei, đồng thời cập nhật tài liệu tổng quan.

## 4. Script cần viết (đặt `docs/ai/`)

| Script | Nhiệm vụ |
|---|---|
| `download_lsrw_full.py` (hoặc sửa `download_missing_datasets.py`) | Tải đủ Huawei Train và Nikon Train, có phân trang, bỏ qua file đã có |
| `prepare_llie_data_v4.py` | Dẫn xuất từ `prepare_llie_data_v3.py`, lệnh `select` và `package` |
| `build_notebook_v4.py`, `verify_notebook_v4.py` | Xem tài liệu notebook |

`prepare_llie_data_v4.py select`:
- Đọc raw từ `D:\Temp_LOL-Blur` (zip), `datasets\` (Real, Synthetic, Huawei), `D:\Temp_LSRW` (Nikon).
- Chọn theo §2, giải nén phần blur mới vào `D:\Temp_LOL-Blur\LOLBlur_extracted`.
- Ghi `docs/ai/lolblur_v4_selection.json`, `docs/ai/lolblur_v4_preview.jpg`, in bảng thống kê theo miền (số ảnh, kích thước, mean_in → mean_gt, độ nét).

`prepare_llie_data_v4.py package`:
- Manifest `docs/ai/split_manifest_v4.json`, cùng lược đồ v3 nhưng thêm khóa miền `nikon` và `lolv2syn` vào `train`; mỗi mục `[low, high, sha256_low, sha256_high]`.
- Kiểm tra: val/test **bằng đúng** v3 (so từng đường dẫn); mọi cặp v3 còn trong train; không rò đường dẫn/chuỗi giữa các tập; test Nikon không trùng train Nikon; SHA-256 khớp.
- **Chỉ ghi vào zip các file có trong manifest + `DICM`/`LIME`** (khác v3 vốn zip nguyên thư mục `LSRW`/`LOL-v2/Real_captured`) để zip gọn và kiểm soát được; đối chiếu số cặp theo manifest, không suy từ công thức.
- Ghi zip ra `datasets\llie_data_clean_v4.zip` (E:, kỳ vọng ≈ 5 GB ≤ 15.9 GB trống; nếu không đủ thì ghi sang `D:` rồi upload từ đó). Không xóa file trung gian khi chưa được người dùng xác nhận.

## 5. Ngân sách đĩa tại máy (ước tính)

| Hạng mục | Dung lượng | Ổ |
|---|---:|---|
| Blur giải nén mới | ≈ 2.4 GB (+0.72 GB cũ) | D: |
| Nikon Train tải về | ≈ 0.8 GB | D: |
| Huawei Train bổ sung (nếu có) | ≈ 0.3 GB | E: (datasets) |
| Real-LOLBlur | chưa biết | D: |
| `llie_data_clean_v4.zip` | ≈ 5 GB | E: (hoặc D:) |
| **Tổng cần thêm** | ≈ 9 GB (chưa tính Real-LOLBlur) | ≤ 15.9 GB (E:) và 30.4 GB (D:) |

## 6. Tiêu chí chấp nhận (checklist cho bên implement)

- [ ] Val và test của v4 trùng v3 từng đường dẫn; `nikon_test_in_domain` được ghi.
- [ ] Số lượng theo miền khớp bảng §2 (blur ≥ 1,800; nikon ≈ 1,000; syn 900; huawei 940 hoặc nhiều hơn; lolv2 639).
- [ ] Không rò rỉ đường dẫn/chuỗi train–val–test; Real-LOLBlur không có trong manifest.
- [ ] Đã xem preview ảnh thật cho từng miền mới (blur, nikon, syn).
- [ ] Zip có `split_manifest.json` ở gốc và mọi đường dẫn manifest tồn tại trong zip.
- [ ] Báo cáo bước chọn dữ liệu ghi số thực tế và mọi sai khác so với kế hoạch (gap, số lượng, link tải).
