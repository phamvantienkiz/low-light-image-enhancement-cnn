# Kế hoạch v3 (chỉ là hướng dẫn, chưa implement)

Mục tiêu: kiểm tra độ nhạy theo dữ liệu và loại confound độ dài huấn luyện của v2. **Test giữ nguyên (190 ảnh).** Mọi mô hình cùng dữ liệu, cùng lịch.

## 0. Quyết định đã chốt
- **Bỏ Early Stopping** (`use_early_stopping=False`).
- **70 epoch cố định cho mọi mô hình**, `epochs = 70`, `T_max = 70` (lr hạ hết về `lr_min`).
- `repeat`: xem mục 0b (mặc định PA-A = 2). Giữ 2 seed [42, 123], 3 họ mô hình (darkir_edge, darkir_edge_nofft, unet) → 6 run.
- Giữ val_interval = 5 (chỉ để theo dõi/ghi `best.pt` phụ), **checkpoint đánh giá chính thức = `final.pt` EMA** (đăng ký trước). `best.pt` chỉ là phân tích phụ.

## 0b. `repeat` là gì và có nên giảm xuống 1?

- `repeat` = số lần **mỗi ảnh gốc xuất hiện trong một epoch** (notebook: `len(dataset) = số ảnh × repeat`). Với `repeat=2`, mỗi ảnh được lấy **2 lần/epoch, mỗi lần cắt một patch 256×256 ngẫu nhiên khác nhau** (không nhân đôi file dữ liệu). Ảnh gốc lớn (ví dụ 640×1120) nên 1 lần cắt chỉ thấy một phần ảnh.
- `repeat=1` → mỗi epoch chỉ **một nửa số bước cập nhật**: v3 còn 1,600/8 = **200 bước/epoch**, 70 epoch = **14,000 bước**. Điều này **có ảnh hưởng**, không phải "gần như không": tổng số cập nhật thấp hơn v2 (Edge ~18.5k bước thực tế; U-Net s123 ~30k bước) → nguy cơ chưa hội tụ, đặc biệt U-Net (cần nhiều bước hơn). Nó làm 70 epoch của v3 *nhẹ hơn* hẳn so với 70 epoch cùng `repeat=2`.
- **Khuyến nghị:** nếu vẫn dùng `repeat=1`, giữ nguyên cho **mọi mô hình** và coi 70 epoch ≈ 14k bước; hoặc bù lại bằng cách đặt epoch = 140 với `repeat=1` (cùng 28k bước nhưng vẫn ~10 h — không tiết kiệm gì). Vì vậy `repeat=1` chỉ có ý nghĩa khi bạn chấp nhận **ít bước hơn**. Đây là đánh đổi thời gian ↔ độ hội tụ; phải ghi rõ trong report v3.
- Lựa chọn thay thế **không đổi số bước**: giữ `repeat=2` nhưng chia nhiều phiên Colab (mục 0c).

## 0c. Colab: mất kết nối có train tiếp được không?

**Có thể tiếp tục, vì notebook đã có cơ chế resume** (`last.pt` lưu lên Google Drive mỗi lần val, tức mỗi 5 epoch, kèm optimizer/scheduler/EMA/history). Khi mất kết nối hoặc hết giờ:
1. Máy ảo bị xóa (`/content` mất, kể cả dữ liệu đã giải nén), nhưng **Drive còn nguyên** checkpoint.
2. Kết nối lại → chạy lại từ đầu các cell cần thiết (mount Drive, cài thư viện, giải nén zip về `/content/datasets`, định nghĩa mô hình, khởi tạo `CFG`) → chạy lại cell train. Run đã có `final.pt` sẽ `[SKIP]`; run dở sẽ `[RESUME]` từ `last.pt`.
3. **Mất tối đa 5 epoch** (từ lần lưu `last.pt` gần nhất) — nên ghi nhận; trạng thái RNG không khôi phục → chạy resume không tái lập bit-by-bit (đã xảy ra ở v2, chấp nhận được, nên nêu trong report).

**Về giới hạn thời gian (nói thẳng những gì không biết):** Google *không công bố* quota cụ thể. Mô tả chính thức: phiên chạy có thể tối đa ~12 h (miễn phí thường ngắn hơn, tùy tải/lịch sử dùng), và **giới hạn sử dụng GPU thay đổi động**, không có "mốc cố định sau bao lâu thì hồi lại". Hệ quả thực tế: sau khi hết phiên, thường *có thể* kết nối lại ngay, nhưng **không đảm bảo** có GPU L4 (có thể bị chuyển GPU khác hoặc bị chặn GPU tạm thời nếu dùng quá nhiều). Không nên giả định "reconnect là train tiếp ngay được". Dự phòng:
- Chia v3 thành **2 phiên theo seed** (mục 0d): mỗi phiên ~6 h, dư rất nhiều so với trần 12 h.
- Sắp thứ tự cell để hoàn tất từng run trước (đã có `final.pt` ngay khi xong) và luôn kiểm tra `last.pt` trên Drive.
- Nếu GPU khác L4 ở lần kết nối lại: ghi rõ trong report (ảnh hưởng latency/efficiency, không ảnh hưởng PSNR) và **đo lại latency trên cùng một loại GPU** cho cả 3 mô hình.
- Có thể dùng `/schedule` hoặc tự canh để bấm chạy lại khi phiên kết thúc; không nên dựa vào chạy qua đêm không giám sát.

> [!WARNING]
> **Ngân sách thời gian chưa chắc đủ 12 h nếu chạy một phiên** với `repeat=2`: v2 Edge ≈ 0.3 s/bước (1.68 phút/337 bước). v3: 1,600×2/8 = 400 bước/epoch → 70 epoch = 28,000 bước ≈ **140 phút/run Edge-họ × 4 = 9.3 h**; U-Net (ước ~0.15 s/bước, **chưa đo**) ≈ 70 phút × 2 = 2.3 h → **≈ 11.6 h nếu chạy một lượt** (quá sát → chia theo seed, xem 0d), chưa tính val 150 ảnh/5 epoch. Phải đo thật (mục 5, bước 1) trước khi chọn phương án:
> - **PA-A (khuyến nghị, đã chọn):** `repeat=2`, 70 epoch, **chia 2 phiên theo seed** (mục 0d), resume khi cần.
> - **PA-B:** `repeat=1` (14k bước/run, ~6 h tổng) — nhưng ít cập nhật hơn v2; xem mục 0b, phải nêu rõ.
> - **PA-C:** `repeat=2`, no-FFT chỉ 1 seed — chỉ khi bất khả kháng, nêu rõ.

## 0d. Cách chạy 2 phiên (chia theo seed, 70 epoch)

**Vì sao chia theo seed:** mỗi seed gồm đủ 3 mô hình (Edge, no-FFT, U-Net) → nếu phiên 2 hỏng thì phiên 1 vẫn là một bộ so sánh hoàn chỉnh. Ước lượng (repeat=2, 400 bước/epoch, 70 epoch = 28,000 bước; Edge ≈ 0.3 s/bước ≈ 140 phút/run; U-Net ước ≈ 70 phút/run, chưa đo): **mỗi seed ≈ 2×140 + 70 = 350 phút ≈ 5.8 h** (+ val, giải nén) → **~6–6.5 h/phiên**, còn dư ~5 h an toàn.

**Phiên = một lần kết nối Colab độc lập; không cần "train xong rồi disconnect rồi chạy lại ngay".** Quy trình:

*Phiên 1 (seed 42):*
1. Mở notebook v3, kết nối L4. Chạy các cell chuẩn bị (mount Drive, cài thư viện, giải nén zip v3, định nghĩa mô hình/`CFG`...).
2. Trong cell `CFG`, đặt `darkir_edge_seeds=[42]`, `darkir_edge_nofft_seeds=[42]`, `unet_seeds=[42]`.
3. Chạy các cell train (Edge → no-FFT → U-Net). Mỗi run xong sẽ có `final.pt` trên Drive.
4. **Chưa chạy các cell đánh giá/kiểm định** (cần cả 2 seed). Khi cả 3 run xong, đóng phiên: *Runtime → Disconnect and delete runtime* (Drive vẫn giữ checkpoint). Kiểm tra trên Drive có đủ 3 thư mục `*_seed42/final.pt`.

*Phiên 2 (seed 123), có thể ngay sau hoặc hôm sau:*
1. Kết nối lại (nên L4; GPU khác vẫn train được, chỉ ảnh hưởng thời gian). Chạy lại các cell chuẩn bị như trên (VM mới nên phải giải nén lại dữ liệu).
2. Đặt 3 danh sách seed = `[123]`, chạy cell train. Run seed 42 không bị chạm tới.
3. Khi xong, **đặt lại cả 3 danh sách seed = `[42, 123]`** rồi chạy các cell đánh giá/kiểm định/efficiency (find_checkpoint đọc cả 2 seed từ Drive). Đo latency trong phiên này (cả 3 mô hình trên cùng một GPU) để so sánh công bằng.

*Nếu giữa chừng mất kết nối (chung cho cả 2 phiên):* kết nối lại → chạy lại cell chuẩn bị → chạy lại cell train với **đúng seed của phiên đó**: run xong sẽ `[SKIP]`, run dở `[RESUME]` từ `last.pt` (mất ≤ 5 epoch). Không tiến sang seed khác khi seed hiện tại chưa đủ 3 `final.pt`.

*Lưu ý quan trọng khi cập nhật notebook (kế hoạch, bạn/bên triển khai thực hiện):*
- Vì sửa danh sách seed ngay trong `CFG`, **`run_config.json` ghi cấu hình khác nhau ở 2 phiên** — khi đánh giá cuối đảm bảo `CFG` có đủ cả 2 seed và ghi chú trong report: "huấn luyện chia 2 phiên theo seed".
- Cả 2 phiên dùng cùng manifest/zip/hyper-parameter, chỉ khác seed; không đổi gì khác giữa 2 phiên.
- Khi kết thúc phiên 1, **không** dùng đường cong/`table_main` tạm (thiếu seed) để rút kết luận.
- Nếu muốn có kết quả sớm: sau phiên 1 chỉ xem lướt `history.csv`; phân tích chính thức chờ đủ 2 seed.

## 1. Dữ liệu v3
| Tập | lolv2 | huawei | blur | Tổng |
|---|---:|---:|---:|---:|
| Train | 639 | 561 | **400** (v2: 150) | 1,600 |
| Val | 50 | 50 | **50** (v2: 10) | 150 |
| Test | 100 | 30 (+nikon 20) | 40 | 190 (giữ nguyên) |

Nguyên tắc chọn thêm 250 train + 40 val blur từ LOL-Blur train (170 chuỗi):
1. **Giữ nguyên** 150 train + 10 val + 40 test đã chọn ở v2; chỉ *thêm*.
2. Tách chuỗi theo tập: chuỗi val không được trùng chuỗi train (cách ly theo chuỗi như v2). Test dùng 30 chuỗi riêng của `test.zip` → không thể trùng.
3. Mỗi chuỗi có 60 frame gần giống nhau → lấy tối đa vài frame/chuỗi, cách nhau ≥ 10 frame để tránh ảnh gần trùng lặp (170 chuỗi → khoảng 2–3 frame/chuỗi cho train 400).
4. Lấy seed cố định (42), lưu `split_manifest` mới có SHA-256, bắt buộc kiểm tra: không trùng đường dẫn/chuỗi giữa train/val/test.
5. Dung lượng thêm ~290 cặp × ~2 MB ≈ 0.6 GB (ổ E: gần đầy → giải nén/chọn ra `D:\Temp_LOL-Blur\` như v2; chỉ ghi `datasets/llie_data_clean_v3.zip` khi E: còn chỗ). **Việc đóng gói do bạn thực hiện; phần dưới là kế hoạch từng bước.**
6. Dùng lại `docs/ai/prepare_llie_data_v2.py` làm khuôn (tạo `prepare_llie_data_v3.py`): các bước `select` (chọn + kiểm tra ảnh thực + preview) rồi `package`.

### 1b. Kế hoạch đóng gói dữ liệu v3 (bạn tự thực hiện)
1. Kiểm tra dung lượng trống: cần ≈ (zip v2 1.26 GB + 0.6 GB) cho zip v3 + ~0.6 GB tạm; nếu E: không đủ, xuất zip sang `D:\` rồi upload trực tiếp từ đó lên Drive (không bắt buộc đặt trong `datasets/`; nếu bắt buộc, báo lại).
2. Tạo `docs/ai/prepare_llie_data_v3.py` từ `prepare_llie_data_v2.py`: giữ lại lệnh `select` và `package`; `select` đọc lại 3 zip LOL-Blur ở `D:\Temp_LOL-Blur\`, **cộng thêm** 250 train + 40 val blur (không động đến 150/10/40 cũ).
3. Chạy `uv run python docs/ai/prepare_llie_data_v3.py select` → kiểm tra: số lượng (400/50/40), không trùng chuỗi giữa các tập, độ phân giải, độ sáng/độ nét tăng (như v2), xem preview ảnh thật.
4. Chạy `... package` → kiểm tra SHA-256 các cặp v1 (1,450) + blur mới; `split_manifest.json` ở gốc zip; số ảnh: v2 có 4,157 ảnh cho 1,650 cặp (không phải đúng 2 ảnh/cặp), nên **không suy số ảnh bằng công thức**; thay vào đó đối chiếu số cặp theo manifest (1,940 cặp = 1,600 train + 150 val + 190 test) và so sánh cấu trúc với zip v2.
5. Upload `llie_data_clean_v3.zip` lên `/content/drive/MyDrive/llie_project_v3/` (cùng tên thư mục mà notebook v3 sẽ trỏ tới).
6. Chỉ xóa file trung gian trên E: khi bạn đã xác nhận (không tự xóa).

## 2. Thay đổi notebook (`LLIE_DarkIR_Edge_Colab_v3.ipynb`, dẫn xuất từ v2, không viết lại)
1. `EXP_VERSION='v3'`, thư mục Drive `MyDrive/llie_project_v3/` (zip, manifest, experiments/v3).
2. `CFG`: `epochs=70`, `use_early_stopping=False`, `n_train`/`n_val` cập nhật theo manifest.
3. Cell train: giữ cơ chế `last.pt` resume; **vô hiệu hóa khối [AUTO-CONTINUE]** liên quan `early_stop_min_epochs` (không còn ý nghĩa). Đảm bảo run chỉ dừng khi đủ 70 epoch.
4. Val: 150 ảnh (macro trên 3 miền, vẫn EMA); nhớ kiểm tra thời gian một lần val.
5. Cell đánh giá: đổi checkpoint mặc định sang `final.pt` (`find_checkpoint(..., ckpt_name='final.pt')` và các nơi gọi `nn_fn(..., ckpt_name=...)`); **giữ thêm** một lượt đánh giá `best.pt` xuất ra file riêng (`table_main_best.csv`) làm phân tích phụ.
6. Xuất thêm checkpoint/history đầy đủ; log vào Drive.
7. Xây dựng bằng script `docs/ai/build_notebook_v3.py` từ v2 và kiểm bằng bản `verify_notebook_v3.py` (ast.parse mọi cell, chạy `LLIE_TEST=1`, kiểm tra không còn dừng sớm, kiểm tra kích thước tập).

## 3. Quy tắc kết luận đăng ký trước (viết vào docs/ trước khi chạy)
- Cùng quy tắc v2: kết luận theo 95% CI bootstrap của ΔPSNR ghép cặp (190 ảnh, trung bình 2 seed) cho Edge vs U-Net và Edge vs no-FFT; kiểm định theo từng miền là phụ.
- Checkpoint chính thức: `final.pt` EMA ở epoch 70.
- Nếu U-Net ≥ Edge ở v3: giữ kết luận "U-Net mạnh hơn" (mạnh hơn nữa vì đã loại confound độ dài).
- Nếu Edge ≥ U-Net: chỉ báo cáo khi CI loại 0; ngược lại ghi "ngang nhau".
- Không đổi test, không đổi thước đo, không chọn lại seed sau khi xem kết quả.

## 4. Script đo bổ sung (**đã xác nhận cần làm**; chạy sau khi có checkpoint; dùng được cho cả v2 và v3)
Mục đích: biến cảm quan "Edge tông tự nhiên hơn" và "chưa khử mờ" thành số liệu. Chỉ suy luận (inference), không train lại.

`docs/ai/eval_extra.py` (hoặc cell mới cuối notebook), cho mỗi (mô hình, seed, ảnh test):
1. **Sai lệch độ sáng có dấu:** `mean_Y(out) − mean_Y(GT)` (Y = độ chói). Giá trị dương = bị quá sáng; âm = thiếu sáng. Báo cáo trung bình & độ lệch chuẩn theo miền.
2. **Khoảng cách histogram độ chói** (ví dụ Wasserstein/EMD 1D) giữa out và GT → đo "vùng sáng/tối có đúng phân bố".
3. **Sai lệch màu CIEDE2000 (ΔE₀₀)** trung bình theo ảnh.
4. **Độ nét:** phương sai Laplacian của out so với GT và input (tỉ lệ out/GT) → chứng minh khử mờ hay chưa; tính riêng miền blur.
5. **Phân vị sáng theo vùng:** sai lệch trung bình trên 10% pixel tối nhất / 10% sáng nhất của GT (kiểm tra trực tiếp "vùng cần tối thì tối, vùng cần sáng thì sáng").
6. Xuất `results_extra.csv` + kiểm định ghép cặp bootstrap (cùng cách như `paired_tests.csv`) cho Edge vs U-Net và Edge vs no-FFT.

Khảo sát mù (không cần mô hình mới):
- `docs/ai/make_blind_survey.py`: chọn 20–30 ảnh test (phân tầng theo miền), cho mỗi ảnh ghép Input/GT + 3 đầu ra (U-Net, Edge, no-FFT) với **nhãn A/B/C xáo trộn ngẫu nhiên**, lưu khóa giải mã riêng (`survey_key.json`); người chấm chọn ảnh "tự nhiên nhất" và "giống GT nhất"; tính tỉ lệ ưu tiên + khoảng tin cậy nhị thức. Cần ≥3 người chấm để có ý nghĩa; nêu rõ quy mô nhỏ.

## 5. Trình tự thực hiện
1. **Đo trước:** chạy 1 epoch Edge và U-Net trên dữ liệu v3 để lấy giây/bước thật và thời gian val 150 ảnh → chọn PA-A/B/C.
2. Viết `prepare_llie_data_v3.py` → `select` → xem preview ảnh thật + kiểm tra chồng chéo → `package` → upload Drive.
3. Dẫn xuất notebook v3 + verify (mục 2).
4. Đăng ký trước quy tắc kết luận (mục 3) vào `docs/` và commit **trước khi** chạy.
5. Chạy Colab (PA đã chọn; phiên 1: Edge-họ, phiên 2: U-Net). Không bấm dừng/đổi giữa chừng; nếu mất kết nối, chạy lại cell train để resume.
6. Tải kết quả về `colab-results/`, chạy `eval_extra.py`, (tùy chọn) khảo sát mù.
7. Viết report v3 trong `presentation/report/` theo cùng khuôn report v2, đối chiếu quy tắc đăng ký; cập nhật slide/paper.

## 6. Rủi ro
| Rủi ro | Giảm thiểu |
|---|---|
| Quá 12 h Colab | Đo trước; chia 2 phiên; PA-B/C |
| Hết dung lượng ổ E: | Chỉ đóng gói zip khi đủ chỗ; xin xác nhận trước khi xóa |
| Ảnh blur gần trùng giữa train/val | Cách ly chuỗi + gap ≥ 10 frame |
| Có người cho rằng "chỉnh đến khi thắng" | Đăng ký trước, giữ test, cùng quy tắc cho mọi mô hình |
| 70 epoch vẫn chưa đủ cho U-Net (v2: U-Net s123 còn cải thiện đến epoch 90) | Ghi nhận hạn chế; không kéo dài riêng cho một mô hình |


