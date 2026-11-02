# Kế hoạch v4 — Mở rộng dữ liệu huấn luyện (tổng quan & phương án)

> Trạng thái: **kế hoạch, chưa implement** (việc implement giao bên khác).
> Tài liệu con: [ke_hoach_v4_du_lieu.md](file:///E:/MyProject/low-light-image-enhancement-cnn/presentation/report/ke_hoach_v4_du_lieu.md) (dữ liệu, script) và [ke_hoach_v4_notebook.md](file:///E:/MyProject/low-light-image-enhancement-cnn/presentation/report/ke_hoach_v4_notebook.md) (notebook Colab).
> Căn cứ: [Bao_Cao_Ket_Qua_v3_LLIE_DarkIR_Edge.md](file:///E:/MyProject/low-light-image-enhancement-cnn/presentation/report/Bao_Cao_Ket_Qua_v3_LLIE_DarkIR_Edge.md), `docs/ai/prepare_llie_data_v3.py`, notebook v3, paper DarkIR (mục 4.1), phiếu chấm `docs/research/PhieudanhgiaMonhoc.pdf`.

## 1. Vì sao v4 là "mở rộng dữ liệu" chứ không chỉ "thêm blur"

Phản hồi của bạn đúng: bản nháp trước chỉ tăng blur trong khi paper DarkIR (DarkIR-mt) còn dùng LOLv2-Synthetic, LSRW-Huawei, LSRW-Nikon. Đối chiếu từng nguồn:

| Nguồn | Có dùng trong v4? | Lý do |
|---|---|---|
| **LOL-Blur train** (150 chuỗi trong v3) | **Có** — pool 400 → ≈ 2,000 cặp | Tăng đa dạng blur/nhiễu; cảnh vẫn chỉ 150 nên không tăng thêm quá nhiều (xem tài liệu dữ liệu) |
| **LOLv2-Synthetic** (900 cặp train, 384×384, **đã có sẵn** `datasets/LOL-v2/Synthetic`) | **Có** — thêm một phần nhỏ (≈ 8% mỗi epoch) | Paper có dùng; là dữ liệu tăng sáng; nhưng là ảnh tổng hợp nên không để chiếm tỉ trọng lớn |
| **LSRW-Huawei** | **Có** — dùng hết pool hiện có (940 ảnh thay vì 561); nếu tải được đủ 2,450 thì dùng cả phần mới | Đây là dữ liệu ảnh thật cho tăng sáng, quan trọng nhất với demo |
| **LSRW-Nikon train** (3,150 cặp, chưa tải) | **Có** — một tập con ≈ 1,000 ảnh | Thêm cảm biến DSLR → demo ổn định hơn; **đánh đổi:** test Nikon (20 ảnh) không còn là "miền chưa thấy" (xem mục 3) |
| **Real-LOLBlur** (482 ảnh RealBlur-J, ảnh thật, **không có ảnh GT**) | **Không train**; dùng để **đánh giá/demo** | Không có ảnh đích nên không tính được L1/SSIM loss; dùng đánh giá không tham chiếu và minh họa định tính; tuyệt đối không đưa vào train để giữ nó là tập kiểm tra ngoài |

> [!NOTE]
> Phát hiện khi rà soát: thư mục `datasets/LSRW/Train/Huawei/low` hiện chỉ có **đúng 1,000 ảnh**, trong khi LSRW-Huawei gốc có 2,450 cặp train. Số 1,000 trùng với giới hạn trang của API liệt kê file Hugging Face, và `docs/ai/download_missing_datasets.py` không xử lý phân trang. Đây là **giả thuyết chưa kiểm chứng**; bước đầu tiên của bên implement là xác nhận (xem tài liệu dữ liệu §3).

## 2. Nguyên tắc thiết kế

1. **Bảo vệ khả năng tăng sáng (ưu tiên của đề tài):** mỗi epoch, lượng mẫu LOLv2-Real và LSRW-Huawei **giữ đúng bằng v3** (1,278 và 1,122 mẫu/epoch). Dữ liệu mới chỉ được **thêm vào**, không giành chỗ của dữ liệu tăng sáng cũ.
2. **Blur chiếm tỉ trọng lớn hơn v3** (25% → 33%), nhưng không làm giảm lượng luyện tăng sáng nhờ nguyên tắc 1.
3. **Test và val giữ nguyên 100% như v3** (190 và 150 ảnh) để so sánh v3 ↔ v4 trực tiếp.
4. **Đối chứng = kết quả v3 đã có**, không train lại.
5. Chỉ huấn luyện **Edge và U-Net**, 2 seed (42, 123) = 4 run. Bỏ no-FFT (không còn là câu hỏi của v4).

**Cái giá phải nêu rõ trong báo cáo:** do giữ nguyên lượng luyện cũ và thêm dữ liệu mới, tổng số bước tăng từ 28,000 lên **42,000 (+50%)**. Như vậy v4 thay đổi đồng thời *dữ liệu* và *số bước*. Không thể tách riêng hai yếu tố này trong ngân sách còn lại, nên kết luận chỉ được phát biểu là "gói mở rộng dữ liệu + thêm bước" so với v3. Phương án thuần khiết hơn (giữ 28,000 bước, chia lại tỉ trọng) bị loại vì buộc phải giảm lượng luyện tăng sáng — trái với ưu tiên của bạn.

## 3. Cấu hình huấn luyện v4 (so với v3)

| Mục | v3 | v4 |
|---|---|---|
| Epoch | 70 | 70 (cố định, không Early Stopping) |
| Bước/epoch | 400 | **600** (4,800 mẫu / batch 8) |
| Tổng bước | 28,000 | **42,000**; `CosineAnnealingLR` với `T_max = 42,000` |
| Patch / batch / lr / loss / EMA | 256 / 8 / 2e-4→1e-6 / L1+0.2·SSIM+0.1·color(+0.5·side) / 0.999 | **không đổi** |
| Mô hình | Edge, no-FFT, U-Net | **Edge, U-Net** |
| Checkpoint đánh giá | `final.pt` (EMA) | **không đổi** |

Số mẫu mỗi epoch theo miền (nguồn → bảng chi tiết ở tài liệu dữ liệu):

| Miền | Pool train | Mẫu/epoch | Tỉ trọng | v3 (mẫu/epoch) |
|---|---:|---:|---:|---:|
| lolv2 (Real) | 639 | 1,278 | 26.6% | 1,278 |
| huawei | 940 (hoặc nhiều hơn nếu tải đủ) | 1,122 | 23.4% | 1,122 |
| **blur** | ≈ 2,000 | **1,600** | **33.3%** | 800 |
| nikon | ≈ 1,000 | 400 | 8.3% | — |
| lolv2-syn | 900 | 400 | 8.3% | — |
| **Tổng** | | **4,800** | 100% | 3,200 |

**Thời gian (ước tính, cần đo lại bằng `probe_speed`):** Edge ≈ 0.3 s/bước ⇒ 42,000 bước ≈ 210 phút/run; U-Net ước ≈ 0.15 s/bước (**chưa đo**) ⇒ ≈ 105 phút/run. Mỗi seed (Edge + U-Net) ≈ 5.3 giờ + val/giải nén ≈ **6 giờ/phiên**, dưới trần 12 giờ. Chia 2 phiên theo seed như v3.

## 4. Ngân sách bộ nhớ Colab (L4, High RAM)

Theo bạn: VRAM ≈ 22.5 GB dùng được, RAM ≈ 25 GB, đĩa ≈ 150 GB.

- **VRAM:** huấn luyện patch 256, batch 8, AMP như v3 đã chạy; v4 không đổi kích thước patch/batch ⇒ **không phát sinh vấn đề VRAM** (đo `torch.cuda.max_memory_allocated` ở đầu run để ghi nhận).
- **RAM:** v3 nạp *toàn bộ* ảnh train ở độ phân giải gốc vào RAM. Với v4, nếu làm y như vậy: LOLv2-Real ≈ 0.9 GB, LOLv2-Syn ≈ 0.8 GB, Huawei 940 ảnh ≈ 3.9 GB, Nikon 1,000 ảnh ≈ 3.7 GB, Blur 2,000 cặp ≈ 8.6 GB ⇒ **≈ 18 GB**, cộng Python/PyTorch/DataLoader ≈ 4–6 GB ⇒ sát 25 GB, và không an toàn nếu Huawei mở rộng lên 2,450 ảnh. Quyết định: **nạp RAM cho các miền nhỏ, nạp lười (đọc từ đĩa Colab theo từng mẫu) cho miền blur** (xem tài liệu notebook §3).
- **Đĩa:** zip v4 ≈ 5 GB giải nén ra `/content/datasets` (≈ 5 GB) — dư rất nhiều so với 150 GB.

## 5. Quy tắc kết luận đăng ký trước

Kiểm định ghép cặp trên cùng ảnh test, ΔPSNR = v4 − v3 theo từng mô hình (trung bình 2 seed theo ảnh), bootstrap 95% CI (2,000 lần, seed 42), như v3.

1. **Điều kiện bảo vệ tăng sáng (bắt buộc):** nhóm **lolv2 + huawei (130 ảnh) pooled**: cận dưới CI của ΔPSNR ≥ **−0.3 dB**. Ngưỡng −0.3 dB là quyết định thiết kế, dựa trên độ lệch giữa seed 0.1–0.5 dB ở v3. Báo thêm `lum_bias`, `lum_abs_err`, `dark10_mae` (từ `eval_extra`) để theo dõi quá/thiếu sáng.
2. **Nikon (20 ảnh):** báo riêng và ghi nhãn "v3: miền chưa thấy / v4: miền đã thấy" — **không** dùng để so sánh v3 ↔ v4 như một phép thử đối chứng.
3. **Điểm cộng blur (40 ảnh):** ΔPSNR blur, `sharp_ratio`, LPIPS. Chỉ gọi là "cải thiện" khi CI loại 0 **và** điều kiện 1 đạt; ngược lại ghi "không phân biệt được".
4. **Real-LOLBlur (482 ảnh, không tham chiếu):** chỉ mô tả (không kiểm định mạnh): NIQE/NRQM/MUSIQ như Bảng 5 của paper, kèm độ chói trung bình đầu ra/đầu vào và lưới ảnh định tính, so v3 với v4 (và U-Net với Edge).
5. **Chọn trọng số cho demo:** version đạt điều kiện 1 và có PSNR lolv2+huawei cao hơn; nếu hòa, chọn v4 chỉ khi blur cải thiện có ý nghĩa, ngược lại giữ v3.
6. Không đổi test/thước đo/seed sau khi xem kết quả; nêu kết quả trung tính/âm trung thực.

## 6. Liên hệ phiếu chấm và demo

- Mô hình demo chính: **U-Net** (PSNR ngang Edge ở v3, nhanh hơn ≈ 2.4× trên L4); Edge là mô hình so sánh.
- Demo: nhập ảnh thiếu sáng → xuất ảnh sáng hơn; chuẩn bị mẫu cho cả LOLv2/LSRW (có ảnh đích) và Real-LOLBlur (không có ảnh đích, chỉ định tính).
- Lưu ý đã biết ở v3: LOLv2 bị quá sáng (≈ +12–13 trên thang 0–255), LSRW-Huawei bị thiếu sáng; nêu như hạn chế trong báo cáo và chọn ảnh demo đại diện.
- Phục vụ phiếu chấm: 4.1 mô tả dataset (10đ) có bảng dữ liệu phong phú, 4.4 thực nghiệm khác tài liệu tham khảo (5đ), 5 nhận xét (5đ). Dành thời gian cho demo, slide và báo cáo paper; **không** thêm version khác sau v4.

## 7. Trình tự

1. Kiểm tra giả thuyết cắt 1,000 ảnh Huawei và tải đủ LSRW (nếu có) — tài liệu dữ liệu §3.
2. Viết `docs/ai/prepare_llie_data_v4.py` → `select` → xem preview ảnh thật → `package` → upload Drive.
3. Dẫn xuất notebook v4 từ v3 (script `docs/ai/build_notebook_v4.py`) và xác minh (`verify_notebook_v4.py`).
4. Đăng ký trước quy tắc mục 5 vào `docs/ai/preregistration_v4.md`, commit **trước khi** chạy.
5. Chạy Colab 2 phiên theo seed; tải kết quả về `colab-results/`; chạy phân tích bổ sung và đánh giá Real-LOLBlur.
6. Viết đoạn so sánh v3 ↔ v4 trong báo cáo paper.

## 8. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| RAM Colab (≈ 25 GB) không đủ nếu nạp hết | Nạp lười miền blur; đo RAM trước khi train; hạ pool blur nếu cần |
| Dữ liệu mới làm lệch phong cách tăng sáng (đặc biệt Syn) | Giữ tỉ trọng Syn nhỏ (8%); điều kiện bảo vệ mục 5 |
| v3 ↔ v4 lẫn yếu tố số bước | Nêu rõ trong báo cáo (mục 2) |
| Test Nikon không còn "chưa thấy" | Báo cáo riêng, thay bằng Real-LOLBlur/DICM/LIME làm kiểm tra ngoài |
| Cần checkpoint v3 để đánh giá Real-LOLBlur | Xác nhận `final.pt` v3 còn trên Drive trước khi bắt đầu; nếu mất thì bỏ so sánh v3 trên tập này |
| Link tải Real-LOLBlur/LSRW không còn | Implement kiểm tra link sớm (tài liệu dữ liệu §3, §5) |
