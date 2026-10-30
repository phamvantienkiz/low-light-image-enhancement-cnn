# Khung slide + khung bài báo khoa học từ kết quả v2

> Nguồn số liệu duy nhất: [report v2](file:///E:/MyProject/low-light-image-enhancement-cnn/presentation/report/Bao_Cao_Ket_Qua_v2_LLIE_DarkIR_Edge.md) và thư mục `colab-results/llie_project_v2-20261029/`. Mọi số dưới đây đã khớp report đó. Chỗ nào ghi **[TODO]** là việc bạn phải tự làm/kiểm tra (ví dụ nhận xét ảnh, xác minh trích dẫn).

**Khung câu chuyện (một câu):** *"Với ngân sách ~1.9 M tham số và ~1.3k ảnh huấn luyện, một U-Net gọn là baseline rất mạnh; biến thể lấy cảm hứng DarkIR (DarkIR-Edge) chưa thắng về số liệu, nhưng cho tông sáng–tối tự nhiên hơn theo quan sát định tính, và việc khử mờ cần nhiều dữ liệu blur hơn."*

Thông điệp phải **nhất quán** giữa slide, paper, Q&A. Các câu CẤM: "Edge tốt hơn U-Net", "FFT cải thiện độ sáng/khử mờ", "thời gian thực trên thiết bị biên", "edge-friendly".

---

# PHẦN A — KHUNG SLIDE (theo đúng yêu cầu của giảng viên)

**Bộ sản phẩm nộp:** (1) Báo cáo chi tiết dạng paper khoa học (Phần B), (2) Source code (repo + notebook), (3) Slide trình bày (Phần A này).

**Thứ tự bắt buộc theo yêu cầu giảng viên → ánh xạ vào slide:**

| Yêu cầu giảng viên | Slide |
|---|---|
| Thành viên + mức độ hoàn thành | S2 |
| Lý do / tầm quan trọng của bài toán | S3 |
| Phát biểu bài toán (Input, Output, Ví dụ) | S4–S5 |
| Phương pháp giải quyết | S6–S10 |
| Kết quả thực nghiệm: Dataset / Độ đo / Môi trường / Bảng kết quả / Minh họa | S11–S20 |
| Đánh giá, nhận xét: ưu điểm & hạn chế | S21–S23 |
| Tài liệu tham khảo | S24 |

Tổng 24 slide chính + 4 slide dự phòng (~18–20 phút). Mỗi slide: **Nội dung → Hình/bảng nguồn → Lời thuyết trình.**

### S1. Trang bìa
- Tên đề tài (đúng tên đăng ký), phụ đề: *So sánh U-Net gọn và biến thể lấy cảm hứng DarkIR trên 4 miền dữ liệu*. Môn học, GVHD, ngày. **[TODO]**

### S2. Thông tin thành viên & đánh giá mức độ hoàn thành
- Bảng: Họ tên | MSSV | Công việc phụ trách | % hoàn thành. **[TODO] điền; ghi trung thực.**
- Gợi ý bảng "mức độ hoàn thành theo hạng mục" (dựa trên thực tế dự án):

| Hạng mục | Trạng thái |
|---|---|
| Chuẩn bị dữ liệu 4 miền (1,650 cặp, split cách ly chuỗi) | Hoàn thành |
| Cài đặt U-Net, DarkIR-Edge, no-FFT, baseline truyền thống | Hoàn thành |
| Huấn luyện 6 run (3 mô hình × 2 seed) + đánh giá + kiểm định | Hoàn thành |
| Báo cáo paper + slide | **[TODO]** |
| Thí nghiệm bổ sung v3 (nhiều blur hơn, 70 epoch cố định) | Đang thực hiện / dự kiến **[TODO]** |

### S3. Lý do và tầm quan trọng của bài toán
- Ảnh thiếu sáng phổ biến (chụp đêm, trong nhà, camera giám sát, điện thoại): tối, nhiễu, mờ do rung/phơi sáng lâu, màu lệch.
- Ảnh hưởng đến các tác vụ phía sau (nhận dạng, phát hiện) và trải nghiệm người dùng.
- Ràng buộc thực tế: cần mô hình nhỏ/nhanh → nhu cầu CNN nhẹ.
- Hình: 1–2 ảnh Input vs GT. Nguồn: `visual_samples/*_input.png`, `*_gt.png`.

### S4. Phát biểu bài toán
- **Input:** ảnh RGB thiếu sáng (có thể kèm nhiễu và nhòe chuyển động) `I_low ∈ R^{H×W×3}`.
- **Output:** ảnh `Î` có độ sáng/màu/chi tiết gần ảnh chuẩn sáng bình thường `I_gt`, cùng kích thước.
- **Mục tiêu:** học ánh xạ `f_θ: I_low → Î` sao cho `Î ≈ I_gt` (cực đại PSNR/SSIM, cực tiểu LPIPS) với số tham số ~1.9 M.
- Nêu RQ1–RQ4 (U-Net vs Edge; FFT; chi phí; khử mờ).

### S5. Ví dụ minh họa bài toán
- Một lưới 3 cột: Input | Kết quả mô hình | Ground truth (cho 1 ảnh không blur và 1 ảnh blur). Nguồn: `visual_samples/`, `lolblur_visual_samples/`. **[TODO] chọn ảnh.**

### S6. Tổng quan phương pháp & công trình liên quan
- Nhóm truyền thống: Gamma, CLAHE, MSRCR. Nhóm CNN: U-Net. Nhóm hiện đại: DarkIR (CVPR 2025) — paper cảm hứng chính; LEDNet/LOL-Blur cho dữ liệu low-light + blur.
- Sơ đồ pipeline chung: dữ liệu → huấn luyện (patch 256, EMA) → đánh giá (PSNR/SSIM/LPIPS) → kiểm định ghép cặp.

### S7. Mô hình 1 — Standard U-Net (baseline CNN đơn giản)
- Encoder–decoder + skip connection, base=32, 1,928,483 tham số, 18.45 GFLOPs @256².
- Vai trò: baseline mạnh, đúng tên đề tài "CNN đơn giản/U-Net". Hình: sơ đồ khối **[TODO]**.

### S8. Mô hình 2 — DarkIR-Edge (lấy cảm hứng DarkIR)
- width=24; khối không gian (depthwise), nhánh miền tần số FFT/FreMLP, gating, side-loss; 1,924,038 tham số, 8.75 GFLOPs.
- Nêu rõ: không phải bản gốc DarkIR; DarkIR-m chính thức (3.32 M) chỉ dùng làm tham chiếu trần.

### S9. Mô hình 3 — Ablation no-FFT & hàm mất mát
- no-FFT: tắt FFT/FreMLP, giữ mọi thứ khác (1,495,422 tham số, −22%).
- Loss: L1 + 0.2·SSIM + 0.1·color (+0.5·side-loss cho họ Edge). **[TODO] công thức**.

### S10. Quy trình huấn luyện & cách kết luận
- AdamW, lr 2e-4→1e-6 cosine, batch 8, patch 256, EMA 0.999, 2 seed (42/123); early stopping val macro (patience 15, sau epoch 40); đánh giá bằng `best.pt` EMA.
- Quy tắc pre-registered: kết luận theo 95% CI bootstrap của ΔPSNR ghép cặp.

### S11. Kết quả thực nghiệm — Dataset
- Bảng: lolv2 (639/50/100), huawei (561/50/30), nikon (0/0/20, chỉ test), blur/LOL-Blur (150/10/40); tổng 1,350/110/190 (1,650 cặp).
- Mô tả từng nguồn (LOLv2-Real, LSRW-Huawei/Nikon, LOL-Blur) và cách chọn blur theo chuỗi video. Nêu thẳng hạn chế: blur 11% train, val blur 10 ảnh.

### S12. Độ đo đánh giá
- PSNR (dB, ↑), SSIM (↑), LPIPS (↓). Giải thích ngắn: PSNR sai số điểm ảnh, SSIM cấu trúc, LPIPS tương đồng cảm quan.
- Pooled (190 ảnh) và macro (4 miền); ΔPSNR ghép cặp + bootstrap CI + t-test/Wilcoxon.

### S13. Môi trường thực nghiệm
- Google Colab, GPU NVIDIA L4; PyTorch; 600×400 cho đo latency FP32/FP16; seed cố định; thư viện/phiên bản **[TODO] lấy từ notebook**. Dữ liệu xử lý bằng `uv`/`.venv` cục bộ.

### S14. Bảng kết quả — PSNR
- Bảng report §2.1: Input 9.75; Gamma 16.90; CLAHE 11.59; MSRCR 12.12; Edge no-FFT 20.76; Edge 20.74; **U-Net 21.36** (pooled); DarkIR-m ref 20.96.
- Điểm nhấn: học sâu vượt xa truyền thống (≥ +3.8 dB so với Gamma).

### S15. Bảng kết quả — SSIM/LPIPS và theo miền
- U-Net SSIM 0.749, LPIPS 0.277; Edge 0.728/0.368; no-FFT 0.730/0.353. Bảng theo miền (blur/huawei/lolv2/nikon).

### S16. Bảng kết quả — kiểm định thống kê
- Forest plot ΔPSNR: Edge−U-Net pooled −0.615 [−0.891, −0.335]; lolv2 −1.050; blur −0.442; huawei/nikon CI chứa 0.
- Edge−no-FFT pooled −0.014 [−0.231, +0.199]. **[TODO]** vẽ từ `paired_tests.csv`.

### S17. Bảng kết quả — chi phí tính toán
- Edge 8.75 G / 31.5 ms / 308 MB; no-FFT 7.50 G / 27.1 ms / 293 MB; U-Net 18.45 G / **13.0 ms** / 342 MB (L4, 600×400, FP32). FLOPs −53% nhưng latency ×2.4.

### S18. Minh họa kết quả — chỉ tăng sáng
- Lưới: Input | CLAHE | U-Net | no-FFT | Edge | GT (`visual_samples/`, `qualitative.png`).
- Quan sát tác giả (1 người chấm, không mù): Edge có tông sáng–tối hài hòa, màu chân thực hơn; PSNR/LPIPS vẫn nghiêng về U-Net. **[TODO]** chọn ảnh.

### S19. Minh họa kết quả — có blur
- Lưới: Input | U-Net | no-FFT | Edge | DarkIR-m | GT (`qualitative_lolblur_5samples.png`, `qualitative_with_darkir_m.png`). Cả 3 mô hình còn mờ/nhiễu; DarkIR-m 27.37 dB nhưng huấn luyện trên toàn bộ LOL-Blur (không công bằng).

### S20. Minh họa — đường cong huấn luyện & chẩn đoán độ sáng
- `curves.png`; gap độ sáng LOLv2 ~5.7–6.0 dB (sai số chủ yếu là sai sáng toàn cục); Edge ≈ no-FFT.

### S21. Đánh giá — Ưu điểm
1. CNN nhẹ ~1.9 M vượt xa phương pháp truyền thống (+3.8 dB trở lên).
2. U-Net gọn: chất lượng tốt nhất trong thí nghiệm, nhanh nhất (13.0 ms), đơn giản.
3. DarkIR-Edge: tông sáng–tối tự nhiên theo quan sát định tính; no-FFT giảm 22% tham số/~14% latency mà không mất chất lượng đo được.
4. Quy trình chặt chẽ: split cách ly chuỗi, 2 seed, kiểm định ghép cặp pre-registered, công bố cả kết quả âm.

### S22. Đánh giá — Hạn chế
1. Edge không thắng U-Net (ΔPSNR −0.62 dB, LPIPS kém hơn), chậm ×2.4; FFT chưa có bằng chứng giúp ích.
2. Khử mờ chưa đạt (còn mờ/nhiễu).
3. Dữ liệu nhỏ: blur 150 train/10 val/40 test (30 chuỗi); nikon 20 ảnh; chỉ 2 seed.
4. Độ dài huấn luyện không đồng đều (55–90 epoch; `best.pt` ở epoch 25–90) → có thể thiên về U-Net.
5. Latency chỉ đo trên L4; quan sát định tính chủ quan.

### S23. Nhận xét & giải thích kết quả
- Giả thuyết (chưa kiểm chứng): U-Net tính toán gấp ~2.1×/cùng tham số; dữ liệu/epoch ít; PSNR thưởng sáng toàn cục; DarkIR cần dữ liệu lớn (DarkIR-m +4 dB trên blur).
- **Kết luận** (5 gạch như report §9) và **hướng phát triển**: v3 (blur 400 train, val 50:50:50, 70 epoch cố định), chỉ số bổ sung (độ sáng có dấu, ΔE, độ nét), khảo sát mù. **[TODO]** nếu v3 xong, thêm slide kết quả.

### S24. Tài liệu tham khảo
- DarkIR (CVPR 2025); LEDNet/LOL-Blur (ECCV 2022); LOLv2 (TIP 2021); LSRW; U-Net (MICCAI 2015); LPIPS (CVPR 2018); CLAHE; MSRCR. **[TODO] xác minh trích dẫn đầy đủ (Phần B).**

### Slide dự phòng
- B1: Bảng đầy đủ từng seed (report §2.3). B2: bảng epoch dừng và epoch `best.pt`. B3: chi tiết t-test/Wilcoxon/% ảnh thắng. B4: `run_config.json`.

### Bộ câu hỏi dự kiến (Q&A)
1. **Sao mô hình đề xuất thua U-Net?** — Đo thực tế (CI loại 0); giả thuyết ở S23; v3 kiểm tra độ nhạy theo dữ liệu.
2. **Bỏ FFT thì sao?** — Không đổi chất lượng đo được (−0.014 dB), −22% tham số, ~−14% latency.
3. **Edge có real-time/edge-friendly không?** — ≥30 FPS trên L4 (GPU server), chưa đo thiết bị biên; không tuyên bố.
4. **Val nhỏ, early stopping có ảnh hưởng?** — Val blur 10 ảnh → nhiễu; epoch 55–90 khác nhau là hạn chế đã nêu; v3 khắc phục.
5. **Sao nikon chỉ để test?** — Đo tổng quát hóa sang miền chưa thấy; mọi mô hình ~17 dB.
6. **Sao không so SOTA khác?** — Phạm vi đề tài là CNN nhẹ; DarkIR-m làm trần tham chiếu.
7. **Chất lượng thị giác?** — Quan sát chủ quan 1 người; LPIPS nghiêng về U-Net; có kế hoạch đo khách quan.
8. **Đánh giá bằng best hay final?** — `best.pt` EMA theo val macro (epoch Edge 30/55, no-FFT 25/80, U-Net 55/90).

---

# PHẦN B — KHUNG BÀI BÁO KHOA HỌC (IMRaD)

**Tiêu đề gợi ý (EN):** *A Controlled Comparison of a Compact U-Net and a DarkIR-Inspired Lightweight CNN for Low-Light Image Enhancement with Blur*
**Tiêu đề (VI):** *So sánh có kiểm soát giữa U-Net gọn và CNN nhẹ lấy cảm hứng DarkIR cho tăng cường ảnh thiếu sáng có mờ nhòe*

### Abstract (~180–220 từ)
Cấu trúc: bối cảnh (1–2 câu) → khoảng trống (thiếu so sánh có kiểm soát CNN nhẹ vs biến thể DarkIR ở cùng số tham số) → phương pháp (3 mô hình ~1.9 M, 4 miền, 1,350/110/190, 2 seed, kiểm định ghép cặp bootstrap) → kết quả (U-Net 21.36 dB vs Edge 20.74 dB pooled, ΔPSNR −0.615 [−0.891, −0.335]; no-FFT Δ −0.014 [−0.231, +0.199]; latency 13.0 vs 31.5 ms) → kết luận (U-Net gọn là baseline mạnh; FFT chưa có bằng chứng; khử mờ cần nhiều dữ liệu hơn).
Từ khóa: low-light image enhancement; lightweight CNN; U-Net; DarkIR; frequency-domain; ablation.

### 1. Giới thiệu
- Bối cảnh LLIE, động cơ mô hình nhẹ.
- Khoảng trống & câu hỏi RQ1–RQ4 (như S3).
- Đóng góp (3 gạch): (i) bộ dữ liệu hợp nhất 4 miền, 1,650 cặp, split cách ly chuỗi; (ii) so sánh có kiểm soát U-Net / DarkIR-Edge / no-FFT với kiểm định ghép cặp pre-registered; (iii) báo cáo trung thực cả kết quả âm và phân tích mối đe dọa tính hợp lệ.

### 2. Công trình liên quan
2.1 LLIE truyền thống (Gamma, CLAHE, Retinex/MSRCR). 2.2 CNN/U-Net cho LLIE. 2.3 DarkIR và xử lý miền tần số. 2.4 Dữ liệu LOL, LSRW, LOL-Blur.

### 3. Phương pháp
3.1 Standard U-Net (base 32, skip, 1,928,483 tham số).
3.2 DarkIR-Edge (width 24; khối, FFT/FreMLP, gating, side-loss; 1,924,038 tham số) — nêu rõ khác bản gốc DarkIR.
3.3 Biến thể no-FFT (1,495,422 tham số).
3.4 Hàm mất mát: L1 + 0.2·SSIM + 0.1·color (+0.5·side). Công thức viết đầy đủ. **[TODO]** lấy chính xác từ notebook cell loss.
3.5 Tham chiếu DarkIR-m (checkpoint chính thức Cidaut/DarkIR) làm trần, không huấn luyện lại.

### 4. Thiết lập thực nghiệm
4.1 Dữ liệu và chia tập (bảng); cách chọn LOL-Blur (`low_blur_noise` → `high_sharp_scaled`), cách ly theo chuỗi; SHA-256 manifest `split_manifest.json`.
4.2 Huấn luyện (siêu tham số trong `run_config.json`; early stopping; EMA; seed).
4.3 Đánh giá: PSNR, SSIM, LPIPS (backbone nào **[TODO]**: kiểm tra notebook), pooled vs macro; checkpoint `best.pt` EMA.
4.4 Thống kê: ΔPSNR ghép cặp trên trung bình 2 seed; bootstrap 95% CI (số lần lặp **[TODO]** kiểm tra notebook), t-test, Wilcoxon; quy tắc pre-registered; lưu ý đa kiểm định.
4.5 Phần cứng: NVIDIA L4; đo latency 600×400 FP32/FP16.

### 5. Kết quả
5.1 So sánh tổng thể (Bảng 2: PSNR; Bảng 3: SSIM/LPIPS).
5.2 Kiểm định (Bảng 4; Hình forest plot).
5.3 Ablation FFT (Bảng 5).
5.4 Hiệu năng (Bảng 6).
5.5 Chẩn đoán độ sáng (Bảng 7).
5.6 Định tính (Hình: lưới ảnh tăng sáng; lưới ảnh blur) — nêu rõ chủ quan, 1 người chấm. **[TODO]** nếu có khảo sát mù thì thay bằng kết quả khảo sát.
5.7 Đường cong huấn luyện (`curves.png`).

### 6. Thảo luận
6.1 Vì sao U-Net mạnh (các giả thuyết, chưa kiểm chứng).
6.2 FFT không thể hiện lợi ích ở quy mô này.
6.3 Mâu thuẫn số liệu–cảm quan (PSNR vs tông tự nhiên).
6.4 Khử mờ chưa đạt: so với DarkIR-m 27.37 dB; vai trò dữ liệu.
6.5 Ý nghĩa thực tiễn (no-FFT rẻ hơn mà không mất chất lượng đo được).

### 7. Mối đe dọa tính hợp lệ (copy từ report §6, §8)
Số seed, độ dài huấn luyện không đồng đều và epoch của `best.pt`, kích thước val/test, tương quan trong chuỗi blur, DarkIR-m không công bằng, latency chỉ trên L4, quan sát chủ quan.

### 8. Kết luận & công việc tương lai
Như S21–S22; nêu v3 (nếu có số).

### Tài liệu tham khảo (**[TODO] xác minh từng mục trước khi đưa vào**)
- DarkIR: Feijoo et al., "DarkIR: Robust Low-Light Image Restoration", CVPR 2025.
- LEDNet / LOL-Blur: Zhou et al., "Lednet: Joint low-light enhancement and deblurring in the dark", ECCV 2022.
- LOLv2: Yang et al., "Sparse gradient regularized deep Retinex network for robust low-light image enhancement", IEEE TIP 2021.
- LSRW: Hai et al., "R2RNet: Low-light image enhancement via real-low to real-normal network", 2023.
- U-Net: Ronneberger et al., MICCAI 2015.
- LPIPS: Zhang et al., CVPR 2018.
- CLAHE: Zuiderveld, Graphics Gems IV, 1994. MSRCR: Jobson et al., IEEE TIP 1997.

### Danh mục hình/bảng cần chuẩn bị
| Mã | Nội dung | Nguồn |
|---|---|---|
| Fig.1 | Sơ đồ U-Net và DarkIR-Edge | docs/brainstorm/03 **[TODO]** |
| Fig.2 | Quy trình dữ liệu/split | tự vẽ |
| Fig.3 | Forest plot ΔPSNR | `paired_tests.csv` **[TODO]** script |
| Fig.4 | Lưới định tính tăng sáng | `visual_samples/`, `qualitative.png` |
| Fig.5 | Lưới định tính blur | `qualitative_lolblur_5samples.png`, `qualitative_with_darkir_m.png` |
| Fig.6 | Đường cong huấn luyện | `curves.png` |
| Tab.1–7 | Dữ liệu, PSNR, SSIM/LPIPS, kiểm định, ablation, hiệu năng, độ sáng | `table_main.csv`, `paired_tests.csv`, `efficiency.csv`, `diagnostic_gtmean.csv` |

### Mẹo trung thực khi viết (nhắc lại)
- Mọi "tốt hơn/kém hơn" đi kèm ΔPSNR + CI. Không dùng "significantly" nếu CI chứa 0.
- Giá trị ± giữa 2 seed chỉ mô tả, không phải CI.
- Khi nói nguyên nhân (vì sao U-Net mạnh) luôn dùng "giả thuyết".
