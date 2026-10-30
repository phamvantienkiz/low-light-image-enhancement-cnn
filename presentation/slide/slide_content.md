# TÀI LIỆU NỘI DUNG & KỊCH BẢN THUYẾT TRÌNH CÁC TRANG SLIDE (PHIÊN BẢN V2 — TINH GỌN CHO MÁY CHIẾU)
> **Đề tài:** Tăng cường chất lượng ảnh trong điều kiện thiếu sáng bằng mạng nơ-ron tích chập (So sánh có kiểm soát Standard U-Net và biến thể DarkIR-Edge)
> **Môn học:** Xử lý ảnh và Thị giác máy tính (CS2203) — Khóa 2026
> **Đơn vị:** Trường Đại học Công nghệ Thông tin, ĐHQG-HCM (UIT - VNUHCM)
> **Học viên thực hiện:** Phạm Văn Tiến — **MSHV:** 26848223
> **Tổng số lượng slide:** 23 slide chuẩn 16:9 (1920x1080), chữ lớn (26px–40px), cô đọng tối đa để hỗ trợ người thuyết trình trên máy chiếu.

---

## 1. ÁNH XẠ CẤU TRÚC SLIDE THEO YÊU CẦU BÁO CÁO

| Thứ tự yêu cầu | Trang Slide | Tiêu đề Header trên Slide |
|---|---|---|
| Trang bìa (Giữ nguyên 100%) | **Slide 01** | *(Trang bìa gốc)* |
| Nội dung thuyết trình / báo cáo | **Slide 02** | `Nội Dung Báo Cáo` |
| Thông tin thành viên & đánh giá mức độ hoàn thành | **Slide 03** | `Thông Tin Thành Viên & Đánh Giá Mức Độ Hoàn Thành` |
| Lý do hoặc tầm quan trọng của bài toán | **Slide 04** | `Vấn Đề Bài Toán Cần Giải Quyết` |
| Phát biểu bài toán: Input & Output | **Slide 05** | `Phát Biểu Bài Toán` |
| Ví dụ minh họa bài toán | **Slide 06** | `Ví Dụ Minh Họa Bài Toán: Input → Output → Ground Truth` |
| Phương pháp giải quyết (Tổng quan) | **Slide 07** | `Phương Pháp Giải Quyết: Tổng Quan Các Nhóm Tiếp Cận` |
| Phương pháp giải quyết (Standard U-Net) | **Slide 09** | `Phương Pháp 1: Mô hình Standard U-Net` |
| Phương pháp giải quyết (DarkIR-Edge) | **Slide 10** | `Phương Pháp 2: Mô hình DarkIR-Edge` |
| Phương pháp giải quyết (Ablation no-FFT & Loss) | **Slide 10** | `Phương Pháp Giải Quyết: Biến Thể no-FFT & Quy Trình Huấn Luyện` |
| Kết quả thực nghiệm: Dataset | **Slide 11** | `Kết Quả Thực Nghiệm: Tập Dữ Liệu Đa Miền (Dataset)` |
| Kết quả thực nghiệm: Độ đo & Môi trường | **Slide 12** | `Kết Quả Thực Nghiệm: Độ Đo Đánh Giá & Môi Trường` |
| Kết quả thực nghiệm: Bảng kết quả PSNR | **Slide 13** | `Kết Quả Thực Nghiệm: Bảng Kết Quả PSNR (dB ↑)` |
| Kết quả thực nghiệm: Bảng kết quả SSIM & LPIPS | **Slide 14** | `Kết Quả Thực Nghiệm: Bảng Kết Quả SSIM (↑) & LPIPS (↓)` |
| Kết quả thực nghiệm: Bảng kiểm định thống kê | **Slide 15** | `Kết Quả Thực Nghiệm: Kiểm Định Thống Kê Ghép Cặp (95% CI)` |
| Kết quả thực nghiệm: Bảng chi phí tính toán | **Slide 16** | `Kết Quả Thực Nghiệm: Chi Phí Tính Toán & Độ Trễ (NVIDIA L4)` |
| Kết quả thực nghiệm: Minh họa tăng sáng | **Slide 17** | `Minh Họa Kết Quả: Phục Hồi Ảnh Thiếu Sáng (Miền LOLv2)` |
| Kết quả thực nghiệm: Minh họa khử mờ | **Slide 18** | `Minh Họa Kết Quả: Thiếu Sáng Kèm Nhòe Chuyển Động (LOL-Blur)` |
| Kết quả thực nghiệm: Đường cong & Độ sáng | **Slide 19** | `Minh Họa Kết Quả: Đường Cong Huấn Luyện & Chẩn Đoán Độ Sáng` |
| Đánh giá, nhận xét: Ưu điểm | **Slide 20** | `Đánh Giá & Nhận Xét: Những Ưu Điểm Đạt Được` |
| Đánh giá, nhận xét: Hạn chế & Hướng phát triển | **Slide 21** | `Đánh Giá & Nhận Xét: Hạn Chế & Hướng Phát Triển` |
| Tài liệu tham khảo | **Slide 22** | `Tài Liệu Tham Khảo` |

---

## 2. TÓM TẮT NỘI DUNG TRÊN SLIDE & GỢI Ý LỜI NÓI CHO NGƯỜI THUYẾT TRÌNH

### Slide 01: Trang Bìa
- **Giữ nguyên 100% nội dung gốc.**
- **Lời nói gợi ý:** Chào thầy/cô và các bạn. Hôm nay em xin trình bày báo cáo đồ án môn Xử lý ảnh và Thị giác máy tính (CS2203) với đề tài Tăng cường chất lượng ảnh trong điều kiện thiếu sáng bằng mạng nơ-ron tích chập.

### Slide 02: Nội Dung Báo Cáo
- **Hiển thị trên slide:** Chia làm 2 phần rõ ràng: (I) Đặt vấn đề & Phương pháp giải quyết; (II) Kết quả thực nghiệm & Đánh giá phản biện.
- **Lời nói gợi ý:** Bài báo cáo gồm 7 phần chính theo đúng yêu cầu môn học: từ thông tin thực hiện, lý do chọn đề tài, phát biểu bài toán kèm ví dụ, phương pháp giải quyết, kết quả thực nghiệm (dataset, độ đo, môi trường, bảng kết quả, minh họa), đến phần nhận xét ưu/nhược điểm và tài liệu tham khảo.

### Slide 03: Thông Tin Thành Viên & Đánh Giá Mức Độ Hoàn Thành
- **Header:** `Thông Tin Thành Viên & Đánh Giá Mức Độ Hoàn Thành` | Tag: `THÔNG TIN`
- **Hiển thị trên slide:** Học viên Phạm Văn Tiến (26848223) — 100% HOÀN THÀNH, cùng bảng trạng thái 4 hạng mục công việc đã thực hiện (giữ thuật ngữ chuẩn: domains, sequence-isolated split, baselines, 6 lần huấn luyện...).
- **Lời nói gợi ý:** Đề tài do cá nhân em thực hiện độc lập 100%, hoàn thành đầy đủ từ khâu chuẩn bị dữ liệu 1,650 cặp ảnh cách ly chuỗi, cài đặt 3 mô hình học sâu + 3 baseline truyền thống, huấn luyện 6 runs (3 mô hình × 2 seed) đến kiểm định thống kê và viết báo cáo.

### Slide 04: Vấn Đề Bài Toán Cần Giải Quyết
- **Header:** `Vấn Đề Bài Toán Cần Giải Quyết` | Tag: `Đặt vấn đề`
- **Hiển thị trên slide:** 3 thẻ cô đọng: (1) Suy thoái kép ban đêm (Tối + Nhiễu + Nhòe), (2) Tác vụ thị giác suy giảm, (3) Hướng phát thực hiện đề tài (Mô hình tối ưu tài nguyên ~1.9M).
- **Lời nói gợi ý:** Ảnh chụp đêm không chỉ thiếu sáng mà còn chịu nhiễu ISO cao và nhòe do phơi sáng lâu. Điều này làm giảm mạnh độ chính xác của các mô hình nhận diện phía sau. Trong thực tế, ta cần mô hình nhỏ gọn thay vì các mạng hàng chục triệu tham số. Do đó, câu hỏi đặt ra là với ngân sách ~1.9M tham số và ~1.3k ảnh huấn luyện, một U-Net gọn hay biến thể lấy cảm hứng từ DarkIR sẽ hiệu quả hơn?

### Slide 05: Phát Biểu Bài Toán
- **Header:** `Phát Biểu Bài Toán` | Tag: `Phát biểu bài toán`
- **Hiển thị trên slide:** 3 cột chuẩn LaTeX: Input $I_{\text{low}}$ (thiếu sáng + nhiễu/nhòe) $\in \mathbb{R}^{H \times W \times 3}$ → Mô hình $f_\theta$ ($|\theta| \approx 1.9\text{M}$ params, cực đại PSNR/SSIM, cực tiểu LPIPS) → Output $\hat{I} = f_\theta(I_{\text{low}}) \approx I_{\text{gt}}$.
- **Lời nói gợi ý:** Bài toán đặt ra là học ánh xạ $f_\theta$ biến ảnh RGB thiếu sáng $I_{\text{low}}$ thành ảnh phục hồi $\hat{I}$ sát với Ground Truth $I_{\text{gt}}$ nhất ở ngân sách ~1.9M tham số.

### Slide 06: Minh Họa Bài Toán — Kịch bản 1 (Thiếu sáng & Nhiễu)
- **Hiển thị trên slide:** 2 ảnh khổ lớn kích thước đồng nhất, căn giữa slide theo chiều dọc: Input - Ảnh thiếu sáng và Ground Truth - Ảnh đủ sáng trên mẫu `low00690` tập LOLv2-Real.
- **Lời nói gợi ý:** Đây là ví dụ minh họa trực quan kịch bản 1 của bài toán: ảnh chụp thực tế trong điều kiện thiếu sáng sâu kèm nhiễu hạt cảm biến từ miền LOLv2-Real đối chiếu với ảnh chuẩn Ground Truth đủ sáng.

### Slide 07: Minh Họa Bài Toán — Kịch bản 2 (Thiếu sáng & Nhòe)
- **Hiển thị trên slide:** 2 ảnh khổ lớn kích thước đồng nhất, căn giữa slide theo chiều dọc: Input - Ảnh thiếu sáng và mờ nhòe và Ground Truth - Ảnh đủ sáng và khử nhòe trên mẫu `0023` tập LOL-Blur.
- **Lời nói gợi ý:** Đây là kịch bản thách thức thứ 2: ảnh chụp thiếu sáng kèm nhòe chuyển động do thời gian phơi sáng dài trên miền LOL-Blur đối chiếu với ảnh chuẩn Ground Truth sắc nét.

### Slide 08: Tổng Quan Các Giải Pháp và Hướng Tiếp Cận
- **Hiển thị trên slide:** 3 nhóm: (1) Truyền thống (Gamma 2.2, CLAHE, MSRCR), (2) Baseline CNN (Standard U-Net 1.93M), (3) Họ DarkIR-Edge (cảm hứng CVPR'25, có và không có FFT).
- **Lời nói gợi ý:** Để đánh giá khách quan, em đối chiếu 3 nhóm phương pháp: các thuật toán truyền thống không học sâu, mạng tích chập chuẩn Standard U-Net làm baseline, và biến thể DarkIR-Edge tích hợp nhánh tần số FFT cùng bản ablation tắt FFT.

### Slide 09: Phương Pháp 1: Mô hình Standard U-Net
- **Hiển thị trên slide:** Cột trái: Thiết kế kiến trúc UNetBaseline chuẩn mã nguồn thực nghiệm (khối `_cb: 2 × (Conv 3×3 + GroupNorm(4) + ReLU)`, 3 tầng downsample MaxPool đối xứng 3 tầng upsample ConvTranspose2d, ghép kênh `torch.cat`, và kết nối thặng dư toàn cục `+ Input`) cùng khung Tổng số tham số (1.928 M). Cột phải: Visualizer sơ đồ kiến trúc UNetBaseline (base=32) phản ánh chính xác 1:1 luồng tính toán từ đầu vào, các khối `_cb`, đường `Skip Concats` đến phép cộng `Global Residual (+X)` ở ngõ ra.
- **Lời nói gợi ý:** Mô hình baseline đối chuẩn là Standard U-Net với kênh cơ sở 32, tổng cộng 1.928M tham số. Mạng sử dụng các khối tích chập kép chuẩn kết hợp chuẩn hóa nhóm GroupNorm, truyền đặc trưng độ phân giải cao qua các đường nối tắt Skip Concatenation và cộng thặng dư toàn cục với ảnh đầu vào để ổn định quá trình tối ưu.

### Slide 10: Phương Pháp 2: Mô hình DarkIR-Edge
- **Hiển thị trên slide:** Cột trái: Thiết kế mô hình đề xuất DarkIREdge chuẩn mã nguồn thực nghiệm (kênh C=24, cải tiến Max-RGB Illumination Prior `[R,G,B,I_max]`, cấu trúc bất đối xứng Encoder [1,2,3] tích hợp `FreMLP (2D-rFFT)` và Decoder [3,1,1] dùng `Di-SpAM [1,4,9]` + `PixelShuffle`, cổng `Residual GatedSkip` và nhánh `Side-loss 8C→3`) cùng khung Tổng số tham số (1.924 M). Cột phải: Visualizer sơ đồ kiến trúc DarkIREdge thể hiện rõ ràng các tầng EBlock, khối Bottleneck kết hợp EBlock-DBlock & giám sát sâu, các khối DBlock, cổng lọc nhiễu Gated Skip và Global Residual.
- **Lời nói gợi ý:** Mô hình đề xuất chính là DarkIR-Edge với kênh C=24, đạt 1.924M tham số tương đương U-Net. Mô hình kế thừa nguyên vẹn thiết kế bất đối xứng của DarkIR gồm Encoder 3 tầng có nhánh tần số Fourier FreMLP và Decoder giải mờ đa tỉ lệ Di-SpAM, đồng thời tích hợp 2 cải tiến vệ tinh: bổ sung kênh tiên nghiệm độ rọi Max-RGB I_max ở đầu vào và cổng lọc nhiễu Residual Gated Skip trên các đường nối tắt.

### Slide 11: Phương Pháp Giải Quyết: Biến Thể no-FFT & Quy Trình Huấn Luyện
- **Hiển thị trên slide:** Cột trái: DarkIR-Edge (no-FFT) 1.495M params (−22.3%) và công thức hàm Loss tổng hợp. Cột phải: Huấn luyện AdamW, Cosine LR, EMA 0.999, Early Stopping val macro, 2 seed (42, 123) và quy tắc 95% Bootstrap CI.
- **Lời nói gợi ý:** Để kiểm chứng nhánh FFT có thực sự giúp ích không, em xây dựng biến thể thứ ba là no-FFT bằng cách tắt khối FFT/FreMLP, giúp giảm 22.3% tham số. Cả 3 mạng cùng học với hàm mất mát L1 + SSIM + Color trên 2 seed độc lập, và mọi kết luận đều tuân theo quy tắc khoảng tin cậy 95% CI đã đăng ký trước.

### Slide 12: Kết Quả Thực Nghiệm: Tập Dữ Liệu Đa Miền (Dataset)
- **Hiển thị trên slide:** Bảng phân chia dữ liệu: `lolv2` (639/50/100), `huawei` (561/50/30), `nikon` (0/0/20 — chỉ test), `blur` (150/10/40); Tổng: `1,350 train / 110 val / 190 test` (1,650 cặp).
- **Lời nói gợi ý:** Tập dữ liệu hợp nhất 4 miền cảm biến với tổng cộng 1,650 cặp ảnh. Riêng miền Nikon chỉ dùng để test khả năng tổng quát hóa sang cảm biến chưa từng thấy. Em cũng nêu rõ hạn chế là miền blur chỉ có 150 ảnh train (11.1%) và 10 ảnh val.

### Slide 13: Kết Quả Thực Nghiệm: Độ Đo Đánh Giá & Môi Trường
- **Hiển thị trên slide:** Cột 1: Độ đo PSNR (↑), SSIM (↑), LPIPS (↓), cách tính Pooled (190 ảnh) & Macro (4 miền). Cột 2: Môi trường Google Colab GPU NVIDIA L4, đo latency ở 600×400 FP32/FP16, môi trường cục bộ cô lập bằng `uv`/`.venv`.
- **Lời nói gợi ý:** Chúng em đánh giá trên 3 chỉ số chuẩn PSNR, SSIM, LPIPS ở cả mức gộp Pooled và trung bình miền Macro. Toàn bộ tốc độ suy luận được đo công bằng trên GPU NVIDIA L4 ở độ phân giải 600×400; do chưa đo trên chip di động nên đề tài không tuyên bố thời gian thực trên thiết bị biên.

### Slide 14: Kết Quả Thực Nghiệm: Bảng Kết Quả PSNR (dB ↑)
- **Hiển thị trên slide:** Bảng số liệu chuẩn xác từ `table_main.csv`: Input (9.75), Gamma (16.90), CLAHE/MSRCR (11.59/12.12), Edge no-FFT (20.76), DarkIR-Edge (20.74), **Standard U-Net (21.36)**, DarkIR-m ref (20.96).
- **Lời nói gợi ý:** Kết quả PSNR cho thấy cả 3 mạng CNN đều vượt xa thuật toán truyền thống ít nhất +3.8 dB. Trong các mô hình huấn luyện cùng điều kiện, Standard U-Net dẫn đầu với 21.36 dB pooled, cao hơn DarkIR-Edge 0.62 dB. Đáng chú ý, bản no-FFT đạt 20.76 dB, ngang ngửa bản có FFT.

### Slide 15: Kết Quả Thực Nghiệm: Bảng Kết Quả SSIM (↑) & LPIPS (↓)
- **Hiển thị trên slide:** Bảng đối chiếu SSIM và LPIPS (Pooled & Macro). Standard U-Net đạt SSIM `0.749` và LPIPS `0.277` (tốt hơn rõ rệt so với `0.368` của DarkIR-Edge).
- **Lời nói gợi ý:** Xu hướng tương tự lặp lại trên SSIM và LPIPS: U-Net dẫn đầu cả về bảo toàn cấu trúc (SSIM 0.749) lẫn chất lượng cảm quan đặc trưng sâu (LPIPS 0.277 thấp hơn nhiều so với 0.368 của Edge).

### Slide 16: Kết Quả Thực Nghiệm: Kiểm Định Thống Kê Ghép Cặp (95% CI)
- **Hiển thị trên slide:** Bảng kiểm định từ `paired_tests.csv`: `Edge − U-Net` pooled ΔPSNR = `−0.615 dB`, CI `[−0.891, −0.335]`, p = 4.4e-5; `Edge − no-FFT` pooled ΔPSNR = `−0.014 dB`, CI `[−0.231, +0.199]`, p = 0.431.
- **Lời nói gợi ý:** Áp dụng quy tắc kiểm định ghép cặp đã đăng ký trước: khoảng tin cậy 95% của hiệu số Edge trừ U-Net nằm hoàn toàn dưới 0, khẳng định U-Net vượt trội có ý nghĩa thống kê (đặc biệt trên LOLv2 −1.05 dB). Ngược lại, mọi khoảng CI giữa Edge và no-FFT đều chứa 0, chứng tỏ không có bằng chứng nhánh FFT giúp cải thiện PSNR ở quy mô này.

### Slide 17: Kết Quả Thực Nghiệm: Chi Phí Tính Toán & Độ Trễ (NVIDIA L4)
- **Hiển thị trên slide:** Bảng `efficiency.csv`: Edge (8.75 GFLOPs, 31.52 ms), no-FFT (7.50 GFLOPs, 27.14 ms), U-Net (18.45 GFLOPs, **12.98 ms**). Hai thẻ giải thích nghịch lý FLOPs vs Latency và lợi ích thực tế của no-FFT.
- **Lời nói gợi ý:** Bảng hiệu năng chỉ ra một thực tế quan trọng: dù DarkIR-Edge cắt giảm 53% FLOPs so với U-Net, nhưng độ trễ thực tế lại chậm hơn 2.43 lần (31.5 ms so với 13.0 ms) vì các toán tử Depthwise, FFT và Gating bị giới hạn bởi băng thông bộ nhớ và chi phí gọi kernel. Khi bỏ FFT, ta tiết kiệm ngay 22% tham số và ~14% độ trễ mà không mất chất lượng.

### Slide 18: Minh Họa Kết Quả: Phục Hồi Ảnh Thiếu Sáng (Miền LOLv2)
- **Hiển thị trên slide:** 4 ảnh lớn đối chiếu trên mẫu LOLv2 #00690: Input (8.09 dB) | Standard U-Net (25.31 dB) | DarkIR-Edge (28.49 dB) | Ground Truth.
- **Lời nói gợi ý:** Khi quan sát định tính trên ảnh chỉ thiếu sáng, U-Net kéo sáng mạnh và đồng đều. Trong khi đó, theo quan sát chủ quan của tác giả, DarkIR-Edge cho tông chuyển sáng–tối hài hòa và màu tự nhiên hơn trên một số ảnh, dù xét trên toàn bộ 190 ảnh test thì PSNR và LPIPS vẫn nghiêng về U-Net.

### Slide 19: Minh Họa Kết Quả: Thiếu Sáng Kèm Nhòe Chuyển Động (LOL-Blur)
- **Hiển thị trên slide:** 4 ảnh lớn mẫu LOL-Blur #0010: Input | Standard U-Net (23.37 dB) | DarkIR-Edge (22.93 dB) | DarkIR-m Ref Ceiling (27.37 dB).
- **Lời nói gợi ý:** Trên miền ảnh vừa tối vừa nhòe chuyển động, cả U-Net và DarkIR-Edge mới chỉ giải quyết tốt phần tăng sáng, còn biên cạnh chữ vẫn còn mờ và nhiễu. Đặt cạnh checkpoint chính thức DarkIR-m (27.37 dB, hơn ~4 dB), ta thấy rõ việc khử mờ đòi hỏi quy mô dữ liệu lớn hơn nhiều so với 150 ảnh blur trong tập train hiện tại.

### Slide 20: Minh Họa Kết Quả: Đường Cong Huấn Luyện & Chẩn Đoán Độ Sáng
- **Hiển thị trên slide:** Biểu đồ `curves.png` bên trái; cột phải phân tích chẩn đoán độ sáng (`diagnostic_gtmean.csv`: gap LOLv2 ~5.65–5.97 dB) và hiện tượng hội tụ sớm của Edge.
- **Lời nói gợi ý:** Biểu đồ huấn luyện và bảng chẩn đoán độ sáng cho thấy: trên LOLv2, khi chuẩn hóa độ sáng trung bình theo GT, PSNR tăng vọt gần 6 dB — tức phần lớn sai số là do lệch độ sáng toàn cục chứ không phải mất chi tiết cấu trúc, và nhánh FFT không làm giảm độ lệch này.

### Slide 21: Đánh Giá & Nhận Xét: Những Ưu Điểm Đạt Được
- **Hiển thị trên slide:** 2 cột: (1) Ưu điểm về mô hình & hiệu năng (CNN nhẹ vượt xa truyền thống ≥ +3.8 dB; U-Net gọn nhanh và chuẩn xác nhất; no-FFT gọn hơn 22%); (2) Ưu điểm về phương pháp luận (split cách ly chuỗi, 2 seed, Bootstrap CI, báo cáo trung thực kết quả âm).
- **Lời nói gợi ý:** Tổng kết lại, đề tài có 4 ưu điểm chính: chứng minh mạng CNN nhẹ ~1.9M vượt xa các phương pháp truyền thống; xác lập U-Net gọn là baseline cực mạnh cả về chất lượng lẫn tốc độ; chỉ ra biến thể no-FFT tiết kiệm tài nguyên mà không giảm chất lượng; và tuân thủ quy trình thực nghiệm có kiểm soát, báo cáo trung thực kết quả âm.

### Slide 22: Đánh Giá & Nhận Xét: Hạn Chế & Hướng Phát Triển
- **Hiển thị trên slide:** Cột 1: Hạn chế tồn tại (Edge thua U-Net −0.62 dB & chậm ×2.4; khử mờ chưa đạt do ít dữ liệu; confound số epoch huấn luyện 55 vs 90; quy mô 2 seed). Cột 2: Giả thuyết giải thích & Kế hoạch mở rộng v3 (tăng lên 400 ảnh blur train / 50 val, chạy cố định 70 epochs, bổ sung độ đo ΔE/Laplacian & khảo sát mù).
- **Lời nói gợi ý:** Về hạn chế, ngoài việc DarkIR-Edge chưa vượt U-Net và chưa khử mờ triệt để do chỉ có 150 ảnh blur, một mối đe dọa tính hợp lệ quan trọng là Early Stopping khiến số epoch huấn luyện không đồng đều (Edge dừng ở epoch 55 trong khi U-Net chạy tới 90). Do đó, hướng phát triển v3 sẽ mở rộng tập blur lên 400 ảnh train, 50 ảnh val và huấn luyện cố định 70 epochs cho mọi mô hình.

### Slide 23: Tài Liệu Tham Khảo
- **Hiển thị trên slide:** 8 công trình nền tảng chia làm 2 cột: [1] DarkIR (CVPR 2025), [2] U-Net (MICCAI 2015), [3] LEDNet/LOL-Blur (ECCV 2022), [4] LOL-v2 (IEEE TIP 2021), [5] LSRW (2023), [6] LPIPS (CVPR 2018), [7] CLAHE (1994), [8] MSRCR (IEEE TIP 1997).
- **Lời nói gợi ý:** Trên đây là các tài liệu tham khảo chính được kế thừa và trích dẫn trong báo cáo. Em xin chân thành cảm ơn thầy/cô và hội đồng đã lắng nghe, em rất mong nhận được các câu hỏi và góp ý ạ!
