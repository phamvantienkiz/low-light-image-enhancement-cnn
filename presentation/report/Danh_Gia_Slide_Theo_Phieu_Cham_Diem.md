# Đánh giá Slide theo Phiếu đánh giá đồ án (CS2203)

Phạm vi: `presentation/slide/index.html` (23 slide, bản v3+v4), đối chiếu với `table_main.csv`, `efficiency.csv`, `split_manifest.json` (v3/v4) và phiếu `PhieudanhgiaMonhoc.pdf`.
Điểm dưới đây là **ước lượng của người rà soát**, không phải điểm chính thức.

## 1. Kết luận nhanh

- **Mức đáp ứng ước lượng hiện tại: ~75–84/100; chưa đạt 90%.** Sau khi sửa 4 điểm ở mục 5 có thể đạt ~88–92. Lý do chính: 2 lỗ hổng rõ (Hỏi đáp, Demo), phát biểu bài toán thiếu công thức và vài chỗ lệch số liệu.
- **Tính trung thực: tốt.** Các bảng PSNR/SSIM/LPIPS và hiệu năng khớp CSV. Slide báo cáo cả kết quả âm: Edge không thắng U-Net, FFT không có ích, v4 vi phạm QT1, blur bão hòa.
- **Cần sửa trước khi bảo vệ:** (1) thêm slide Hỏi đáp/Cảm ơn, (2) nút Demo trỏ tới `127.0.0.1:7860` nhưng repo không có app, (3) sửa các số liệu mâu thuẫn ở mục 3, (4) bổ sung công thức bài toán.

## 2. Chấm theo từng tiêu chí

| # | Tiêu chí | Tối đa | Ước lượng | Nhận xét |
|---|---|---|---|---|
| 1 | Lý do, cơ sở | 5 | 3.5–4 | Slide 4 nêu đúng động cơ (tối + nhiễu + nhòe, tác vụ hạ nguồn, ngân sách ~1.9M). Nhưng chỉ có khẳng định định tính, **không có trích dẫn hay con số** (ví dụ mức sụt giảm độ chính xác của detector ban đêm). |
| 2.1 | Phát biểu bài toán | 15 | 10–11 | Slide 5 chỉ có Input/Output. **Thiếu hàm ánh xạ f_θ, ràng buộc \|θ\|≈1.9M, mục tiêu tối ưu (cực đại PSNR/SSIM, cực tiểu LPIPS)**. `slide_content.md` có mô tả cột này nhưng bản HTML không có. Đây là mục 15 điểm nên rủi ro lớn nhất. |
| 2.2 | Hình minh họa | 5 | 5 | Slide 6–7: Input vs GT cho 2 kịch bản (LOLv2, LOL-Blur). Tốt. |
| 3 | Phương pháp, làm nổi bật khác biệt | 30 | 23–26 | Slide 7–10 rõ ràng, có sơ đồ kiến trúc U-Net/DarkIR-Edge, ablation no-FFT, số tham số. **Thiếu bảng "khác DarkIR gốc ở đâu"** (DarkIR-m 5.54M vs Edge C=24 1.92M, thêm I_max, GatedSkip, side-loss). Các khác biệt rải rác trong slide 10. Điểm yếu thực chất: phương pháp đề xuất **không vượt baseline**. Cần diễn đạt khác biệt như "đóng góp kiểm chứng/ablation", đừng như ưu thế. |
| 4.1 | Dataset | 10 | 9 | Slide 11 đầy đủ: 5 miền, train/val/test, chống rò rỉ. Số liệu khớp manifest (v3 640+...=1600/150/190; v4 tổng 6,777). Thiếu độ phân giải ảnh và số lượng gốc của mỗi bộ. |
| 4.2 | Độ đo | 10 | 8–9 | Slide 12: PSNR/SSIM/LPIPS, pooled/macro, kiểm định. Thiếu **công thức/ý nghĩa ngắn** của từng độ đo; một số độ đo mở rộng (ΔE00, lum_bias…) chỉ nêu tên. |
| 4.3 | Kết quả các phương pháp | 5 | 5 | Slide 13–16: đủ baseline truyền thống, 3 CNN, DarkIR-m ref, CI, latency. |
| 4.4 | Demo + thực nghiệm khác tài liệu | 5 | 2–3 | Có thực nghiệm khác bài gốc (ablation FFT, đa miền, v3→v4). **Nhưng demo không kiểm chứng được:** nút trỏ `localhost:7860`; trong `src/`, `README.md` không tìm thấy app Gradio/Streamlit. `docs/ai/task_plan_demo_presentation.md` mới chỉ là kế hoạch. Nếu chạy được trên máy thì giữ, nếu không phải thay bằng ảnh chụp/video hoặc `infer.py`. |
| 5 | Nhận xét, kết luận | 5 | 4.5 | Slide 20–21 cân bằng, nêu hạn chế và hướng phát triển. |
| 6 | Slide đầy đủ, dễ hiểu, đánh số trang | 5 | 3.5–4 | Có đánh số `NN/23`. Nhưng slide 14–16 là bảng rất dày (7–11 cột, nhiều dòng v3/v4 xen kẽ), khó đọc trên máy chiếu; slide 20, 22 nhiều chữ. |
| 7 | Hỏi đáp | 5 | 1–2 | **Không có slide Hỏi đáp/Cảm ơn**; slide cuối (23) là Tài liệu tham khảo. Chưa có bộ câu hỏi dự phòng (có thể nằm trong báo cáo nhưng không thấy trong slide). |
| | **Tổng** | 100 | **~75–84** theo cận trên/dưới ở trên | Cận trên đạt ~85–88 nếu sửa các lỗi dưới. |

> Khi cộng cận dưới là ~75, cận trên ~84 (chưa sửa). Sau khi sửa 4 điểm ở mục 5 có thể đạt ~88–92.

## 3. Kiểm tra tính trung thực & nhất quán số liệu

### 3.1 Đã khớp nguồn (đã đối chiếu CSV)
- Bảng PSNR/SSIM/LPIPS v3 (Input, Gamma, CLAHE, MSRCR, 3 CNN) khớp `table_main.csv`.
- Hiệu năng: Params, GFLOPs 8.75/7.50/18.45, latency FP32 31.71/27.31/13.23 ms (v3) và 32.29/27.79/13.37 ms (v4) khớp `efficiency.csv`. Tỷ lệ ×2.4 đúng (31.71/13.23=2.40).
- Tổng dữ liệu v3: 1600/150/190 = 1940 cặp; v4: train 6,777 (639+2380+1858+1000+900) + 150 + 190 = 7,117 cặp. Đúng.
- Tăng PSNR so Gamma: 20.97→+4.07, 21.31→+4.41 dB. Slide ghi "+4.1 đến +4.4". Đúng.

### 3.2 Mâu thuẫn / lỗi cần sửa
| Slide | Vấn đề | Đề xuất |
|---|---|---|
| 19 | Giai đoạn v3 ghi "150 cặp ảnh Blur", v4 ghi "Mở rộng lên 400 cặp". Slide 11 và manifest: v3 = **400**, v4 = **1,858**. | Sửa thành 400 → 1,858. |
| 21 | "Tăng gấp 4.88× (từ 1,389/1,600 …)". 6,777/1,600 = 4.24×; 4.88× chỉ đúng với 1,389. Số 1,389 không còn xuất hiện ở nơi khác. | Thống nhất: 6,777/1,600 ≈ 4.2×. |
| 22 | "Tăng 4.1× dữ liệu blur (1,858)". 1,858/400 = **4.65×**. | Sửa thành ~4.6×. |
| 15 | Dòng `[v3] Edge − U-Net` ghi "Ngang nhau", trong khi Wilcoxon p=0.028 (<0.05) và t-test p=0.222. | Ghi rõ "CI chứa 0 nhưng Wilcoxon p=0.028, nghiêng về U-Net" để tránh bị bắt lỗi. |
| 18–19 | Chọn đúng 1 ảnh minh họa. Ảnh LOLv2 #00690: Edge v3 27.88 > U-Net 26.31, nhưng v4 ngược lại (24.92 < 27.15); LOL-Blur v4: Edge 28.36 > U-Net 25.01 trong khi **trung bình** thì U-Net 24.01 > Edge 23.54. Ảnh đơn lẻ dễ gây hiểu nhầm là cherry-pick. | Ghi chú "mẫu đơn lẻ, không đại diện; xem trung bình ở slide 13". Có thể thêm 1 ảnh Edge kém hơn. |
| 19 | Nhãn DarkIR-m: "Mẫu 25.55 dB / Mean 27.37" ở v3, v4 chỉ ghi "Trần tham chiếu". Mẫu thấp hơn trung bình có thể gây thắc mắc. | Bổ sung giải thích. |
| 10 / 21 | "DarkIR-Edge" được mô tả như "mô hình đề xuất" nhưng kết quả cho thấy không tốt hơn U-Net. Tên "ours" trong bảng dễ hiểu là đề xuất thắng. | Đổi cách gọi: "biến thể thử nghiệm" hoặc nêu rõ "kết quả trung tính". |
| 3 | "Báo cáo ... ứng dụng Demo ... Hoàn thành" nhưng demo chưa kiểm chứng. | Chỉ ghi Hoàn thành nếu demo chạy được. |
| 16/ nguồn | `slide_content.md` vẫn là bản **v2** (23 slide cũ: 1,650 cặp, U-Net dẫn đầu −0.62 dB, p=4.4e-5…) khác hẳn HTML v3/v4. | Cập nhật hoặc xóa để tránh dùng nhầm làm kịch bản nói. |

### 3.3 Điểm trung thực đáng khen
- Công bố kết quả không có lợi cho phương pháp chính (FFT không giúp, Edge chậm ×2.4).
- Dùng CI bootstrap ghép cặp, đăng ký trước quy tắc kết luận, báo cả "vi phạm QT1".
- Ghi rõ không đo trên thiết bị biên (hướng phát triển).

## 4. Mức độ "dễ hiểu" (tiêu chí 3 và 6)

- Tốt: sơ đồ kiến trúc trực quan, lộ trình rõ (đặt vấn đề → phương pháp → thực nghiệm → kết luận).
- Chưa tốt: nhiều thuật ngữ nội bộ khó với người nghe (QT1/QT2/QT3, "Quy tắc 5", lum_bias, QuotaSampler, gt-mean) mà không có giải thích ngắn; slide 14–16 quá dày. Khuyến nghị: tách v3 và v4 ra 2 slide, hoặc chỉ giữ cột Pooled/Macro, chuyển bảng đầy đủ vào phụ lục.

## 5. Việc cần làm (ưu tiên giảm dần)

1. **Thêm slide "Cảm ơn & Hỏi đáp"** (và 1–2 slide phụ lục trả lời trước các câu hỏi dự kiến, xem mục 6). (+3–4 điểm)
2. **Bổ sung slide 5:** `f_θ: R^{H×W×3}→R^{H×W×3}`, ràng buộc ≈1.9M tham số, hàm mục tiêu/loss, độ đo. (+3–4 điểm)
3. **Demo:** làm app thật (ví dụ `infer.py` kèm ảnh mẫu/video quay màn hình), hoặc bỏ nút nếu không có. Thêm 1 slide demo với ảnh in-the-wild. (+2 điểm)
4. **Sửa các số liệu ở mục 3.2** (blur 400/1,858, 4.2×, 4.6×, nhãn Wilcoxon).
5. **Thêm 1 bảng "DarkIR gốc vs DarkIR-Edge vs U-Net"** để làm nổi bật khác biệt (tiêu chí 3). Nói rõ đóng góp là kiểm chứng có kiểm soát.
6. Giảm mật độ slide 14–16, giải thích QT1–QT3.
7. Đồng bộ `slide_content.md` với bản HTML hiện tại.

## 6. Câu hỏi hội đồng dự kiến (gợi ý cho phần Hỏi đáp)

1. Vì sao DarkIR-Edge không vượt U-Net dù tinh vi hơn? (giả thuyết: dữ liệu nhỏ, dung lượng 1.9M, FFT không phù hợp, lệch sáng toàn cục chiếm phần lớn sai số: PSNR tăng ~5–6 dB khi chuẩn hóa độ sáng.)
2. FLOPs giảm 53% nhưng latency tăng ×2.4, giải thích? (DW-Conv/rFFT/Gating bị giới hạn bởi băng thông bộ nhớ và số lần gọi kernel.)
3. Vì sao dùng 2 seed? CI có đủ tin cậy không? (hạn chế, đề xuất ≥5 seed.)
4. Test set 190 ảnh có rò rỉ với train không? (split cách ly theo chuỗi/video, buffer gap=10.)
5. Nikon là zero-shot ở v3 nhưng in-domain ở v4: so sánh v3–v4 có công bằng? (chỉ so trên cùng test set cố định.)
6. Vì sao DarkIR-m tốt hơn trên blur? (5.54M tham số, huấn luyện trên dữ liệu lớn; so sánh không cùng ngân sách.)
7. Ứng dụng thực tế/thiết bị di động? (chưa đo trên thiết bị biên.)

## 7. Giới hạn của lần rà soát này

- Chưa kiểm tra từng ảnh minh họa bằng mắt thường, chưa kiểm tra render/độ dễ đọc thực tế trên máy chiếu.
- Chỉ đối chiếu có chọn lọc với CSV (bảng PSNR/SSIM/LPIPS v3, hiệu năng v3/v4, manifest v3). Số liệu v4 chi tiết, paired tests và bảng v3−v4 chưa đối chiếu từng ô.
- Chưa mở hai notebook để xác nhận cấu hình huấn luyện (70 epoch, AdamW, EMA).
