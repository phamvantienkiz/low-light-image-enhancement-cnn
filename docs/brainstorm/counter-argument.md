Tôi so sánh bản mới của file 02 và 03 với bản trước.

Môi trường thử chưa có torch nên đang cài nền. Trong lúc đó tôi viết notebook.

Tôi đang chờ cài torch trong sandbox để chạy thử notebook trên dữ liệu giả trước khi gửi cho bạn.

Torch đã chạy được. Giờ chạy thử toàn bộ notebook (dưới dạng script) trên dữ liệu giả ở chế độ siêu nhỏ.

Mọi thứ chạy đến bước LPIPS (cần tải trọng số AlexNet, sandbox không có mạng nên lỗi này chỉ xảy ra ở đây). Tôi cho LPIPS dự phòng an toàn rồi chạy lại.

Tôi đã viết xong notebook `LLIE_DarkIR_Edge_Colab.ipynb`, chạy trọn luồng được trên dữ liệu giả trong sandbox không có GPU. Chưa thử trên Colab với dữ liệu thật.

## Về 3 file bản mới

Các lỗi lớn lần trước đã được sửa, và khớp với nhau:

- LOLv2-Real: 639 + 50 + 100 = 789.
- LSRW-Huawei: 561 train + 50 val + 30 test.
- Tổng 1.450 cặp (train 1.200, val 100, test 150), 100 epoch × 300 bước = 30.000 bước.
- Quy tắc chọn checkpoint là checkpoint cuối, và câu "FFT chiếm <4%" đã bỏ.

Chỉ còn vài chi tiết nhỏ:

- File 02 phần IV vẫn nhắc `D_train_0` (450 cặp) và các kịch bản +Imax/+Gate/+color loss mà bạn không còn chạy.
- File 03 ghi DarkIR-m là "~3.32M" nhưng bảng tham số cộng ra 3.389M.
- Câu Q3 còn viết "chứng minh vượt trội rõ rệt" trước khi có kết quả; nên đổi thành "đánh giá liệu…".

## Notebook làm gì

Chạy lần lượt: setup → dữ liệu → mô hình → **đo it/s** → train → evaluate → bảng/hình → hiệu năng.

- **Dữ liệu:** ghép cặp ảnh, báo lỗi nếu số ảnh low/high không khớp, chia train/val theo khối cuối (hạn chế ảnh cùng cảnh bị chia sang hai tập), kiểm tra không giao nhau giữa các tập, lưu manifest kèm SHA-256.
- **Mô hình:** `DarkIREdge` (EBlock có FreMLP, DBlock có Di-SpAM d=[1,4,9], SCA, side-loss ở đáy, kênh Imax, GatedSkip) và U-Net baseline có số kênh tự chọn để gần cùng số tham số.
- **FreMLP:** chạy FP32, dùng phasor đơn vị thay cho atan2, mặc định `norm='ortho'`.
- **Train:** ngân sách bước cố định, AdamW, cosine theo bước, AMP, clip gradient, EMA, tự resume nếu Colab ngắt, bỏ qua run đã xong.
- **Đánh giá:** gamma, CLAHE, MSRCR, U-Net, DarkIR-Edge trên 150 ảnh test. Báo PSNR/SSIM/LPIPS theo miền, pooled và macro, dạng mean ± std qua seed. Có paired t-test, Wilcoxon, bootstrap CI, hình minh họa, đường cong học, và bảng tham số/FLOPs/latency.

## Việc bạn cần làm trước khi chạy

1. Điền `SOURCES` ở cell 5 cho đúng cấu trúc thư mục thật. Tôi không biết cấu trúc zip của bạn nên để đường dẫn mẫu; cell 3 sẽ in cây thư mục để bạn đối chiếu.
2. Chạy cell đo tốc độ trước khi train. Ước lượng "35–45 phút/run" trong file 02 chưa có cơ sở đo, nên hãy xem it/s thật rồi quyết định số seed.

## Mức độ đã kiểm chứng

Đã kiểm chứng:

- Mọi cell chạy từ đầu đến cuối, kể cả kiểm định thống kê và các file kết quả.
- Cả hai mô hình học được và gradient chảy qua mọi tham số, kể cả FFT, gate và side-loss.
- Xử lý được ảnh có kích thước không chia hết cho 8.
- DarkIR-Edge có 1.996M tham số (tài liệu ghi 1.93M) và U-Net 1.928M.

Chưa kiểm chứng:

- **GPU/AMP FP16:** sandbox không có GPU.
- **LPIPS:** cần tải trọng số AlexNet, sandbox không có mạng. Nếu lỗi, notebook tự bỏ LPIPS và vẫn chạy.
- **Cấu trúc dữ liệu thật** của bạn.
- Cách ghép tên file low/high với dữ liệu thật.

## Điều cần ghi trung thực trong báo cáo

- **Đây là bản cài đặt lại theo mô tả DarkIR, không phải bản sao 1:1 mã gốc.** Số tham số lệch nhẹ so với 1.93M là bình thường. Hãy dùng số tham số mà notebook in ra.
- **Cùng số tham số nhưng không cùng chi phí tính toán.** U-Net cùng cỡ tham số có khoảng 18.5 GFLOPs so với khoảng 9.6 GFLOPs của Edge (chưa tính phần FFT). Hãy báo cả hai con số khi so sánh.
- **Chỉ có một mô hình đề xuất,** nên không kết luận được từng cải tiến vệ tinh đóng góp bao nhiêu. Nếu còn thời gian, đặt `use_imax=False, use_gate=False` cho một run đối chứng.
- **Mốc DarkIR chính thức:** ở cuối notebook có khung mẫu để chạy checkpoint gốc trên đúng 150 ảnh test. Bạn cần chỉnh theo README của repo vì tôi chưa xác minh API của nó.

Khi có kết quả, bạn gửi lại `results_per_image.csv`, `table_main.csv` và `qualitative.png`. Tôi sẽ giúp bạn đọc kết quả và viết phần phân tích, ca thất bại và hạn chế.
