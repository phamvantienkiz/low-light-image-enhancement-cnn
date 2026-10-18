# Image Processing & Computer Vision (CS2203)

Master of Science in Computer Science 2026

University of Information Technology - VNUHCM

## Đồ án môn học

### 1. Tên đề tài

Tên tiếng Việt:

> **Tăng cường chất lượng ảnh trong điều kiện thiếu sáng bằng mạng nơ-ron tích chập (CNN)**

Tên tiếng Anh:

> **Low-Light Image Enhancement Using a Lightweight Convolutional Neural Network**

Tên ngắn hơn:

> **Low-Light Image Enhancement Using CNN**

---

### 2. Tóm tắt nội dung đề tài

> **Đề tài nghiên cứu bài toán tăng cường chất lượng ảnh trong điều kiện thiếu sáng (Low-Light Image Enhancement -- LLIE), trong đó ảnh đầu vào có độ sáng thấp, tương phản kém, nhiễu và mất chi tiết. Nhóm sẽ xây dựng một mô hình mạng nơ-ron tích chập (CNN) nhẹ theo hướng image-to-image learning để học ánh xạ từ ảnh thiếu sáng sang ảnh có chất lượng/độ sáng tốt hơn. Mô hình dự kiến được xây dựng trên kiến trúc CNN dạng U-Net rút gọn, kết hợp các lớp convolution, activation và skip connection nhằm bảo toàn thông tin không gian và chi tiết ảnh. Nhóm sẽ huấn luyện và đánh giá mô hình trên các bộ dữ liệu low-light có cặp ảnh tương ứng, sử dụng các độ đo như PSNR và SSIM, đồng thời so sánh với một số phương pháp xử lý ảnh truyền thống. Các phương pháp LLIE hiện đại trong các công trình CVF giai đoạn 2024--2026 sẽ được sử dụng làm cơ sở tham khảo và đối chiếu, nhưng đồ án tập trung vào việc xây dựng và phân tích một giải pháp CNN cơ bản, nhẹ và phù hợp với phạm vi môn học.**

---

### 3. Yêu cầu của giảng viên

Điều giảng viên muốn:

- có bài toán thực tế;
- có paper hiện đại;
- có dataset;
- có CNN;
- có thực nghiệm;
- **không tuyên bố reproduce SOTA**;
- không cần phát minh phương pháp mới.
- Sử dụng các phương pháp/model ở mức đọ cơ bản
- Cần hiểu rõ đến mức chi tiết bài toán.
- Phải phát biểu được bài toán, phương pháp, thực nghiệm

#### Nội dung thuyết trình/báo cáo

1. Nội báo cáo đồ án gồm:

- Báo cáo chi tiết (viết dạng paper khoa học)
- Source code
- Slide trình bày

2. Nội duy thuyết trình/báo cáo

- Thông tin các thành viên và đánh giá mức độ hoàn thành
- Những lý do hoặc tầm quan trọng của bài toán
- Phát biểu bài toán
  - Input
  - Output
  - Ví dụ minh họa
- Phương pháp giải quyết
- Kết quả thực nghiệm:
  - Dataset
  - Độ đo đánh giá
  - Môi trường thực nghiệm
  - Bảng kết quả
  - Minh họa kết quả
- Đánh giá, nhận xét những ưu điểm và hạn chế
- Tài liệu tham khảo
- Nguồn tham khảo chính `CVF` - "https://openaccess.thecvf.com/menu"

### 4. Các yêu cầu hiện tại của tôi

Hiện tại trên laptop này không có GPU nên việc thực nghiệm tôi sẽ trực tiếp chạy trên Colab hoặc phần cứng khác có GPU, trên laptop này, repo này chỉ thực hiện nghiêu cứu đến bước code, sau khi có code tôi sẽ đem train, và kiểm thử sau đó đem kết quả thực nghiệm về đây.
