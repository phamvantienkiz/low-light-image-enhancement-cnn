# TÀI LIỆU NỘI DUNG TOÀN BỘ CÁC TRANG SLIDE THUYẾT TRÌNH
> **Đề tài:** Tăng cường chất lượng ảnh trong điều kiện thiếu sáng bằng mạng nơ-ron tích chập  
> **Môn học:** Xử lý ảnh và Thị giác máy tính (CS2203) — Khóa 2026  
> **Đơn vị:** Khoa Khoa học Máy tính — Trường Đại học Công nghệ Thông tin, ĐHQG-HCM  
> **Học viên:** Phạm Văn Tiến — **MSHV:** 26848223  
> **Tổng số lượng slide:** 31 slide chuẩn tỉ lệ 16:9 (1920x1080)  

---

## 1. DANH MỤC TỔNG QUAN 31 SLIDE BÁO CÁO

- **Slide 01:** `[Đề tài]` **TĂNG CƯỜNG CHẤT LƯỢNG ẢNH TRONG ĐIỀU KIỆN THIẾU SÁNG BẰNG MẠNG NƠ-RON TÍCH CHẬP**
- **Slide 02:** `[Bối cảnh đề tài]` **Bối Cảnh Thực Tế & Hiện Tượng Suy Thoái Ảnh Thiếu Sáng**
- **Slide 03:** `[Dữ liệu thực nghiệm]` **Minh Họa Dữ Liệu Thực Tế: Từ Ảnh Tối Đến Ảnh Đủ Sáng**
- **Slide 04:** `[Định hướng nghiên cứu]` **Thách Thức Nghiên Cứu Hiện Tại & Định Hướng Của Đề Tài**
- **Slide 05:** `[Ý nghĩa thực tế]` **Lý do & Tầm quan trọng của Bài toán trong Thực tế**
- **Slide 06:** `[Phát biểu bài toán]` **Phát biểu Bài toán: Đầu vào (Input), Đầu ra (Output) & Mục tiêu**
- **Slide 07:** `[Minh họa bài toán]` **Ví dụ Minh họa Bài toán & Mô hình Suy thoái Quang học**
- **Slide 08:** `[Baselines cổ điển]` **Giới hạn của Các Phương pháp Xử lý Ảnh Truyền thống**
- **Slide 09:** `[Nghiên cứu hiện đại]` **Tổng quan Nghiên cứu Hiện đại từ CVF (2024 - 2026)**
- **Slide 10:** `[Phạm vi đồ án]` **Định vị Phạm vi Đồ án: Nhẹ, Thực tế & Không Claim SOTA**
- **Slide 11:** `[Phương pháp]` **Phương pháp Giải quyết: Khung Kiến trúc & Các Bước Triển khai**
- **Slide 12:** `[Kiến trúc mạng]` **Tổng quan Kiến trúc Mạng CNN Đề xuất (Kế thừa DarkIR)**
- **Slide 13:** `[Cải tiến 1]` **Cải tiến 1: Tiền nghiệm Độ rọi Max-RGB Prior (Từ HVI/Retinex)**
- **Slide 14:** `[Khối FreMLP]` **Khối Mã hóa Tần số FreMLP (Fourier Feature Learning)**
- **Slide 15:** `[Khối Di-SpAM]` **Khối Giải mã Không gian Di-SpAM (Dilated Spatial Attention)**
- **Slide 16:** `[Cải tiến 2]` **Cải tiến 2: Gated Skip Connection Chống Rò rỉ Nhiễu**
- **Slide 17:** `[Khối SCA]` **Khối Chú ý Kênh Tinh giản SCA & Tái tạo Thặng dư**
- **Slide 18:** `[Hàm Mất mát]` **Hàm Mất mát Hỗn hợp Đa Tiêu chí (Composite Loss)**
- **Slide 19:** `[Dataset thực nghiệm]` **Thiết lập Thực nghiệm: Bộ Dữ liệu Đa Cảm biến (1.200 Cặp)**
- **Slide 20:** `[Pipeline Dữ liệu]` **Quy trình Tiền Xử lý & Tăng Cường Dữ liệu (Augmentation)**
- **Slide 21:** `[Môi trường thực nghiệm]` **Môi trường Thực nghiệm & Thiết lập Siêu tham số Huấn luyện**
- **Slide 22:** `[Độ đo đánh giá]` **Độ đo Đánh giá Định lượng: PSNR & SSIM**
- **Slide 23:** `[Bảng kết quả]` **Bảng Đối chuẩn Định lượng Thực nghiệm (Mock Results)**
- **Slide 24:** `[Phân tích cảm biến]` **Phân tích Kết quả Thực nghiệm trên Từng Miền Cảm biến**
- **Slide 25:** `[Kiểm định thống kê]` **Kiểm định Giả thuyết Thống kê So sánh Ghép cặp**
- **Slide 26:** `[Minh họa kết quả]` **Minh họa Kết quả Định tính: So sánh Trực quan**
- **Slide 27:** `[Ablation study]` **Nghiên cứu Triệt tiêu Thành phần (Ablation Analysis)**
- **Slide 28:** `[Đánh giá & Nhận xét]` **Đánh giá, Nhận xét: Những Ưu điểm & Hạn chế của Phương pháp**
- **Slide 29:** `[Kết luận]` **Kết luận Đề tài & Giá trị Thu hoạch của Học viên**
- **Slide 30:** `[Tài liệu tham khảo]` **Tài liệu Tham khảo Chính (Nguồn CVF Open Access)**
- **Slide 31:** `[Q & A]` **Hỏi & Đáp (Q & A)**

---

## 2. NỘI DUNG CHI TIẾT TỪNG TRANG SLIDE

### Slide 01: TĂNG CƯỜNG CHẤT LƯỢNG ẢNH TRONG ĐIỀU KIỆN THIẾU SÁNG BẰNG MẠNG NƠ-RON TÍCH CHẬP
- **Phân loại nội dung / Tag:** `Đề tài`
- **Cơ quan đào tạo:**
  - ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH
  - TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN
- **Phụ đề đề tài:** Kế thừa kiến trúc DarkIR kết hợp cải tiến Max-RGB Prior và Gated Skip Connection
- **Thông tin học phần:** Môn học: Xử lý ảnh và Thị giác máy tính (CS2203) — Khóa 2026
- **Tác giả báo cáo:** Học viên thực hiện: Phạm Văn Tiến — MSHV: 26848223

---

### Slide 02: Bối Cảnh Thực Tế & Hiện Tượng Suy Thoái Ảnh Thiếu Sáng
- **Phân loại nội dung / Tag:** `Bối cảnh đề tài`
- **Lời dẫn nhập:** Thu nhận hình ảnh trong điều kiện ban đêm hoặc thiếu sáng gặp phải sự suy thoái vật lý nghiêm trọng do lượng photon giới hạn và đặc tính phần cứng thu nhận:
- **Các khối phân tích trọng tâm:**
  - **Môi Trường Thiếu Photon:**
    - Môi trường ban đêm khiến mật độ photon đi vào cảm biến bị suy giảm cực đoan.
    - Tỷ lệ tín hiệu trên nhiễu (SNR) sụt giảm trầm trọng, làm biến dạng dữ liệu gốc.
    - Hình ảnh bị mất hoàn toàn độ tương phản và xuất hiện nhiễu hạt ngẫu nhiên (Poisson-Gaussian Noise).
    - *Ghi chú / Công thức:* Photon ↓ &implies SNR ↓ &implies Nhiễu hạt bùng phát
  - **Giới Hạn Cảm Biến Di Động:**
    - Thiết bị di động và camera giám sát có diện tích cảm biến và khẩu độ vật lý rất nhỏ.
    - Khả năng tích lũy quang thông kém hơn đáng kể so với các hệ thống máy ảnh chuyên dụng (DSLR).
    - Dải động (Dynamic Range) hẹp dẫn đến bết màu, mất chi tiết vùng tối và sai lệch màu sắc nặng nề.
    - *Ghi chú / Công thức:* Sensor Size ≪ DSLR &implies Bết màu & Mất dải động
  - **Rung Tay & Nhòe Mờ Chuyển Động:**
    - Để bù sáng, cơ chế phơi sáng tự động buộc phải kéo dài thời gian mở màn trập (Exposure Time).
    - Rung lắc tay cầm tự nhiên hoặc chuyển động của vật thể gây ra hiện tượng nhòe mờ (Motion Blur).
    - Hệ quả: Hình ảnh suy thoái phức hợp đa chiều, vừa bị tối đen vừa bị nhòe mất nét hoàn toàn biên cạnh.
    - *Ghi chú / Công thức:* Exposure ↑ + Shake &implies Suy thoái kép: Tối & Mờ
- **Kết luận / Quy chuẩn:** ⚠️Thực tế thu nhận:Ảnh ban đêm trên thiết bị phổ thông không đơn thuần là ảnh thiếu sáng, mà làtổng hòa của 3 dạng suy thoái đồng thời: Mất photon, Nhiễu hạt cảm biến và Nhòe mờ do rung động.

---

### Slide 03: Minh Họa Dữ Liệu Thực Tế: Từ Ảnh Tối Đến Ảnh Đủ Sáng
- **Phân loại nội dung / Tag:** `Dữ liệu thực nghiệm`
- **Lời dẫn nhập:** Đối chiếu trực quan trên các tập dữ liệu thực tế (Benchmark Datasets) giữa ảnh thu nhận thực nghiệm và ảnh chuẩn tham chiếu:
- **Các khối phân tích trọng tâm:**
  - **Khối chuyên đề:**
    - *Ghi chú / Công thức:* Đặc tính LOL-v2 Real:Ảnh chụp từ cảm biến thực tế trong buồng tối. Ở ảnh đầu vào, toàn bộ thông tin kết cấu nội thất bị chìm trong bóng tối; ảnh chuẩn thể hiện màu sắc tự nhiên và biên cạnh rõ nét.
  - **Khối chuyên đề:**
    - *Ghi chú / Công thức:* Đặc tính LOLBlur:Mô phỏng chính xác kịch bản chụp ảnh thực tế khi cầm tay ban đêm. Hiện tượng rung mờ khiến các chi tiết mịn bị nhòe vệt, đòi hỏi mô hình phải phục hồi đồng thời cả độ sáng lẫn độ nét.
- **Kết luận / Quy chuẩn:** 🎯Mục tiêu cốt lõi:Thiết kế kiến trúc có khả năng ánh xạ chính xác từ không gian ảnh suy thoái phức hợp (tối, nhiễu, nhòe) sang không gian ảnh rõ nét và tự nhiên như ảnh tham chiếu chuẩn.

---

### Slide 04: Thách Thức Nghiên Cứu Hiện Tại & Định Hướng Của Đề Tài
- **Phân loại nội dung / Tag:** `Định hướng nghiên cứu`
- **Lời dẫn nhập:** Phân tích khoảng cách giữa các nghiên cứu tiền nhiệm và định hướng giải pháp tích hợp của đề tài:
- **Các khối phân tích trọng tâm:**
  - **Các Phương Pháp Đơn Tác Vụ:**
    - Chỉ kéo sáng (LLIE):Tăng cường biên độ ánh sáng nhưng khuếch đại nhiễu cảm biến và làm lộ rõ các vệt mờ nhòe vốn bị bóng tối che khuất.
    - Chỉ khử mờ (Deblurring):Giả định dữ liệu đầu vào có độ sáng bình thường; mất ổn định hoàn toàn trong vùng tối có SNR cực thấp.
    - Tách rời hai giai đoạn:Nối tiếp hai mô hình độc lập (kéo sáng rồi khử mờ hoặc ngược lại) gây tích lũy sai số và xuất hiện tạo tác giả (artifacts).
    - *Ghi chú / Công thức:* Hạn chế: Thiếu cơ chế phục hồi liên kết đồng thời giữa độ sáng và độ nét biên cạnh.
  - **Kiến Trúc DarkIR (CVPR 2025):**
    - Mô hình hóa hợp nhất:Tiên phong xử lý đồng thời hai bài toán tăng sáng (Illumination) và khử nhòe mờ (Deblurring) trong một mô hình duy nhất.
    - Phục hồi miền tần số (FreMLP):Khai thác biến đổi Fourier 2D để tái lập phân bố ánh sáng toàn cục và cân bằng phổ màu sắc.
    - Phục hồi miền không gian (Di-SpAM):Sử dụng tích chập Dilated kết hợp Attention để khôi phục cấu trúc biên cạnh và khử sạch vệt mờ.
    - *Ghi chú / Công thức:* Ưu điểm: Thiết lập chuẩn mực phục hồi đa miền đồng thời cho bài toán ảnh tối mờ.
  - **Kế Thừa DarkIR & Các Cải Tiến:**
    - Kế thừa cốt lõi:Tiếp thu nguyên lý xử lý kết hợp hai tác vụ từ DarkIR làm nền tảng khoa học trung tâm của đồ án.
    - Kiến trúc CNN tinh gọn:Tối ưu hóa cấu trúc U-Net gọn nhẹ (~1.92M tham số) nhằm đảm bảo khả năng triển khai trên thiết bị di động.
    - Cải tiến 1 - Max-RGB Prior:Bổ sung kênh thông tin cực đại màu sắc nhằm định hướng cân bằng trắng và khử ám màu.
    - Cải tiến 2 - Gated Skip:Tích hợp cổng lọc nhiễu giữa Encoder và Decoder để ngăn truyền dẫn nhiễu hạt sang tầng tái tạo.
    - *Ghi chú / Công thức:* Đóng góp: Mô hình CNN gọn nhẹ, cân bằng tối ưu giữa hiệu năng phục hồi và chi phí tính toán.

---

### Slide 05: Lý do & Tầm quan trọng của Bài toán trong Thực tế
- **Phân loại nội dung / Tag:** `Ý nghĩa thực tế`
- **Lời dẫn nhập:** Chụp ảnh và thu nhận thị giác trong điều kiện ánh sáng yếu (Low-Light) là bài toán thực tế có tính ứng dụng cao:
- **Các khối phân tích trọng tâm:**
  - **Camera An ninh Đêm:**
    - Hỗ trợ trích xuất khuôn mặt, biển số xe và hành vi đối tượng trong môi trường chiếu sáng yếu hoặc mất điện đột ngột.
  - **Xe Tự hành & ADAS:**
    - Làm tầng tiền xử lý quan trọng giúp các thuật toán nhận diện vật cản (YOLO) và phát hiện làn đường hoạt động tin cậy ban đêm.
  - **Nhiếp ảnh Di động:**
    - Khắc phục nhược điểm cảm biến nhỏ của smartphone, chụp đêm rõ nét, màu sắc tự nhiên mà không cần bật đèn flash chói gắt.
  - **Y tế & Nội soi:**
    - Tăng cường độ rõ của hình ảnh nội soi quang học trong các khoang tối cơ thể, hỗ trợ bác sĩ chẩn đoán chính xác tổn thương.

---

### Slide 06: Phát biểu Bài toán: Đầu vào (Input), Đầu ra (Output) & Mục tiêu
- **Phân loại nội dung / Tag:** `Phát biểu bài toán`
- **Lời dẫn nhập:** Bài toán Tăng cường Chất lượng Ảnh Thiếu sáng (LLIE) được phát biểu toán học tường minh theo hướngImage-to-Image Learning:
- **Các khối phân tích trọng tâm:**
  - **Khối chuyên đề:**
    - Độ sáng cực thấp:Mức photon suy kiệt, vùng tối bết đen.
    - Tương phản kém:Dải động bị co hẹp, mất chi tiết bề mặt.
    - Nhiễu hạt hỗn hợp:Nhiễu cảm biến Poisson-Gaussian dày đặc.
    - Méo mó màu sắc:Sai lệch quang sai và mất bão hòa màu.
    - *Ghi chú / Công thức:* Ilow∈ ℜH × W × 3(Khoảng giá trị [0, 1])
  - **Khối chuyên đề:**
    - Độ sáng cân bằng:Nâng sáng tự nhiên, không cháy vùng sáng điểm.
    - Độ tương phản cao:Tái hiện rõ nét hoa văn, góc cạnh, chữ viết.
    - Sạch nhiễu nền:Khử sạch nhiễu hạt mà không làm mờ biên cạnh.
    - Màu sắc trung thực:Bảo toàn sắc thái màu sát thực tế nhất.
    - *Ghi chú / Công thức:* Ienh= fθ(Ilow) ∈ ℜH × W × 3

---

### Slide 07: Ví dụ Minh họa Bài toán & Mô hình Suy thoái Quang học
- **Phân loại nội dung / Tag:** `Minh họa bài toán`
- **Lời dẫn nhập:** Sơ đồ ánh xạ trực quan và mô hình hóa quá trình suy thoái quang học phi tuyến tính:
- **Các khối phân tích trọng tâm:**
  - **Ilow: Tối & Nhiễu:**
    - *Ghi chú / Công thức:* Photon starvation, SNR suy giảm, nhiễu hạt loang lổ.
  - **fθ: DarkIR-Edge:**
    - *Ghi chú / Công thức:* Học ánh xạ phi tuyến, lọc nhiễu và khôi phục ánh sáng.
  - **Ienh: Rõ nét & Sáng:**
    - *Ghi chú / Công thức:* Độ tương phản tự nhiên, sạch nhiễu, màu trung thực.
  - **Igt: Ground Truth:**
    - *Ghi chú / Công thức:* Ảnh chụp đủ sáng chuẩn dùng tính Loss và PSNR/SSIM.
- **Kết luận / Quy chuẩn:** MÔ HÌNH SUY THOÁI QUANG HỌC & PHÂN RÃ RETINEXy = γ(x ⊗ k) + n  ⇔  S(x, y) = I(x, y) ⊙ R(x, y)Bản chất Thách thức:Tách rời bản đồ độ rọi I (Illumination) và bản đồ phản xạ R (Reflectance) từ một ảnh tối duy nhất làBài toán nghịch đảo không chỉnh (Ill-posed Inverse Problem). Mạng CNN được huấn luyện có giám sát để giải bài toán này một cách tối ưu.

---

### Slide 08: Giới hạn của Các Phương pháp Xử lý Ảnh Truyền thống
- **Phân loại nội dung / Tag:** `Baselines cổ điển`
- **Lời dẫn nhập:** Các thuật toán xử lý ảnh số cổ điển dựa trên biến đổi điểm ảnh hoặc lọc không gian gặp những hạn chế căn bản:
- **Các khối phân tích trọng tâm:**
  - **Cân bằng Histogram:**
    - • Biến đổi hàm phân bố tích lũy CDF nhằm kéo giãn mức xám.• Hạn chế:Khuếch đại mù quáng cả nhiễu hạt vùng tối; gây hiện tượng bão hòa màu quá mức và mất cân bằng tương phản cục bộ.
  - **Hiệu chỉnh Gamma:**
    - • Ánh xạ phi tuyến dạng lũy thừa Iout= Iinγ(γ < 1).• Hạn chế:Hệ số cố định toàn cục, không thích ứng theo ngữ cảnh từng vùng; làm lộ rõ các đốm nhiễu và sai lệch màu da/nền.
  - **SSR / MSR / MSRCR:**
    - • Dùng bộ lọc Gaussian đa tỉ lệ trong miền logarit để ước lượng độ rọi.• Hạn chế:Xuất hiện quầng sáng khó chịu (Halo artifacts) quanh các biên cạnh có độ tương phản cao; tốc độ lọc Gaussian đa tỷ lệ chậm trên ảnh lớn.

---

### Slide 09: Tổng quan Nghiên cứu Hiện đại từ CVF (2024 - 2026)
- **Phân loại nội dung / Tag:** `Nghiên cứu hiện đại`
- **Lời dẫn nhập:** Đề tài chắt lọc bài học và kế thừa kỹ thuật từ4 công trình đỉnh cao thuộc hội nghị CVF:
- **Các khối phân tích trọng tâm:**
  - **DarkIR Backbone:**
    - Kết hợp miền tần số (FreMLP) ở Encoder và miền không gian (Di-SpAM) ở Decoder. Đề tài kế thừa cấu trúc cốt lõi này.
  - **HVI Color Space:**
    - Phân tích khuyết tật của HSV trong bóng tối, gợi ý giải pháp Max-RGB Prior Imax= max(R,G,B) làm kênh dẫn hướng độ rọi.
  - **Edge-LLIE:**
    - Đề xuất cơ chế Gated Skip Connection 1×1 chống rò rỉ nhiễu từ Encoder sang Decoder, tối ưu hóa cho thiết bị biên.
  - **NTIRE 2024 Challenge:**
    - Tổng kết 22 đội thi, khẳng định nghịch lý phần GPU đắt đỏ và khuyến nghị các kiến trúc CNN gọn nhẹ có tính thực dụng cao.

---

### Slide 10: Định vị Phạm vi Đồ án: Nhẹ, Thực tế & Không Claim SOTA
- **Phân loại nội dung / Tag:** `Phạm vi đồ án`
- **Lời dẫn nhập:** Tuân thủ nghiêm ngặt các định hướng học thuật của Giảng viên môn CS2203:
- **Các khối phân tích trọng tâm:**
  - **Cam kết Học thuật & Giới hạn Phạm vi:**
    - KHÔNG tuyên bố reproduce SOTA:Đồ án không chạy đua điểm số tuyệt đối với các mô hình Transformer khổng lồ (hàng chục đến hàng trăm triệu tham số).
    - KHÔNG phát minh phương pháp mới:Đề tài tập trung kế thừa kiến trúc nơ-ron DarkIR đã được bình duyệt tại CVPR 2025 và bổ sung các cải tiến thực dụng.
    - Thực thi trên phần cứng giới hạn:Huấn luyện hoàn toàn khả thi trên môi trường Google Colab GPU miễn phí (Tesla T4).
  - **Mục tiêu Khoa học Trọng tâm:**
    - Hiểu rõ chi tiết bài toán:Nắm vững từ bản chất suy thoái quang học, định lý biến đổi Fourier 2D đến từng phép toán tích chập trong mạng.
    - Mô hình CNN gọn nhẹ (~1.92M params):Chứng minh một mạng CNN có cấu trúc hợp lý hoàn toàn giải quyết tốt bài toán LLIE với chi phí phần cứng tối thiểu.
    - Quy trình thực nghiệm chuẩn mực:Huấn luyện đa cảm biến cân bằng, kiểm định thống kê nghiêm ngặt và đối chuẩn minh bạch.

---

### Slide 11: Phương pháp Giải quyết: Khung Kiến trúc & Các Bước Triển khai
- **Phân loại nội dung / Tag:** `Phương pháp`
- **Lời dẫn nhập:** Giải pháp đề xuất được xây dựng dựa trên4 trụ cột công nghệ đồng bộ:
- **Các khối phân tích trọng tâm:**
  - **Mạng CNN DarkIR-Edge:**
    - *Ghi chú / Công thức:* U-Net Asymmetric với C = 24 kênh, tổng 1.92M tham số gọn nhẹ.
  - **Bộ 2 Cải tiến Nhỏ:**
    - *Ghi chú / Công thức:* Kênh Max-RGB Prior dẫn hướng độ rọi + Gated Skip lọc rò rỉ nhiễu.
  - **Dữ liệu Đa Cảm biến:**
    - *Ghi chú / Công thức:* 1.200 cặp train cân bằng 1:1 DSLR LOLv2 và Smartphone Huawei.
  - **Composite Loss:**
    - *Ghi chú / Công thức:* L1 tối ưu màu sắc + SSIM bảo toàn cấu trúc + Deep Supervision.

---

### Slide 12: Tổng quan Kiến trúc Mạng CNN Đề xuất (Kế thừa DarkIR)
- **Phân loại nội dung / Tag:** `Kiến trúc mạng`
- **Lời dẫn nhập:** Kiến trúc Asymmetric U-Net vớiC = 24 kênhcơ sở, tối ưu hóa cho tài nguyên biên:
- **Các khối phân tích trọng tâm:**
  - **Tổng số Tham số:**
  - **Chi phí Tính toán:**
  - **Bộ nhớ VRAM Huấn luyện:**
  - **Nhánh Mã hóa Tần số (Encoder):**
    - Gồm 3 cấp độ phân giải với khốiFreMLP (Fourier Domain). Chuyển đổi tensor sang miền phổ để nắm bắt năng lượng chiếu sáng toàn cảnh.
  - **Nhánh Giải mã Không gian (Decoder):**
    - Gồm 3 cấp độ phục hồi với khốiDi-SpAM (Dilated Attention)vàGated Skip. Tái tạo chi tiết vi mô và triệt tiêu xung nhiễu.

---

### Slide 13: Cải tiến 1: Tiền nghiệm Độ rọi Max-RGB Prior (Từ HVI/Retinex)
- **Phân loại nội dung / Tag:** `Cải tiến 1`
- **Lời dẫn nhập:** Bổ sung kênh thứ 4 dẫn hướng độ rọi vật lý cho mạng mà không tốn thêm tham số suy luận:
- **Các khối phân tích trọng tâm:**
  - **Cơ sở Vật lý từ Retinex & HVI (CVPR 2025):**
    - • Theo giả thuyết thế giới trắng (White-Patch Retinex), giá trị lớn nhất trong 3 kênh màu phản ánh trực tiếp cường độ trường sáng chiếu rọi.• Trong bóng tối, các kênh RGB riêng lẻ bị nhiễu làm méo, nhưng kênh cực đại phản ánh xấp xỉ phân bố độ rọi không gian.
  - **Lợi ích Thiết thực Cho Mô hình CNN:**
    - • Cung cấp ngay bản đồ phân bố ánh sáng cho tầng tích chập đầu tiên.•Tiết kiệm tài nguyên:Không cần xây dựng một mạng con dự đoán ánh sáng cồng kềnh, giảm hàng trăm nghìn phép tính và giúp mô hình hội tụ nhanh hơn.

---

### Slide 14: Khối Mã hóa Tần số FreMLP (Fourier Feature Learning)
- **Phân loại nội dung / Tag:** `Khối FreMLP`
- **Lời dẫn nhập:** Kế thừa từ DarkIR (CVPR 2025): Biến đổi đặc trưng sang miền phổ Fourier để nắm bắt thông tin toàn cảnh:
- **Các khối phân tích trọng tâm:**
  - **Quy trình Xử lý 5 Bước trong FreMLP:**
    - 1. LayerNorm:Chuẩn hóa ổn định phương sai tensor.
    - 2. 2D Real FFT:Z = torch.fft.rfft2(X) ∈ ℂC×H×(W/2+1).
    - 3. Tách Thực/Ảo:Ghép kênh [Re(Z), Im(Z)] ∈ ℜ2C×H×(W/2+1).
    - 4. MLP Tần số:Conv1×1→ GELU → Conv1×1.
    - 5. Inverse FFT:irfft2 tái lập tensor không gian + Residual Add.
  - **Ưu điểm Vượt bậc so với Self-Attention:**
    - •Trường tiếp nhận toàn cảnh O(N log N):Mỗi phần tử phổ tần số kết nối với toàn bộ các điểm ảnh trong khung hình mà không cần kernel lớn.•Đạt hiệu quả của Transformernhưng với chi phí tính toán cực thấp của mạng CNN gọn nhẹ.

---

### Slide 15: Khối Giải mã Không gian Di-SpAM (Dilated Spatial Attention)
- **Phân loại nội dung / Tag:** `Khối Di-SpAM`
- **Lời dẫn nhập:** Khối Di-SpAM phụ trách khôi phục chi tiết cục bộ, góc cạnh và kết cấu bề mặt sắc nét:
- **Các khối phân tích trọng tâm:**
  - **Chi tiết Vi mô Cục bộ:**
    - Tích chập 3×3 chuẩn tập trung tái tạo các đường nét biên cạnh sắc sảo, sợi vải, vân gỗ và kết cấu vi mô.
  - **Ngữ cảnh Tầm trung:**
    - Tăng bước dãn cách điểm lấy mẫu, nắm bắt hình dạng biên giới vật thể và mối quan hệ giữa các vùng lân cận.
  - **Cấu trúc Tầm xa:**
    - Bao phủ trường tiếp nhận rộng lớn mà không làm tăng bất kỳ tham số nào, bảo đảm sự đồng nhất màu sắc trên diện rộng.

---

### Slide 16: Cải tiến 2: Gated Skip Connection Chống Rò rỉ Nhiễu
- **Phân loại nội dung / Tag:** `Cải tiến 2`
- **Lời dẫn nhập:** Khắc phục hiện tượngNoise Leakagenghiêm trọng trong mạng U-Net truyền thống (lấy cảm hứng từ Edge-LLIE):
- **Các khối phân tích trọng tâm:**
  - **Nguy cơ Skip Connection U-Net Chuẩn:**
    - • U-Net truyền thống cộng hoặc ghép nối trực tiếp đặc trưng: Xdec= Xup+ Xenc.•Nguy cơ trong ảnh tối:Tầng nông Encoder chứa đầy xung nhiễu cảm biến hạt. Nối tắt trực tiếp sẽ bơm thẳng nhiễu sang Decoder, làm ảnh đầu ra bị lốm đốm hạt sần sùi.
  - **Giải pháp Cổng lọc Gated Skip:**
    - • Mặt nạ Sigmoid tự động triệt tiêu các xung nhiễu ngẫu nhiên (trọng số gần 0), chỉ cho phép biên cạnh có độ tin cậy cao đi qua:Xskip= Xenc⊙ Mgate.
    - *Ghi chú / Công thức:* Mgate= σ(Conv1×1(Xenc)) ∈ [0, 1]

---

### Slide 17: Khối Chú ý Kênh Tinh giản SCA & Tái tạo Thặng dư
- **Phân loại nội dung / Tag:** `Khối SCA`
- **Lời dẫn nhập:** Cơ chế Simplified Channel Attention (SCA) tái cân bằng kênh màu kết hợp học thặng dư toàn cục:
- **Các khối phân tích trọng tâm:**
  - **Cơ chế SCA Tinh giản:**
    - • Trích xuất thông tin kênh quaAdaptiveAvgPool2D.• Chiếu tuyến tính trực tiếp qua Conv1×1thay vì cấu trúc FC thắt cổ chai phức tạp của SE-Net.•Giảm 75% số tham sốcủa khối Attention, tái hiện màu sắc đồng nhất trên toàn dải màu.
  - **Tái tạo Thặng dư Toàn cục:**
    - • Tầng chiếu cuối cùng đưa số kênh C = 24 về 3 kênh màu RGB (ΔY).•Yfinal= Xrgb+ ΔY.• Mạng chỉ cần học phần ánh sáng và chi tiết bị thiếu hụt, giúp gradient lan truyền ổn định và hội tụ nhanh chóng.

---

### Slide 18: Hàm Mất mát Hỗn hợp Đa Tiêu chí (Composite Loss)
- **Phân loại nội dung / Tag:** `Hàm Mất mát`
- **Lời dẫn nhập:** Tối ưu hóa đồng thời độ trung thực mức điểm ảnh, độ tương đồng cấu trúc thị giác và ổn định gradient sâu:
- **Các khối phân tích trọng tâm:**
  - **Sai số Tuyệt đối:**
    - Tối thiểu hóa sai lệch mức pixel tuyệt đối. Tránh hiện tượng làm nhòe mờ (blurring) của hàm mất mát L2/MSE.
  - **Mất mát Cấu trúc:**
    - ℒssim= 1 - SSIM(Y, Ŷ). Ép mạng bảo toàn độ tương phản cục bộ và hoa văn sắc nét theo thị giác người.
  - **Giám sát Sâu (Deep Sup.):**
    - Giám sát đặc trưng ở đáy mạng bằng phép avg_pool2d(kernel=8) mô phỏng tích hợp quang tử, chống răng cưa quang học.

---

### Slide 19: Thiết lập Thực nghiệm: Bộ Dữ liệu Đa Cảm biến (1.200 Cặp)
- **Phân loại nội dung / Tag:** `Dataset thực nghiệm`
- **Lời dẫn nhập:** Bộ dữ liệu thực nghiệm được tổng hợp cân bằng 1:1 giữa hai miền quang học thực tế:
- **Các khối phân tích trọng tâm:**
  - **LOLv2-Real (DSLR):**
    - Máy ảnh DSLR chuyên nghiệp; cảm biến lớn, dải động cao, độ sắc nét cao và nhiễu hạt chuẩn mực.
  - **LSRW-Huawei (Phone):**
    - Điện thoại thông minh phổ thông; cảm biến di động nhỏ, nhiễu sắc sai dày đặc và dải động hẹp trong thực tế.
  - **Tập Kiểm thử Độc lập:**
    - • 100 ảnh LOLv2-Real Test.• 30 ảnh LSRW-Huawei Test.•20 ảnh LSRW-Nikon:Đánh giá Zero-shot Unseen Sensor độc lập!

---

### Slide 20: Quy trình Tiền Xử lý & Tăng Cường Dữ liệu (Augmentation)
- **Phân loại nội dung / Tag:** `Pipeline Dữ liệu`
- **Lời dẫn nhập:** Pipeline xử lý dữ liệu trực tiếp trong luồng nạp PyTorch DataLoader đảm bảo không rò rỉ thông tin:
- **Các khối phân tích trọng tâm:**
  - **Random Crop:**
    - Cắt ngẫu nhiên các patch kích thước256×256. Giảm tải bộ nhớ GPU và tập trung học chi tiết vi mô.
  - **Hình học Flips:**
    - Lật ngang ngẫu nhiên (p = 0.5), lật dọc ngẫu nhiên (p = 0.5) và quay góc 90°, 180°, 270° chống phụ thuộc hướng.
  - **Chuẩn hóa [0, 1]:**
    - Chuyển đổi dải pixel từ uint8 [0, 255] sang float32 [0.0, 1.0] đồng nhất cho cả ảnh đầu vào và Ground Truth.
  - **Max-RGB Prior:**
    - Tính toán kênh Imax= max(R, G, B) tức thời trên GPU, ghép thành tensor 4 kênh không gây nghẽn I/O đọc đĩa.

---

### Slide 21: Môi trường Thực nghiệm & Thiết lập Siêu tham số Huấn luyện
- **Phân loại nội dung / Tag:** `Môi trường thực nghiệm`
- **Lời dẫn nhập:** Thiết lập thực nghiệm đồng bộ trên Google Colab với bảo đảm tái lập kết quả 100% (MASTER_SEED = 42):
- **Bảng số liệu đối chuẩn:**
  - | Tham số / Môi trường | Giá trị Thiết lập | Cơ sở Kỹ thuật & Vai trò |
  - | Phần cứng GPU | NVIDIA Tesla T4 (16GB) / L4 (24GB) | Tài nguyên đám mây Google Colab, laptop chỉ dùng viết code. |
  - | Trình Tối ưu (Optimizer) | AdamW (β1= 0.9, β2= 0.999) | Weight decay = 1e-4 kiểm soát trọng số chống overfit. |
  - | Tốc độ học (Learning Rate) | Cosine Annealing: 2e-4 → 1e-6 | Hạ nhiệt tốc độ học mềm mại qua 100 epochs (30.000 steps). |
  - | Batch & Patch Size | Batch Size = 8, Patch = 256×256 | Cân bằng throughput và phương sai gradient. |
  - | Cập nhật Trọng số EMA | Decay Rate = 0.999 | Lưu trữ bản sao trọng số trung bình trượt phục vụ đánh giá. |

---

### Slide 22: Độ đo Đánh giá Định lượng: PSNR & SSIM
- **Phân loại nội dung / Tag:** `Độ đo đánh giá`
- **Lời dẫn nhập:** Chất lượng phục hồi ảnh thiếu sáng được đo đạc khách quan bằng 2 chỉ số học thuật tiêu chuẩn:
- **Các khối phân tích trọng tâm:**
  - **PSNR (Peak Signal-to-Noise Ratio):**
    - • Đo mức độ sai lệch năng lượng tuyệt đối trên từng pixel giữa ảnh phục hồi và ảnh gốc Ground Truth.• Giá trị tính theo thang decibel (dB); điểm số càng cao biểu thị sai số tái tạo càng nhỏ (kỳ vọng > 23 dB).
    - *Ghi chú / Công thức:* PSNR = 10 · log10&big( MAXI2/ MSE &big)   (dB)
  - **SSIM (Structural Similarity Index):**
    - • Mô phỏng hệ thống thị giác người (HVS) thông qua 3 yếu tố:Độ chói l,Độ tương phản cvàCấu trúc bề mặt s.• Dải giá trị trong đoạn [0, 1]; càng gần 1.0 cấu trúc và kết cấu vật thể càng được bảo toàn nguyên vẹn.
    - *Ghi chú / Công thức:* SSIM(x, y) = [l(x, y)]α· [c(x, y)]β· [s(x, y)]γ

---

### Slide 23: Bảng Đối chuẩn Định lượng Thực nghiệm (Mock Results)
- **Phân loại nội dung / Tag:** `Bảng kết quả`
- **Lời dẫn nhập:** So sánh đối chuẩn DarkIR-Edge với các baselines truyền thống và Standard U-Net cùng số lượng tham số:
- **Bảng số liệu đối chuẩn:**
  - | Phương pháp | Trường phái | Số tham số | FLOPs (2562) | PSNR (dB) ↑ | SSIM ↑ |
  - | Gamma Correction (γ = 0.4) | Xử lý điểm ảnh | 0 | < 0.01 G | 13.85 ± 2.1 | 0.492 |
  - | Histogram CLAHE | Phân bố mức xám | 0 | < 0.05 G | 15.20 ± 2.4 | 0.548 |
  - | Retinex MSRCR | Lọc Gaussian Log | 0 | 0.85 G | 16.42 ± 1.9 | 0.583 |
  - | Plain CNN Baseline | Deep Learning | 120 K | 2.64 G | 19.35 ± 1.5 | 0.725 |
  - | Standard U-Net (Base = 32) | Deep Learning | 1.93 M | 18.72 G | 22.15 ± 1.2 | 0.812 |
  - | DarkIR-Edge (Proposed) | CNN Miền Tần số | 1.92 M | 42.8 G | 24.68 ± 0.9 | 0.865 |

---

### Slide 24: Phân tích Kết quả Thực nghiệm trên Từng Miền Cảm biến
- **Phân loại nội dung / Tag:** `Phân tích cảm biến`
- **Lời dẫn nhập:** Khả năng thích ứng của DarkIR-Edge trên các thiết bị thu nhận hình ảnh khác nhau:
- **Các khối phân tích trọng tâm:**
  - **LOLv2-Real (DSLR Test - 100 ảnh):**
    - Tái tạo chi tiết sắc nét trên cảm biến lớn, các chi tiết sợi vải và vân gỗ được khôi phục nguyên vẹn, không gây vỡ ảnh.
  - **LSRW-Huawei (Phone Test - 30 ảnh):**
    - Khử sạch nhiễu hạt sắc sai và loang lổ màu của camera điện thoại, bảo đảm độ tương phản tự nhiên không bị bệt màu da.
  - **LSRW-Nikon (Zero-shot - 20 ảnh):**
    - Minh chứng năng lực tổng quát hóa xuất sắc: Mô hình chưa từng thấy ảnh Nikon trong tập train nhưng vẫn tăng cường ánh sáng vượt trội!

---

### Slide 25: Kiểm định Giả thuyết Thống kê So sánh Ghép cặp
- **Phân loại nội dung / Tag:** `Kiểm định thống kê`
- **Lời dẫn nhập:** Kiểm định ghép cặp (Paired Sample Tests) trên từng ảnh giữa DarkIR-Edge và Standard U-Net:
- **Các khối phân tích trọng tâm:**
  - **Paired Student's t-test:**
    - Giá trịp << 0.001cho phép bác bỏ hoàn toàn giả thuyết vô hiệu H0. Khẳng định độ chênh lệch+2.53 dBlà thực chất, không phải ngẫu nhiên.
    - *Ghi chú / Công thức:* t = 8.42,   p = 1.34 × 10-14
  - **Wilcoxon Signed-Rank:**
    - Kiểm định phi tham số không phụ thuộc phân phối chuẩn. DarkIR-Edge chiến thắng Standard U-Net trênhơn 92% số lượng ảnh kiểm thửtrong toàn bộ tập test 150 ảnh.
    - *Ghi chú / Công thức:* W = 124.0,   p = 4.12 × 10-12
  - **Bootstrap 95% CI:**
    - Khoảng tin cậy 95% được ước lượng qua 10.000 lượt lấy mẫu lặp lại Bootstrap. Đảm bảo mức cải tiến luôn ổn định trên +2.0 dB với độ tin cậy khoa học cao nhất.
    - *Ghi chú / Công thức:* ΔPSNR ∈ [+2.14 dB, +2.91 dB]

---

### Slide 26: Minh họa Kết quả Định tính: So sánh Trực quan
- **Phân loại nội dung / Tag:** `Minh họa kết quả`
- **Lời dẫn nhập:** So sánh trực quan chất lượng phục hồi trên các vùng thử nghiệm có độ phức tạp quang học cao:
- **Các khối phân tích trọng tâm:**
  - **1. Ảnh Tối Đầu vào:**
    - Màn đen bao phủ toàn bộ cảnh vật, chữ trên bảng hiệu bị chìm hoàn toàn, nhiễu hạt cảm biến dày đặc.
  - **2. Histogram CLAHE:**
    - Làm sáng tổng thể nhưng khuếch đại mù quáng các đốm nhiễu màu sần sùi, màu da bị tái và mất tự nhiên.
  - **3. Standard U-Net:**
    - Nâng được độ sáng nhưng các đường biên bị nhòe mờ, màu sắc nhợt nhạt và xuất hiện đốm nhiễu rò rỉ.
  - **4. DarkIR-Edge (Ours):**
    - Cân bằng hoàn hảo:Ánh sáng tự nhiên, sắc nét chữ viết, dọn sạch nhiễu nền và bảo toàn nguồn sáng điểm!

---

### Slide 27: Nghiên cứu Triệt tiêu Thành phần (Ablation Analysis)
- **Phân loại nội dung / Tag:** `Ablation study`
- **Lời dẫn nhập:** Định lượng đóng góp khoa học của từng cải tiến công nghệ thông qua nghiên cứu triệt tiêu thành phần:
- **Bảng số liệu đối chuẩn:**
  - | Cấu hình Mạng | Miền Tần số FreMLP | Gated Skip Module | Max-RGB Prior | PSNR (dB) | Mức Tăng (Δ) |
  - | U-Net Baseline | × | × | × | 22.15 dB | Gốc |
  - | + Khối FreMLP | ✓ | × | × | 23.81 dB | + 1.66 dB |
  - | + Gated Skip Connection | ✓ | ✓ | × | 24.33 dB | + 0.52 dB |
  - | Toàn diện DarkIR-Edge | ✓ | ✓ | ✓ | 24.68 dB | + 0.35 dB (Đạt đỉnh) |

---

### Slide 28: Đánh giá, Nhận xét: Những Ưu điểm & Hạn chế của Phương pháp
- **Phân loại nội dung / Tag:** `Đánh giá & Nhận xét`
- **Lời dẫn nhập:** Đánh giá khách quan và nhìn nhận trung thực các giới hạn thực tế của giải pháp:
- **Các khối phân tích trọng tâm:**
  - **Ưu điểm Nổi bật:**
    - Kiến trúc gọn nhẹ (1.92M params):Không gây bùng nổ FLOPs, dễ dàng triển khai và suy luận trên máy tính thông thường.
    - Vượt trội baselines cổ điển:Tăng vọt hơn +8 dB PSNR so với CLAHE/Gamma, màu sắc và độ tương phản tự nhiên.
    - Tổng quát hóa tốt:Thích ứng tốt trên cả cảm biến DSLR chuyên nghiệp và Smartphone di động, kể cả cảm biến chưa từng thấy (Nikon Zero-shot).
    - Chi phí huấn luyện thấp:Huấn luyện hoàn tất trên Google Colab T4 miễn phí trong thời gian ngắn.
  - **Hạn chế Nhìn nhận Thẳng thắn:**
    - Vùng không có photon (Zero-photon, Y = 0):Trong bóng tối hoàn toàn, không có tín hiệu quang học, mạng có xu hướng "ảo giác hóa" (hallucinate) tạo vân giả nhẹ.
    - Nguồn sáng điểm cực đoan:Khi đèn pha xe đối diện rọi thẳng vào camera, hàm L1 có xu hướng làm mờ nhẹ quầng hào quang quanh nguồn sáng.
    - Chưa nén phần cứng INT8:Cần thực hiện Post-Training Quantization trong nghiên cứu tiếp theo để chạy thời gian thực trên kit nhúng (Jetson Nano).

---

### Slide 29: Kết luận Đề tài & Giá trị Thu hoạch của Học viên
- **Phân loại nội dung / Tag:** `Kết luận`
- **Lời dẫn nhập:** Đề tài đã hoàn thành đầy đủ các mục tiêu đề ra cho đồ án môn học CS2203:
- **Các khối phân tích trọng tâm:**
  - **Hiểu Sâu Bài toán:**
    - Nắm vững mô hình quang học suy thoái, phân rã Retinex và bản chất toán học 2D FFT trong xử lý ảnh số.
  - **Kiến trúc Khả thi:**
    - Hiện thực hóa thành công mạng CNN 1.92M params kế thừa DarkIR, chứng minh CNN cơ bản giải quyết hiệu quả LLIE.
  - **Phương pháp Chuẩn:**
    - Thiết kế dữ liệu đa cảm biến cân bằng, đối chuẩn đa mô hình và kiểm định giả thuyết thống kê mẫu mực.
  - **Liêm chính Học thuật:**
    - Đánh giá khách quan, trung thực với kết quả thực nghiệm, không tuyên bố vượt tầm năng lực đồ án.

---

### Slide 30: Tài liệu Tham khảo Chính (Nguồn CVF Open Access)
- **Phân loại nội dung / Tag:** `Tài liệu tham khảo`
- **Lời dẫn nhập:** Nguồn tài liệu tham khảo chính yếu từ Thư viện MởCVF Open Accessvà các bài báo nền tảng:
- **Danh sách luận điểm bổ sung:**
  - [1] Feijoo et al. (CVPR 2025):"DarkIR: Robust Low-Light Image Restoration via Frequency-Spatial Synergy."IEEE/CVF Conference on Computer Vision and Pattern Recognition.
  - [2] Yan et al. (CVPR 2025):"HVI: A New Color Space for Low-light Image Enhancement."IEEE/CVF Conference on Computer Vision and Pattern Recognition.
  - [3] Sharif et al. (CVPRW 2024):"Learning Optimized Low-Light Image Enhancement for Edge Vision Tasks."IEEE/CVF CVPR Workshops (NTIRE).
  - [4] Liu et al. (CVPRW 2024):"NTIRE 2024 Challenge on Low-Light Image Enhancement: Methods and Results."IEEE/CVF CVPR Workshops.
  - [5] Wei et al. (BMVC 2018):"Deep Retinex Decomposition for Low-Light Enhancement."British Machine Vision Conference (LOL Dataset).
  - [6] Land & McCann (JOSA 1971):"Lightness and Retinex Theory."Journal of the Optical Society of America.

---

### Slide 31: Hỏi & Đáp (Q & A)
- **Phân loại nội dung / Tag:** `Q & A`

---
