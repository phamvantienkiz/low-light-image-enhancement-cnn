# CẨM NANG TOÀN DIỆN VỀ CÁC ĐỘ ĐO CHẤT LƯỢNG ẢNH VÀ THỐNG KÊ THỰC NGHIỆM
> **Dành cho người mới bắt đầu / Giải thích trực quan, dễ hiểu từ bản chất đến ứng dụng**
> Tài liệu hỗ trợ báo cáo đồ án: *Low-Light Image Enhancement with Lightweight CNNs* (UIT - VNUHCM)

---

## MỤC LỤC
1. [Lời mở đầu: Tại sao cần độ đo và thống kê?](#1-lời-mở-đầu-tại-sao-cần-độ-đo-và-thống-kê)
2. [Phần I: Các độ đo chất lượng ảnh (Image Quality Metrics)](#phần-i-các-độ-đo-chất-lượng-ảnh-image-quality-metrics)
   - [2.1. PSNR (Peak Signal-to-Noise Ratio)](#21-psnr-peak-signal-to-noise-ratio--tỷ-số-tín-hiệu-trên-nhiễu-cực-đại)
   - [2.2. SSIM (Structural Similarity Index)](#22-ssim-structural-similarity-index--chỉ-số-tương-đồng-cấu-trúc)
   - [2.3. LPIPS (Learned Perceptual Image Patch Similarity)](#23-lpips-chỉ-số-cảm-nhận-thị-giác-sâu)
   - [2.4. PSNR_gtmean (PSNR sau khi chuẩn hóa độ sáng)](#24-psnr_gtmean--chẩn-đoán-sai-lệch-độ-sáng-toàn-cục)
   - [2.5. Lum Bias / Mean Brightness Error](#25-lum-bias--độ-lệch-sáng-toàn-cục-có-dấu)
   - [2.6. Sharp Ratio (Độ nét biên cạnh Laplacian)](#26-sharp-ratio--độ-đo-khôi-phục-độ-nét-biên-cạnh)
3. [Phần II: Các độ đo chi phí tính toán & phần cứng biên](#phần-ii-các-độ-đo-chi-phí-tính-toán--phần-cứng-biên)
   - [3.1. Params (Số lượng tham số mô hình)](#31-params--dung-lượng-mô-hình)
   - [3.2. FLOPs / GFLOPs (Khối lượng phép tính)](#32-flops--gflops--chi-phí-tính-toán-lý-thuyết)
   - [3.3. Latency (Độ trễ suy luận, ms) & FPS](#33-latency-độ-trễ-ms-và-fps-tốc-độ-khung-hình)
   - [3.4. VRAM (Bộ nhớ đồ họa: Infer vs Train)](#34-vram-bộ-nhớ-đồ-họa-suy-luận-vs-huấn-luyện)
4. [Phần III: Các khái niệm kiểm định thống kê & 95% CI](#phần-iii-các-khái-niệm-kiểm-định-thống-kê--95-ci)
   - [3.1. Delta PSNR ($\Delta\text{PSNR}$)](#41-delta-psnr--mức-chênh-lệch-chất-lượng)
   - [3.2. Bootstrap 95% Confidence Interval (Khoảng tin cậy 95%)](#42-bootstrap-95-confidence-interval-ci--khoảng-tin-cậy-95)
   - [3.3. P-value: $p_t$ (Paired t-test) và $p_W$ (Wilcoxon signed-rank)](#43-giá-trị-p-value-p_t-và-p_w)
   - [3.4. Tỷ lệ % A thắng (% Win rate)](#44-tỷ-lệ--a-thắng--tính-nhất-quán-trên-từng-ảnh)
   - [3.5. Quy tắc quyết định đã đăng ký trước (Pre-registered Decision Rules)](#45-quy-tắc-kiểm-định-đăng-ký-trước-pre-registered-rules)
5. [Bảng tóm tắt tra cứu nhanh](#5-bảng-tóm-tắt-tra-cứu-nhanh)

---

## 1. Lời mở đầu: Tại sao cần độ đo và thống kê?

Khi bạn thiết kế một mô hình AI phục hồi ảnh thiếu sáng:
- Nhìn bằng mắt thường: Một người có thể khen *"Ảnh này sáng hơn, đẹp hơn"*, người khác lại chê *"Ảnh này hơi chói hoặc vỡ hạt"*. Nhận xét bằng mắt có tính **chủ quan** cao.
- Vì vậy, khoa học cần các **độ đo định lượng (Metrics)**: các con số toán học khách quan đo lường độ sáng, độ nét, độ tương đồng màu sắc.
- Đồng thời, nếu mô hình của bạn đạt $21.10\text{ dB}$ còn mô hình đối thủ đạt $21.03\text{ dB}$ (chênh lệch chỉ $0.07\text{ dB}$), liệu mô hình bạn có thực sự tốt hơn hay chỉ do may mắn ngẫu nhiên? Để trả lời, ta cần **kiểm định thống kê (Statistical Hypothesis Testing)**.

---

## Phần I: Các độ đo chất lượng ảnh (Image Quality Metrics)

### 2.1. PSNR (Peak Signal-to-Noise Ratio — Tỷ số tín hiệu trên nhiễu cực đại)

#### Bản chất đời thường
Hãy tưởng tượng bạn đang nghe đài radio. 
- "Tín hiệu" (Signal) là giọng nói rõ ràng của phát thanh viên.
- "Nhiễu" (Noise) là tiếng xì xèo rè rè phía sau.
- Tỷ số tín hiệu trên nhiễu càng lớn $\rightarrow$ tiếng nói càng trong trẻo, tiếng rè càng nhỏ $\rightarrow$ nghe càng chuẩn.

Trong xử lý ảnh:
- "Tín hiệu chuẩn": Là bức ảnh gốc chụp đủ sáng đẹp đẽ (gọi là **Ground Truth - GT**).
- "Nhiễu": Là sự sai khác (lệch màu, tối, mờ) giữa ảnh mô hình tạo ra và ảnh GT.

#### Công thức toán học đơn giản
Trước tiên, máy tính đo độ lệch bình phương trung bình giữa từng điểm ảnh (**MSE - Mean Squared Error**):
$$\text{MSE} = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} (I_{\text{pred}}(i,j) - I_{\text{gt}}(i,j))^2$$
Sau đó tính PSNR (đo bằng đơn vị **Decibel - dB**):
$$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}^2}{\text{MSE}} \right)$$
*(Với ảnh thông thường, giá trị điểm ảnh tối đa $\text{MAX} = 255$ hoặc $1.0$)*.

#### Cách đọc hiểu kết quả
- **Càng cao càng tốt** (ký hiệu $\uparrow$).
- **Đơn vị:** dB.
- Dưới $15\text{ dB}$: Ảnh rất tệ, tối đen hoặc sai màu nghiêm trọng (ảnh input ban đầu thường chỉ khoảng $8\text{–}10\text{ dB}$).
- Từ $20\text{–}25\text{ dB}$: Ảnh phục hồi tốt, nhận diện rõ vật thể, màu sắc tương đối chuẩn.
- Trên $28\text{–}30\text{ dB}$: Ảnh rất xuất sắc, gần như mắt người không phân biệt được với ảnh gốc đủ sáng.

---

### 2.2. SSIM (Structural Similarity Index — Chỉ số tương đồng cấu trúc)

#### Bản chất đời thường
Con người khi nhìn ảnh không so sánh từng chấm điểm ảnh (pixel) rời rạc như máy tính, mà nhìn vào **đường nét, hình khối, góc cạnh và cấu trúc vật thể**.
- Một bức ảnh bị lệch sáng nhẹ nhưng mọi đồ vật, chữ viết, khuôn mặt vẫn nguyên vẹn cấu trúc $\rightarrow$ Mắt người vẫn thấy dễ chịu.
- **SSIM** được sinh ra để mô phỏng cách mắt người đánh giá: So sánh độ sáng (luminance), độ tương phản (contrast), và cấu trúc đường nét (structure).

#### Công thức toán học (Tóm lược 3 thành phần)
$$\text{SSIM}(x, y) = [l(x,y)]^\alpha \cdot [c(x,y)]^\beta \cdot [s(x,y)]^\gamma$$
- $l(x,y)$: So sánh độ sáng trung bình.
- $c(x,y)$: So sánh độ lệch chuẩn (độ tương phản).
- $s(x,y)$: So sánh hiệp phương sai (cấu trúc hình khối).

#### Cách đọc hiểu kết quả
- **Càng cao càng tốt** (ký hiệu $\uparrow$).
- **Thang điểm:** Từ $0.0$ đến $1.0$ (hoặc từ $0\%$ đến $100\%$).
- $\text{SSIM} = 1.0$: Hai bức ảnh giống nhau tuyệt đối từng đường nét cấu trúc.
- Dưới $0.40$: Cấu trúc bị phá vỡ, nhòe nhoẹt.
- Từ $0.70\text{–}0.85$: Cấu trúc giữ lại rất tốt (các mô hình như U-Net hay DarkIR-Edge đạt $\sim 0.74\text{–}0.83$).

---

### 2.3. LPIPS (Learned Perceptual Image Patch Similarity — Chỉ số cảm nhận thị giác sâu)

#### Bản chất đời thường
PSNR và SSIM là các công thức toán học cổ điển, đôi khi bức ảnh bị mờ nhẹ hoặc hơi mịn da quá mức thì PSNR vẫn có thể cao, nhưng mắt người nhìn vào lại thấy "giả tạo".
- **LPIPS** sử dụng một mạng nơ-ron học sâu (như VGG hoặc AlexNet đã học trên hàng triệu ảnh) đóng vai trò như **đôi mắt của một chuyên gia**.
- Mạng nơ-ron này trích xuất đặc trưng sâu của 2 ảnh và đo xem *"Đôi mắt AI cảm thấy 2 ảnh này giống nhau đến mức nào"*.

#### Cách đọc hiểu kết quả
- **Càng THẤP càng tốt** (ký hiệu $\downarrow$).
- Vì đây là độ đo sự **khác biệt (khoảng cách)**. Khoảng cách bằng $0$ nghĩa là giống hệt.
- LPIPS nhỏ hơn $0.20$: Ảnh rất tự nhiên, chân thực với mắt người.
- LPIPS lớn hơn $0.40$: Ảnh có cảm giác giả tạo, nhiễu hạt hoặc nhòe mờ mắt người dễ nhận ra.

---

### 2.4. PSNR_gtmean — Chẩn đoán sai lệch độ sáng toàn cục

#### Bản chất đời thường (Cực kỳ quan trọng trong đề tài này!)
Giả sử mô hình phục hồi bức ảnh căn phòng: mọi cuốn sách, bàn ghế, góc tường đều rõ nét tuyệt đối, nhưng mô hình lại cho ra màu hơi tối hơn ảnh Ground Truth một chút:
- Vì bị lệch độ sáng toàn bức ảnh, chỉ số **PSNR truyền thống sẽ bị tụt thảm hại** (vì phép trừ điểm ảnh bị lệch hàng loạt).
- Để kiểm tra xem: *"Mô hình dở do không khôi phục được chi tiết, hay chỉ đơn giản là kéo sáng non tay hơn ảnh mẫu một chút?"*, các nhà nghiên cứu dùng **$\text{PSNR}_{\text{gtmean}}$**.
- **Cách làm:** Ta lấy ảnh mô hình tạo ra, nhân chỉnh độ sáng trung bình của nó cho bằng đúng độ sáng của ảnh Ground Truth, rồi mới bấm máy tính PSNR.

#### Ý nghĩa trong bài báo cáo của bạn:
- Trên tập `LOL-v2`, PSNR thô chỉ đạt $\sim 21.0\text{ dB}$, nhưng khi tính $\text{PSNR}_{\text{gtmean}}$ thì vọt lên tận **$26.5\text{–}27.1\text{ dB}$ (tăng vọt $+5.5\text{–}+6.1\text{ dB}$)**!
- **Ý nghĩa rút ra:** Mô hình của chúng ta khôi phục chi tiết cực kỳ nét và chuẩn, phần lớn điểm bị trừ oan uổng là do lệch độ sáng trung bình với ảnh mẫu.

---

### 2.5. Lum Bias — Độ lệch sáng toàn cục có dấu

- **Khái niệm:** Lấy độ sáng trung bình của ảnh mô hình trừ đi độ sáng trung bình của ảnh GT:
  $$\text{Lum Bias} = \text{Mean}(I_{\text{pred}}) - \text{Mean}(I_{\text{gt}})$$
- **Giá trị dương ($+$):** Mô hình có xu hướng làm ảnh sáng hơn Ground Truth.
- **Giá trị âm ($-$):** Mô hình có xu hướng làm ảnh tối hơn Ground Truth.
- Trong thực nghiệm v4: Khi thêm dữ liệu Nikon và Huawei (vốn là ảnh sáng), mô hình bị tăng độ lệch sáng toàn cục (`+12.8` lên `+15.3`), làm tăng điểm ở miền này nhưng lại kéo giảm điểm ở miền `LOL-v2`.

---

### 2.6. Sharp Ratio — Độ đo khôi phục độ nét biên cạnh

- **Bản chất:** Dùng toán tử vi phân Laplacian ($\nabla^2$) quét qua ảnh để tìm các đường viền, mép chữ, cạnh bàn.
- Lấy phương sai Laplacian của ảnh tạo ra chia cho ảnh Ground Truth:
  $$\text{Sharp Ratio} = \frac{\text{Var}(\text{Laplacian}(I_{\text{pred}}))}{\text{Var}(\text{Laplacian}(I_{\text{gt}}))}$$
- **Ý nghĩa:**
  - Bằng $1.0$: Độ sắc nét biên cạnh tương đương 100% ảnh gốc.
  - Càng nhỏ (ví dụ $0.20$): Ảnh bị mịn mờ, viền cạnh bị nhòe (trong báo cáo, U-Net đạt $\sim 0.30$ cao hơn Edge $\sim 0.20$, giải thích vì sao U-Net nhìn sắc cạnh hơn khi khử mờ).

---

## Phần II: Các độ đo chi phí tính toán & Phần cứng biên

Khi đưa mô hình lên các thiết bị như điện thoại, camera giám sát, máy bay không người lái (edge devices), mô hình không chỉ cần chất lượng ảnh cao mà còn phải **nhẹ và chạy nhanh**.

### 3.1. Params (Số lượng tham số mô hình)
- **Bản chất:** Là tổng số các "trọng số" (weights) mà mạng nơ-ron học được và lưu trong file `.pt` hoặc `.pth`.
- **Đơn vị:** Triệu tham số (M — Million).
- **So sánh trực quan:**
  - Mô hình khổng lồ (Diffusion / LLM): Hàng trăm triệu đến hàng tỷ tham số $\rightarrow$ Cần siêu máy tính server GPU khủng.
  - Đề tài của chúng ta: **Mô hình siêu nhẹ (Lightweight CNN)** với ngân sách giới hạn $\le 2.0\text{M}$ tham số:
    - `Standard U-Net`: $1.928\text{M}$
    - `DarkIR-Edge`: $1.924\text{M}$
    - `DarkIR-Edge (no-FFT)`: $1.495\text{M}$ (tiết kiệm $22\%$ dung lượng!).

---

### 3.2. FLOPs / GFLOPs — Chi phí tính toán lý thuyết
- **FLOPs (Floating Point Operations):** Số lượng phép tính số thực (cộng, nhân ma trận) cần thực hiện để xử lý 1 bức ảnh.
- **Đơn vị:** **GFLOPs** ($10^9$ phép tính — Giga FLOPs).
- **Trong bài báo cáo:** Đo trên kích thước ảnh chuẩn $256 \times 256$:
  - `Standard U-Net`: $18.45\text{ GFLOPs}$
  - `DarkIR-Edge`: $8.75\text{ GFLOPs}$ (**giảm $53\%$ phép tính** nhờ dùng Depthwise Convolution).

---

### 3.3. Latency (Độ trễ, ms) và FPS (Tốc độ khung hình)
- **Latency:** Thời gian phần cứng (GPU NVIDIA L4) cần để xử lý xong 1 bức ảnh. Tính bằng mili-giây (**ms** — $1\text{ giây} = 1000\text{ ms}$).
- **FPS (Frames Per Second):** Số bức ảnh xử lý được trong 1 giây:
  $$\text{FPS} = \frac{1000}{\text{Latency (ms)}}$$
- **Nghịch lý thực tế quan trọng trong slide 17:**
  - *Lý thuyết:* DarkIR-Edge có số phép tính FLOPs ít hơn U-Net $53\%$.
  - *Thực tế:* DarkIR-Edge lại **chạy chậm hơn U-Net 2.4 lần** ($32\text{ ms}$ so với $13\text{ ms}$)!
  - *Nguyên nhân:* Các phép toán chia nhỏ kênh (Depthwise), phép biến đổi Fourier (FFT), và cổng Gating phải đọc/ghi bộ nhớ liên tục (nghẽn băng thông bộ nhớ - memory bandwidth bound) thay vì tính toán thuần túy như phép tích chập chuẩn của U-Net.

---

### 3.4. VRAM — Bộ nhớ đồ họa (Suy luận vs Huấn luyện)
- **Infer VRAM (Bộ nhớ khi chạy suy luận):** Lượng RAM của card màn hình dùng để tăng sáng 1 ảnh (khoảng $220\text{–}270\text{ MB}$).
- **Train VRAM (Bộ nhớ khi huấn luyện):** Lượng RAM card đồ họa dùng khi train (chứa batch size, lưu lại các gradient và đồ thị đạo hàm để cập nhật trọng số):
  - U-Net chỉ tốn: $1,471\text{ MB}$.
  - DarkIR-Edge tốn tới: $4,816\text{ MB}$ (gấp hơn 3 lần vì các nhánh FFT và Gating phải lưu trạng thái trung gian rất lớn).

---

## Phần III: Các khái niệm kiểm định thống kê & 95% CI

Đây là phần mang tính học thuật cao nhất trong đề tài của bạn, đảm bảo kết quả báo cáo không bị quy chụp là "nói mò" hay "ăn may".

### 4.1. Delta PSNR ($\Delta\text{PSNR}$) — Mức chênh lệch chất lượng
- Khi so sánh mô hình A với mô hình B trên cùng một bức ảnh:
  $$\Delta\text{PSNR} = \text{PSNR}_A - \text{PSNR}_B$$
- $\Delta\text{PSNR} > 0$: Mô hình A tốt hơn mô hình B.
- $\Delta\text{PSNR} < 0$: Mô hình A kém hơn mô hình B.
- Ví dụ: `Edge - U-Net = -0.063 dB` nghĩa là trung bình DarkIR-Edge thấp hơn U-Net $0.063\text{ dB}$.

---

### 4.2. Bootstrap 95% Confidence Interval (CI) — Khoảng tin cậy 95%

#### Bản chất đời thường
Tập kiểm thử có 190 ảnh. Nếu ta thử chọn ngẫu nhiên 190 ảnh khác thì kết quả có còn đúng không?
- **Phương pháp Bootstrap:** Máy tính tự động bốc ngẫu nhiên (có lặp lại) hàng nghìn tập con từ 190 ảnh đó, tính chênh lệch điểm số hàng nghìn lần để tạo ra một **khoảng tin cậy 95%** $[CI_{\text{thấp}}, CI_{\text{cao}}]$.
- Ý nghĩa: Chúng ta tin tưởng chắc chắn $95\%$ rằng chênh lệch thực sự giữa 2 mô hình nằm trong khoảng này.

#### Cách nhìn khoảng CI để biết mô hình nào thắng/hòa:
- **Nếu khoảng CI CHỨA SỐ 0 (ví dụ $[-0.360, +0.229]$):**
  - Số $0$ nghĩa là hai mô hình bằng điểm nhau.
  - Vì khoảng tin cậy bao trùm cả giá trị âm lẫn giá trị dương và chứa cả số $0$, ta kết luận: **Hai mô hình HÒA NHAU VỀ MẶT THỐNG KÊ (Ngang ngửa nhau, không có khác biệt có ý nghĩa)**.
- **Nếu khoảng CI HOÀN TOÀN DƯƠNG (ví dụ $[+0.150, +0.650]$):**
  - Không chứa số $0$.
  - Ta kết luận: **Mô hình A THẮNG THẬT SỰ với độ tin cậy 95%**.
- **Nếu khoảng CI HOÀN TOÀN ÂM (ví dụ $[-0.765, -0.067]$):**
  - Không chứa số $0$.
  - Ta kết luận: **Mô hình A THUA THẬT SỰ với độ tin cậy 95%**.

---

### 4.3. Giá trị P-value: $p_t$ và $p_W$

Trong thống kê, người ta đặt ra giả thuyết đảo ngược (Null Hypothesis $H_0$): *"Hai mô hình chẳng khác gì nhau cả, chênh lệch điểm chỉ là do ăn may ngẫu nhiên"*.
- **P-value** là xác suất để sự "ăn may" đó xảy ra.
- Nếu **$p < 0.05$ (nhỏ hơn 5%)**: Khả năng ăn may quá nhỏ $\rightarrow$ Bác bỏ $H_0$, tuyên bố **sự khác biệt là có thật (Statistically Significant)**.
- Nếu **$p \ge 0.05$**: Chênh lệch chưa đủ sức thuyết phục, có thể do ngẫu nhiên $\rightarrow$ Tuyên bố **hai mô hình tương đương nhau**.

Trong đề tài sử dụng song song 2 bài kiểm tra:
1. **$p_t$ (Paired Student's t-test):** Kiểm định tham số dựa trên giả định phân phối chuẩn.
2. **$p_W$ (Wilcoxon Signed-Rank test):** Kiểm định phi tham số dựa trên thứ hạng (an toàn hơn, không sợ ảnh bị ngoại lai).

---

### 4.4. Tỷ lệ % A thắng — Tính nhất quán trên từng ảnh
- Trên tập 190 bức ảnh kiểm thử, có bao nhiêu bức ảnh mà mô hình A đạt điểm PSNR cao hơn mô hình B?
- Ví dụ trong slide 16:
  - Ở v4, `Edge - U-Net` đạt **$50.5\%$**: Nghĩa là trong 190 ảnh, có khoảng 96 ảnh Edge nhỉnh hơn và 94 ảnh U-Net nhỉnh hơn.
  - Tỷ lệ xấp xỉ $50/50$ chứng minh thực tế hai mô hình **ngang tài ngang sức** trên từng bức ảnh riêng lẻ.

---

### 4.5. Quy tắc kiểm định đăng ký trước (Pre-registered Rules)

Để tránh việc sau khi chạy ra kết quả mới "chữa cháy" số liệu, đồ án đã đăng ký trước 3 quy tắc vàng (QT1, QT2, QT3) để nghiệm thu:
1. **QT1 (Đa miền - Non-inferiority):** Mở rộng tập dữ liệu không được làm suy giảm chất lượng quá mức ($\Delta\text{PSNR} \ge -0.3\text{ dB}$).
   - *Kết quả thực nghiệm:* Vi phạm QT1 trên miền LOL-v2 do hiện tượng kéo co độ sáng giữa các miền. Báo cáo trung thực kết quả âm này là một điểm cộng học thuật rất lớn.
2. **QT2 (Nikon - Exposure Ambiguity):** Kiểm tra xem việc thêm ảnh Nikon vào train có giúp mô hình nắm bắt độ phơi sáng không.
   - *Kết quả:* Không đổi ($p = 0.987$, CI chứa 0) do tính bất định độ phơi sáng của ảnh cảm biến Nikon.
3. **QT3 (Blur - Saturation):** Kiểm tra xem tăng gấp 4 lần dữ liệu ảnh nhòe mờ có giúp mô hình khử mờ tốt như mô hình DarkIR-m lớn không.
   - *Kết quả:* Bão hòa dung lượng (~1.9M tham số không đủ sức chứa thêm tri thức khử mờ).

---

## 5. Bảng tóm tắt tra cứu nhanh

| Tên độ đo / Thuật ngữ | Ký hiệu | Ý nghĩa đời thường | Hướng tối ưu | Ứng dụng trong đồ án |
| :--- | :---: | :--- | :---: | :--- |
| **PSNR** | dB | Tỷ số tín hiệu/nhiễu (Độ tương đồng điểm ảnh) | Cao hơn là tốt ($\uparrow$) | Đo chất lượng tái tạo ảnh tổng quát |
| **SSIM** | $0 \rightarrow 1$ | Độ giữ nguyên hình khối, đường nét cấu trúc | Cao hơn là tốt ($\uparrow$) | Đảm bảo ảnh không bị méo mó vật thể |
| **LPIPS** | $0 \rightarrow 1$ | Độ lệch cảm nhận theo thị giác con người | Thấp hơn là tốt ($\downarrow$) | Đánh giá độ tự nhiên, không bị giả |
| **$\text{PSNR}_{\text{gtmean}}$** | dB | PSNR sau khi triệt tiêu lệch sáng toàn cục | Cao hơn là tốt ($\uparrow$) | Chứng minh mạng khôi phục chi tiết cực tốt |
| **Params** | M | Dung lượng kích thước mạng nơ-ron | Nhỏ hơn là tốt ($\downarrow$) | Kiểm soát mạng gọn nhẹ $\le 2.0\text{M}$ |
| **FLOPs** | G | Số lượng phép tính lý thuyết cho 1 ảnh | Nhỏ hơn là tốt ($\downarrow$) | DarkIR-Edge giảm $53\%$ phép tính |
| **Latency** | ms | Thời gian thực tế để card đồ họa xử lý 1 ảnh | Thấp hơn là tốt ($\downarrow$) | U-Net dẫn đầu về tốc độ ($13\text{ ms}$) |
| **95% Bootstrap CI** | $[a, b]$ | Khoảng tin cậy 95% của chênh lệch điểm số | Chứa số 0 $\rightarrow$ Hòa | Khẳng định khách quan hai mô hình hòa nhau |
| **P-value** ($p_t, p_W$) | $0 \rightarrow 1$ | Xác suất xảy ra do ngẫu nhiên ăn may | $< 0.05 \rightarrow$ Thật sự khác biệt | Chứng minh kết luận vững chắc về mặt toán học |

---
*Tài liệu được biên soạn phục vụ chuyên môn môn học CS2203 - UIT.*
