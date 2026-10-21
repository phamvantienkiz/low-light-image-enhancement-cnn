Được. Sau khi bạn chốt **Đề tài 2 -- Low-Light Image Enhancement**, tôi nghĩ chúng ta nên **khóa phạm vi ngay từ đầu** để tránh việc sau này đề tài bị trượt sang Transformer/Diffusion hoặc trở thành một bài nghiên cứu quá sức.

Tôi đã kiểm tra lại các paper CVF **2024--2026** và đặc biệt chú ý yêu cầu mới của thầy. Với yêu cầu đó, tôi đề xuất phương án chính thức như sau.

---

# 1\. Tên đề tài

Tôi đề xuất tên tiếng Việt:

> **Tăng cường chất lượng ảnh trong điều kiện thiếu sáng bằng mạng nơ-ron tích chập (CNN)**

Tên tiếng Anh:

> **Low-Light Image Enhancement Using a Lightweight Convolutional Neural Network**

Nếu form yêu cầu tên ngắn hơn:

> **Low-Light Image Enhancement Using CNN**

Tôi thích tên **"Lightweight CNN"** hơn "CNN" đơn thuần vì nó thể hiện ngay định hướng của đồ án: **không chạy theo mô hình quá lớn**, tập trung hiểu bản chất giải pháp.

---

# 2\. Tóm tắt nội dung đề tài --- phiên bản có thể đưa vào form

Bạn có thể dùng gần như nguyên văn đoạn sau:

> **Đề tài nghiên cứu bài toán tăng cường chất lượng ảnh trong điều kiện thiếu sáng (Low-Light Image Enhancement -- LLIE), trong đó ảnh đầu vào có độ sáng thấp, tương phản kém, nhiễu và mất chi tiết. Nhóm sẽ xây dựng một mô hình mạng nơ-ron tích chập (CNN) nhẹ theo hướng image-to-image learning để học ánh xạ từ ảnh thiếu sáng sang ảnh có chất lượng/độ sáng tốt hơn. Mô hình dự kiến được xây dựng trên kiến trúc CNN dạng U-Net rút gọn, kết hợp các lớp convolution, activation và skip connection nhằm bảo toàn thông tin không gian và chi tiết ảnh. Nhóm sẽ huấn luyện và đánh giá mô hình trên các bộ dữ liệu low-light có cặp ảnh tương ứng, sử dụng các độ đo như PSNR và SSIM, đồng thời so sánh với một số phương pháp xử lý ảnh truyền thống. Các phương pháp LLIE hiện đại trong các công trình CVF giai đoạn 2024--2026 sẽ được sử dụng làm cơ sở tham khảo và đối chiếu, nhưng đồ án tập trung vào việc xây dựng và phân tích một giải pháp CNN cơ bản, nhẹ và phù hợp với phạm vi môn học.**

Tôi cố tình viết theo hướng này vì nó **khớp chính xác với tinh thần thầy nói**:

- có bài toán thực tế;
- có paper hiện đại;
- có dataset;
- có CNN;
- có thực nghiệm;
- **không tuyên bố reproduce SOTA**;
- không cần phát minh phương pháp mới.

---

# 3\. Chúng ta thực sự sẽ làm gì?

Đây là phần quan trọng nhất.

Tôi đề xuất **không sử dụng RetNet, Transformer, Diffusion, foundation model hay transfer learning làm giải pháp chính.**

### Phương án chính:

> **Supervised Image-to-Image Enhancement bằng một Lightweight U-Net/CNN.**

Pipeline:

```
             TRAINING

Low-light image ────────┐
                        │
                        ▼
                  Lightweight
                     CNN/U-Net
                        │
                        ▼
                 Enhanced image
                        │
                        ▼
                 Loss function
                        │
                        ▼
             Ground-truth image
```

Còn khi inference:

```
Low-light image
       │
       ▼
 Lightweight CNN
       │
       ▼
Enhanced image
```

---

# 4\. Tại sao tôi chọn U-Net thay vì CNN tuần tự đơn giản?

Chúng ta **vẫn đang làm CNN**.

U-Net không phải một hướng hoàn toàn khác với CNN. Nó chủ yếu là một kiến trúc CNN encoder-decoder có **skip connection**.

Ví dụ:

```
                  ┌───────────────┐
                  │               │
Input             │               │
  │               │               │
  ▼               │               │
Conv              │               │
  ↓               │               │
Conv ─────────────┼──────────────► Conv
  ↓               │               │
Downsample        │               │
  ↓               │               │
Conv              │               │
  ↓               │               │
Bottleneck        │               │
  │               │               │
  ▼               │               │
Upsample          │               │
  │               │               │
  └───────────────┘               │
                                  ▼
                                Output
```

Điểm này rất phù hợp với image enhancement vì chúng ta không muốn chỉ phân loại ảnh:

> "Đây là ảnh tối."

mà phải tạo ra **một ảnh mới có cùng kích thước**, đồng thời cố giữ:

- cạnh;
- texture;
- cấu trúc;
- màu sắc;
- chi tiết.

Skip connection giúp đưa thông tin không gian từ encoder sang decoder.

---

# 5\. Kiến trúc cụ thể tôi đề xuất

Không nên bắt đầu bằng U-Net lớn.

Tôi đề xuất:

## **Lightweight U-Net**

Ví dụ:

```
Input RGB 256×256×3
          │
          ▼
     Conv 3×3
     32 channels
          │
          ▼
     Conv Block
          │
          ├────────────────────┐
          ▼                    │
     Downsample                │
          │                    │
          ▼                    │
     Conv Block                │
     64 channels               │
          │                    │
          ▼                    │
     Downsample                │
          │                    │
          ▼                    │
     Bottleneck                │
     128 channels              │
          │                    │
          ▼                    │
       Upsample                │
          │                    │
          └──── Skip ──────────┘
          │
          ▼
     Conv Block
          │
          ▼
       Upsample
          │
          ▼
     Conv 3×3
          │
          ▼
    Enhanced RGB
```

Không cần:

- Transformer;
- self-attention;
- diffusion;
- CLIP;
- pretrained foundation model;
- RetNet.

---

# 6\. Có nên dùng Transfer Learning không?

### Tôi đề xuất: **Không dùng trong phiên bản chính.**

Lý do rất quan trọng.

Nếu chúng ta lấy:

> ResNet/EfficientNet pretrained ImageNet → fine-tuning → enhancement

thì sẽ rất dễ bị hỏi:

> "Vậy bản chất CNN của nhóm nằm ở đâu?"

Trong khi mục tiêu của môn là **hiểu giải pháp**.

Tôi muốn nhóm có thể đứng trước lớp và giải thích được:

> Conv layer làm gì?

> Feature map là gì?

> Vì sao downsampling?

> Vì sao upsampling?

> Vì sao skip connection?

> CNN học cách tăng sáng như thế nào?

> Loss được tính giữa ảnh nào và ảnh nào?

> Tại sao PSNR tăng?

> Tại sao SSIM tăng?

Đây mới là giá trị của đồ án.

---

# 7\. Có nên sử dụng RetNet không?

### Không.

Nếu bạn đang nghĩ đến **RetNet** như một alternative architecture thì tôi sẽ loại khỏi phạm vi đề tài.

RetNet không mang lại lợi ích cần thiết cho mục tiêu của môn này.

Thay vào đó:

> **CNN → U-Net → residual block**

là đủ.

---

# 8\. Có nên dùng một mô hình CNN "hiện đại" hơn không?

Có, nhưng **chỉ như extension**.

Ví dụ sau khi hoàn thành baseline:

### Model A

Plain CNN

### Model B

Lightweight U-Net

### Model C --- optional

U-Net + Residual Block

Khi đó chúng ta có:

| Model             | Params | PSNR | SSIM | Time |
| ----------------- | ------ | ---- | ---- | ---- |
| Traditional       | --     | ...  | ...  | ...  |
| Plain CNN         | ...    | ...  | ...  | ...  |
| Lightweight U-Net | ...    | ...  | ...  | ...  |
| U-Net + Residual  | ...    | ...  | ...  | ...  |

Đây là **thí nghiệm rất đẹp**.

Không cần phải chứng minh:

> "Model của chúng tôi tốt hơn paper."

Chỉ cần chứng minh:

> "CNN cơ bản có khả năng giải quyết bài toán ở mức nào và kiến trúc tăng độ phức tạp ảnh hưởng thế nào đến kết quả."

---

# 9\. Loss function

Tôi đề xuất bắt đầu rất đơn giản:

## L1 Loss

\[ L*{1} = \frac{1}{N}\sum |I*{pred}-I\_{GT}| \]

Sau đó có thể thử:

## MSE

\[ L*{2} = \frac{1}{N}\sum(I*{pred}-I\_{GT})^2 \]

Không cần ngay lập tức dùng perceptual loss.

Nếu còn thời gian, có thể thử:

> **L1 + SSIM loss**

nhưng đây là extension.

---

# 10\. Dataset chính

Tôi đề xuất:

## Training

### **LOL-v2 Real**

và có thể kết hợp:

### **LOL-v1**

Đây là hai dataset phù hợp nhất với hướng supervised CNN.

Sau đó có thể dùng:

### **LOL-v2 Synthetic**

để kiểm tra khả năng học trên dữ liệu synthetic.

---

# 11\. Dataset kiểm tra mở rộng

Sau khi model chạy ổn:

### FiveK

để kiểm tra khả năng tổng quát hóa.

### ExDark

để tạo demo:

> "Model được train bằng paired low-light data nhưng khi gặp ảnh low-light thực tế khác domain thì thế nào?"

Đây là một thí nghiệm rất hay cho phần **Limitations**.

---

# 12\. Metric

Tôi đề xuất chính thức:

### PSNR

\[ PSNR=10\log\_{10}\frac{MAX_I^2}{MSE} \]

### SSIM

Để đo độ tương đồng cấu trúc.

Ngoài ra:

### LPIPS

chỉ nên là **optional**.

NTIRE 2024 LLIE challenge sử dụng **PSNR, SSIM và LPIPS**, với PSNR/SSIM là các tiêu chí chính và LPIPS hỗ trợ đánh giá perceptual similarity. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Liu_NTIRE_2024_Challenge_on_Low_Light_Image_Enhancement_Methods_and_CVPRW_2024_paper.pdf?utm_source=chatgpt.com)

Như vậy việc nhóm chọn PSNR + SSIM là hoàn toàn có cơ sở từ benchmark CVF gần đây.

---

# 13\. Ba paper chính cho form --- tôi đã thay đổi lựa chọn

Vì thầy yêu cầu **paper 2024--2026**, tôi sẽ **không dùng SCI 2022 làm 1 trong 3 paper chính nữa**.

Tôi đề xuất chính thức:

---

## PAPER 1 --- quan trọng nhất

### **DarkIR: Robust Low-Light Image Restoration**

**Daniel Feijoo et al., CVPR 2025**

Đây là paper tôi muốn đặt **số 1**.

Paper rất phù hợp với định hướng của chúng ta vì tác giả **không đi theo Transformer**, mà phát triển các attention mechanism để mở rộng receptive field của **efficient CNNs**. DarkIR hướng tới low-light restoration và đồng thời xử lý cả low-light + blur; paper đánh giá trên LOLBlur, LOLv2 và Real-LOLBlur. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper.html?utm_source=chatgpt.com)

### Tại sao chọn?

Vì nó cho phép nhóm nói:

> "Các nghiên cứu hiện đại vẫn đang phát triển các kiến trúc CNN hiệu quả cho low-light image restoration; ví dụ DarkIR."

Sau đó:

> "Đồ án của chúng tôi không reproduce DarkIR mà xây dựng một lightweight CNN/U-Net đơn giản hơn để nghiên cứu nguyên lý LLIE."

Đây là **một câu chuyện học thuật rất sạch**.

[DarkIR --- CVPR 2025 Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper.html?utm_source=chatgpt.com)

---

# PAPER 2 --- cực kỳ phù hợp cho phần phương pháp hiện đại

### **HVI: A New Color Space for Low-light Image Enhancement**

**Qingsen Yan et al., CVPR 2025**

Paper giải quyết một vấn đề quan trọng:

> LLIE không chỉ là "làm sáng ảnh".

Khi tăng sáng, có thể xuất hiện:

- color bias;
- red artifacts;
- black artifacts.

HVI đề xuất một color space mới --- **Horizontal/Vertical-Intensity (HVI)** --- và sử dụng **Color and Intensity Decoupling Network (CIDNet)** để xử lý enhancement. Code được công khai. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_HVI_A_New_Color_Space_for_Low-light_Image_Enhancement_CVPR_2025_paper.html?utm_source=chatgpt.com)

### Vì sao tôi chọn paper này?

Nó cho nhóm một phần **lý thuyết rất hay để trình bày**:

```
Low-light enhancement
        │
        ├── Brightness
        │
        ├── Color
        │
        └── Artifacts
```

Sau đó giải thích:

> "Một CNN đơn giản có thể tăng brightness nhưng chưa chắc bảo toàn màu tốt."

Đây là một limitation rất hợp lý để nhóm thảo luận.

[HVI --- CVPR 2025 Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_HVI_A_New_Color_Space_for_Low-light_Image_Enhancement_CVPR_2025_paper.html?utm_source=chatgpt.com)

---

# PAPER 3 --- dùng để làm benchmark/dataset/reference

### **NTIRE 2024 Challenge on Low Light Image Enhancement: Methods and Results**

**Xiaoning Liu et al., CVPR Workshops 2024**

Đây không phải một model duy nhất mà là **benchmark/challenge paper**, và tôi rất muốn đưa nó vào form.

Paper xây dựng một dataset LLIE với:

- **230 training scenes**
- **35 validation scenes**
- **35 testing scenes**

bao phủ:

- dim environments;
- extreme darkness;
- non-uniform illumination;
- backlighting;
- night scenes;
- indoor/outdoor;
- độ phân giải lên tới 4K+. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Liu_NTIRE_2024_Challenge_on_Low_Light_Image_Enhancement_Methods_and_CVPRW_2024_paper.pdf?utm_source=chatgpt.com)

Challenge sử dụng:

- PSNR
- SSIM
- LPIPS

để đánh giá. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Liu_NTIRE_2024_Challenge_on_Low_Light_Image_Enhancement_Methods_and_CVPRW_2024_paper.pdf?utm_source=chatgpt.com)

### Tại sao paper này quan trọng?

Nó giúp nhóm chứng minh:

> **Bài toán LLIE không phải một bài toán tự nghĩ ra; nó là một benchmark được cộng đồng CV nghiên cứu và đánh giá có hệ thống.**

Và quan trọng hơn:

> Nó giúp chúng ta xác định **dataset + evaluation protocol**.

[NTIRE 2024 Low Light Image Enhancement Challenge --- CVF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Liu_NTIRE_2024_Challenge_on_Low_Light_Image_Enhancement_Methods_and_CVPRW_2024_paper.html?utm_source=chatgpt.com)

---

# 14\. Vì sao tôi không chọn một số paper 2024 khác?

Có một số paper rất nổi bật nhưng tôi **cố tình không đưa vào 3 paper chính**.

### Zero-Reference Low-Light Enhancement via Physical Quadruple Priors --- CVPR 2024

Rất hay, nhưng framework sử dụng **pretrained generative diffusion model** và bypass decoder. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Zero-Reference_Low-Light_Enhancement_via_Physical_Quadruple_Priors_CVPR_2024_paper.html?utm_source=chatgpt.com)

→ Không phù hợp với baseline CNN của nhóm.

---

### Light the Night --- CVPR 2024

Dùng:

> multi-condition diffusion + depth + RGB + text + reinforcement learning. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Light_the_Night_A_Multi-Condition_Diffusion_Framework_for_Unpaired_Low-Light_CVPR_2024_paper.html?utm_source=chatgpt.com)

→ Quá phức tạp.

---

### Event-guided LLIE --- CVPR 2024

Rất thú vị và có **hơn 30K paired image-event samples**, nhưng cần event camera và event-image fusion. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_Towards_Robust_Event-guided_Low-Light_Image_Enhancement_A_Large-Scale_Real-World_Event-Image_CVPR_2024_paper.html?utm_source=chatgpt.com)

→ Không phù hợp.

---

### Specularity Factorization --- CVPR 2024

Dùng RSFNet, zero-reference factorization. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Saini_Specularity_Factorization_for_Low-Light_Enhancement_CVPR_2024_paper.html?utm_source=chatgpt.com)

→ Hay để đọc thêm, nhưng không cần đưa vào form.

---

# 15\. Có một paper 2024 tôi muốn giữ làm "paper phụ"

Có thể bạn sẽ gặp paper này trong quá trình tìm hiểu:

### **Learning Optimized Low-Light Image Enhancement for Edge Vision Tasks**

CVPR Workshop 2024.

Điểm hay là paper tập trung vào **lightweight/mobile-friendly deep network**, thậm chí tối ưu INT8 và triển khai trên edge device. Tác giả báo cáo hơn 199 FPS trên một low-power edge board. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Sharif_Learning_Optimized_Low-Light_Image_Enhancement_for_Edge_Vision_Tasks_CVPRW_2024_paper.html?utm_source=chatgpt.com)

Tôi **không chọn nó vào top 3**, nhưng nó cực kỳ hữu ích nếu nhóm muốn biện minh:

> **Tại sao đồ án lại chọn lightweight CNN thay vì một mạng khổng lồ?**

Nó cho chúng ta một lập luận rất tốt:

> Lightweight LLIE không chỉ có giá trị về mặt học thuật mà còn có ý nghĩa triển khai thực tế trên thiết bị hạn chế tài nguyên.

---

# 16\. Vậy chính xác đề tài của chúng ta sẽ là gì?

Tôi muốn **khóa specification** như sau:

### Tên

> **Low-Light Image Enhancement Using a Lightweight Convolutional Neural Network**

### Bài toán

```
Low-light image
       ↓
Lightweight CNN / U-Net
       ↓
Enhanced image
```

### Phương pháp

> **Supervised image-to-image learning**

### Kiến trúc

> **Lightweight U-Net**

### Thành phần

- Convolution
- ReLU/LeakyReLU
- Downsampling
- Upsampling
- Skip connection
- Residual block --- optional

### Loss

> **L1 hoặc MSE**

### Dataset chính

> **LOL-v1 + LOL-v2 Real**

### Dataset mở rộng

> LOL-v2 Synthetic / FiveK

### Evaluation

> **PSNR + SSIM**

### Baseline truyền thống

> Gamma Correction / CLAHE

### Optional

> LPIPS

### Không dùng

- ❌ RetNet
- ❌ Transformer
- ❌ Diffusion
- ❌ GAN
- ❌ CLIP
- ❌ pretrained foundation model
- ❌ event camera
- ❌ multimodal input

---

# 17\. Một điểm tôi đặc biệt muốn nhóm làm

Đừng chỉ làm:

> "CNN → kết quả."

Hãy thiết kế đồ án thành **3 mức độ mô hình**:

### Model 0 --- Traditional

```
Low-light
 ↓
Gamma / CLAHE
 ↓
Enhanced
```

### Model 1 --- CNN

```
Low-light
 ↓
Plain CNN
 ↓
Enhanced
```

### Model 2 --- Lightweight U-Net

```
Low-light
 ↓
U-Net
 ↓
Enhanced
```

Sau đó:

| Method            | PSNR | SSIM | Params | Inference |
| ----------------- | ---- | ---- | ------ | --------- |
| Gamma             |      |      | --     |           |
| CLAHE             |      |      | --     |           |
| CNN               |      |      |        |           |
| Lightweight U-Net |      |      |        |           |

Đây sẽ là **xương sống của toàn bộ đồ án**.

---

# 18\. Và lúc đó 3 paper chính có vai trò rất rõ

| Paper                        | Vai trò trong đồ án                              |
| ---------------------------- | ------------------------------------------------ |
| **DarkIR --- CVPR 2025**     | Related Work về **CNN hiện đại/lightweight CNN** |
| **HVI --- CVPR 2025**        | Related Work về **color/illumination problem**   |
| **NTIRE LLIE --- CVPR 2024** | **Dataset + benchmark + evaluation**             |

Tôi đánh giá bộ ba này **tốt hơn nhiều** so với việc chọn ba paper chỉ vì chúng là "LLIE paper".

Nó tạo ra một câu chuyện:

```
                 LOW-LIGHT IMAGE ENHANCEMENT
                            │
            ┌───────────────┼────────────────┐
            │               │                │
         Problem         Modern CNN        Benchmark
            │               │                │
            ▼               ▼                ▼
          HVI             DarkIR           NTIRE
            │               │                │
            └───────────────┼────────────────┘
                            │
                            ▼
                  OUR APPROACH
                            │
                            ▼
                   Lightweight U-Net
                            │
                            ▼
                       PSNR / SSIM
```

Đây chính là cấu trúc tôi nghĩ **rất hợp để điền form đăng ký và sau này phát triển thành slide/báo cáo**.

---

## 19\. Nếu form đăng ký chỉ cho điền một đoạn "nội dung dự kiến thực hiện"

Bạn có thể dùng phiên bản ngắn này:

> **Nhóm dự kiến nghiên cứu bài toán Low-Light Image Enhancement và xây dựng một mô hình Lightweight Convolutional Neural Network theo kiến trúc U-Net rút gọn để thực hiện ánh xạ từ ảnh thiếu sáng sang ảnh được tăng cường. Mô hình được huấn luyện theo phương pháp supervised image-to-image learning trên các bộ dữ liệu low-light có ảnh ground truth tương ứng. Nhóm sẽ thực hiện các bước tiền xử lý dữ liệu, xây dựng và huấn luyện CNN, đánh giá bằng PSNR và SSIM, đồng thời so sánh với một số phương pháp xử lý ảnh truyền thống như Gamma Correction và CLAHE. Các công trình DarkIR (CVPR 2025), HVI (CVPR 2025) và NTIRE 2024 Low Light Image Enhancement Challenge sẽ được sử dụng làm tài liệu tham khảo chính để tìm hiểu bài toán, phương pháp và tiêu chuẩn đánh giá. Đồ án tập trung vào việc phân tích và triển khai một giải pháp CNN nhẹ ở mức cơ bản, không nhằm tái hiện các mô hình SOTA phức tạp.**

### Ba paper để điền form:

1.  **Feijoo, D., Benito, J. C., Garcia, A., & Conde, M. V. (2025). _DarkIR: Robust Low-Light Image Restoration_. CVPR 2025.** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper.html?utm_source=chatgpt.com) [PDF / CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper.html?utm_source=chatgpt.com)

2.  **Yan, Q., et al. (2025). _HVI: A New Color Space for Low-light Image Enhancement_. CVPR 2025.** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_HVI_A_New_Color_Space_for_Low-light_Image_Enhancement_CVPR_2025_paper.html?utm_source=chatgpt.com) [PDF / CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_HVI_A_New_Color_Space_for_Low-light_Image_Enhancement_CVPR_2025_paper.html?utm_source=chatgpt.com)

3.  **Liu, X., et al. (2024). _NTIRE 2024 Challenge on Low Light Image Enhancement: Methods and Results_. CVPR Workshops 2024.** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Liu_NTIRE_2024_Challenge_on_Low_Light_Image_Enhancement_Methods_and_CVPRW_2024_paper.html?utm_source=chatgpt.com) [PDF / CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Liu_NTIRE_2024_Challenge_on_Low_Light_Image_Enhancement_Methods_and_CVPRW_2024_paper.html?utm_source=chatgpt.com)

**Tôi sẽ chốt kiến trúc ban đầu là `Lightweight U-Net`, không phải RetNet/Transformer/transfer learning.** Đây là điểm tôi nghĩ chúng ta nên giữ ổn định từ lúc đăng ký đề tài, rồi sau khi được duyệt mới tiến hành thử nghiệm xem Plain CNN → U-Net → Residual U-Net khác nhau thế nào. Như vậy nhóm vừa **đúng yêu cầu "CNN cơ bản" của thầy**, vừa có đủ chiều sâu để trình bày, và nếu U-Net không đạt như mong muốn thì vẫn có đường lùi về CNN đơn giản hơn.

---

# PAPER

---

Được. Với đề tài **Low-Light Image Enhancement (LLIE) bằng CNN**, nếu xét tiêu chí **dataset public + phù hợp để sinh viên huấn luyện CNN/U-Net + có ground truth để tính PSNR/SSIM**, thì danh sách nên chia thành **paired** và **unpaired** như sau.

### 1. Các dataset public quan trọng

| Dataset                   |                              Số lượng / quy mô |      Paired?      |  Real/Synthetic  | Phù hợp cho project của bạn |
| ------------------------- | ---------------------------------------------: | :---------------: | :--------------: | :-------------------------: |
| **LOL / LOL-v1**          |                                        500 cặp |        ✅         |       Real       |         ⭐⭐⭐⭐⭐          |
| **LOL-v2 Real**           |                       689 train + 100 test cặp |        ✅         |       Real       |         ⭐⭐⭐⭐⭐          |
| **LOL-v2 Synthetic**      |                          ~900 train + 100 test |        ✅         |    Synthetic     |          ⭐⭐⭐⭐           |
| **MIT-Adobe FiveK**       |                 5,000 ảnh, 5 phiên bản retouch |       ✅\*        |       Real       |          ⭐⭐⭐⭐           |
| **SID (See-in-the-Dark)** | 5,094 RAW/short-exposure images, 424 reference |        ✅         |       Real       |          ⭐⭐⭐⭐           |
| **SICE**                  |                                     ~4,400 ảnh | ✅/multi-exposure |       Real       |           ⭐⭐⭐            |
| **SMID**                  |                     Low-light RAW/video frames |        ✅         |       Real       |           ⭐⭐⭐            |
| **SDSD-In / SDSD-Out**    |                    Low-light video/image pairs |        ✅         |       Real       |           ⭐⭐⭐            |
| **LSRW**                  |                                         ~5,650 |        ✅         | Real + Synthetic |           ⭐⭐⭐            |
| **VE-LOL-L**              |                                         ~2,500 |        ✅         | Real + Synthetic |           ⭐⭐⭐            |
| **ELD**                   |                                           ~480 |        ✅         |       Real       |           ⭐⭐⭐            |
| **ExDark**                |                                          7,363 |        ❌         |       Real       |            ⭐⭐             |
| **DICM**                  |                                            ~69 |        ❌         |       Real       |            ⭐⭐             |
| **LIME**                  |                                             10 |        ❌         |       Real       |             ⭐              |
| **NPE**                   |                 ~85–156 tùy phiên bản/thống kê |        ❌         |       Real       |             ⭐              |
| **MEF**                   |                                  ~17 sequences |        ❌         |       Real       |             ⭐              |
| **VV**                    |                                            ~24 |        ❌         |       Real       |             ⭐              |

Các survey gần đây xác nhận LOL, FiveK, SID, SICE, ExDark… là những dataset public được sử dụng phổ biến trong LLIE. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1319157824003239?utm_source=chatgpt.com)

\* FiveK có ảnh input và các phiên bản được chuyên gia chỉnh sửa, nhưng bản chất không hoàn toàn giống cặp “low-light → ground-truth normal-light” như LOL.

---

## 2. Với đồ án của bạn: tôi khuyên dùng 3 dataset chính

### 🥇 LOL-v1

**LOL (LOw-Light dataset)** là lựa chọn rất phù hợp để bắt đầu.

- 500 cặp ảnh **low-light / normal-light**
- Ảnh thực tế.
- Có ground truth.
- Kích thước khoảng 400×600.
- Được sử dụng rất rộng rãi trong các nghiên cứu LLIE. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1319157824003239?utm_source=chatgpt.com)

Đặc biệt phù hợp với kiến trúc:

```text
Low-light image
      ↓
   CNN/U-Net
      ↓
Enhanced image
      ↓
Normal-light ground truth
```

Sau đó tính:

```text
PSNR
SSIM
```

---

### 🥈 LOL-v2 Real

Đây là dataset tôi **rất khuyến nghị dùng làm dataset chính hoặc validation/test bổ sung**.

LOLv2 Real có:

- **689 cặp training**
- **100 cặp testing**
- low-light ↔ normal-light
- ảnh thực tế.

Nguồn tổng hợp dataset hiện liệt kê chính xác 689 training pairs và 100 testing pairs. [Papers with Code](https://paperswithcode.com/datasets?page=1&task=low-light-image-enhancement&utm_source=chatgpt.com)

Điểm hay là bạn có thể làm:

```text
Train:
LOL-v1 + LOL-v2 Real

Test:
LOL-v2 Real test
```

hoặc đơn giản hơn:

```text
Train → LOL-v1
Test  → LOL-v1 test
```

rồi dùng LOL-v2 Real để kiểm tra **generalization**.

---

### 🥉 LOL-v2 Synthetic

Cũng thuộc LOL-v2 nhưng low-light được tạo theo quy trình tổng hợp.

Nó có thể dùng để:

- tăng dữ liệu training;
- kiểm tra khả năng tổng quát;
- so sánh **real vs synthetic low-light**.

Tuy nhiên, với project môn học, tôi **không khuyên lấy nó làm dataset duy nhất**, vì bạn muốn chứng minh CNN xử lý được ảnh thiếu sáng thực tế.

LOLv2 gồm cả phiên bản Real và Synthetic và hiện được sử dụng rộng rãi trong các benchmark LLIE. [Papers with Code](https://paperswithcode.com/datasets?page=1&task=low-light-image-enhancement&utm_source=chatgpt.com)

---

# 3. Các dataset khác đáng biết

### MIT-Adobe FiveK

Rất lớn:

**5,000 ảnh × 5 phiên bản chỉnh sửa bởi các chuyên gia.**

Dataset bao phủ nhiều cảnh, chủ thể và điều kiện ánh sáng khác nhau. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1319157824003239?utm_source=chatgpt.com)

Có thể dùng cho image enhancement, nhưng **không “sạch” về bài toán low-light → normal-light** bằng LOL.

→ Tôi sẽ xem nó là **dataset mở rộng**, không phải dataset chính của đồ án.

---

### SID — See-in-the-Dark

Dataset rất nổi tiếng cho extreme low-light.

Có:

- 5,094 RAW images
- 424 long-exposure reference images
- indoor + outdoor
- camera Sony và Fujifilm. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1319157824003239?utm_source=chatgpt.com)

Nhưng SID thiên nhiều về **RAW image / extreme low-light reconstruction**, nên pipeline phức tạp hơn RGB CNN thông thường.

→ **Không nên chọn SID làm dataset đầu tiên** nếu mục tiêu là một project CNN cơ bản.

---

### SICE

SICE là dataset multi-exposure với hàng nghìn ảnh, được sử dụng cho contrast/image enhancement và cũng xuất hiện trong LLIE. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1319157824003239?utm_source=chatgpt.com)

Có thể dùng để mở rộng experiment, nhưng vấn đề là nó không trực tiếp tương ứng với bài toán:

> low-light RGB → normal-light RGB

một cách đơn giản như LOL.

---

### ExDark

Có khoảng **7,363 ảnh low-light** và chủ yếu là **unpaired**. Dataset còn có annotation cho object detection. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1319157824003239?utm_source=chatgpt.com)

Do đó ExDark rất hữu ích nếu bạn muốn:

```text
CNN enhancement
       ↓
ảnh low-light thực tế
       ↓
kiểm tra visual/generalization
```

nhưng **không phù hợp làm dataset supervised training chính**, vì không có normal-light ground truth tương ứng.

---

# 4. Có thể chia dataset thành 3 nhóm như này

### Nhóm A — Rất phù hợp cho CNN/U-Net supervised

```text
LOL-v1
LOL-v2 Real
LOL-v2 Synthetic
MIT-Adobe FiveK
SID
SICE
SMID
SDSD
LSRW
VE-LOL-L
ELD
```

### Nhóm B — Chủ yếu để test/generalization

```text
ExDark
DICM
LIME
NPE
MEF
VV
```

Các dataset như LIME, NPE, DICM, MEF, VV thường được dùng dưới dạng **unpaired low-light images**, nên phù hợp hơn để đánh giá trực quan/generalization thay vì tính PSNR/SSIM với ground truth. [GitHub](https://github.com/Elin24/Awesome-Low-Light-Enhancement?utm_source=chatgpt.com)

### Nhóm C — Đặc biệt/không nên dùng cho project cơ bản

```text
SID
SMID
SDSD
VE-LOL
ELD
```

Không phải vì chúng không tốt, mà vì chúng có các vấn đề như **RAW, video, extreme low-light, sensor noise...**, khiến pipeline phức tạp hơn.

---

# 5. Nếu đặt vào project của bạn, tôi sẽ chọn thế này

Với mục tiêu **"Lightweight CNN/U-Net cho môn Xử lý ảnh và thị giác máy tính"**, tôi đề xuất:

```text
                    DATASETS
                       │
          ┌────────────┴────────────┐
          │                         │
       Training                  Testing
          │                         │
     LOL-v1                     LOL-v1 test
          │
     LOL-v2 Real
          │
          └──────────────┐
                         │
                  Generalization
                         │
              ┌──────────┴──────────┐
              │                     │
           ExDark              LOL-v2 Real
```

**Phương án đơn giản nhất:**

> **Train:** LOL-v1  
> **Test:** LOL-v1 test  
> **Additional test:** LOL-v2 Real / ExDark

**Phương án đẹp hơn cho báo cáo:**

> **Train:** LOL-v1 + LOL-v2 Real  
> **Validation/Test:** LOL-v2 Real  
> **Generalization:** ExDark

Như vậy bạn có thể trình bày rất rõ:

> _“Mô hình được huấn luyện trên các ảnh low-light có ground truth, sau đó đánh giá cả trên dữ liệu cùng phân phối và dữ liệu low-light thực tế ngoài phân phối.”_

Điều này **hợp với mục tiêu môn học hơn nhiều** so với việc cố nhồi 5–10 dataset vào project.

Một nguồn tổng hợp hiện cũng cung cấp LOL, LOLv2-Real, LSRW, DICM, LIME, MEF và NPE dưới dạng dataset có thể tải công khai. [Figshare](https://figshare.com/articles/dataset/_Information_zip/27192921?utm_source=chatgpt.com)

**Nếu chỉ cần chọn 3 cái để ghi vào proposal:**  
**LOL-v1 + LOL-v2 Real + ExDark** là bộ tôi đề xuất. LOL-v1/LOLv2 Real dùng cho supervised training/evaluation; ExDark dùng để kiểm tra khả năng tổng quát trên ảnh thiếu sáng thực tế không có ground truth. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1319157824003239?utm_source=chatgpt.com)
