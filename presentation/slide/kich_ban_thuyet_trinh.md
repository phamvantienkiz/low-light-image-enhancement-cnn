# KỊCH BẢN THUYẾT TRÌNH CHI TIẾT (bản v3 + v4)

> **Đề tài:** Tăng cường chất lượng ảnh trong điều kiện thiếu sáng bằng mạng nơ-ron tích chập
> **Người trình bày:** Phạm Văn Tiến — MSHV 26848223
> **Thời lượng gợi ý:** ~18–20 phút nói + ~3 phút demo + Hỏi đáp
>
> **Cách dùng:** Mỗi slide có **[Trên slide]** (tóm tắt để đối chiếu), **[Lời nói]** (đọc/nhớ gần như nguyên văn) và **[Lưu ý]** (thao tác, số cần nhớ). Số slide theo bản `index.html` hiện tại (23 slide). Nếu thêm slide "DarkIR gốc" (xem Phụ lục A), các slide sau đó dịch lên 1 trang. Script của slide đó là **Slide 10b**.
>
> _Văn phong: nói chậm, ngắt câu ngắn. Dấu `…` là chỗ nghỉ nhịp. Chữ **đậm** là con số/ý phải nhấn._

| Phần                           | Slide                             | Thời gian |
| ------------------------------ | --------------------------------- | --------- |
| Mở đầu                         | 1–3                               | ~1′30″    |
| Đặt vấn đề, phát biểu bài toán | 4–7                               | ~3′       |
| Phương pháp                    | 8–11                              | ~5′30″    |
| Thực nghiệm: dataset, độ đo    | 12–13                             | ~3′       |
| Thực nghiệm: kết quả           | 14–20                             | ~5′       |
| Demo                           | (chuyển từ slide 17–19 hoặc cuối) | ~3′       |
| Kết luận, Hỏi đáp              | 21–23                             | ~2′       |

---

## Slide 1 — Trang bìa

**[Lời nói]**
Em xin chào thầy/cô và các bạn. Em là Phạm Văn Tiến, học viên lớp môn Xử lý ảnh và Thị giác máy tính, mã CS2203.
Hôm nay em xin trình bày đồ án với đề tài **"Tăng cường chất lượng ảnh trong điều kiện thiếu sáng bằng mạng nơ-ron tích chập"**.
Nói ngắn gọn, em so sánh một cách có kiểm soát hai mạng nhẹ cỡ **1,9 triệu tham số**: một mạng U-Net gọn, và một biến thể em đặt tên là DarkIR-Edge, lấy cảm hứng từ bài báo DarkIR của hội nghị CVPR 2025.

---

## Slide 2 — Nội dung báo cáo

**[Trên slide]** Phần I: đặt vấn đề và phương pháp. Phần II: thực nghiệm (v3 & v4) và tổng kết.

**[Lời nói]**
Bài trình bày gồm hai phần.
Phần một là đặt vấn đề, phát biểu bài toán và phương pháp giải quyết.
Phần hai là thực nghiệm: dữ liệu, độ đo, kết quả định lượng, minh họa định tính, một đoạn demo, rồi nhận xét và kết luận.
Một điểm em muốn nói trước: thực nghiệm của em chạy **hai giai đoạn**. Giai đoạn v3 dùng khoảng 1.600 ảnh huấn luyện. Giai đoạn v4 mở rộng lên khoảng **6.800 ảnh** để kiểm tra xem thêm dữ liệu có giúp ích không.

---

## Slide 3 — Thông tin & mức độ hoàn thành

**[Lời nói]**
Đề tài do em thực hiện độc lập, hoàn thành 100%.
Công việc gồm bốn hạng mục: chuẩn bị dữ liệu đa miền; cài đặt mô hình và các baseline; huấn luyện và kiểm định thống kê; cuối cùng là báo cáo, slide và ứng dụng demo.
Tổng cộng em huấn luyện **12 lần**: 3 mô hình, nhân 2 seed, nhân 2 giai đoạn v3 và v4.

**[Lưu ý]** Chỉ nói "hoàn thành" khi chắc chắn chạy được demo (xem phần Demo). Chạy thử `run_demo.bat` trước giờ báo cáo.

---

## Slide 4 — Vấn đề bài toán cần giải quyết _(tiêu chí: lý do, cơ sở)_

**[Lời nói]**
Chúng ta thường gặp ảnh chụp ban đêm hoặc trong nhà thiếu sáng. Vấn đề không chỉ là **tối**.
Để chụp được trong điều kiện ít ánh sáng, máy ảnh phải tăng ISO, và khi tăng ISO thì **nhiễu hạt** bùng lên. Nếu không tăng ISO thì phải phơi sáng lâu, và ảnh bị **nhòe chuyển động**. Vậy ảnh đêm bị ba lỗi cùng lúc: tối, nhiễu và nhòe.
Hệ quả là các bài toán phía sau, như nhận diện hay phát hiện vật thể, đều giảm độ chính xác. Camera giám sát, điện thoại hay xe tự hành đều cần bước tăng cường ảnh này làm tiền xử lý.
Các mô hình hiện đại thường có hàng chục triệu tham số, rất nặng cho thiết bị nhỏ. Vì vậy em đặt ra một câu hỏi cụ thể: **nếu chỉ có ngân sách khoảng 1,9 triệu tham số, kiến trúc nào tốt hơn?** Một U-Net đơn giản, hay một thiết kế tinh vi hơn lấy cảm hứng từ DarkIR?

**[Lưu ý]** Nếu muốn mạnh hơn, thêm 1 câu có con số/trích dẫn về việc detector giảm độ chính xác trên ảnh tối. Chỉ nói nếu có nguồn.

---

## Slide 5 — Phát biểu bài toán _(tiêu chí 15 điểm — nói kỹ)_

**[Lời nói]**
Em phát biểu bài toán một cách hình thức như sau.
**Đầu vào** là một ảnh RGB thiếu sáng, ký hiệu $I_{low}$, kích thước $H \times W \times 3$. Ảnh này có thể kèm nhiễu hạt cảm biến và nhòe chuyển động.
**Đầu ra** là một ảnh $\hat{I}$ cùng kích thước, có độ sáng cân bằng, ít nhiễu, màu sắc và chi tiết gần với ảnh chuẩn đủ sáng, gọi là Ground Truth $I_{gt}$.
Nhiệm vụ của chúng ta là học một hàm $f_\theta$ sao cho $\hat{I} = f_\theta(I_{low})$ gần $I_{gt}$ nhất, **với ràng buộc số tham số của $\theta$ khoảng 1,9 triệu**.
"Gần nhất" được đo bằng ba độ đo: PSNR và SSIM càng cao càng tốt, LPIPS càng thấp càng tốt. Em sẽ giải thích ba độ đo này ở phần thực nghiệm.

**[Lưu ý]** Slide hiện chỉ có Input/Output. Nên thêm 1 dòng $f_\theta$ và ràng buộc $|\theta| \approx 1.9M$ để lời nói khớp slide.

---

## Slide 6 — Minh họa bài toán, kịch bản 1

**[Lời nói]**
Đây là kịch bản thứ nhất: ảnh thiếu sáng sâu kèm nhiễu hạt, lấy từ bộ **LOLv2-Real**. Bên trái là ảnh đầu vào, gần như đen, và nếu nhìn kỹ sẽ thấy hạt nhiễu. Bên phải là ảnh Ground Truth chụp cùng cảnh nhưng đủ sáng.
Mô hình phải biến ảnh bên trái thành ảnh gần giống bên phải.

---

## Slide 7 — Minh họa bài toán, kịch bản 2

**[Lời nói]**
Kịch bản thứ hai khó hơn: ảnh vừa tối vừa **nhòe chuyển động**, từ bộ **LOL-Blur**. Ngoài việc tăng sáng, mô hình phải khôi phục lại biên cạnh sắc nét, ví dụ nét chữ hoặc đường viền. Đây là bài toán kết hợp tăng sáng và khử mờ.

---

## Slide 8 — Tổng quan các giải pháp

**[Lời nói]**
Em so sánh ba nhóm phương pháp.
**Nhóm một: không học sâu.** Gồm Gamma 2,2 là một phép biến đổi phi tuyến độ sáng; CLAHE là cân bằng histogram cục bộ; và MSRCR là Retinex đa tỉ lệ, tách thành phần chiếu sáng và phản xạ. Nhóm này đơn giản, nhưng khuếch đại nhiễu và hay lệch màu.
**Nhóm hai: baseline CNN**, là Standard U-Net gọn, 1,93 triệu tham số.
**Nhóm ba: họ DarkIR-Edge**, lấy cảm hứng từ DarkIR CVPR 2025, có thêm nhánh tần số FFT. Em còn có một bản đối chứng tắt FFT, gọi là no-FFT, để kiểm tra nhánh tần số có thực sự có ích không.
Ngoài ra em dùng checkpoint chính thức **DarkIR-m** làm mốc tham chiếu. Lưu ý đây là mô hình lớn hơn nhiều, nên em chỉ coi là _trần tham chiếu_, không phải đối thủ cùng ngân sách.

---

## Slide 9 — Phương pháp 1: Standard U-Net _(giải thích sơ đồ)_

**[Trên slide]** Khối `_cb` = 2×(Conv3×3 + GroupNorm + ReLU); 3 tầng xuống, 3 tầng lên; skip bằng `torch.cat`; cộng thặng dư toàn cục; 1,928M tham số.

**[Lời nói]**
Đây là baseline của em: Standard U-Net với số kênh cơ sở là 32.
Em xin giải thích sơ đồ bên phải, đọc **từ trên xuống rồi lên lại**, giống hình chữ U.

_(chỉ vào đường đi xuống)_
Ảnh vào có 3 kênh. Đi qua khối đầu tiên `e1`, 32 kênh, ở độ phân giải đầy đủ. Mỗi khối gồm **hai lớp tích chập 3×3**, mỗi lớp có chuẩn hóa nhóm GroupNorm và hàm ReLU.
Sau đó qua **MaxPool** thu nhỏ một nửa, vào khối `e2` với 64 kênh; lại MaxPool, vào `e3` với 128 kênh; rồi MaxPool lần nữa xuống đáy chữ U, khối `mid`, **256 kênh** nhưng ảnh chỉ còn 1/8 kích thước.
Càng xuống sâu, số kênh tăng gấp đôi và kích thước giảm một nửa. Mạng "nhìn" được vùng rộng hơn, hiểu ngữ cảnh toàn cảnh: ví dụ vùng nào cần sáng nhiều.

_(chỉ vào đường đi lên)_
Đường đi lên ngược lại. Mỗi bước dùng **ConvTranspose** để phóng to gấp đôi, rồi **ghép (concatenate)** với đặc trưng cùng độ phân giải ở đường đi xuống. Đây chính là **skip connection**. Nó trả lại các chi tiết không gian nhỏ như cạnh, kết cấu, vốn bị mất khi thu nhỏ.
Cuối cùng, mạng cộng thêm **chính ảnh đầu vào** vào kết quả, gọi là **kết nối thặng dư toàn cục**. Nhờ đó mạng chỉ cần học _phần cần thêm vào_ ảnh tối, chứ không phải vẽ lại cả ảnh. Điều này giúp huấn luyện ổn định hơn.

Tổng số tham số là **1,928 triệu**, đúng ngân sách đề ra.

**[Lưu ý]** Ý chính cần nhớ: _xuống = hiểu ngữ cảnh; lên = phục hồi chi tiết; skip = trả lại chi tiết; residual = chỉ học phần chênh lệch._

---

## Slide 10 — Phương pháp 2: DarkIR-Edge _(giải thích sơ đồ)_

**[Trên slide]** C=24; đầu vào 4 kênh [R,G,B,I_max]; Encoder [1,2,3] với FreMLP; Decoder [3,1,1] với Di-SpAM [1,4,9] + PixelShuffle; Residual Gated Skip; side-loss; 1,924M tham số.

**[Lời nói]**
Mô hình thứ hai là DarkIR-Edge, với số kênh cơ sở 24 và 1,924 triệu tham số, tương đương U-Net. Nên khác biệt về chất lượng chủ yếu đến từ **thiết kế**, không phải từ kích thước.

_(chỉ vào đầu vào)_
Đầu tiên là **đầu vào bốn kênh**: ngoài R, G, B, em thêm kênh $I_{max}$, là giá trị lớn nhất trong ba kênh màu tại mỗi điểm ảnh. Đây là một ước lượng thô về **độ rọi**, giúp mạng biết nơi nào tối.

_(chỉ vào encoder)_
Mạng cũng có dạng chữ U nhưng **bất đối xứng**. Phần **Encoder** có ba tầng, lần lượt **1, 2, 3 khối**. Mỗi khối EBlock có chú ý theo kênh và một nhánh **FreMLP**: nhánh này biến đặc trưng sang miền tần số bằng phép biến đổi Fourier, xử lý ở đó, rồi biến đổi ngược về. Ý tưởng là độ sáng toàn cục nằm nhiều ở thành phần tần số thấp, nên xử lý ở miền tần số có thể giúp chỉnh sáng.
Giữa các tầng, ảnh được thu nhỏ một nửa bằng tích chập có bước nhảy 2.

> 3.  MLP Là Gì? FreMLP Có Phải Multi-Layer Perceptron?
>     Chính xác! MLP là viết tắt của Multi-Layer Perceptron (mạng nơ-ron truyền thẳng nhiều lớp gồm các lớp tuyến tính / Conv 1×1 xen kẽ cổng).
>     Còn FreMLP (Frequency-Domain MLP): Là mạng MLP hoạt động hoàn toàn trong miền tần số Fourier! Thay vì tác động lên pixel ảnh trong không gian, FreMLP can thiệp trực tiếp vào ma trận phổ biên độ tần số của bức ảnh.

> Tại Sao Biến Đổi Fourier Lại Là "Chìa Khóa Vàng" Cho Ảnh Thiếu Sáng?
>
> - Khi thực hiện biến đổi Fourier 2 chiều (2D Fast Fourier Transform - FFT), ma trận ảnh được tách thành 2 thành phần số phức:
>
> * Biên độ Fourier A(u,v) (Amplitude): Chứa 90% thông tin về mức độ chiếu sáng và năng lượng toàn cảnh. Khi ảnh bị tối đen, toàn bộ phổ biên độ co rúm lại ở vùng tần số thấp.
> * Pha Fourier P(u,v) (Phase): Chứa trọn vẹn vị trí không gian, hình học và các đường gờ cạnh sắc nét (edges). Dù ảnh tối om hay sáng rực, pha của vật thể vẫn không thay đổi!
> * ✨ Phát hiện thần kỳ của tác giả: Muốn làm sáng ảnh một cách hoàn hảo mà không làm hỏng cấu trúc, ta chỉ việc: (1) Giữ nguyên 100% Pha P, và (2) Dùng FreMLP để kéo phổ Biên độ A giãn nở lên! Khi nghịch đảo IFFT quay về miền không gian, bức ảnh bừng sáng rực rỡ mà từng cọng tóc, sợi lông mi vẫn sắc lẹm, không hề bị mờ nhòe!

_(chỉ vào bottleneck)_
Ở đáy là bottleneck gồm hai EBlock rồi hai DBlock. Ở đây có thêm **một đầu giám sát phụ (side-loss)**, giúp tín hiệu học lan xuống sâu hơn.

_(chỉ vào decoder)_
Phần **Decoder** ngược lại, có **3, 1, 1 khối**. Mỗi khối DBlock dùng các tích chập giãn nở với độ giãn **1, 4 và 9**, gọi là Di-SpAM. Ba tỉ lệ này cho mạng nhìn được vùng nhỏ lẫn vùng lớn cùng lúc, hữu ích cho việc khử mờ chuyển động vì vệt nhòe có độ dài khác nhau. Phóng to bằng **PixelShuffle**, cách sắp xếp lại kênh thành không gian, tránh hiệu ứng ô vuông.

> 4.  SpAM Là Gì? SCA Là Gì?
>     SpAM (Spatial Attention Module): Là mô-đun chú ý không gian trong Encoder. Nó dùng các phép tích chập 3×3 kết hợp SimpleGate để nắm bắt thông tin cấu trúc cục bộ xung quanh từng pixel.

> SCA (Simplified Channel Attention): Cơ chế chú ý kênh đơn giản hóa. Thay vì bắt các pixel tính tích vô hướng O(N2) như Transformer chuẩn, SCA chỉ cần lấy trung bình toàn bộ không gian của mỗi kênh (Global Average Pooling) để biết kênh màu/đặc trưng nào quan trọng, rồi gán trọng số kênh. Độ phức tạp tính toán chỉ là O(C) siêu nhẹ!

_(chỉ vào skip)_
Các đường nối tắt không ghép thẳng như U-Net, mà đi qua một **cổng lọc, Residual Gated Skip**: đặc trưng được nhân với một cổng giữa 0 và 1 để lọc bớt nhiễu trước khi chuyển sang decoder. Cuối cùng cũng có kết nối thặng dư toàn cục như U-Net.

Tóm lại, so với U-Net, DarkIR-Edge **dùng tích chập chiều sâu, nhánh tần số, tích chập giãn nở và cổng lọc**, với mục tiêu ít FLOPs hơn nhưng nhiều khả năng hơn. Em sẽ cho thấy kết quả thực tế có đúng như kỳ vọng hay không.

**[Lưu ý]** Điểm cần nhớ: _encoder 1-2-3 = lo sáng/nhiễu (có FFT); decoder 3-1-1 = lo nhòe (dilated 1-4-9); thêm 2 cải tiến của em: I_max và Gated Skip._

---

## Slide 10b (đề xuất thêm) — DarkIR gốc đã làm gì

**[Lời nói]** _(xem Phụ lục A cho nội dung slide và đánh giá ý tưởng. Nên đặt TRƯỚC slide 10 để người nghe hiểu DarkIR trước rồi mới nghe em thay đổi gì.)_
Trước khi vào mô hình của em, em xin nói ngắn gọn bài báo DarkIR gốc làm gì.
DarkIR nhận xét rằng ảnh đêm bị hai loại lỗi khác nhau: **sai độ sáng và nhiễu**, và **nhòe**. Thay vì xử lý chung, họ thiết kế **encoder và decoder bất đối xứng**: encoder lo phần độ sáng và nhiễu, có nhánh **FFT** vì độ sáng là đặc tính toàn cục nên dễ xử lý ở miền tần số; decoder lo phần nhòe bằng tích chập **giãn nở nhiều tỉ lệ**.
Em kế thừa khung này, rồi thu nhỏ xuống còn khoảng 1,9 triệu tham số để so sánh công bằng với U-Net.

---

## Slide 11 — Biến thể no-FFT & quy trình huấn luyện

**[Lời nói]**
Mô hình thứ ba là một **thí nghiệm cắt bỏ (ablation)**: giữ nguyên DarkIR-Edge nhưng tắt nhánh FFT. Số tham số giảm **22,3%**, còn 1,495 triệu. Nếu FFT thực sự quan trọng thì bản này phải kém hơn rõ.

**Hàm mất mát** là tổng của nhiều thành phần: sai số điểm ảnh L1, cộng với mất mát SSIM để giữ cấu trúc, mất mát màu để giữ tông màu, và mất mát phụ ở đầu giám sát. Ở v4, em thêm mất mát biên cạnh và mất mát miền tần số.

**Huấn luyện** dùng AdamW, tốc độ học giảm theo cosine, EMA 0,999 để làm mượt trọng số. Có hai điểm quan trọng về phương pháp luận.
Một: em **huấn luyện cố định 70 epoch và tắt early stopping**. Vì ở bản trước, mô hình dừng sớm ở các epoch khác nhau nên so sánh không công bằng.
Hai: mỗi mô hình chạy **2 seed** và em kết luận dựa trên **khoảng tin cậy 95%** bằng bootstrap ghép cặp, quy tắc này được đặt ra _trước_ khi xem kết quả.

---

## Slide 12 — Dataset

**[Lời nói]**
Dữ liệu của em gồm nhiều miền để kiểm tra tính tổng quát.

- **LOLv2-Real**: ảnh trong nhà, thiếu sáng sâu và nhiễu. 639 ảnh huấn luyện.
- **LSRW-Huawei**: ảnh từ điện thoại. 561 ảnh ở v3, mở rộng 2.380 ở v4.
- **LOL-Blur**: ảnh tối kèm nhòe chuyển động. 400 ở v3, **1.858** ở v4.
- **LSRW-Nikon**: ảnh từ máy DSLR. Ở v3 chỉ dùng để _test_, tức là cảm biến chưa từng thấy. Ở v4 em đưa 1.000 ảnh vào huấn luyện.
- **LOLv2-Synthetic**: 900 ảnh tổng hợp, chỉ thêm ở v4.

Tổng: v3 là **1.600 train, 150 validation, 190 test**; v4 là **6.777 train** nhưng **vẫn giữ nguyên 150 validation và 190 test**, nên có thể so v3 với v4 trực tiếp.
Em chú ý **chống rò rỉ dữ liệu**: các ảnh cùng một chuỗi hoặc video không bị chia sang hai tập khác nhau, và có vùng đệm giữa tập train và validation.

**[Lưu ý]** Số cần nhớ: **190 ảnh test, 4 miền test** (blur 40, huawei 30, lolv2 100, nikon 20).

---

## Slide 13 — Độ đo & môi trường _(giải thích độ đo)_

**[Lời nói]**
Em dùng ba độ đo chính, mỗi độ đo trả lời một câu hỏi khác nhau.

**PSNR — Peak Signal-to-Noise Ratio**, tính bằng đề-xi-ben, **càng cao càng tốt**. Nó đo sai số trung bình bình phương giữa từng điểm ảnh của ảnh kết quả và ảnh chuẩn: $PSNR = 10\log_{10}(255^2/MSE)$. Dễ hiểu: "mỗi pixel sai bao nhiêu". Hạn chế: PSNR rất nhạy với **lệch độ sáng toàn cục**, và không phản ánh đúng cảm nhận của mắt người. Một ảnh sáng hơn ảnh chuẩn một chút vẫn có thể bị PSNR phạt nặng.

**SSIM — Structural Similarity**, trong khoảng từ 0 đến 1, **càng cao càng tốt**. Nó so sánh _độ sáng, độ tương phản và cấu trúc_ trong từng cửa sổ nhỏ. Nói đơn giản: "hình dạng, cạnh, kết cấu có giống ảnh chuẩn không". Nó phù hợp hơn PSNR để nhận ra mất cấu trúc.

**LPIPS — Learned Perceptual Image Patch Similarity**, **càng thấp càng tốt**. Nó so khoảng cách giữa các **đặc trưng sâu** trích từ một mạng nơ-ron đã huấn luyện sẵn. Nghĩa là "mắt người thấy hai ảnh giống nhau đến đâu". Đây là độ đo gần cảm quan nhất trong ba độ đo.

Vì sao dùng cả ba? Vì mỗi độ đo có điểm mù riêng, nên em chỉ kết luận khi các độ đo **cùng chiều**.

Em báo cáo hai cách tổng hợp. **Pooled** là trung bình trên cả 190 ảnh, nên miền LOLv2 có 100 ảnh chiếm tỷ trọng lớn. **Macro** là trung bình của bốn miền, mỗi miền có trọng số bằng nhau, nên miền nhỏ như Nikon không bị lấn át.
Ngoài ba độ đo trên, em còn dùng vài độ đo chẩn đoán, như sai lệch độ sáng và độ nét theo Laplacian, để giải thích _tại sao_ kết quả như vậy.

**Môi trường:** huấn luyện và đo trên Google Colab với GPU **NVIDIA L4**. Tốc độ đo ở độ phân giải 600×400. Em chưa đo trên thiết bị di động nên **không tuyên bố chạy thời gian thực trên thiết bị biên**.

---

## Slide 14 — Bảng PSNR

**[Lời nói]**
Đây là bảng PSNR. Em xin chỉ ba điểm.

**Một, so với phương pháp truyền thống.** Ảnh đầu vào chỉ **9,75 dB**. Gamma đạt 16,9. CLAHE và MSRCR chỉ khoảng 11,6 và 12,1, thậm chí thấp hơn Gamma vì khuếch đại nhiễu và lệch màu. Còn cả ba mạng CNN đạt khoảng **21 dB**, tức tốt hơn Gamma từ **4,1 đến 4,4 dB**.

**Hai, so giữa ba mạng CNN.** Ở v3, U-Net đạt **21,31**, DarkIR-Edge **21,10**, no-FFT **21,00**. Ở v4, cả ba nằm trong khoảng **21,0 đến 21,1**. Chênh lệch rất nhỏ, và sau đây em sẽ cho thấy về mặt thống kê gần như **hòa**.

**Ba, so với DarkIR-m chính thức.** Trên toàn bộ 190 ảnh, DarkIR-m đạt 20,96, gần như bằng các mô hình của em. Tuy nhiên ở **riêng miền blur**, DarkIR-m đạt **27,4 đến 27,7 dB**, cao hơn hẳn mô hình của em khoảng 23,5 đến 24. Tức là khử mờ là chỗ mô hình nhỏ còn yếu.

**[Lưu ý]** Nếu bị hỏi: bảng có ± là độ lệch giữa 2 seed.

---

## Slide 15 — Bảng SSIM & LPIPS

**[Lời nói]**
Về SSIM, cả ba mạng nằm trong khoảng **0,73 đến 0,75**, rất sát nhau.
Điểm khác biệt rõ hơn nằm ở **LPIPS**. U-Net đạt **0,277 đến 0,279**, tốt hơn DarkIR-Edge ở v3 là 0,341. Ở v4, khoảng cách thu hẹp: DarkIR-Edge còn 0,307, nhưng U-Net vẫn dẫn.
Nghĩa là U-Net cho ảnh _gần cảm quan hơn_ với ảnh chuẩn. Còn DarkIR-m có LPIPS 0,233, tốt nhất, phù hợp với việc nó là mô hình lớn hơn.

---

## Slide 16 — Kiểm định thống kê

**[Lời nói]**
Với chênh lệch nhỏ như vậy, ta cần hỏi: _đây là khác biệt thật hay chỉ là ngẫu nhiên?_ Em dùng **bootstrap ghép cặp**: lấy mẫu lại các ảnh test nhiều lần, tính hiệu PSNR giữa hai mô hình trên cùng ảnh, rồi xem khoảng tin cậy 95%. **Nếu khoảng này chứa số 0 thì coi là chưa có bằng chứng khác biệt.**

- DarkIR-Edge trừ U-Net: ở v3 là −0,21 dB, khoảng tin cậy từ −0,53 đến +0,13; ở v4 là −0,06 dB, khoảng từ −0,36 đến +0,23. **Cả hai đều chứa 0, tức hòa thống kê.** Riêng ở v3, kiểm định Wilcoxon cho p bằng 0,028, hơi nghiêng về U-Net, em xin nói rõ để trung thực.
- DarkIR-Edge trừ no-FFT: cũng chứa 0 ở cả hai giai đoạn. Kết luận: **nhánh FFT không cho thấy lợi ích đo được ở quy mô này**.
- Dòng dưới là so sánh v4 với v3. Em đặt ra các quy tắc trước, ví dụ nếu thêm dữ liệu làm giảm PSNR trên LOLv2 và Huawei quá 0,3 dB thì coi là "vi phạm". Kết quả: **U-Net bị vi phạm** trên hai miền này, còn miền blur **không cải thiện**, em gọi là _bão hòa dung lượng_.

---

## Slide 17 — Chi phí tính toán

**[Lời nói]**
Đây là bảng chi phí, và có một phát hiện thú vị.
DarkIR-Edge chỉ cần **8,75 GFLOPs**, ít hơn U-Net (18,45) tới **53%**. Về lý thuyết, nó phải nhanh hơn.
Nhưng đo thực tế trên GPU L4, DarkIR-Edge mất **khoảng 32 mili-giây**, còn U-Net chỉ **khoảng 13 mili-giây**, tức DarkIR-Edge **chậm hơn gấp 2,4 lần**.
Lý do: FLOPs chỉ đếm phép tính, còn tốc độ thực phụ thuộc cả **băng thông bộ nhớ** và số lần gọi nhân GPU. Tích chập chiều sâu, FFT và các cổng có ít phép tính nhưng truy xuất bộ nhớ nhiều, trong khi tích chập chuẩn của U-Net được GPU tối ưu rất tốt.
Bản no-FFT giảm 22% tham số, nhanh hơn khoảng 14% và tiết kiệm khoảng 32% bộ nhớ huấn luyện, **mà không giảm chất lượng**.
Bài học: _ít FLOPs không đồng nghĩa với nhanh._

---

## Slide 18 — Minh họa: LOLv2 (v3 và v4)

**[Lời nói]**
Đây là ví dụ định tính trên ảnh LOLv2. Bên trái là ảnh vào rất tối, rồi U-Net, DarkIR-Edge, và ảnh chuẩn.
Cả hai mạng đều kéo sáng ảnh tốt. Ở v3, DarkIR-Edge đạt 27,9 dB trên ảnh này, cao hơn U-Net là 26,3. Nhưng ở v4 thì ngược lại.
Em muốn nhấn mạnh: **đây chỉ là một ảnh**. Xếp hạng trên một ảnh có thể đảo chiều, nên kết luận chính luôn dựa vào **trung bình 190 ảnh** ở các bảng trước.

---

## Slide 19 — Minh họa: LOL-Blur

**[Lời nói]**
Đây là ảnh vừa tối vừa nhòe. Cả hai mạng đều sáng lên, nhưng biên cạnh, ví dụ nét chữ, vẫn còn mờ. Cột bên phải là DarkIR-m chính thức, rõ nét hơn.
Về trung bình, DarkIR-m đạt khoảng **27,4 đến 27,7 dB** trên miền blur, còn mô hình của em chỉ đạt khoảng **23,5 đến 24 dB**.
Điều quan trọng: em tăng dữ liệu blur từ 400 lên 1.858 ảnh mà PSNR **gần như không đổi**. Điều này gợi ý nút thắt không phải là dữ liệu mà là **dung lượng mô hình**: 1,9 triệu tham số là chưa đủ để học khử mờ tốt. Đây là giả thuyết em rút ra từ kết quả, chứ chưa kiểm chứng riêng.

---

## Slide 20 — Đường cong huấn luyện & chẩn đoán

**[Lời nói]**
Biểu đồ bên trái cho thấy cả ba mô hình hội tụ mượt với lịch trình cosine 70 epoch. Trọng số cuối và trọng số tốt nhất chênh nhau không quá 0,07 dB, nên kết quả không phụ thuộc cách chọn checkpoint.
Bên phải là chẩn đoán: nếu em chuẩn hóa độ sáng trung bình theo ảnh chuẩn, PSNR trên LOLv2 **tăng 5 đến 6 dB**. Nghĩa là **phần lớn sai số là lệch độ sáng toàn cục**, chứ không phải mất chi tiết. Đây cũng giải thích vì sao PSNR khoảng 21 dB chưa phản ánh chất lượng thị giác thực tế.
Về độ nét, U-Net có tỷ lệ độ nét cao hơn DarkIR-Edge, nghĩa là khôi phục biên cạnh sắc hơn.

---

## DEMO (~3 phút) — chuyển sang ứng dụng

> **Chuẩn bị:** chạy sẵn `presentation\demo\run_demo.bat`, mở `http://127.0.0.1:7860` trên một tab riêng. Có thể bấm nút **🚀 Demo App** ở góc dưới slide. Chuẩn bị ảnh dự phòng/ảnh chụp màn hình nếu app lỗi.

**[Lời nói mở đầu]**
Sau phần số liệu, em xin phép chuyển sang ứng dụng demo chạy cục bộ, xây bằng Gradio.
Điểm quan trọng: các ảnh mẫu trong demo **không thuộc train, validation hay test**. Em đã kiểm tra bằng mã băm SHA-256 và giao ba tập này với ảnh demo là **rỗng**, tức không rò rỉ. Ứng dụng chạy hoàn toàn offline.

**[Thao tác — Tab "Thanh Trượt Trước / Sau"]**

1. Chọn một ảnh mẫu ở ô "Ảnh Mẫu", ví dụ ảnh Nikon hoặc Huawei.
2. Chọn **Phiên bản** v3, **Mô hình** là Standard U-Net, rồi DarkIR-Edge.
3. _"Em kéo thanh trượt: bên trái là ảnh gốc, bên phải là ảnh đã phục hồi."_ Chỉ vào các vùng tối nay đã thấy rõ chi tiết.
4. Chỉ vào các chỉ số (PSNR/SSIM) hiển thị khi có ảnh chuẩn, và dòng huy hiệu "Out-of-Split Verified".

**[Thao tác — Tab "So Sánh Đa Mô Hình"]** 5. _"Ở tab này em đặt cạnh nhau ảnh gốc, CLAHE, U-Net và DarkIR-Edge."_ Nhấn vào chỗ CLAHE bị nhiễu/lệch màu, còn hai mạng CNN sạch hơn.

**[Thao tác — ảnh ngoài thực tế, nếu có thời gian]** 6. Tải một ảnh đêm bất kỳ từ máy, hoặc chọn ảnh LIME/DICM trong danh sách mẫu (không có ảnh chuẩn nên không có chỉ số).

**[Lời kết demo]**
Qua demo có thể thấy các mạng nhẹ phục hồi tốt hơn rõ rệt các thuật toán truyền thống, đúng như số liệu em vừa trình bày. Em xin quay lại slide để kết luận.

**[Lưu ý]** Nói trung thực: demo dùng bộ trọng số **v3 chính thức** là mặc định (theo quy tắc đã đăng ký). Nếu app báo thiếu checkpoint, bỏ qua demo và dùng ảnh dự phòng.

---

## Slide 21 — Ưu điểm & kết luận

**[Lời nói]**
Em xin tổng kết bốn kết luận chính.
**Một,** các mạng CNN nhẹ cỡ 1,5 đến 1,9 triệu tham số đạt khoảng 21 dB PSNR, vượt Gamma 2,2 từ **4,1 đến 4,4 dB**, và vượt CLAHE, MSRCR khoảng **9 dB**.
**Hai,** khi so sánh công bằng, huấn luyện cố định 70 epoch, hai seed, DarkIR-Edge **hòa thống kê** với U-Net về PSNR, nên em chưa chứng minh được thiết kế tinh vi hơn có lợi.
**Ba,** U-Net gọn là baseline rất mạnh: **nhanh nhất, 13 mili-giây, khoảng 75 khung hình mỗi giây**, và LPIPS tốt nhất trong ba mạng. Biến thể no-FFT thì tiết kiệm 22% tham số và 14% độ trễ mà không giảm chất lượng.
**Bốn,** về phương pháp: em kiểm soát biến, chia dữ liệu cách ly, đăng ký trước quy tắc kết luận, và **báo cáo cả kết quả âm**.

---

## Slide 22 — Hạn chế & hướng phát triển

**[Lời nói]**
Em cũng xin nêu thẳng các hạn chế.
**Thứ nhất,** khử mờ vẫn kém DarkIR-m khoảng 4 dB, và tăng dữ liệu không giúp, em cho rằng do giới hạn dung lượng 1,9 triệu tham số.
**Thứ hai,** có sự đánh đổi độ sáng giữa các miền: thêm miền sáng hơn giúp PSNR trên Huawei tăng 0,3 dB nhưng làm giảm trên LOLv2 khoảng 0,3 dB.
**Thứ ba,** nhánh FFT không có lợi ích thống kê và còn chậm hơn U-Net 2,4 lần trên GPU L4.
**Thứ tư,** chỉ dùng 2 seed, và chưa đo trên thiết bị biên.
Hướng phát triển gồm: điều kiện hóa theo độ phơi sáng để giải quyết xung đột độ sáng giữa các miền; tăng độ rộng kênh lên 32 trở lên để khử mờ tốt hơn; và đo trên phần cứng biên như Jetson hay ONNX, với từ 5 seed trở lên.

---

## Slide 23 — Tài liệu tham khảo → Cảm ơn & Hỏi đáp

**[Lời nói]**
Đây là các tài liệu em tham khảo chính: DarkIR, U-Net, LEDNet và bộ LOL-Blur, LOL-v2, LSRW, LPIPS, cùng hai thuật toán truyền thống CLAHE và MSRCR.
Phần trình bày của em đến đây là hết. **Em xin chân thành cảm ơn thầy/cô và các bạn đã lắng nghe. Em xin sẵn sàng nhận câu hỏi và góp ý ạ.**

---

# PHỤ LỤC A — Slide "DarkIR gốc" (ý tưởng của bạn)

## Đánh giá ý tưởng: **Nên làm, và rất có lợi**

**Vì sao tốt**

1. **Tiêu chí 3 (30 điểm)** yêu cầu "làm nổi bật điểm khác biệt so với phương pháp khác". Slide này làm rõ _DarkIR gốc làm gì_ để slide 10 trình bày _em thay đổi gì_. Hiện người nghe chưa từng đọc DarkIR sẽ khó phân biệt.
2. Giải thích trước các thuật ngữ (FreMLP, Di-SpAM, encoder/decoder bất đối xứng), giúp slide 10 đỡ dày chữ.
3. Gỡ được câu hỏi hay gặp: "Cái nào của em, cái nào là của tác giả?"

**Đề xuất chỉnh**

- **Đặt TRƯỚC slide 10** (giữa slide 9 và 10) thay vì sau, vì người nghe hiểu bản gốc rồi mới xem biến thể. Nếu đặt sau thì cũng được, nhưng cần câu nối "để hiểu DarkIR-Edge, em nói rõ hơn ý tưởng gốc".
- Giữ **3–4 gạch đầu dòng**, không dùng đoạn văn. Chữ lớn như các slide khác.
- Tách rõ **"Của DarkIR (tác giả)"** và **"Của em"**; có thể bằng hai cột hoặc hai màu.
- **Đừng khẳng định chi tiết bạn chưa kiểm tra lại với bài báo** (con số tham số, tên khối). Hãy đối chiếu với bài báo DarkIR và `presentation/demo/models/darkir_edge.py` trước khi chốt.
- Sau khi thêm, tổng số slide thành 24, nhớ cập nhật bộ đếm trang `NN/24` và thêm slide Hỏi đáp riêng nếu muốn (xem lưu ý dưới).

**Nội dung gợi ý trên slide (chữ ngắn)**

> **Tiêu đề:** Ý tưởng cốt lõi của DarkIR (CVPR 2025)
>
> - **Hai loại suy thoái, hai cách xử lý:** sáng/nhiễu và nhòe được xử lý riêng
> - **Encoder:** khối chú ý theo kênh, kèm **FreMLP (FFT)** xử lý độ sáng ở miền tần số
> - **Decoder:** tích chập giãn nở đa tỉ lệ **Di-SpAM** để khử mờ
> - **Bất đối xứng:** encoder sâu dần, decoder gọn, nhằm giữ mô hình nhẹ
> - **DarkIR-Edge (của em):** thu nhỏ còn ~1,9M tham số, thêm kênh **I_max** và **Gated Skip**

**Lời nói** đã viết ở mục Slide 10b ở trên.

> [!NOTE]
> Hiện slide cuối (23) là Tài liệu tham khảo, chưa có slide Hỏi đáp. Gợi ý: thêm một slide "Cảm ơn & Hỏi đáp" ở cuối để tiêu chí 7 rõ ràng, rồi chép câu kết ở Slide 23 sang đó.

---

# PHỤ LỤC B — Câu hỏi dự kiến & cách trả lời ngắn

1. **Vì sao DarkIR-Edge không hơn U-Net?**
   _Em chưa kiểm chứng riêng, nhưng có ba giả thuyết: dữ liệu nhỏ; dung lượng 1,9 triệu tham số bị chia cho nhiều khối; và phần lớn sai số là lệch độ sáng toàn cục nên nhánh FFT không giúp. Thực nghiệm cho thấy bỏ FFT không làm giảm chất lượng._
2. **FLOPs ít hơn mà chậm hơn gấp 2,4 lần?**
   _FLOPs chỉ đếm phép tính. Tích chập chiều sâu, FFT và cổng bị giới hạn bởi băng thông bộ nhớ và số lần gọi nhân GPU._
3. **Vì sao chỉ 2 seed? Có đủ không?**
   _Chưa đủ cho kết luận mạnh. Vì vậy em dùng khoảng tin cậy bootstrap ghép cặp trên 190 ảnh và chỉ kết luận khi khoảng không chứa 0. Hướng phát triển là từ 5 seed trở lên._
4. **Có rò rỉ dữ liệu giữa train và test không?**
   _Em chia theo chuỗi/video và có vùng đệm giữa train và val. Ảnh demo thì được kiểm tra bằng SHA-256, giao với train/val/test là rỗng._
5. **Nikon ở v3 là zero-shot, ở v4 là in-domain, so sánh có công bằng không?**
   _Test set giữ cố định 190 ảnh cho cả hai giai đoạn, và em ghi rõ sự khác biệt này khi so v4 với v3._
6. **Vì sao DarkIR-m tốt hơn trên blur?**
   _DarkIR-m là mô hình lớn hơn (khoảng 5,5 triệu tham số theo slide) và được huấn luyện trên dữ liệu lớn hơn, nên em chỉ dùng làm trần tham chiếu, không so cùng ngân sách._
7. **Ý nghĩa thực tế? Chạy trên điện thoại được không?**
   _U-Net gọn đạt khoảng 75 FPS ở 600×400 trên GPU L4. Em chưa đo trên thiết bị di động nên chưa tuyên bố thời gian thực trên thiết bị biên._
8. **PSNR 21 dB có cao không?**
   _Trên bộ dữ liệu hỗn hợp nhiều cảm biến, đây là mức hợp lý. Chẩn đoán cho thấy chuẩn hóa độ sáng sẽ tăng thêm 5–6 dB trên LOLv2, nghĩa là phần lớn sai số là lệch độ sáng._
9. **Đóng góp của em là gì?**
   _Em không nhận là thuật toán mới thắng. Đóng góp là một so sánh có kiểm soát, cùng ngân sách tham số, kèm ablation FFT, kiểm định thống kê và báo cáo cả kết quả âm._

---

> [!WARNING]
> Một số số liệu trong script (ví dụ DarkIR-m trên blur 27,4/27,7 dB, LPIPS 0,233, các con số v4) được lấy từ bản slide hiện tại. Trước khi thuyết trình, hãy rà lại với `colab-results/*/results_latest/table_main.csv`. Ngoài ra, slide 19, 21, 22 đang ghi sai vài tỷ lệ dữ liệu (xem báo cáo `presentation/report/Danh_Gia_Slide_Theo_Phieu_Cham_Diem.md`); script này đã dùng số **đúng** (blur 400 → 1.858, tăng ~4,2× tổng train).
