/**
 * qa-accordion.js - 15 Graduate Defense Q&A Dataset & Interactive Accordion
 * CS2203 - Low-Light Image Enhancement Project
 */

(function() {
    const defenseQAs = [
        {
            q: "1. Tại sao nhóm chọn Lightweight CNN / U-Net mà không theo trào lưu Transformer hay Diffusion đang rất hot năm 2024-2025?",
            a: "Dạ thưa Thầy Cô, nhóm chủ động chọn Lightweight CNN dựa trên 3 lý do khoa học rõ ràng: <br>1) <strong>Phù hợp với mục tiêu môn học:</strong> Giảng viên yêu cầu hiểu sâu bản chất toán học của tích chập (Conv), trích xuất đặc trưng và quá trình học phục hồi ảnh chứ không phải gọi các thư viện khổng lồ có sẵn. <br>2) <strong>Giới hạn phần cứng thiết bị biên:</strong> Như bài báo NTIRE 2024 đã chỉ ra, 100% mô hình Transformer/Diffusion đều bị tràn bộ nhớ (OOM) trên các card đồ họa tiêu chuẩn 8-12GB và không thể chạy thời gian thực trên camera an ninh hay drone. <br>3) <strong>Hiệu quả tối ưu:</strong> Công trình Edge-LLIE (CVPRW 2024) chứng minh một mạng U-Net nhẹ lượng tử hóa INT8 có thể chạy ở tốc độ gần 200 FPS trên NVIDIA Jetson với chất lượng vượt xa các giải thuật truyền thống."
        },
        {
            q: "2. U-Net vốn sinh ra cho bài toán phân vùng ảnh y tế (Semantic Segmentation), tại sao lại áp dụng tốt cho bài toán tăng cường ảnh (LLIE)?",
            a: "Dạ thưa Thầy Cô, cả hai bài toán đều thuộc dạng <em>Image-to-Image translation</em> đòi hỏi ánh xạ từ một ma trận ảnh sang một ma trận ảnh có cùng kích thước không gian $(H \\times W)$. Điểm mấu chốt của U-Net là <strong>Skip Connections</strong>: các lớp tích chập sâu ở Bottleneck học ngữ cảnh toàn cục (nguồn sáng phân bố ở đâu), trong khi Skip Connections chuyển trực tiếp thông tin không gian độ phân giải cao từ Encoder sang Decoder, giúp ảnh đầu ra không bị mất biên cạnh sắc nét và vân bề mặt."
        },
        {
            q: "3. Nhóm có sử dụng Transfer Learning từ các mô hình tiền huấn luyện trên ImageNet (như ResNet, VGG) cho mạng chính không?",
            a: "Dạ thưa Thầy Cô, nhóm <strong>hoàn toàn không dùng Transfer Learning cho mạng tăng cường chính</strong> mà huấn luyện từ đầu (from scratch). Lý do là các mô hình ImageNet được huấn luyện để phân loại đối tượng trên ảnh ban ngày đủ sáng; phân phối dữ liệu (domain distribution) của chúng hoàn toàn khác với ảnh chụp đêm bị suy thoái bởi nhiễu cảm biến Poisson và bóng tối cực đoan. Huấn luyện từ đầu giúp mạng CNN học chính xác các bộ lọc triệt tiêu nhiễu tối và bù sáng phù hợp nhất."
        },
        {
            q: "4. Skip Connections trong U-Net có nhược điểm gì trong bài toán ảnh thiếu sáng và nhóm xử lý như thế nào?",
            a: "Dạ đây là điểm then chốt mà nhóm học được từ bài báo <strong>Edge-LLIE (CVPRW 2024)</strong>: trong ảnh thiếu sáng, Encoder không chỉ mã hóa chi tiết ảnh mà còn vô tình mã hóa cả hạt nhiễu cảm biến rất nặng. Nếu nối tắt thô sơ bằng `torch.cat`, Decoder sẽ vô tình khuếch đại hạt nhiễu này ra ảnh đầu ra. Nhóm giải quyết bằng cách bổ sung một <strong>lớp tích chập điểm $1 \\times 1$ Gating</strong> tại mỗi đường nối tắt để đóng vai trò bộ lọc không gian, ngăn hạt nhiễu rò rỉ sang Decoder."
        },
        {
            q: "5. Tại sao nhóm dùng hàm mất mát L1 (MAE) làm nền tảng thay vì L2 (MSE) phổ biến?",
            a: "Dạ thưa Thầy Cô, hàm mất mát $L_2$ phạt bình phương sai số $(x - \\hat{x})^2$. Khi gặp ảnh thiếu sáng có nhiều hạt nhiễu cực đoan, hàm $L_2$ sẽ ép mô hình hướng tới giá trị trung bình thống kê (mean), làm cho toàn bộ hình ảnh bị nhòe mượt quá đà (oversmoothing) và mất sạch chi tiết kết cấu. Ngược lại, hàm $L_1$ hướng tới giá trị trung vị (median), bảo toàn biên cạnh sắc nét hơn rất nhiều khi kết hợp cùng SSIM Loss."
        },
        {
            q: "6. PSNR và SSIM khác nhau như thế nào? Có trường hợp nào PSNR rất cao nhưng ảnh nhìn vẫn xấu không?",
            a: "Dạ có thưa Thầy Cô: <br>- <strong>PSNR</strong> là độ đo thuần số học dựa trên sai số từng pixel riêng lẻ ($\\text{MSE}$). Nếu một mô hình làm nhẵn mượt toàn bộ ảnh, MSE có thể rất nhỏ dẫn đến PSNR cao, nhưng thực tế ảnh nhìn mờ bệt như tranh sáp. <br>- <strong>SSIM</strong> đánh giá dựa trên hệ thần kinh thị giác con người (HVS), so sánh cấu trúc cục bộ, tương phản và độ chói. Vì vậy, nhóm luôn đánh giá song hành cả PSNR và SSIM để đảm bảo ảnh vừa chuẩn số học vừa tự nhiên về mặt cảm thụ."
        },
        {
            q: "7. Tại sao paper tham khảo chính DarkIR (CVPR 2025) lại xử lý ánh sáng trong miền tần số Fourier ở Encoder?",
            a: "Dạ theo biến đổi Fourier 2D, thông tin độ sáng toàn cảnh của một bức ảnh liên kết trực tiếp với <strong>Biên độ (Amplitude Spectrum)</strong>, còn pha (Phase Spectrum) quyết định vị trí cấu trúc hình học. Bằng cách can thiệp vào phổ biên độ qua khối Fre-MLP ở Encoder, DarkIR điều chỉnh độ sáng toàn cảnh rất mượt mà và đồng nhất ở mọi tỷ lệ không gian mà không làm biến dạng cấu trúc pha của vật thể."
        },
        {
            q: "8. Hai khuyết tật hình học của không gian màu HSV là gì mà bài báo HVI (CVPR 2025) đã chỉ ra?",
            a: "Dạ thưa Thầy Cô: <br>1) <strong>Red Discontinuity:</strong> Trục sắc độ $H \\in [0, 6)$ bị đứt gãy ở màu đỏ ($h=0$ và $h=6$), khiến mạng nơ-ron sinh ra vệt rách màu đen loang lổ trên các vật thể màu đỏ. <br>2) <strong>Black Plane Noise:</strong> Khi độ sáng $V \\to 0$ (vùng tối đen), mặt phẳng sắc độ $HS$ vẫn là một đĩa tròn đơn vị; nhiễu cảm biến nhỏ nhất cũng làm vector màu xoay hỗn loạn sinh ra nhiễu hạt mè màu sặc sỡ. HVI giải quyết bằng phân cực lượng giác và hàm co sụp $C_k$ nén đĩa màu về điểm 0 khi $I \\to 0$."
        },
        {
            q: "9. Mô hình Compact U-Net của nhóm có khả năng đạt tốc độ thời gian thực (Real-time) trên phần cứng thực tế không?",
            a: "Dạ hoàn toàn khả thi! Mô hình được thiết kế với dưới 1.5M tham số và độ phức tạp tính toán dưới 8 GFLOPs (ảnh $256 \\times 256$). Mạng chỉ sử dụng các toán tử chuẩn (`Conv2d`, `LeakyReLU`, `Bilinear Upsample`) rất thân thiện với bộ tăng tốc TensorRT. Khi xuất sang định dạng ONNX/TensorRT, tốc độ trên GPU máy trạm đạt trên 100 FPS và trên chip nhúng như Jetson Orin hoàn toàn vượt ngưỡng 45-60 FPS."
        },
        {
            q: "10. Nếu ảnh đầu vào vừa tối vừa bị mờ rung tay (Motion Blur) thì mô hình xử lý như thế nào?",
            a: "Dạ thưa Thầy Cô, U-Net thông thường nếu không được huấn luyện trên dữ liệu mờ sẽ có xu hướng khuếch đại vệt mờ thành quầng nhòe sáng. Đây chính là lý do nhóm bổ sung thành phần <strong>Gradient Edge Loss ($\\mathcal{L}_{edge}$)</strong> để phạt nặng các gờ biên bị mờ, đồng thời tham khảo cấu trúc tích chập giãn nở đa tỉ lệ (Dilated Convolutions) của DarkIR ở tầng giải mã để mở rộng trường tiếp nhận phục hồi độ sắc nét."
        },
        {
            q: "11. Tại sao nhóm chọn bộ dữ liệu LOL-v1 để huấn luyện chính thay vì tập ExDark có hơn 7.000 ảnh?",
            a: "Dạ thưa Thầy Cô, tập <strong>LOL-v1</strong> là tập dữ liệu có cặp ảnh đối ứng (paired dataset): máy ảnh được cố định trên chân máy (tripod) chụp cùng 1 góc nhìn ở 2 mức phơi sáng khác nhau, đảm bảo độ chuẩn xác pixel-to-pixel phục vụ phương pháp Supervised Learning và tính toán PSNR/SSIM. Trong khi đó, <strong>ExDark</strong> là ảnh chụp đêm tự nhiên không có ảnh chuẩn (unpaired), nhóm sử dụng ExDark cho giai đoạn kiểm thử mở rộng (Zero-shot / In-the-wild test) để đánh giá độ ổn định thực tế."
        },
        {
            q: "12. Kỹ thuật Progressive Patch Training trong cuộc thi NTIRE 2024 có ý nghĩa gì đối với việc huấn luyện?",
            a: "Dạ chiến lược này huấn luyện mạng theo các giai đoạn tăng dần kích thước crop: ban đầu cắt patch nhỏ ($128 \\times 128$) với batch size lớn để mô hình học nhanh các đặc trưng cục bộ và hội tụ ổn định; các epoch cuối tăng patch lên $256 \\times 256$ hoặc $384 \\times 384$ để mạng thích nghi với các dải tương phản không gian rộng mà không làm tràn bộ nhớ VRAM của GPU."
        },
        {
            q: "13. Tại sao ở lớp đầu ra cuối cùng nhóm lại dùng hàm kích hoạt Sigmoid?",
            a: "Dạ vì toàn bộ dữ liệu ảnh đầu vào và ảnh nhãn Ground Truth đều được chuẩn hóa mức xám về khoảng số thực $[0.0, 1.0]$. Hàm `Sigmoid` có miền giá trị chặt chẽ $(0, 1)$, đảm bảo mọi điểm ảnh đầu ra không bao giờ bị tràn ngưỡng âm hoặc vượt quá giới hạn tối đa, triệt tiêu lỗi hiển thị pixel khi chuyển ngược về ảnh số 8-bit $[0, 255]$."
        },
        {
            q: "14. Nếu tiếp tục phát triển đồ án lên thiết bị camera giám sát nhúng, nhóm sẽ thực hiện các bước kỹ thuật nào tiếp theo?",
            a: "Dạ nhóm sẽ thực hiện 3 bước: <br>1) <strong>Export ONNX:</strong> Cố định đồ thị tính toán tĩnh của mô hình PyTorch. <br>2) <strong>Post-Training Quantization (PTQ INT8):</strong> Lượng tử hóa số nguyên 8-bit với Histogram Calibration (học từ Edge-LLIE) để giảm dung lượng mô hình xuống $4\\times$ và tăng tốc độ xử lý $3-5\\times$. <br>3) <strong>Tối ưu hóa TensorRT Engine:</strong> Nạp vào bo mạch NVIDIA Jetson Nano / Orin và ghép trực tiếp vào luồng đọc khung hình OpenCV/GStreamer."
        },
        {
            q: "15. Đâu là hạn chế lớn nhất của đề tài mà nhóm thẳng thắn thừa nhận?",
            a: "Dạ thưa Thầy Cô, hạn chế lớn nhất của mô hình Compact U-Net là khi đối mặt với <strong>vùng ảnh bị thiếu sáng tuyệt đối (Zero-photon / Completely Dark)</strong> — tức cảm biến không nhận được bất kỳ photon ánh sáng nào mà chỉ toàn tín hiệu nhiễu ngẫu nhiên. Vì mô hình không sử dụng mạng sinh khuếch tán (Generative Diffusion Priors) để 'vẽ bịa' ra vật thể mới, ảnh đầu ra tại những vùng này sẽ có màu sắc hơi nhạt hoặc nhiễu nhẹ. Nhóm nhận định đây là sự đánh đổi trung thực cần thiết để giữ mô hình gọn nhẹ và kiểm soát được hành vi tính toán."
        }
    ];

    function initQA() {
        const container = document.getElementById('qa-container');
        if (!container) return;

        container.innerHTML = defenseQAs.map((item, idx) => `
            <div class="qa-card" id="qa-${idx}">
                <div class="qa-question" onclick="toggleQA(${idx})">
                    <span>${item.q}</span>
                    <span class="qa-icon">+</span>
                </div>
                <div class="qa-answer">${item.a}</div>
            </div>
        `).join('');

        if (window.renderMathInElement) {
            renderMathInElement(container, {
                delimiters: [{left: '$', right: '$', display: false}],
                throwOnError: false
            });
        }
    }

    function toggleQA(idx) {
        const card = document.getElementById(`qa-${idx}`);
        if (card) card.classList.toggle('open');
    }

    window.initQA = initQA;
    window.toggleQA = toggleQA;
})();
