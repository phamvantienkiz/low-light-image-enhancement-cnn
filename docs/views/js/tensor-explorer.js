/**
 * tensor-explorer.js - Interactive Model Architecture Layer Inspector
 * CS2203 - Low-Light Image Enhancement Project
 */

(function() {
    const layerDetails = {
        enc1: {
            title: "Encoder Stage 1 (Stem + Conv 3x3)",
            text: "Tiếp nhận ảnh đầu vào $[B, 3, 256, 256]$. Khối Stem sử dụng tích chập $3 \\times 3$, padding 1 đưa số kênh lên 32. Đi kèm là BatchNorm và hàm kích hoạt LeakyReLU $(\\alpha = 0.2)$. Vai trò: Giữ lại toàn vẹn độ phân giải không gian $256 \\times 256$, trích xuất các đường biên cạnh và chi tiết tần số cao cục bộ."
        },
        enc2: {
            title: "Encoder Stage 2 (Strided Conv + ResBlock)",
            text: "Hạ mẫu bằng Conv $3 \\times 3$ với bước nhảy `stride=2` xuống kích thước $[B, 64, 128, 128]$. Kết hợp 1 khối dư Residual Block (gồm 2 lớp Conv $3 \\times 3$ và đường truyền tắt nội bộ). Vai trò: Tăng trường tiếp nhận lên $7 \\times 7$, lọc bỏ các xung nhiễu hạt mè cảm biến cấp thấp."
        },
        enc3: {
            title: "Encoder Stage 3 (Strided Conv + ResBlock)",
            text: "Tiếp tục hạ mẫu không gian $2\\times$ xuống $[B, 128, 64, 64]$. Vai trò: Trích xuất các đặc trưng ngữ cảnh bán toàn cục (cấu trúc hình học vật thể, hướng chiếu sáng chính của căn phòng hoặc bóng tối đổ bóng)."
        },
        bottleneck: {
            title: "Bottleneck Layer (Tầng Đáy Không Gian Sâu)",
            text: "Kích thước tensor đạt độ nén tối đa $[B, 256, 32, 32]$. Đây là tầng có trường tiếp nhận bao phủ toàn bộ bức ảnh (Global Receptive Field). Vai trò: Học bản đồ phân rã độ rọi toàn cục (Illumination Map) và điều phối tỷ lệ tương phản động trước khi bước vào pha giải mã tái tạo."
        },
        skip: {
            title: "Gated Skip Connections (Conv 1x1 Gating)",
            text: "Học từ bài báo Edge-LLIE (CVPRW 2024): Thay vì truyền trực tiếp đặc trưng thô từ Encoder sang Decoder (dẫn đến việc mang cả hạt nhiễu cảm biến nặng sang), ta đặt một tầng tích chập điểm $1 \\times 1$ có trọng số học được. Tầng này đóng vai trò bộ lọc thông dải, chỉ cho phép các chi tiết cấu trúc sạch đi qua để nối vào Decoder."
        },
        dec: {
            title: "Decoder & Upsampling Pyramid (Giải Mã Tái Tạo)",
            text: "Sử dụng phép phóng to song tuyến tính Bilinear Upsampling $2\\times$ kết hợp phép nối kênh `torch.cat` từ Skip Connections tương ứng. Các lớp Conv $3 \\times 3$ tiến hành dung hợp đặc trưng để khôi phục dần kích thước từ $32 \\to 64 \\to 128 \\to 256$. Lớp cuối cùng là Conv $1 \\times 1$ với hàm kích hoạt Sigmoid xuất ra ảnh khôi phục hoàn chỉnh $[B, 3, 256, 256]$ có dải giá trị $[0, 1]$."
        }
    };

    function showLayerDetail(key) {
        const box = document.getElementById('layer-detail-box');
        const data = layerDetails[key];
        if (!box || !data) return;

        // Highlight selected node
        document.querySelectorAll('.layer-node').forEach(node => {
            node.classList.remove('selected');
        });
        if (event && event.currentTarget) {
            event.currentTarget.classList.add('selected');
        }

        box.style.display = 'block';
        document.getElementById('layer-detail-title').innerText = data.title;
        document.getElementById('layer-detail-text').innerHTML = data.text;

        if (window.renderMathInElement) {
            renderMathInElement(box, {
                delimiters: [{left: '$', right: '$', display: false}],
                throwOnError: false
            });
        }
    }

    window.showLayerDetail = showLayerDetail;
})();
