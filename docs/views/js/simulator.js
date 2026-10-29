/**
 * simulator.js - 2D Luminance & Histogram Interactive Simulator
 * CS2203 - Low-Light Image Enhancement Project
 */

(function() {
    function initHistogramSimulator() {
        const canvasImg = document.getElementById('canvas-image');
        const canvasHist = document.getElementById('canvas-hist');
        const modeSelect = document.getElementById('sim-mode-select');
        const noiseSlider = document.getElementById('sim-noise-slider');
        const noiseLabel = document.getElementById('noise-val');

        if (!canvasImg || !canvasHist || !modeSelect || !noiseSlider) return;

        window.renderSimulator = function() {
            const mode = modeSelect.value;
            const noise = parseInt(noiseSlider.value, 10);
            if (noiseLabel) noiseLabel.innerText = noise + '%';

            const ctxImg = canvasImg.getContext('2d');
            const ctxHist = canvasHist.getContext('2d');
            const w = canvasImg.width;
            const h = canvasImg.height;

            const imgData = ctxImg.createImageData(w, h);
            const hist = new Array(256).fill(0);

            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const idx = (y * w + x) * 4;
                    let baseVal = 18;
                    if (x > w * 0.6 && y < h * 0.5) baseVal = 42;
                    if (y > h * 0.65) baseVal = 26;
                    if (Math.abs(x - w * 0.4) < 2 || Math.abs(y - h * 0.65) < 2) baseVal += 22;

                    const noiseVal = (Math.random() - 0.5) * noise * 2.2;
                    let val = baseVal + noiseVal;

                    if (mode === 'dark') {
                        val = Math.max(2, Math.min(60, val));
                    } else if (mode === 'linear') {
                        val = val * 3.5;
                    } else if (mode === 'clahe') {
                        val = Math.pow(val / 60, 0.5) * 190 + noiseVal * 1.8;
                    } else if (mode === 'cnn') {
                        val = Math.pow(baseVal / 60, 0.45) * 180 + (noiseVal * 0.22);
                    }

                    val = Math.max(0, Math.min(255, Math.round(val)));
                    hist[val]++;

                    imgData.data[idx] = val;
                    imgData.data[idx + 1] = Math.round(val * 0.95);
                    imgData.data[idx + 2] = Math.round(val * 1.08);
                    imgData.data[idx + 3] = 255;
                }
            }
            ctxImg.putImageData(imgData, 0, 0);

            // Draw Histogram with theme awareness
            ctxHist.clearRect(0, 0, canvasHist.width, canvasHist.height);
            const maxCount = Math.max(...hist.slice(1, 255)) || 1;
            const histW = canvasHist.width;
            const histH = canvasHist.height;

            // Gradient bar for dark aesthetic
            const grad = ctxHist.createLinearGradient(0, histH, 0, 0);
            grad.addColorStop(0, '#0284C7');
            grad.addColorStop(1, '#38BDF8');
            ctxHist.fillStyle = grad;

            for (let i = 0; i < 256; i++) {
                const barH = (hist[i] / maxCount) * (histH - 15);
                const barX = (i / 256) * histW;
                ctxHist.fillRect(barX, histH - barH, histW / 256 + 1, barH);
            }

            const imgDesc = document.getElementById('canvas-img-desc');
            const histDesc = document.getElementById('canvas-hist-desc');
            if (!imgDesc || !histDesc) return;

            if (mode === 'dark') {
                imgDesc.innerText = 'Ảnh thiếu sáng: chi tiết chìm sâu trong vùng tối, mắt người khó phân biệt đối tượng.';
                histDesc.innerText = 'Biểu đồ Histogram: 95% pixel tập trung dồn cục ở dải giá trị rất thấp [0 - 40].';
            } else if (mode === 'linear') {
                imgDesc.innerText = 'Kéo sáng tuyến tính: ảnh sáng lên nhưng nhiễu hạt bị phóng đại khổng lồ, vùng đèn bị cháy trắng.';
                histDesc.innerText = 'Biểu đồ Histogram: Dải pixel bị kéo giãn thưa thớt, xuất hiện các khoảng trống răng cưa (comb artifacts).';
            } else if (mode === 'clahe') {
                imgDesc.innerText = 'CLAHE: Tương phản tăng mạnh nhưng viền vật thể bị quầng sáng (halos) và hạt sạn lốm đốm.';
                histDesc.innerText = 'Biểu đồ Histogram: Trải đều khắp dải [0 - 255] nhưng đỉnh cực đại bị cắt cụm cưỡng bức.';
            } else if (mode === 'cnn') {
                imgDesc.innerText = 'Compact U-Net (Ours): Vùng tối được kéo sáng tự nhiên, cấu trúc biên sắc nét và nhiễu hạt bị triệt tiêu êm dịu.';
                histDesc.innerText = 'Biểu đồ Histogram: Phân phối hình chuông Gauss chuẩn mực của ảnh phơi sáng tự nhiên.';
            }
        };

        modeSelect.addEventListener('change', window.renderSimulator);
        noiseSlider.addEventListener('input', window.renderSimulator);
        window.renderSimulator();
    }

    window.initHistogramSimulator = initHistogramSimulator;
})();
