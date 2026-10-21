/**
 * Interactive Presentation & Simulation Engine
 * CS2203 UIT - Low-Light Image Enhancement CNN
 */

// State Management
let currentSlide = 0;
const totalSlides = 12;
let isAutoPlaying = false;
let autoPlayTimer = null;
const autoPlayDuration = 18000; // 18 seconds per slide in auto mode

// Slide Init on DOM Loaded
document.addEventListener('DOMContentLoaded', () => {
    initSlideNav();
    initKeyboardShortcuts();
    setupSlide1PixelMatrix();
    setupSlide2DegradationSim();
    setupSlide3ConvAnimation();
    setupSlide6GatedSkipSim();
    setupSlide7SCASim();
    setupSlide11InteractiveLab();
    setupSlide12Accordion();

    // Render KaTeX if available
    renderMathJaxOrKaTeX();
});

function renderMathJaxOrKaTeX() {
    if (window.renderMathInElement) {
        renderMathInElement(document.body, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
            ],
            throwOnError: false
        });
    }
}

// -----------------------------------------------------------------------------
// NAVIGATION ENGINE
// -----------------------------------------------------------------------------
function initSlideNav() {
    const dotsContainer = document.getElementById('nav-dots');
    if (!dotsContainer) return;

    dotsContainer.innerHTML = '';
    for (let i = 0; i < totalSlides; i++) {
        const dot = document.createElement('div');
        dot.className = `nav-dot ${i === 0 ? 'active' : ''}`;
        dot.title = `Chuyển tới Slide ${i + 1}`;
        dot.addEventListener('click', () => goToSlide(i));
        dotsContainer.appendChild(dot);
    }

    updateNavState();
}

function goToSlide(index) {
    if (index < 0 || index >= totalSlides) return;

    const slides = document.querySelectorAll('.slide-card');
    slides.forEach((s, idx) => {
        s.classList.toggle('active', idx === index);
    });

    currentSlide = index;
    updateNavState();

    // Specific slide activation hooks
    if (index === 1) startSlide2Sim();
    if (index === 2) startSlide3Conv();

    // Render math in new slide if needed
    renderMathJaxOrKaTeX();
}

function nextSlide() {
    if (currentSlide < totalSlides - 1) {
        goToSlide(currentSlide + 1);
    }
}

function prevSlide() {
    if (currentSlide > 0) {
        goToSlide(currentSlide - 1);
    }
}

function updateNavState() {
    // Update progress bar
    const progressFill = document.getElementById('slides-progress-fill');
    if (progressFill) {
        const pct = ((currentSlide + 1) / totalSlides) * 100;
        progressFill.style.width = `${pct}%`;
    }

    // Update dots
    const dots = document.querySelectorAll('.nav-dot');
    dots.forEach((dot, idx) => {
        dot.classList.toggle('active', idx === currentSlide);
    });

    // Update counter
    const counterCur = document.getElementById('slide-counter-cur');
    if (counterCur) {
        counterCur.textContent = String(currentSlide + 1).padStart(2, '0');
    }

    // Update buttons disabled state
    const prevBtn = document.getElementById('prev-slide-btn');
    const nextBtn = document.getElementById('next-slide-btn');
    if (prevBtn) prevBtn.disabled = (currentSlide === 0);
    if (nextBtn) nextBtn.disabled = (currentSlide === totalSlides - 1);

    // Scroll to top of slide viewport
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') {
            e.preventDefault();
            nextSlide();
        } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
            e.preventDefault();
            prevSlide();
        } else if (e.key === 'f' || e.key === 'F') {
            toggleFullScreen();
        }
    });
}

function toggleAutoPlay() {
    isAutoPlaying = !isAutoPlaying;
    const btn = document.getElementById('autoplay-btn');
    if (btn) {
        btn.classList.toggle('active', isAutoPlaying);
        btn.innerHTML = isAutoPlaying ? '⏸ Tạm Dừng Video' : '▶ Tự Động Chiếu (Video)';
    }

    if (isAutoPlaying) {
        startAutoPlayLoop();
    } else {
        clearInterval(autoPlayTimer);
    }
}

function startAutoPlayLoop() {
    clearInterval(autoPlayTimer);
    autoPlayTimer = setInterval(() => {
        if (currentSlide < totalSlides - 1) {
            nextSlide();
        } else {
            goToSlide(0); // Loop back
        }
    }, autoPlayDuration);
}

function toggleFullScreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
    } else {
        document.exitFullscreen().catch(() => {});
    }
}

// -----------------------------------------------------------------------------
// SLIDE 1: PIXEL MATRIX & LUX SIMULATOR
// -----------------------------------------------------------------------------
function setupSlide1PixelMatrix() {
    const grid = document.getElementById('slide1-pixel-grid');
    const luxSlider = document.getElementById('slide1-lux-slider');
    const luxVal = document.getElementById('slide1-lux-val');
    const inspectBox = document.getElementById('slide1-inspect-val');
    if (!grid || !luxSlider) return;

    // 5x5 Base Normal-Light Matrix (Values in range 140 - 240)
    const baseMatrix = [
        [160, 185, 210, 190, 165],
        [175, 220, 245, 225, 180],
        [190, 235, 255, 230, 195],
        [170, 215, 240, 210, 175],
        [155, 180, 205, 185, 160]
    ];

    function renderGrid(lux) {
        grid.innerHTML = '';
        for (let r = 0; r < 5; r++) {
            for (let c = 0; c < 5; c++) {
                const base = baseMatrix[r][c];
                // Non-linear low-light darkening
                const degraded = Math.max(1, Math.round(base * Math.pow(lux, 1.8)));
                
                const cell = document.createElement('div');
                cell.className = 'pixel-cell';
                
                // Color mapping: warm amber/white when high, pitch navy/black when low
                const norm = degraded / 255;
                const rCol = Math.round(norm * 240);
                const gCol = Math.round(norm * 220);
                const bCol = Math.round(norm * 210);
                cell.style.backgroundColor = `rgb(${rCol}, ${gCol}, ${bCol})`;
                cell.style.color = norm > 0.4 ? '#0D121D' : '#94A3B8';

                cell.innerHTML = `<span class="val">${degraded}</span>`;
                cell.title = `Tọa độ: (${r}, ${c}) | Cường độ: ${degraded}/255`;

                cell.addEventListener('mouseenter', () => {
                    if (inspectBox) {
                        inspectBox.innerHTML = `Điểm ảnh <strong>(H=${r}, W=${c})</strong>: Giá trị Pixel = <strong>${degraded}</strong>/255 (${(norm*100).toFixed(1)}% độ sáng).`;
                    }
                });

                grid.appendChild(cell);
            }
        }
    }

    luxSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        if (luxVal) luxVal.textContent = `${Math.round(val * 100)}%`;
        renderGrid(val);
    });

    renderGrid(1.0); // Initial normal light
}

// -----------------------------------------------------------------------------
// SLIDE 2: DARKIR DEGRADATION SIMULATOR: y = gamma(x (x) k) + n
// -----------------------------------------------------------------------------
let slide2Step = 1;
let slide2AutoTimer = null;

function setupSlide2DegradationSim() {
    updateSlide2UI(1);
}

function setSlide2Step(step) {
    slide2Step = step;
    updateSlide2UI(step);
}

function nextSlide2Step() {
    let next = slide2Step + 1;
    if (next > 5) next = 1;
    setSlide2Step(next);
}

function autoPlaySlide2Cycle() {
    clearInterval(slide2AutoTimer);
    let s = 1;
    setSlide2Step(s);
    slide2AutoTimer = setInterval(() => {
        s++;
        if (s > 5) {
            clearInterval(slide2AutoTimer);
        } else {
            setSlide2Step(s);
        }
    }, 2500);
}

function startSlide2Sim() {
    autoPlaySlide2Cycle();
}

function updateSlide2UI(step) {
    // Update step pills
    const pills = document.querySelectorAll('#slide2-steps .step-indicator-pill');
    pills.forEach((p, idx) => {
        p.classList.toggle('active', idx + 1 === step);
        p.classList.toggle('completed', idx + 1 < step);
    });

    const canvas = document.getElementById('slide2-canvas');
    const mathFormula = document.getElementById('slide2-math-stage');
    const explainText = document.getElementById('slide2-explain-stage');
    const matrixBox = document.getElementById('slide2-matrix-sample');

    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    // Draw simulation frames based on step
    if (step === 1) {
        // Step 1: Ground Truth x (Crisp, High Dynamic Range, Colorful)
        drawSimulatedScene(ctx, W, H, { blur: 0, darkFactor: 1.0, noise: 0 });
        if (mathFormula) mathFormula.innerHTML = `Trạng thái: <strong>$$\\mathbf{x}$$</strong> (Ảnh Thế Giới Thực - Ground Truth)`;
        if (explainText) explainText.innerHTML = `<strong>Bước 1: Khung cảnh nguyên bản ($\\mathbf{x}$)</strong>. Ánh sáng chan hòa, đường biên sắc nét, không có bất kỳ hạt nhiễu nào. Giá trị điểm ảnh rải đều từ 120 đến 250.`;
        if (matrixBox) renderTinyMatrix(matrixBox, [
            [210, 215, 200],
            [225, 240, 210],
            [195, 205, 180]
        ], 'Sáng Rực & Biên Sắc');
    } else if (step === 2) {
        // Step 2: x (x) k (Motion blur from hand shake / long exposure)
        drawSimulatedScene(ctx, W, H, { blur: 8, darkFactor: 0.95, noise: 0 });
        if (mathFormula) mathFormula.innerHTML = `Trạng thái: <strong>$$\\mathbf{x} \\otimes \\mathbf{k}$$</strong> (Tích chập làm mờ do Rung Tay)`;
        if (explainText) explainText.innerHTML = `<strong>Bước 2: Cú rung tay của máy ảnh ($\\mathbf{k}$)</strong>. Trong đêm, cảm biến cần mở màn trập lâu (vài giây). Tay cầm rung nhẹ tạo nên kernel Point Spread Function $\\mathbf{k}$ làm nhòe mọi đường nét!`;
        if (matrixBox) renderTinyMatrix(matrixBox, [
            [205, 210, 208],
            [212, 215, 210],
            [200, 202, 198]
        ], 'Bị San Phẳng (Nhòe)');
    } else if (step === 3) {
        // Step 3: gamma(x (x) k) (Low photon starvation)
        drawSimulatedScene(ctx, W, H, { blur: 8, darkFactor: 0.08, noise: 0 });
        if (mathFormula) mathFormula.innerHTML = `Trạng thái: <strong>$$\\gamma(\\mathbf{x} \\otimes \\mathbf{k})$$</strong> (Suy giảm Quang Thông Phi Tuyến)`;
        if (explainText) explainText.innerHTML = `<strong>Bước 3: Mất ánh sáng ($\\gamma(\\cdot)$)</strong>. Quá ít hạt photon đập vào cảm biến CMOS. Hàm phi tuyến ép toàn bộ giá trị điểm ảnh từ 200+ sụt thẳng đứng xuống còn 5 - 15. Cảnh vật tối om như mực!`;
        if (matrixBox) renderTinyMatrix(matrixBox, [
            [12, 14, 11],
            [15, 16, 13],
            [9, 10, 8]
        ], 'Tối Thui (Giá Trị Cực Bé)');
    } else if (step === 4) {
        // Step 4: + n (Sensor noise added)
        drawSimulatedScene(ctx, W, H, { blur: 8, darkFactor: 0.08, noise: 35 });
        if (mathFormula) mathFormula.innerHTML = `Trạng thái: <strong>$$\\gamma(\\mathbf{x} \\otimes \\mathbf{k}) + \\mathbf{n}$$</strong> (Nhiễu Cảm Biến Cực Đại)`;
        if (explainText) explainText.innerHTML = `<strong>Bước 4: Tiếng xì xào của vi mạch ($\\mathbf{n}$)</strong>. Để thấy được gì đó, camera đẩy ISO lên 6400+. Dòng điện tử hỗn loạn sinh ra nhiễu Poisson và nhiễu Gaussian đầy hạt muối tiêu và đốm xanh đỏ!`;
        if (matrixBox) renderTinyMatrix(matrixBox, [
            [48, 2, 39],
            [5, 52, 9],
            [41, 11, 46]
        ], 'Hạt Nhiễu Lộn Xộn');
    } else if (step === 5) {
        // Step 5: Final Degraded Image y
        drawSimulatedScene(ctx, W, H, { blur: 8, darkFactor: 0.08, noise: 35 });
        // Draw an alert badge over canvas
        ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
        ctx.fillRect(0, 0, W, H);
        ctx.fillStyle = '#F87171';
        ctx.font = 'bold 16px Inter, sans-serif';
        ctx.fillText('ẢNH QUAN SÁT SUY THOÁI (y) - ĐẦU VÀO CỦA MẠNG CNN', 20, 36);

        if (mathFormula) mathFormula.innerHTML = `Phương trình tổng quát DarkIR: <strong>$$\\mathbf{y} = \\gamma(\\mathbf{x} \\otimes \\mathbf{k}) + \\mathbf{n}$$</strong>`;
        if (explainText) explainText.innerHTML = `<strong>Bước 5: Thảm họa hoàn chỉnh ($\\mathbf{y}$)</strong>. Đây chính là bức ảnh thô mà mô hình CNN của chúng ta nhận được! Nhiệm vụ của mô hình là làm phép màu: <em>vừa tăng sáng, vừa gỡ nhòe, vừa lọc sạch nhiễu</em> để quay ngược về $\\mathbf{x}$!`;
        if (matrixBox) renderTinyMatrix(matrixBox, [
            [48, 2, 39],
            [5, 52, 9],
            [41, 11, 46]
        ], 'Ảnh Quan Sát Đầu Vào');
    }

    renderMathJaxOrKaTeX();
}

function renderTinyMatrix(container, mat, label) {
    let html = `<div style="font-size:0.75rem; color:#94A3B8; margin-bottom:4px; font-weight:600;">Ma trận 3x3 điểm ảnh (${label}):</div>`;
    html += '<div style="display:inline-grid; grid-template-columns:repeat(3, 34px); gap:3px;">';
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            const val = mat[r][c];
            const bg = Math.min(255, val * 3);
            html += `<div style="width:34px; height:34px; border-radius:4px; background:rgb(${val},${Math.round(val*0.9)},${Math.round(val*0.8)}); display:flex; align-items:center; justify-content:center; font-family:var(--font-mono); font-size:0.75rem; font-weight:700; color:${val > 80 ? '#000':'#FFF'}; border:1px solid rgba(255,255,255,0.1);">${val}</div>`;
        }
    }
    html += '</div>';
    container.innerHTML = html;
}

function drawSimulatedScene(ctx, W, H, opts) {
    // Draw synthetic room scene with a lamp, painting on wall, and table
    ctx.save();
    
    // Background wall
    ctx.fillStyle = '#1E293B';
    ctx.fillRect(0, 0, W, H);

    // Painting on wall (crisp details)
    ctx.fillStyle = '#0284C7';
    ctx.fillRect(W * 0.15, H * 0.2, W * 0.3, H * 0.45);
    ctx.fillStyle = '#F59E0B';
    ctx.beginPath();
    ctx.arc(W * 0.25, H * 0.35, 24, 0, Math.PI * 2);
    ctx.fill();

    // Table
    ctx.fillStyle = '#B45309';
    ctx.fillRect(W * 0.55, H * 0.5, W * 0.35, H * 0.35);

    // Lamp on table
    ctx.fillStyle = '#FDE047';
    ctx.beginPath();
    ctx.arc(W * 0.72, H * 0.4, 28, 0, Math.PI * 2);
    ctx.fill();

    // Shading & Floor
    ctx.fillStyle = '#0F172A';
    ctx.fillRect(0, H * 0.82, W, H * 0.18);

    // Apply Blur if needed
    if (opts.blur > 0) {
        ctx.filter = `blur(${opts.blur}px)`;
        ctx.drawImage(ctx.canvas, 0, 0);
        ctx.filter = 'none';
    }

    // Apply Darkening (gamma curve)
    if (opts.darkFactor < 1.0) {
        const imgData = ctx.getImageData(0, 0, W, H);
        const d = imgData.data;
        for (let i = 0; i < d.length; i += 4) {
            d[i] = Math.pow(d[i] / 255, 2.2) * 255 * opts.darkFactor;     // R
            d[i+1] = Math.pow(d[i+1] / 255, 2.2) * 255 * opts.darkFactor; // G
            d[i+2] = Math.pow(d[i+2] / 255, 2.2) * 255 * opts.darkFactor; // B
        }
        ctx.putImageData(imgData, 0, 0);
    }

    // Apply Noise
    if (opts.noise > 0) {
        const imgData = ctx.getImageData(0, 0, W, H);
        const d = imgData.data;
        for (let i = 0; i < d.length; i += 4) {
            const n = (Math.random() - 0.5) * opts.noise * 2;
            d[i] = Math.min(255, Math.max(0, d[i] + n));
            d[i+1] = Math.min(255, Math.max(0, d[i+1] + n));
            d[i+2] = Math.min(255, Math.max(0, d[i+2] + n));
        }
        ctx.putImageData(imgData, 0, 0);
    }

    ctx.restore();
}

// -----------------------------------------------------------------------------
// SLIDE 3: CONVOLUTION 2D SLIDING KERNEL ENGINE
// -----------------------------------------------------------------------------
let convStep = 0;
let convInterval = null;

function setupSlide3ConvAnimation() {
    drawConvStep(0);
}

function startSlide3Conv() {
    clearInterval(convInterval);
    convStep = 0;
    convInterval = setInterval(() => {
        convStep = (convStep + 1) % 9; // 3x3 positions in a 5x5 grid with valid padding
        drawConvStep(convStep);
    }, 1400);
}

function drawConvStep(step) {
    const inputMatrix = [
        [3, 0, 1, 2, 7],
        [1, 5, 8, 9, 3],
        [2, 7, 2, 5, 1],
        [0, 1, 3, 1, 7],
        [4, 3, 1, 6, 2]
    ];

    // Edge detector Kernel 3x3
    const kernel = [
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ];

    const row = Math.floor(step / 3);
    const col = step % 3;

    // Calculate dot product
    let sum = 0;
    let mathTerms = [];
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            const inVal = inputMatrix[row + r][col + c];
            const kVal = kernel[r][c];
            sum += inVal * kVal;
            if (mathTerms.length < 4) {
                mathTerms.push(`(${inVal}×${kVal})`);
            }
        }
    }

    // Render input grid with highlighted kernel
    const inGrid = document.getElementById('slide3-in-grid');
    if (inGrid) {
        inGrid.innerHTML = '';
        for (let r = 0; r < 5; r++) {
            for (let c = 0; c < 5; c++) {
                const cell = document.createElement('div');
                const isUnderKernel = (r >= row && r < row + 3 && c >= col && c < col + 3);
                cell.className = `pixel-cell ${isUnderKernel ? 'highlight-kernel' : ''}`;
                cell.innerHTML = `<span class="val">${inputMatrix[r][c]}</span>`;
                cell.style.backgroundColor = isUnderKernel ? '#1E293B' : '#0B111D';
                inGrid.appendChild(cell);
            }
        }
    }

    // Render Output grid (3x3)
    const outGrid = document.getElementById('slide3-out-grid');
    if (outGrid) {
        outGrid.innerHTML = '';
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                const isCurrent = (r === row && c === col);
                const isCalculated = (r * 3 + c <= step);
                const cell = document.createElement('div');
                cell.className = 'pixel-cell';
                if (isCurrent) {
                    cell.style.border = '2px solid #34D399';
                    cell.style.backgroundColor = 'rgba(52, 211, 153, 0.2)';
                    cell.innerHTML = `<span class="val" style="color:#34D399;">${sum}</span>`;
                } else if (isCalculated) {
                    cell.style.backgroundColor = '#131A29';
                    cell.innerHTML = `<span class="val" style="color:#94A3B8;">✓</span>`;
                } else {
                    cell.style.backgroundColor = '#06090E';
                    cell.innerHTML = `<span class="val" style="color:#334155;">?</span>`;
                }
                outGrid.appendChild(cell);
            }
        }
    }

    // Render calculation text
    const calcBox = document.getElementById('slide3-calc-trace');
    if (calcBox) {
        calcBox.innerHTML = `Vị trí Kernel: Hàng <strong>${row+1}</strong>, Cột <strong>${col+1}</strong> &rarr; Tổng tích: <code>${mathTerms.join(' + ')} + ... = <strong>${sum}</strong></code> (Đặc trưng biên cạnh tại vị trí này).`;
    }
}

// -----------------------------------------------------------------------------
// SLIDE 6: GATED SKIP CONNECTION SIMULATOR
// -----------------------------------------------------------------------------
function setupSlide6GatedSkipSim() {
    const slider = document.getElementById('slide6-gate-threshold');
    const label = document.getElementById('slide6-gate-val');
    const canvas = document.getElementById('slide6-canvas');
    if (!slider || !canvas) return;

    slider.addEventListener('input', (e) => {
        const threshold = parseFloat(e.target.value);
        if (label) label.textContent = threshold.toFixed(2);
        drawGatedSkip(canvas, threshold);
    });

    drawGatedSkip(canvas, 0.5);
}

function drawGatedSkip(canvas, gateThreshold) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Left box: Encoder Shallow Feature (Contains Crisp Edge + High Sensor Noise)
    ctx.fillStyle = '#0F172A';
    ctx.fillRect(20, 30, 140, 160);
    ctx.strokeStyle = '#38BDF8';
    ctx.strokeRect(20, 30, 140, 160);
    ctx.fillStyle = '#38BDF8';
    ctx.font = 'bold 12px Inter';
    ctx.fillText('Encoder L1 Feature', 30, 52);
    ctx.fillStyle = '#94A3B8';
    ctx.font = '11px Inter';
    ctx.fillText('• Cạnh sắc nét (Edge)', 30, 75);
    ctx.fillText('• Nhiễu cảm biến n', 30, 95);

    // Draw noisy points inside
    for (let i = 0; i < 40; i++) {
        ctx.fillStyle = Math.random() > 0.5 ? '#F87171' : '#FBBF24';
        ctx.fillRect(35 + (i * 11) % 110, 110 + (i * 7) % 60, 3, 3);
    }

    // Center: Gating Gate
    ctx.fillStyle = '#1E293B';
    ctx.fillRect(210, 60, 100, 100);
    ctx.strokeStyle = '#34D399';
    ctx.lineWidth = 2;
    ctx.strokeRect(210, 60, 100, 100);
    ctx.fillStyle = '#34D399';
    ctx.font = 'bold 12px Inter';
    ctx.fillText('Cổng Sigmoid', 220, 85);
    ctx.fillText(`Gate = ${gateThreshold}`, 220, 105);

    // Arrow to gate
    ctx.strokeStyle = '#64748B';
    ctx.beginPath();
    ctx.moveTo(160, 110);
    ctx.lineTo(210, 110);
    ctx.stroke();

    // Right: Passed to Decoder
    ctx.fillStyle = '#0F172A';
    ctx.fillRect(360, 30, 140, 160);
    ctx.strokeStyle = '#A78BFA';
    ctx.strokeRect(360, 30, 140, 160);
    ctx.fillStyle = '#A78BFA';
    ctx.font = 'bold 12px Inter';
    ctx.fillText('Đưa vào Decoder', 370, 52);

    // Arrow out of gate
    ctx.strokeStyle = '#64748B';
    ctx.beginPath();
    ctx.moveTo(310, 110);
    ctx.lineTo(360, 110);
    ctx.stroke();

    // Filter effect: Higher threshold = Filter out noise, keep edge
    ctx.fillStyle = '#34D399';
    ctx.fillText('✓ Cạnh bàn (0.95)', 370, 80);
    if (gateThreshold < 0.4) {
        ctx.fillStyle = '#F87171';
        ctx.fillText('⚠️ Nhiễu rò rỉ (0.35)', 370, 105);
        ctx.fillText('Ảnh bị lốm đốm màu!', 370, 130);
    } else {
        ctx.fillStyle = '#34D399';
        ctx.fillText('🛡 Nhiễu bị dập tắt (0.02)', 370, 105);
        ctx.fillText('Ảnh mịn màng & sạch!', 370, 130);
    }
}

// -----------------------------------------------------------------------------
// SLIDE 7: SIMPLIFIED CHANNEL ATTENTION (SCA)
// -----------------------------------------------------------------------------
function setupSlide7SCASim() {
    const btn = document.getElementById('slide7-recalc-btn');
    if (btn) {
        btn.addEventListener('click', () => randomizeSCAChannels());
    }
    randomizeSCAChannels();
}

function randomizeSCAChannels() {
    const container = document.getElementById('slide7-channel-bars');
    if (!container) return;

    // 6 Channels with semantic meanings
    const channels = [
        { name: 'Kênh 1: Vùng Bầu Trời Đêm', noise: 0.8, signal: 0.1 },
        { name: 'Kênh 2: Biên Cạnh Tòa Nhà', noise: 0.1, signal: 0.9 },
        { name: 'Kênh 3: Nhiễu Hạt Cảm Biến Đỏ', noise: 0.95, signal: 0.05 },
        { name: 'Kênh 4: Ánh Sáng Đèn Đường', noise: 0.05, signal: 0.98 },
        { name: 'Kênh 5: Khuôn Mặt Người', noise: 0.1, signal: 0.92 },
        { name: 'Kênh 6: Vệt Mờ Chuyển Động', noise: 0.7, signal: 0.25 }
    ];

    container.innerHTML = '';
    channels.forEach((ch, idx) => {
        // SCA calculates weight w = Conv1x1(GAP(x))
        const weight = Math.max(0.05, Math.min(0.99, ch.signal * (1.1 - ch.noise * 0.8)));
        const isImportant = weight > 0.5;

        const row = document.createElement('div');
        row.style.marginBottom = '12px';
        row.innerHTML = `
            <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:4px;">
                <span style="color:#F8FAFC; font-weight:600;">${ch.name}</span>
                <span style="font-family:var(--font-mono); color:${isImportant ? '#34D399':'#F87171'}; font-weight:700;">
                    Trọng số w = ${weight.toFixed(2)} (${isImportant ? 'Khuếch Đại':'Dập Tắt'})
                </span>
            </div>
            <div style="height:10px; background:rgba(255,255,255,0.08); border-radius:5px; overflow:hidden;">
                <div style="height:100%; width:${Math.round(weight*100)}%; background:${isImportant ? 'linear-gradient(90deg, #38BDF8, #34D399)':'#F87171'}; transition:width 0.4s ease;"></div>
            </div>
        `;
        container.appendChild(row);
    });
}

// -----------------------------------------------------------------------------
// SLIDE 11: INTERACTIVE SIMULATION LAB
// -----------------------------------------------------------------------------
function setupSlide11InteractiveLab() {
    const lux = document.getElementById('lab-lux');
    const noise = document.getElementById('lab-noise');
    const modelSel = document.getElementById('lab-model');
    const canvas = document.getElementById('lab-canvas');

    if (!lux || !noise || !modelSel || !canvas) return;

    function renderLab() {
        const luxVal = parseFloat(lux.value);
        const noiseVal = parseFloat(noise.value);
        const model = modelSel.value;

        const ctx = canvas.getContext('2d');
        const W = canvas.width;
        const H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        // Simulation parameters based on chosen model
        let enhanceFactor = 1.0;
        let residualNoise = noiseVal;
        let edgeSharpness = 1.0;
        let estimatedPSNR = 14.5;
        let estimatedSSIM = 0.52;

        if (model === 'plain') {
            enhanceFactor = 1.6;
            residualNoise = noiseVal * 0.8;
            edgeSharpness = 0.6; // Blurry edges
            estimatedPSNR = 18.2 + luxVal * 3;
            estimatedSSIM = 0.68 + luxVal * 0.1;
        } else if (model === 'compact') {
            enhanceFactor = 2.4;
            residualNoise = noiseVal * 0.65; // Skip leakage
            edgeSharpness = 0.85;
            estimatedPSNR = 21.8 + luxVal * 4;
            estimatedSSIM = 0.79 + luxVal * 0.08;
        } else if (model === 'proposed') {
            enhanceFactor = 3.2;
            residualNoise = noiseVal * 0.15; // Gated skip cleans noise!
            edgeSharpness = 1.2; // SCA sharpens edges!
            estimatedPSNR = 24.6 + luxVal * 4.5;
            estimatedSSIM = 0.88 + luxVal * 0.06;
        }

        // Draw preview
        ctx.fillStyle = '#0F172A';
        ctx.fillRect(0, 0, W, H);

        // Simulated picture element
        const bright = Math.min(255, Math.round(luxVal * 40 * enhanceFactor));
        ctx.fillStyle = `rgb(${bright}, ${Math.round(bright*0.9)}, ${Math.round(bright*1.1)})`;
        ctx.beginPath();
        ctx.arc(W / 2, H / 2, 60 * edgeSharpness, 0, Math.PI * 2);
        ctx.fill();

        // Overlay noise
        const imgData = ctx.getImageData(0, 0, W, H);
        const d = imgData.data;
        for (let i = 0; i < d.length; i += 4) {
            const n = (Math.random() - 0.5) * residualNoise * 1.5;
            d[i] = Math.min(255, Math.max(0, d[i] + n));
            d[i+1] = Math.min(255, Math.max(0, d[i+1] + n));
            d[i+2] = Math.min(255, Math.max(0, d[i+2] + n));
        }
        ctx.putImageData(imgData, 0, 0);

        // Update score tags
        const psnrTag = document.getElementById('lab-psnr-tag');
        const ssimTag = document.getElementById('lab-ssim-tag');
        if (psnrTag) psnrTag.textContent = `${estimatedPSNR.toFixed(2)} dB`;
        if (ssimTag) ssimTag.textContent = `${estimatedSSIM.toFixed(3)}`;
    }

    lux.addEventListener('input', renderLab);
    noise.addEventListener('input', renderLab);
    modelSel.addEventListener('change', renderLab);

    renderLab();
}

// -----------------------------------------------------------------------------
// SLIDE 12: ACCORDION FOR DEFENSE QUESTIONS
// -----------------------------------------------------------------------------
function setupSlide12Accordion() {
    const cards = document.querySelectorAll('.qa-card-slide');
    cards.forEach((c) => {
        const q = c.querySelector('.qa-card-question');
        if (q) {
            q.addEventListener('click', () => {
                c.classList.toggle('open');
            });
        }
    });
}
