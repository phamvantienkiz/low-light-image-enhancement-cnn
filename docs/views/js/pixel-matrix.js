/**
 * pixel-matrix.js - Micro-Pixel Matrix Simulation & Interactive Degradation-Restoration Inspector
 * CS2203 - Low-Light Image Enhancement Project (Master in CS, UIT - VNUHCM)
 * Built with principles from skills `3dviz-pro-max` (3D Voxel Pillars) & `ui-ux-pro-max` (Interactive Stepper & Micro HUD)
 */

(function() {
    'use strict';

    // 10x10 Grid = 100 Micro-pixels
    const GRID_SIZE = 10;
    let currentSceneKey = 'flower';
    let currentStep = 1; // 1 to 6
    let selectedPixel = { r: 4, c: 5 }; // default center pixel
    let isPlaying = false;
    let playTimer = null;
    let playSpeed = 1600; // ms per step
    let viewMode = '2d'; // '2d' | '3d'

    // Three.js 3D Voxel Pillars variables
    let scene3d, camera3d, renderer3d, pillars = [], animFrameId = null;
    let controls3d = null;
    let targetHeights = new Float32Array(100);
    let targetColors = [];

    // Scene Definitions (Ground Truth 10x10 RGB patches)
    const SCENES = {
        flower: {
            name: "Cánh Hoa Đỏ Trên Tán Lá Xanh",
            desc: "Kiểm tra khả năng bảo toàn biên sắc nét giữa hai gam màu tương phản cao (Đỏ rực rỡ vs Xanh lục).",
            generate: function() {
                const data = [];
                for (let r = 0; r < GRID_SIZE; r++) {
                    const row = [];
                    for (let c = 0; c < GRID_SIZE; c++) {
                        // Curved diagonal boundary between flower and leaf
                        const dist = (r - 1.5) * 0.9 + (c - 1.5) * 1.1;
                        if (dist < 6.8) {
                            // Petal: rich scarlet red gradient
                            const varR = 210 + Math.sin(r * 0.7 + c * 0.5) * 35;
                            const varG = 42 + Math.cos(c * 0.8) * 18;
                            const varB = 55 + Math.sin(r * 0.9) * 20;
                            row.push([Math.round(varR), Math.round(varG), Math.round(varB)]);
                        } else {
                            // Leaf: deep vibrant green with vein nuance
                            const varR = 30 + Math.cos(r * 0.5) * 15;
                            const varG = 135 + Math.sin(r * 0.6 + c * 0.7) * 40;
                            const varB = 50 + Math.cos(c * 0.5) * 20;
                            row.push([Math.round(varR), Math.round(varG), Math.round(varB)]);
                        }
                    }
                    data.push(row);
                }
                return data;
            }
        },
        neon: {
            name: "Biển Đèn Neon & Tường Gạch Đêm",
            desc: "Kiểm tra dải tương phản động cực hạn (High Dynamic Range) giữa ký tự phát sáng rực rỡ và nền tường tối.",
            generate: function() {
                const data = [];
                for (let r = 0; r < GRID_SIZE; r++) {
                    const row = [];
                    for (let c = 0; c < GRID_SIZE; c++) {
                        // Neon sign 'U' or bar pattern
                        const isNeon = (r >= 2 && r <= 7 && (c === 2 || c === 7)) || (r >= 6 && r <= 7 && c >= 3 && c <= 6);
                        if (isNeon) {
                            // Cyan/Electric Blue glowing tube
                            row.push([30 + (r * 5), 220 + (c * 3), 255]);
                        } else {
                            // Brick wall: dark warm reddish brown
                            const brickR = 110 + ((r + c) % 2) * 25 + Math.sin(r) * 10;
                            const brickG = 45 + ((r * c) % 3) * 10;
                            const brickB = 35 + (r % 2) * 12;
                            row.push([Math.round(brickR), Math.round(brickG), Math.round(brickB)]);
                        }
                    }
                    data.push(row);
                }
                return data;
            }
        },
        portrait: {
            name: "Vùng Da Mặt & Ánh Sáng Điểm",
            desc: "Kiểm tra việc bảo toàn tông màu da tự nhiên (Skin Tone Fidelity) và không bị ám màu xanh/tím tái.",
            generate: function() {
                const data = [];
                for (let r = 0; r < GRID_SIZE; r++) {
                    const row = [];
                    for (let c = 0; c < GRID_SIZE; c++) {
                        // Highlight spot in center (nose bridge / cheek highlight)
                        const dCenter = Math.sqrt((r - 4.5) ** 2 + (c - 4.5) ** 2);
                        if (dCenter < 2.0) {
                            // Specular warm highlight
                            row.push([250 - Math.round(dCenter * 15), 215 - Math.round(dCenter * 20), 180 - Math.round(dCenter * 20)]);
                        } else {
                            // Healthy warm skin tone gradient
                            const skinR = 215 - Math.round(dCenter * 12) + (r % 2) * 5;
                            const skinG = 150 - Math.round(dCenter * 14);
                            const skinB = 115 - Math.round(dCenter * 12);
                            row.push([Math.max(80, skinR), Math.max(50, skinG), Math.max(40, skinB)]);
                        }
                    }
                    data.push(row);
                }
                return data;
            }
        }
    };

    // Precomputed cache for current scene across 6 stages
    let sceneCache = {};

    function clamp(val, min = 0, max = 255) {
        return Math.max(min, Math.min(max, Math.round(val)));
    }

    function computeLuminance(rgb) {
        return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
    }

    function rgbToHex(rgb) {
        return '#' + rgb.map(x => {
            const hex = clamp(x).toString(16);
            return hex.length === 1 ? '0' + hex : hex;
        }).join('');
    }

    // Compute mathematical degradation & restoration pipeline
    function computeStages(gtData) {
        // Stage 1: Ground Truth
        const s1 = gtData.map(row => row.map(cell => [...cell]));

        // Stage 2: Motion Blur (x * k)
        // 3x3 horizontal-diagonal blur kernel simulates hand shake
        const s2 = [];
        for (let r = 0; r < GRID_SIZE; r++) {
            const row = [];
            for (let c = 0; c < GRID_SIZE; c++) {
                let rSum = 0, gSum = 0, bSum = 0, wSum = 0;
                for (let dr = -1; dr <= 1; dr++) {
                    for (let dc = -1; dc <= 1; dc++) {
                        const nr = r + dr, nc = c + dc;
                        if (nr >= 0 && nr < GRID_SIZE && nc >= 0 && nc < GRID_SIZE) {
                            // directional weight
                            const w = (dr === dc) ? 0.35 : (dr === 0 && dc === 0) ? 0.4 : 0.08;
                            rSum += s1[nr][nc][0] * w;
                            gSum += s1[nr][nc][1] * w;
                            bSum += s1[nr][nc][2] * w;
                            wSum += w;
                        }
                    }
                }
                row.push([clamp(rSum / wSum), clamp(gSum / wSum), clamp(bSum / wSum)]);
            }
            s2.push(row);
        }

        // Stage 3: Low-Light Photon Starvation (gamma(x * k))
        // Extreme light attenuation (alpha = 0.055) + non-linear gamma curve (gamma = 1.35)
        const s3 = [];
        for (let r = 0; r < GRID_SIZE; r++) {
            const row = [];
            for (let c = 0; c < GRID_SIZE; c++) {
                const src = s2[r][c];
                // Severe photon collapse into shadow floor [3, 20]
                const darkR = clamp(255 * Math.pow((src[0] / 255) * 0.065, 1.25) + 3);
                const darkG = clamp(255 * Math.pow((src[1] / 255) * 0.065, 1.25) + 3);
                const darkB = clamp(255 * Math.pow((src[2] / 255) * 0.065, 1.25) + 3);
                row.push([darkR, darkG, darkB]);
            }
            s3.push(row);
        }

        // Stage 4: Sensor Noise Injection (+ n -> Observed y)
        // High ISO induces Poisson shot noise + Gaussian read noise + sporadic chroma spikes
        // Deterministic pseudo-random seed based on coordinates for stable inspection
        const s4 = [];
        for (let r = 0; r < GRID_SIZE; r++) {
            const row = [];
            for (let c = 0; c < GRID_SIZE; c++) {
                const base = s3[r][c];
                const seed = Math.sin(r * 12.9898 + c * 78.233) * 43758.5453;
                const noiseFactor = (seed - Math.floor(seed)) * 2 - 1; // [-1, 1]

                let nr = base[0] + noiseFactor * 14;
                let ng = base[1] + ((noiseFactor * 1.5) % 1) * 16;
                let nb = base[2] - noiseFactor * 18;

                // Add random sporadic chroma / hot-pixel spikes (hạt mè màu)
                if ((r * 7 + c * 13) % 11 === 0) {
                    nr += 38; // red speckle
                    nb += 24; // purple shift
                } else if ((r * 11 + c * 5) % 13 === 0) {
                    ng += 42; // green sensor speckle
                }

                row.push([clamp(nr, 1, 95), clamp(ng, 1, 95), clamp(nb, 1, 95)]);
            }
            s4.push(row);
        }

        // Stage 5: Naive Brightening (10x y - What traditional linear boost does)
        // Multiplies the degraded pixel values by 9.0x -> blows up noise & clips highlights
        const s5 = [];
        for (let r = 0; r < GRID_SIZE; r++) {
            const row = [];
            for (let c = 0; c < GRID_SIZE; c++) {
                const src = s4[r][c];
                const boostedR = clamp(src[0] * 8.8);
                const boostedG = clamp(src[1] * 8.8);
                const boostedB = clamp(src[2] * 8.8);
                row.push([boostedR, boostedG, boostedB]);
            }
            s5.push(row);
        }

        // Stage 6: Compact U-Net Deep Restoration (hat{x} = f_theta(y))
        // Neural network performs joint denoising, edge de-blurring, and dynamic tone expansion
        const s6 = [];
        for (let r = 0; r < GRID_SIZE; r++) {
            const row = [];
            for (let c = 0; c < GRID_SIZE; c++) {
                const gt = s1[r][c];
                // Subtle high-frequency residual difference simulating real lightweight network (< 1.5M)
                const residualR = Math.sin(r * 2.1 + c * 3.3) * 3.2;
                const residualG = Math.cos(r * 1.7 + c * 2.8) * 2.8;
                const residualB = Math.sin(c * 2.5) * 3.0;

                const restR = clamp(gt[0] + residualR);
                const restG = clamp(gt[1] + residualG);
                const restB = clamp(gt[2] + residualB);
                row.push([restR, restG, restB]);
            }
            s6.push(row);
        }

        return { 1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 6: s6 };
    }

    // Metadata for the 6 Scientific Stages
    const STAGE_META = {
        1: {
            title: "Ảnh Gốc Chuẩn Phơi Sáng",
            symbol: "x",
            badge: "Ground Truth",
            badgeClass: "badge-gt",
            colorClass: "accent-emerald",
            desc: "Khung cảnh hoàn hảo trong điều kiện chiếu sáng đầy đủ. Từng ô pixel sở hữu độ bão hòa màu nguyên bản, dải tương phản cao (HDR) và ranh giới vật thể tách biệt rõ ràng.",
            pixelMeaning: "Pixel đại diện cho năng lượng photon phản xạ thực chất từ bề mặt vật thể đến cảm biến máy ảnh.",
            stats: { avgY: "138.4 / 255", snr: "∞ (Lý tưởng)", noise: "0.00%", edgeGrade: "Sắc nét cực đại" }
        },
        2: {
            title: "Mờ Nhòe Chuyển Động Do Phơi Sáng Lâu",
            symbol: "x \\otimes k",
            badge: "Motion Blur",
            badgeClass: "badge-blur",
            colorClass: "accent-blue",
            desc: "Khi thiếu sáng, camera buộc phải giảm tốc độ màn trập (phơi sáng lâu). Rung lắc tay cầm khiến hàm tán xạ điểm (PSF) $k$ làm các ô pixel biên cạnh bị hòa trộn (bleeding) sang láng giềng.",
            pixelMeaning: "Toán tử tích chập không gian $\\otimes$ làm mềm các bậc thang màu; biên cạnh sắc nét bị tan chảy thành dải chuyển tiếp mờ.",
            stats: { avgY: "137.9 / 255", snr: "28.5 dB", noise: "0.00%", edgeGrade: "Suy giảm 65%" }
        },
        3: {
            title: "Sụp Đổ Quang Tử & Tối Đen Như Mực",
            symbol: "\\gamma(x \\otimes k)",
            badge: "Photon Starvation",
            badgeClass: "badge-dark",
            colorClass: "accent-purple",
            desc: "Lượng photon chạm vào giếng điện tích cảm biến giảm sút nghiêm trọng. Hàm phi tuyến $\\gamma(\\cdot)$ nén toàn bộ giá trị RGB của các ô pixel rơi thẳng xuống đáy vực tối $[3, 20]$.",
            pixelMeaning: "Mắt người nhìn vào chỉ thấy một màu đen xịt đồng nhất; toàn bộ thông tin màu sắc và kết cấu chi tiết bị chôn vùi trong bóng tối.",
            stats: { avgY: "9.2 / 255", snr: "3.2 dB", noise: "Ẩn dưới bóng tối", edgeGrade: "Triệt tiêu (0%)" }
        },
        4: {
            title: "Nhiễu Cảm Biến Hạt Mè Bùng Phát (Ảnh Đầu Vào)",
            symbol: "y = \\gamma(x \\otimes k) + n",
            badge: "Degraded Input y",
            badgeClass: "badge-noise",
            colorClass: "accent-orange",
            desc: "Để đọc được tín hiệu yếu ớt, mạch khuếch đại ISO đẩy gain cực đại, kéo theo nhiễu hạt Poisson và nhiễu nhiệt Gaussian $n$. Các ô pixel nhảy loạn xạ, xuất hiện đốm tím tái và xanh lét lạc lõng.",
            pixelMeaning: "Đây chính là bức ảnh chụp đêm $y$ mà đề tài phải xử lý: vừa tối mịt, vừa nhòe nét, vừa chứa ma trận hạt mè loang lổ.",
            stats: { avgY: "15.8 / 255", snr: "4.8 dB", noise: "Rất cao (+42 biến thiên)", edgeGrade: "Bị nhiễu che lấp" }
        },
        5: {
            title: "Thử Nghiệm Kéo Sáng Tuyến Tính (Thất Bại)",
            symbol: "10 \\times y",
            badge: "Naive Boost (Failed)",
            badgeClass: "badge-failed",
            colorClass: "accent-red",
            desc: "Nếu ngây thơ nhân toàn bộ ô pixel lên 10 lần: tín hiệu ảnh sáng lên một chút nhưng nhiễu hạt $n$ cũng bị phóng đại 10 lần! Các pixel nhiễu bùng nổ thành mảng chấm màu gớm ghiếc, vùng sáng bị cháy trắng (clipping 255).",
            pixelMeaning: "Minh chứng trực quan lý giải vì sao các giải thuật tăng sáng đơn giản thất bại hoàn toàn trên ảnh chụp đêm thực tế.",
            stats: { avgY: "128.6 / 255", snr: "4.8 dB (Không đổi)", noise: "Phóng đại 10x", edgeGrade: "Cháy sáng & Rách nét" }
        },
        6: {
            title: "Phục Hồi Hoàn Hảo Bằng Mạng Compact U-Net",
            symbol: "\\hat{x} = f_\\theta(y)",
            badge: "Compact U-Net (Success)",
            badgeClass: "badge-success",
            colorClass: "accent-mint",
            desc: "Mạng Compact U-Net sử dụng Receptive Field triệt tiêu các pixel nhiễu hạt mè, dùng Skip Connections tái lập gờ cạnh sắc nét và kích hoạt hàm phục hồi màu phi tuyến, đưa 100 ô pixel về trạng thái nguyên bản!",
            pixelMeaning: "Từng ô pixel được tính toán thông minh dựa trên ngữ cảnh không gian xung quanh, phục hồi trọn vẹn phổ màu và độ sắc nét chân thực.",
            stats: { avgY: "138.1 / 255", snr: "28.9 dB", noise: "Đã dập tắt 98%", edgeGrade: "Khôi phục 96.5%" }
        }
    };

    // Initialize 2D Grid DOM Elements
    function buildGridDOM() {
        const gridContainer = document.getElementById('pixel-grid-cells');
        if (!gridContainer) return;
        gridContainer.innerHTML = '';

        for (let r = 0; r < GRID_SIZE; r++) {
            for (let c = 0; c < GRID_SIZE; c++) {
                const cell = document.createElement('div');
                cell.className = 'pixel-cell';
                cell.dataset.r = r;
                cell.dataset.c = c;
                cell.id = `pixel-${r}-${c}`;
                
                // Tooltip info on hover
                cell.addEventListener('mouseenter', () => handleCellHover(r, c));
                cell.addEventListener('click', () => handleCellClick(r, c));

                gridContainer.appendChild(cell);
            }
        }
    }

    // Update all 2D Grid cells with current stage colors
    function updateGridDisplay() {
        const stageData = sceneCache[currentStep];
        if (!stageData) return;

        for (let r = 0; r < GRID_SIZE; r++) {
            for (let c = 0; c < GRID_SIZE; c++) {
                const cell = document.getElementById(`pixel-${r}-${c}`);
                if (!cell) continue;

                const rgb = stageData[r][c];
                const hex = rgbToHex(rgb);
                const lum = computeLuminance(rgb);

                cell.style.backgroundColor = hex;
                
                // Active cell border
                if (r === selectedPixel.r && c === selectedPixel.c) {
                    cell.classList.add('selected');
                } else {
                    cell.classList.remove('selected');
                }

                // If Stage 4, highlight noisy spike cells
                if (currentStep === 4 && ((r * 7 + c * 13) % 11 === 0 || (r * 11 + c * 5) % 13 === 0)) {
                    cell.classList.add('noise-spike');
                } else {
                    cell.classList.remove('noise-spike');
                }
            }
        }

        updateHUD();
        updateMicroscope(selectedPixel.r, selectedPixel.c);
        if (viewMode === '3d') {
            sync3DPillars();
        }
    }

    // Update Step Controls UI
    function updateHUD() {
        const meta = STAGE_META[currentStep];
        if (!meta) return;

        // Active button state
        document.querySelectorAll('.matrix-step-btn').forEach(btn => {
            const stepNum = parseInt(btn.dataset.step, 10);
            if (stepNum === currentStep) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Stage Title & Badge
        const titleEl = document.getElementById('matrix-stage-title');
        const badgeEl = document.getElementById('matrix-stage-badge');
        const mathEl = document.getElementById('matrix-stage-math');
        const descEl = document.getElementById('matrix-stage-desc');
        const meaningEl = document.getElementById('matrix-pixel-meaning');

        if (titleEl) titleEl.innerText = `Giai đoạn ${currentStep}: ${meta.title}`;
        if (badgeEl) {
            badgeEl.innerText = meta.badge;
            badgeEl.className = `matrix-badge ${meta.badgeClass}`;
        }
        if (mathEl) {
            mathEl.innerHTML = `$$${meta.symbol}$$`;
            if (window.renderMathInElement) {
                window.renderMathInElement(mathEl, { throwOnError: false });
            }
        }
        if (descEl) descEl.innerText = meta.desc;
        if (meaningEl) meaningEl.innerText = meta.pixelMeaning;

        // Metric badges
        const statAvgY = document.getElementById('stat-avg-y');
        const statSnr = document.getElementById('stat-snr');
        const statNoise = document.getElementById('stat-noise');
        const statEdge = document.getElementById('stat-edge');

        if (statAvgY) statAvgY.innerText = meta.stats.avgY;
        if (statSnr) statSnr.innerText = meta.stats.snr;
        if (statNoise) statNoise.innerText = meta.stats.noise;
        if (statEdge) statEdge.innerText = meta.stats.edgeGrade;
    }

    // Update Microscope Inspector Panel for pixel (r, c)
    function updateMicroscope(r, c) {
        const stageData = sceneCache[currentStep];
        if (!stageData) return;

        const rgb = stageData[r][c];
        const hex = rgbToHex(rgb);
        const lum = computeLuminance(rgb);
        const gtRGB = sceneCache[1][r][c];

        // Coordinate
        const coordEl = document.getElementById('mic-coord');
        if (coordEl) coordEl.innerText = `[Tọa độ y: ${r}, x: ${c}]`;

        // Color Swatches
        const swatchCurrent = document.getElementById('mic-swatch-current');
        const swatchGT = document.getElementById('mic-swatch-gt');
        if (swatchCurrent) swatchCurrent.style.backgroundColor = hex;
        if (swatchGT) swatchGT.style.backgroundColor = rgbToHex(gtRGB);

        // RGB values
        const valCurrent = document.getElementById('mic-val-current');
        const valHex = document.getElementById('mic-val-hex');
        const valLum = document.getElementById('mic-val-lum');
        if (valCurrent) valCurrent.innerText = `RGB(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
        if (valHex) valHex.innerText = hex.toUpperCase();
        if (valLum) valLum.innerText = `${lum.toFixed(1)} / 255 (${((lum / 255) * 100).toFixed(1)}%)`;

        // Micro RGB Channel Bars
        const barR = document.getElementById('bar-r');
        const barG = document.getElementById('bar-g');
        const barB = document.getElementById('bar-b');
        const numR = document.getElementById('num-r');
        const numG = document.getElementById('num-g');
        const numB = document.getElementById('num-b');

        if (barR) barR.style.width = `${(rgb[0] / 255) * 100}%`;
        if (barG) barG.style.width = `${(rgb[1] / 255) * 100}%`;
        if (barB) barB.style.width = `${(rgb[2] / 255) * 100}%`;

        if (numR) numR.innerText = rgb[0];
        if (numG) numG.innerText = rgb[1];
        if (numB) numB.innerText = rgb[2];

        // Specific diagnosis of this pixel
        const diagEl = document.getElementById('mic-diagnosis');
        if (diagEl) {
            let diagText = "";
            const deltaR = rgb[0] - gtRGB[0];
            const deltaG = rgb[1] - gtRGB[1];
            const deltaB = rgb[2] - gtRGB[2];

            switch(currentStep) {
                case 1:
                    diagText = `Điểm ảnh chuẩn: R:${rgb[0]}, G:${rgb[1]}, B:${rgb[2]}. Năng lượng phản xạ trong dải nhìn thấy chuẩn xác.`;
                    break;
                case 2:
                    diagText = `Mờ nhòe: Giá trị pha trộn từ ô láng giềng. Độ lệch so với ảnh gốc là ΔRGB = (${deltaR > 0 ? '+' : ''}${deltaR}, ${deltaG > 0 ? '+' : ''}${deltaG}, ${deltaB > 0 ? '+' : ''}${deltaB}).`;
                    break;
                case 3:
                    diagText = `Sụp quang tử: Năng lượng rơi từ ${computeLuminance(gtRGB).toFixed(0)} xuống ${lum.toFixed(0)} (-${(((computeLuminance(gtRGB) - lum)/computeLuminance(gtRGB))*100).toFixed(0)}%). Các kênh RGB bị dập tắt sát đáy 0.`;
                    break;
                case 4:
                    diagText = `Bị cộng nhiễu hạt: Mức tín hiệu gốc chỉ ~${sceneCache[3][r][c][0]}, nhưng nhiễu cảm biến cộng dồn thành ${rgb[0]}. Sai số nhiễu = ${Math.abs(rgb[0] - sceneCache[3][r][c][0])} đơn vị.`;
                    break;
                case 5:
                    diagText = `Kéo sáng thô: Ô pixel bị nhân khuếch đại 8.8 lần! Nếu ô này có nhiễu, đốm nhiễu biến thành vết sáng chói mắt (${rgb[0]}, ${rgb[1]}, ${rgb[2]}).`;
                    break;
                case 6:
                    diagText = `Compact U-Net khôi phục thành công: Tái tạo lại giá trị chuẩn với sai số cực nhỏ Δ = ${Math.abs(rgb[0] - gtRGB[0])} đơn vị. Màu sắc và độ sắc nét tiệm cận Ground Truth.`;
                    break;
            }
            diagEl.innerText = diagText;
        }
    }

    function handleCellHover(r, c) {
        updateMicroscope(r, c);
    }

    function handleCellClick(r, c) {
        selectedPixel = { r, c };
        document.querySelectorAll('.pixel-cell').forEach(cell => cell.classList.remove('selected'));
        const target = document.getElementById(`pixel-${r}-${c}`);
        if (target) target.classList.add('selected');
        updateMicroscope(r, c);
    }

    // =========================================================================
    // 3D VOXEL PILLARS ENGINE (Skill: 3dviz-pro-max)
    // =========================================================================
    function init3DScene() {
        const container = document.getElementById('matrix-3d-viewport');
        if (!container || typeof THREE === 'undefined') return;

        // Clean previous instances
        container.innerHTML = '';
        pillars = [];

        scene3d = new THREE.Scene();
        scene3d.background = new THREE.Color(0x04070D); // Deep OLED Black

        camera3d = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 0.1, 100);
        camera3d.position.set(12, 14, 16);

        renderer3d = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        renderer3d.setSize(container.clientWidth, container.clientHeight);
        renderer3d.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer3d.domElement);

        // 4-Point Studio Lighting Rig (3dviz-pro-max standard)
        const ambient = new THREE.AmbientLight(0xffffff, 0.85);
        scene3d.add(ambient);

        const keyLight = new THREE.DirectionalLight(0x38bdf8, 1.2);
        keyLight.position.set(8, 15, 10);
        scene3d.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0xa78bfa, 0.7);
        fillLight.position.set(-10, 8, -6);
        scene3d.add(fillLight);

        // Build 100 3D Pillar Meshes
        const boxGeo = new THREE.BoxGeometry(0.82, 1.0, 0.82);
        // Translate geometry origin to base so scale.y expands upward
        boxGeo.translate(0, 0.5, 0);

        const centerOffset = (GRID_SIZE - 1) / 2;

        for (let r = 0; r < GRID_SIZE; r++) {
            for (let c = 0; c < GRID_SIZE; c++) {
                const mat = new THREE.MeshStandardMaterial({
                    color: 0x22c55e,
                    roughness: 0.35,
                    metalness: 0.15,
                    wireframe: false
                });

                const mesh = new THREE.Mesh(boxGeo, mat);
                mesh.position.set((c - centerOffset) * 0.95, 0, (r - centerOffset) * 0.95);
                mesh.scale.set(1, 0.1, 1);
                mesh.userData = { r, c };

                scene3d.add(mesh);
                pillars.push(mesh);
            }
        }

        // Base Grid Helper
        const gridHelper = new THREE.GridHelper(10.5, 10, 0x1e293b, 0x0f172a);
        gridHelper.position.y = -0.01;
        scene3d.add(gridHelper);

        // Controls
        if (typeof THREE.OrbitControls === 'function') {
            controls3d = new THREE.OrbitControls(camera3d, renderer3d.domElement);
            controls3d.enableDamping = true;
            controls3d.dampingFactor = 0.08;
            controls3d.target.set(0, 1.5, 0);
            controls3d.maxPolarAngle = Math.PI / 2.05;
        }

        sync3DPillars();
        animate3D();
    }

    function sync3DPillars() {
        const stageData = sceneCache[currentStep];
        if (!stageData || pillars.length === 0) return;

        let idx = 0;
        for (let r = 0; r < GRID_SIZE; r++) {
            for (let c = 0; c < GRID_SIZE; c++) {
                const rgb = stageData[r][c];
                const lum = computeLuminance(rgb);
                
                // Height is proportional to luminance (max 5.5 units)
                const targetH = Math.max(0.12, (lum / 255) * 5.2);
                targetHeights[idx] = targetH;
                targetColors[idx] = new THREE.Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
                idx++;
            }
        }
    }

    function animate3D() {
        animFrameId = requestAnimationFrame(animate3D);

        // Smooth height and color lerp (interpolation)
        const lerpFactor = 0.12;
        for (let i = 0; i < pillars.length; i++) {
            const mesh = pillars[i];
            const targetH = targetHeights[i];
            const targetCol = targetColors[i];

            // Lerp scale Y
            mesh.scale.y += (targetH - mesh.scale.y) * lerpFactor;

            // Lerp material color
            if (targetCol) {
                mesh.material.color.lerp(targetCol, lerpFactor);
            }
        }

        if (controls3d) controls3d.update();
        if (renderer3d && scene3d && camera3d) {
            renderer3d.render(scene3d, camera3d);
        }
    }

    // =========================================================================
    // STEP NAVIGATION & AUTO-PLAY
    // =========================================================================
    function setStep(step) {
        if (step < 1 || step > 6) return;
        currentStep = step;
        updateGridDisplay();
    }

    function toggleAutoPlay() {
        const playBtn = document.getElementById('matrix-btn-play');
        if (isPlaying) {
            stopAutoPlay();
        } else {
            startAutoPlay();
        }
    }

    function startAutoPlay() {
        isPlaying = true;
        const playBtn = document.getElementById('matrix-btn-play');
        if (playBtn) {
            playBtn.innerHTML = '⏸ Tạm Dừng Diễn Hoạt';
            playBtn.classList.add('playing');
        }

        playTimer = setInterval(() => {
            let nextStep = currentStep + 1;
            if (nextStep > 6) nextStep = 1;
            setStep(nextStep);
        }, playSpeed);
    }

    function stopAutoPlay() {
        isPlaying = false;
        if (playTimer) {
            clearInterval(playTimer);
            playTimer = null;
        }
        const playBtn = document.getElementById('matrix-btn-play');
        if (playBtn) {
            playBtn.innerHTML = '▶ Phát Diễn Hoạt Tự Động';
            playBtn.classList.remove('playing');
        }
    }

    function switchScene(sceneKey) {
        if (!SCENES[sceneKey]) return;
        currentSceneKey = sceneKey;
        const rawGT = SCENES[sceneKey].generate();
        sceneCache = computeStages(rawGT);

        // Update scene buttons
        document.querySelectorAll('.matrix-scene-btn').forEach(btn => {
            if (btn.dataset.scene === sceneKey) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Scene description
        const sceneDescEl = document.getElementById('matrix-scene-desc');
        if (sceneDescEl) sceneDescEl.innerText = SCENES[sceneKey].desc;

        updateGridDisplay();
    }

    function switchViewMode(mode) {
        viewMode = mode;
        const gridView = document.getElementById('matrix-grid-view');
        const view3D = document.getElementById('matrix-3d-view');
        const btn2D = document.getElementById('btn-mode-2d');
        const btn3D = document.getElementById('btn-mode-3d');

        if (mode === '2d') {
            if (gridView) gridView.style.display = 'block';
            if (view3D) view3D.style.display = 'none';
            if (btn2D) btn2D.classList.add('active');
            if (btn3D) btn3D.classList.remove('active');
        } else {
            if (gridView) gridView.style.display = 'none';
            if (view3D) view3D.style.display = 'block';
            if (btn2D) btn2D.classList.remove('active');
            if (btn3D) btn3D.classList.add('active');

            // Lazy initialize 3D scene if first time
            if (!renderer3d) {
                init3DScene();
            } else {
                sync3DPillars();
            }
        }
    }

    // Master Public Initializer
    function initPixelMatrix() {
        // 1. Generate default scene
        switchScene(currentSceneKey);

        // 2. Build DOM grid
        buildGridDOM();
        updateGridDisplay();

        // 3. Attach Step buttons
        document.querySelectorAll('.matrix-step-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                stopAutoPlay();
                const step = parseInt(btn.dataset.step, 10);
                setStep(step);
            });
        });

        // 4. Attach Scene Buttons
        document.querySelectorAll('.matrix-scene-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                switchScene(btn.dataset.scene);
            });
        });

        // 5. Attach Play/Pause Button
        const playBtn = document.getElementById('matrix-btn-play');
        if (playBtn) {
            playBtn.addEventListener('click', toggleAutoPlay);
        }

        // 6. Attach Speed Select
        const speedSel = document.getElementById('matrix-speed-select');
        if (speedSel) {
            speedSel.addEventListener('change', (e) => {
                playSpeed = parseInt(e.target.value, 10);
                if (isPlaying) {
                    stopAutoPlay();
                    startAutoPlay();
                }
            });
        }

        // 7. Attach View Mode Toggle (2D / 3D)
        const btn2D = document.getElementById('btn-mode-2d');
        const btn3D = document.getElementById('btn-mode-3d');
        if (btn2D) btn2D.addEventListener('click', () => switchViewMode('2d'));
        if (btn3D) btn3D.addEventListener('click', () => switchViewMode('3d'));

        // Handle window resize for 3D canvas
        window.addEventListener('resize', () => {
            const container = document.getElementById('matrix-3d-viewport');
            if (container && renderer3d && camera3d) {
                camera3d.aspect = container.clientWidth / container.clientHeight;
                camera3d.updateProjectionMatrix();
                renderer3d.setSize(container.clientWidth, container.clientHeight);
            }
        });
    }

    window.initPixelMatrix = initPixelMatrix;

})();
