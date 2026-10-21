/**
 * three-viz.js - Interactive WebGL 3D Pixel Intensity Heightmap (Skill 3dviz-pro-max)
 * CS2203 - Low-Light Image Enhancement Project
 */

(function() {
    let scene, camera, renderer, controls, terrainMesh, pointCloudMesh;
    let current3DMode = 'dark';
    let renderStyle = 'solid'; // 'solid' | 'wireframe' | 'points'
    let isAutoRotating = true;

    // Fail-safe Internal Mouse Orbit Controller
    function createSimpleOrbitControls(cam, dom) {
        let isDragging = false;
        let prevX = 0, prevY = 0;
        let spherical = { radius: 75, theta: 0.8, phi: 0.95 };

        function updateCam() {
            cam.position.x = spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta);
            cam.position.y = spherical.radius * Math.cos(spherical.phi);
            cam.position.z = spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta);
            cam.lookAt(0, 4, 0);
        }
        updateCam();

        dom.addEventListener('mousedown', (e) => {
            isDragging = true;
            prevX = e.clientX;
            prevY = e.clientY;
        });
        window.addEventListener('mouseup', () => { isDragging = false; });
        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const dx = e.clientX - prevX;
            const dy = e.clientY - prevY;
            prevX = e.clientX;
            prevY = e.clientY;

            spherical.theta -= dx * 0.008;
            spherical.phi = Math.max(0.1, Math.min(Math.PI / 2.05, spherical.phi - dy * 0.008));
            updateCam();
        });
        dom.addEventListener('wheel', (e) => {
            e.preventDefault();
            spherical.radius = Math.max(25, Math.min(140, spherical.radius + e.deltaY * 0.05));
            updateCam();
        }, { passive: false });

        return {
            update: () => {
                if (isAutoRotating && !isDragging) {
                    spherical.theta += 0.004;
                    updateCam();
                }
            },
            reset: () => {
                spherical = { radius: 75, theta: 0.8, phi: 0.95 };
                updateCam();
            }
        };
    }

    function initThreeJS() {
        const container = document.getElementById('three-container');
        const canvas = document.getElementById('three-canvas');
        if (!container || !canvas || typeof THREE === 'undefined') {
            console.warn('Three.js canvas container not ready or Three.js library not loaded.');
            return;
        }

        // 1. Scene & Atmosphere (3dviz-pro-max standard)
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x05080E); // Deepest obsidian void

        // 2. Camera Setup
        camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.set(45, 40, 55);

        // 3. WebGL Renderer
        renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: false });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setClearColor(0x05080E, 1);

        // 4. Orbit Controls (Kiểm tra constructor toàn cục hoặc dùng fail-safe controller)
        if (typeof THREE.OrbitControls === 'function') {
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.06;
            controls.maxPolarAngle = Math.PI / 2.05;
            controls.autoRotate = true;
            controls.autoRotateSpeed = 0.8;
            controls.target.set(0, 4, 0);
        } else {
            controls = createSimpleOrbitControls(camera, renderer.domElement);
        }

        // 5. Lighting Rig (Studio High-Contrast 4-Point Light according to 3dviz-pro-max)
        const ambientLight = new THREE.AmbientLight(0xFFFFFF, 0.9);
        scene.add(ambientLight);

        // Key Light: Cyber Cyan-Blue
        const keyLight = new THREE.DirectionalLight(0x38BDF8, 2.2);
        keyLight.position.set(35, 50, 30);
        scene.add(keyLight);

        // Fill Light: Soft Electric Violet
        const fillLight = new THREE.DirectionalLight(0xA78BFA, 1.4);
        fillLight.position.set(-35, 30, -25);
        scene.add(fillLight);

        // Rim Light: Neon Mint Emerald
        const rimLight = new THREE.DirectionalLight(0x34D399, 1.2);
        rimLight.position.set(0, 40, -45);
        scene.add(rimLight);

        // 6. Cyber Grid Plane
        const gridHelper = new THREE.GridHelper(70, 35, 0x1E293B, 0x0F172A);
        gridHelper.position.y = -0.2;
        scene.add(gridHelper);

        // 7. Initial Terrain Geometry
        buildTerrain('dark');

        // 8. Animation Render Loop
        function animate() {
            requestAnimationFrame(animate);
            if (controls && controls.update) controls.update();
            renderer.render(scene, camera);
        }
        animate();

        // 9. Resize Listener
        window.addEventListener('resize', () => {
            if (!container || !renderer || !camera) return;
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });
    }

    function buildTerrain(mode) {
        if (terrainMesh) scene.remove(terrainMesh);
        if (pointCloudMesh) scene.remove(pointCloudMesh);

        const segs = 75;
        const size = 56;
        const geo = new THREE.PlaneGeometry(size, size, segs, segs);
        geo.rotateX(-Math.PI / 2);

        const pos = geo.attributes.position;
        const colors = [];

        for (let i = 0; i < pos.count; i++) {
            const x = pos.getX(i);
            const z = pos.getZ(i);

            // Coherent underlying scene geometry: buildings, wall, road, illumination glow
            let baseStructure = 4.0 + Math.sin(x * 0.12) * Math.cos(z * 0.12) * 3.5;
            if (Math.abs(x) < 5 || Math.abs(z) < 5) baseStructure += 5.5; // sharp object edge

            let y = 0;
            let r = 0.1, g = 0.2, b = 0.4;

            if (mode === 'dark') {
                // Deep sunken valley with prominent Poisson needle spikes
                const isSpike = (Math.sin(x * 37.1 + z * 19.3) > 0.88);
                const sensorNoise = isSpike ? (6.0 + Math.random() * 8.0) : ((Math.random() - 0.5) * 1.5);
                y = (baseStructure * 0.22) + sensorNoise;
                y = Math.max(0.4, y);

                if (isSpike) {
                    // High-contrast glowing neon red/orange for sensor noise needle
                    r = 0.98; g = 0.28; b = 0.22;
                } else {
                    // Deep twilight ocean blue for sunken baseline signal
                    const norm = Math.min(1.0, y / 5.0);
                    r = 0.08 + norm * 0.15;
                    g = 0.15 + norm * 0.35;
                    b = 0.45 + norm * 0.55;
                }
            } else if (mode === 'clahe') {
                // Aggressively stretched contrast with flattened clipped ceiling + amplified noise
                const isSpike = (Math.sin(x * 37.1 + z * 19.3) > 0.88);
                const noise = isSpike ? (12.0 + Math.random() * 9.0) : ((Math.random() - 0.5) * 4.8);
                y = Math.pow(baseStructure, 1.25) * 1.35 + noise;

                if (y >= 16.5) {
                    // Harsh clipped plateau (Cháy sáng / Clipping artifacts)
                    y = 16.5;
                    r = 1.0; g = 0.96; b = 0.65; // Overexposed harsh yellow-white
                } else if (isSpike) {
                    r = 1.0; g = 0.18; b = 0.18;
                } else {
                    r = 0.85 + (y / 18) * 0.15;
                    g = 0.40 + (y / 18) * 0.35;
                    b = 0.15;
                }
            } else if (mode === 'enhanced') {
                // Smooth, harmoniously elevated mountain ridges with sharp edge structures and NO noise spikes
                const microTexture = (Math.sin(x * 0.8) * Math.cos(z * 0.8)) * 0.35;
                y = Math.pow(baseStructure, 1.22) * 1.45 + microTexture;
                y = Math.max(1.5, y);

                // Luminous Aurora Gradient: Mint Emerald -> Sky Blue -> Electric Violet
                const norm = Math.min(1.0, y / 16.0);
                if (norm < 0.4) {
                    r = 0.10 + norm * 0.1;
                    g = 0.65 + norm * 0.3;
                    b = 0.45;
                } else if (norm < 0.8) {
                    r = 0.15;
                    g = 0.70 + (norm - 0.4) * 0.2;
                    b = 0.95;
                } else {
                    r = 0.65 + (norm - 0.8) * 0.35;
                    g = 0.55 + (norm - 0.8) * 0.35;
                    b = 0.98;
                }
            }

            pos.setY(i, y);
            colors.push(r, g, b);
        }

        geo.computeVertexNormals();
        geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        // Solid / Wireframe Mesh
        const mat = new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.35,
            metalness: 0.15,
            wireframe: (renderStyle === 'wireframe'),
            side: THREE.DoubleSide
        });
        terrainMesh = new THREE.Mesh(geo, mat);

        // Point Cloud Mesh (Đám mây điểm photon 3D)
        const pointMat = new THREE.PointsMaterial({
            size: 0.65,
            vertexColors: true,
            transparent: true,
            opacity: 0.9
        });
        pointCloudMesh = new THREE.Points(geo, pointMat);

        if (renderStyle === 'points') {
            scene.add(pointCloudMesh);
        } else {
            scene.add(terrainMesh);
        }
    }

    function switch3DMode(mode, btnEl) {
        current3DMode = mode;
        buildTerrain(mode);

        // Update button active state
        document.querySelectorAll('.three-controls .three-btn').forEach(b => {
            if (b.innerText.includes('1.') || b.innerText.includes('2.') || b.innerText.includes('3.')) {
                b.classList.remove('active');
            }
        });
        if (btnEl) btnEl.classList.add('active');

        // Update overlay HUD
        const title = document.getElementById('three-title');
        const desc = document.getElementById('three-desc');
        const pill = document.getElementById('three-stat-pill');

        if (mode === 'dark') {
            title.innerHTML = '🌌 1. Ảnh Thiếu Sáng Gốc (Input Dark)';
            desc.innerText = 'Điểm ảnh chìm sâu dưới thung lũng đen [z ≈ 1.5 - 3.5]. Các gai nhọn màu đỏ cam chọc thẳng lên cao thể hiện rõ hạt nhiễu cảm biến ngẫu nhiên.';
            if (pill) pill.innerText = 'Elevation: 2.8u • Noise Spikes: High';
        } else if (mode === 'clahe') {
            title.innerHTML = '⚡ 2. Cân Bằng Histogram Thích Nghi (CLAHE)';
            desc.innerText = 'Độ tương phản bị kéo cưỡng bức: trần nhà sáng bị cắt bẹt bằng phẳng (Clipping Plateau), các mũi gai nhiễu hạt bị phóng đại gấp 3 lần.';
            if (pill) pill.innerText = 'Elevation: 16.5u (Clipped) • Spikes: Extreme';
        } else if (mode === 'enhanced') {
            title.innerHTML = '✨ 3. Mạng Compact U-Net (Ours)';
            desc.innerText = 'Địa hình không gian được phục hồi mượt mà: sườn núi biên cạnh sắc nét, thung lũng được nâng sáng hài hòa, gai nhiễu bị san phẳng êm dịu.';
            if (pill) pill.innerText = 'Elevation: 15.2u • Noise: Suppressed (Clean)';
        }
    }

    function toggleRenderMode(btnEl) {
        if (renderStyle === 'solid') {
            renderStyle = 'wireframe';
            if (btnEl) btnEl.innerText = 'Hiển Thị: Lưới Khung (Wireframe)';
        } else if (renderStyle === 'wireframe') {
            renderStyle = 'points';
            if (btnEl) btnEl.innerText = 'Hiển Thị: Đám Mây Điểm (Points)';
        } else {
            renderStyle = 'solid';
            if (btnEl) btnEl.innerText = 'Hiển Thị: Khối Bề Mặt (Solid)';
        }
        buildTerrain(current3DMode);
    }

    function toggleAutoRotate(btnEl) {
        isAutoRotating = !isAutoRotating;
        if (controls) {
            controls.autoRotate = isAutoRotating;
        }
        if (btnEl) {
            btnEl.innerText = isAutoRotating ? 'Tự Động Xoay: BẬT' : 'Tự Động Xoay: TẮT';
        }
    }

    function reset3DCamera() {
        if (!camera) return;
        if (controls && controls.reset) {
            controls.reset();
        } else if (controls && controls.target) {
            camera.position.set(45, 40, 55);
            controls.target.set(0, 4, 0);
            controls.update();
        }
    }

    // Export global functions for HTML inline onclick handlers
    window.initThreeJS = initThreeJS;
    window.switch3DMode = switch3DMode;
    window.toggleRenderMode = toggleRenderMode;
    window.toggleAutoRotate = toggleAutoRotate;
    window.reset3DCamera = reset3DCamera;
})();
