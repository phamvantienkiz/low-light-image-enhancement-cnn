/**
 * papers-reader.js - Academic Paper Accordion Reader & Lightbox
 * CS2203 - Master in CS, UIT VNUHCM
 */

(function() {
    'use strict';

    // Toggle Paper Expansion (Accordion mechanism)
    function togglePaperReader(paperKey) {
        const wrapper = document.getElementById(`reader-${paperKey}`);
        const btn = document.querySelector(`button[data-paper="${paperKey}"]`);
        const arrow = document.getElementById(`arrow-${paperKey}`);

        if (!wrapper) return;

        const isCurrentlyActive = wrapper.classList.contains('active');

        if (isCurrentlyActive) {
            // Collapse
            wrapper.classList.remove('active');
            if (btn) btn.classList.remove('expanded');
            if (arrow) arrow.innerText = '▼';
        } else {
            // Expand & Inject Content if empty
            if (!wrapper.dataset.loaded) {
                if (window.PAPERS_DATA && window.PAPERS_DATA[paperKey]) {
                    wrapper.innerHTML = window.PAPERS_DATA[paperKey];
                    wrapper.dataset.loaded = 'true';

                    // Trigger KaTeX rendering on this paper content
                    if (window.renderMathInElement) {
                        window.renderMathInElement(wrapper, {
                            delimiters: [
                                {left: '$$', right: '$$', display: true},
                                {left: '$', right: '$', display: false},
                                {left: '\\[', right: '\\]', display: true},
                                {left: '\\(', right: '\\)', display: false}
                            ],
                            throwOnError: false
                        });
                    }

                    // Attach image zoom listeners
                    wrapper.querySelectorAll('.paper-figure img').forEach(img => {
                        img.addEventListener('click', () => {
                            const caption = img.closest('.paper-figure').querySelector('.paper-figcaption');
                            openLightbox(img.src, caption ? caption.innerText : '');
                        });
                    });
                }
            }

            wrapper.classList.add('active');
            if (btn) btn.classList.add('expanded');
            if (arrow) arrow.innerText = '▲';

            // Smooth scroll into paper header
            wrapper.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    // Toggle 2-column / 1-column layout for reading comfort
    function togglePaperColumns(paperKey) {
        const wrapper = document.getElementById(`reader-${paperKey}`);
        if (!wrapper) return;

        const colContainer = wrapper.querySelector('.paper-content-columns');
        const colBtn = wrapper.querySelector('.btn-col-toggle');
        if (!colContainer) return;

        if (colContainer.classList.contains('single-column')) {
            colContainer.classList.remove('single-column');
            if (colBtn) colBtn.innerText = '📄 Chế Độ 1 Cột (Dọc)';
        } else {
            colContainer.classList.add('single-column');
            if (colBtn) colBtn.innerText = '📰 Chế Độ 2 Cột (CVPR)';
        }
    }

    // Close paper reader
    function closePaperReader(paperKey) {
        const wrapper = document.getElementById(`reader-${paperKey}`);
        const btn = document.querySelector(`button[data-paper="${paperKey}"]`);
        const arrow = document.getElementById(`arrow-${paperKey}`);

        if (wrapper) wrapper.classList.remove('active');
        if (btn) btn.classList.remove('expanded');
        if (arrow) arrow.innerText = '▼';
    }

    // Lightbox modal functionality
    function initLightbox() {
        let lightbox = document.getElementById('figure-lightbox');
        if (!lightbox) {
            lightbox = document.createElement('div');
            lightbox.id = 'figure-lightbox';
            lightbox.innerHTML = `
                <img id="lightbox-img" src="" alt="Zoomed Figure">
                <div id="lightbox-caption"></div>
            `;
            document.body.appendChild(lightbox);

            lightbox.addEventListener('click', () => {
                lightbox.classList.remove('active');
            });
        }
    }

    function openLightbox(src, captionText) {
        let lightbox = document.getElementById('figure-lightbox');
        if (!lightbox) {
            initLightbox();
            lightbox = document.getElementById('figure-lightbox');
        }
        const img = document.getElementById('lightbox-img');
        const cap = document.getElementById('lightbox-caption');

        if (img) img.src = src;
        if (cap) cap.innerText = captionText;
        if (lightbox) lightbox.classList.add('active');
    }

    // Export functions globally
    window.togglePaperReader = togglePaperReader;
    window.togglePaperColumns = togglePaperColumns;
    window.closePaperReader = closePaperReader;
    window.openLightbox = openLightbox;

    document.addEventListener('DOMContentLoaded', () => {
        initLightbox();
    });

})();
