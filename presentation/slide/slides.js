/* ==========================================================================
   SLIDES.JS — Simple, Fast Presentation Controller
   Auto 16:9 scaling, keyboard navigation, and responsive 2-digit counter
   ========================================================================== */

(function () {
  'use strict';

  let currentSlide = 0;
  const slides = document.querySelectorAll('.slide');
  const totalSlides = slides.length;
  const slideCanvas = document.getElementById('slide-canvas');
  const currentNumEl = document.getElementById('current-slide-num');
  const totalNumEl = document.getElementById('total-slide-num');

  if (totalNumEl) totalNumEl.textContent = String(totalSlides).padStart(2, '0');

  // Auto 16:9 Scale (1920x1080)
  function resizeCanvas() {
    if (!slideCanvas) return;
    const w = window.innerWidth;
    const h = window.innerHeight;
    const scale = Math.min(w / 1920, h / 1080);
    slideCanvas.style.transform = `scale(${scale})`;
  }

  window.addEventListener('resize', resizeCanvas);
  window.addEventListener('DOMContentLoaded', resizeCanvas);
  resizeCanvas();

  function showSlide(index) {
    if (index < 0) index = 0;
    if (index >= totalSlides) index = totalSlides - 1;

    slides.forEach((s, i) => {
      if (i === index) s.classList.add('active');
      else s.classList.remove('active');
    });

    currentSlide = index;

    const curStr = String(currentSlide + 1).padStart(2, '0');
    const totStr = String(totalSlides).padStart(2, '0');

    if (currentNumEl) {
      currentNumEl.textContent = curStr;
    }

    // Update footers matching 1.png & 2.png: 01/34, 02/34...
    if (slides[index]) {
      slides[index].querySelectorAll('.ppt-slide-counter').forEach(el => {
        el.textContent = `${curStr}/${totStr}`;
      });
    }

    window.location.hash = `#slide-${currentSlide + 1}`;
  }

  function next() {
    if (currentSlide < totalSlides - 1) showSlide(currentSlide + 1);
  }

  function prev() {
    if (currentSlide > 0) showSlide(currentSlide - 1);
  }

  // Keyboard navigation
  window.addEventListener('keydown', (e) => {
    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowDown':
      case ' ':
      case 'PageDown':
      case 'Enter':
        e.preventDefault();
        next();
        break;
      case 'ArrowLeft':
      case 'ArrowUp':
      case 'PageUp':
      case 'Backspace':
        e.preventDefault();
        prev();
        break;
      case 'Home':
        e.preventDefault();
        showSlide(0);
        break;
      case 'End':
        e.preventDefault();
        showSlide(totalSlides - 1);
        break;
      case 'f':
      case 'F':
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen().catch(() => {});
        } else {
          if (document.exitFullscreen) document.exitFullscreen();
        }
        break;
    }
  });

  // Buttons
  const btnNext = document.getElementById('btn-next');
  const btnPrev = document.getElementById('btn-prev');
  const btnFs = document.getElementById('btn-fullscreen');

  if (btnNext) btnNext.addEventListener('click', next);
  if (btnPrev) btnPrev.addEventListener('click', prev);
  if (btnFs) btnFs.addEventListener('click', () => {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {});
    else if (document.exitFullscreen) document.exitFullscreen();
  });

  // Hash Navigation Init
  function initHash() {
    const h = window.location.hash;
    if (h && h.startsWith('#slide-')) {
      const n = parseInt(h.replace('#slide-', ''), 10);
      if (!isNaN(n) && n >= 1 && n <= totalSlides) currentSlide = n - 1;
    }
    showSlide(currentSlide);
  }

  window.addEventListener('load', () => {
    resizeCanvas();
    initHash();
  });
})();
