/**
 * navigation.js - Reading Progress, ScrollSpy & Mobile Drawer
 * CS2203 - Low-Light Image Enhancement Project
 */

(function() {
    function initProgress() {
        const progressBar = document.getElementById('progress-bar');
        if (!progressBar) return;

        window.addEventListener('scroll', () => {
            const winScroll = document.documentElement.scrollTop;
            const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
            if (height > 0) {
                const scrolled = (winScroll / height) * 100;
                progressBar.style.width = scrolled + '%';
            }
        });
    }

    function initScrollSpy() {
        const sections = document.querySelectorAll('section > div[id], section[id], header[id]');
        const navLinks = document.querySelectorAll('.nav-item');

        window.addEventListener('scroll', () => {
            let current = '';
            const scrollPos = window.scrollY + 140;

            sections.forEach(sec => {
                const top = sec.offsetTop;
                const height = sec.offsetHeight;
                if (scrollPos >= top && scrollPos < top + height) {
                    current = sec.getAttribute('id');
                }
            });

            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === '#' + current) {
                    link.classList.add('active');
                }
            });
        });
    }

    function initMobileMenu() {
        const btn = document.getElementById('mobile-menu-btn');
        const sidebar = document.getElementById('sidebar');
        if (btn && sidebar) {
            btn.addEventListener('click', () => {
                sidebar.classList.toggle('open');
            });

            // Close sidebar when clicking a navigation link on mobile
            document.querySelectorAll('.nav-item').forEach(link => {
                link.addEventListener('click', () => {
                    if (window.innerWidth <= 1024) {
                        sidebar.classList.remove('open');
                    }
                });
            });
        }
    }

    window.initNavigation = function() {
        initProgress();
        initScrollSpy();
        initMobileMenu();
    };
})();
