/**
 * theme.js - Bespoke Dark/Light Mode Theme Management
 * CS2203 - Low-Light Image Enhancement Project
 */

(function() {
    function initTheme() {
        const savedTheme = localStorage.getItem('cs2203_theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);
        updateThemeButton(savedTheme);
    }

    function toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('cs2203_theme', newTheme);
        updateThemeButton(newTheme);

        // Notify simulator to re-render with updated theme colors if available
        if (window.renderSimulator) {
            window.renderSimulator();
        }
    }

    function updateThemeButton(theme) {
        const icon = document.getElementById('theme-icon');
        const label = document.getElementById('theme-label');
        if (icon && label) {
            if (theme === 'dark') {
                icon.innerText = '🌙';
                label.innerText = 'Dark';
            } else {
                icon.innerText = '☀️';
                label.innerText = 'Light';
            }
        }
    }

    // Export functions globally
    window.initTheme = initTheme;
    window.toggleTheme = toggleTheme;

    // Run early to prevent theme flash
    initTheme();
})();
