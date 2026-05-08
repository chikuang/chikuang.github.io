(function () {
  var STORAGE_KEY = 'ck-site-theme';

  function getMetaThemeColor() {
    return document.querySelector('meta[name="theme-color"]');
  }

  function applyTheme(theme) {
    if (theme !== 'light' && theme !== 'dark') theme = 'light';
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch (e) {}

    var meta = getMetaThemeColor();
    if (meta) {
      meta.setAttribute('content', theme === 'dark' ? '#16181d' : '#ffffff');
    }

    var btn = document.getElementById('theme-toggle');
    if (btn) {
      btn.setAttribute(
        'aria-label',
        theme === 'dark' ? '切換為淺色模式' : '切換為深色模式'
      );
      btn.setAttribute('title', theme === 'dark' ? '淺色模式' : '深色模式');
    }
  }

  function toggleTheme() {
    var cur = document.documentElement.getAttribute('data-theme') || 'light';
    applyTheme(cur === 'dark' ? 'light' : 'dark');
  }

  function init() {
    var btn = document.getElementById('theme-toggle');
    if (btn) {
      btn.addEventListener('click', toggleTheme);
    }
    var stored = null;
    try {
      stored = localStorage.getItem(STORAGE_KEY);
    } catch (e) {}
    if (stored === 'light' || stored === 'dark') {
      applyTheme(stored);
    } else {
      applyTheme(
        document.documentElement.getAttribute('data-theme') || 'light'
      );
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
