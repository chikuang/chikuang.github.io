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
      btn.removeAttribute('title');
    }
  }

  function toggleTheme() {
    var cur = document.documentElement.getAttribute('data-theme') || 'light';
    applyTheme(cur === 'dark' ? 'light' : 'dark');
  }

  function initSections() {
    var content = document.querySelector('[data-site-panels]');
    var tablist = document.querySelector('[data-site-tabs]');
    if (!content || !tablist) return;

    var tabs = Array.from(tablist.querySelectorAll('[data-panel-link]'));
    var panels = Array.from(content.querySelectorAll('[data-panel-key]'));
    var defaultKey = content.getAttribute('data-default-panel');
    var homeUrl = content.getAttribute('data-home-url');
    var activeKey;

    function hasPanel(key) {
      return panels.some(function (panel) {
        return panel.getAttribute('data-panel-key') === key;
      });
    }

    function showPanel(key) {
      if (!hasPanel(key)) key = defaultKey;
      activeKey = key;
      panels.forEach(function (panel) {
        var selected = panel.getAttribute('data-panel-key') === key;
        panel.hidden = !selected;
        if (selected) {
          document.title = key === 'home'
            ? content.getAttribute('data-home-title')
            : panel.getAttribute('data-panel-title') + ' - ' + content.getAttribute('data-site-title');
        }
      });
      tabs.forEach(function (tab) {
        var selected = tab.getAttribute('data-panel-link') === key;
        tab.classList.toggle('is-active', selected);
        tab.setAttribute('aria-selected', String(selected));
        tab.tabIndex = selected ? 0 : -1;
      });
    }

    function syncFromUrl() {
      showPanel(window.location.hash.slice(1));
    }

    function activate(tab) {
      var key = tab.getAttribute('data-panel-link');
      if (key !== activeKey) {
        window.history.pushState(null, '', tab.href);
        showPanel(key);
      }
    }

    // Ordinary section links remain usable when JavaScript is unavailable.
    tablist.setAttribute('role', 'tablist');
    tablist.setAttribute('aria-label', 'Main sections');
    tabs.forEach(function (tab) {
      var key = tab.getAttribute('data-panel-link');
      tab.href = homeUrl + '#' + key;
      tab.setAttribute('role', 'tab');
      tab.setAttribute('aria-controls', 'panel-' + key);
      tab.removeAttribute('aria-current');
      tab.addEventListener('click', function (event) {
        if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        activate(tab);
      });
      tab.addEventListener('keydown', function (event) {
        var index = tabs.indexOf(tab);
        var next;
        if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
        else if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
        else if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = tabs.length - 1;
        else return;
        event.preventDefault();
        tabs[next].focus();
        activate(tabs[next]);
      });
    });
    panels.forEach(function (panel) {
      panel.setAttribute('role', 'tabpanel');
      panel.tabIndex = 0;
    });
    window.addEventListener('popstate', syncFromUrl);
    window.addEventListener('hashchange', syncFromUrl);
    syncFromUrl();
  }

  function init() {
    initSections();
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
