(function () {
  var counter = document.getElementById('site-visitor-count');
  if (!counter) return;

  // Preview traffic must never enter the public website's visitor count.
  if (window.location.hostname !== counter.getAttribute('data-counter-host') || window.location.port) {
    counter.title = 'Visitor count is available on the published website';
    return;
  }

  var script = document.createElement('script');
  script.src = 'https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js';
  script.async = true;
  script.onerror = function () {
    counter.title = 'Visitor count is temporarily unavailable';
  };
  document.head.appendChild(script);
})();
