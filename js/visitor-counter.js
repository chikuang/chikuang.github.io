(function () {
  var counter = document.getElementById('site-visitor-count');
  var value = document.getElementById('site-view-count');
  if (!counter || !value) return;

  // Preview traffic must never enter the public website's visitor count.
  if (window.location.hostname !== counter.getAttribute('data-counter-host') || window.location.port) {
    value.textContent = 'Preview';
    counter.title = 'Local preview is not counted. Open the published website for the total.';
    return;
  }

  var lastTotal = null;
  var requestNumber = 0;

  function recordView() {
    var requestId = ++requestNumber;
    var callback = 'ChikuangViews_' + Date.now() + '_' + requestId;
    var script = document.createElement('script');
    var finished = false;
    var timeout;
    value.setAttribute('aria-busy', 'true');

    function cleanup() {
      finished = true;
      window.clearTimeout(timeout);
      script.remove();
      // A timed-out script may still arrive. Its callback must remain harmless.
      window[callback] = function () {};
      window.setTimeout(function () { delete window[callback]; }, 60000);
    }

    function unavailable() {
      if (finished) return;
      cleanup();
      if (requestId !== requestNumber) return;
      value.setAttribute('aria-busy', 'false');
      if (lastTotal === null) value.textContent = 'Unavailable';
      counter.title = lastTotal === null
        ? 'The counting service is temporarily unavailable. Please try again on your next visit.'
        : 'The counting service is temporarily unavailable; the last retrieved total is shown.';
    }

    window[callback] = function (data) {
      if (finished) return;
      if (!data || !Number.isSafeInteger(data.site_pv) || data.site_pv < 0) {
        unavailable();
        return;
      }
      cleanup();
      // Responses can arrive out of order after rapid section navigation.
      lastTotal = lastTotal === null ? data.site_pv : Math.max(lastTotal, data.site_pv);
      value.textContent = lastTotal.toLocaleString('en-US');
      if (requestId === requestNumber) {
        value.setAttribute('aria-busy', 'false');
        counter.title = 'Cumulative recorded page views, including repeat visits and section views';
      }
    };

    // Use the same persistent site_pv total as the official Busuanzi client.
    // One request both records this view and returns the site's running total.
    script.src = 'https://busuanzi.ibruce.info/busuanzi?jsonpCallback=' + callback;
    script.referrerPolicy = 'no-referrer-when-downgrade';
    script.async = true;
    script.onerror = unavailable;
    timeout = window.setTimeout(unavailable, 15000);
    document.head.appendChild(script);
  }

  document.addEventListener('site:pageview', recordView);
  window.addEventListener('pageshow', function (event) {
    if (event.persisted) recordView();
  });
  recordView();
})();
