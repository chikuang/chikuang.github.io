(function () {
  var counter = document.getElementById('site-visitor-count');
  var value = document.getElementById('site-view-count');
  var status = document.getElementById('site-view-status');
  if (!counter || !value || !status) return;

  var lastTotal = Number(counter.getAttribute('data-counter-total'));
  var lastUpdated = counter.getAttribute('data-counter-updated');
  var storageKey = 'site-view-total:' + counter.getAttribute('data-counter-host');
  var requestNumber = 0;

  function validTotal(total) {
    return Number.isSafeInteger(total) && total >= 0;
  }

  function showSaved(preview) {
    value.textContent = lastTotal.toLocaleString('en-US');
    status.textContent = ' (' + (preview ? 'preview · ' : '') + 'saved ' + lastUpdated.slice(0, 10) + ')';
    counter.title = 'Last confirmed cumulative total, saved ' + lastUpdated + '. '
      + (preview ? 'Local preview visits are not counted.'
        : 'Live counting may be blocked or unavailable; blocked visits are not recorded.');
  }

  // The HTML snapshot also works on a first visit with scripts or trackers blocked.
  // Keep it visible while loading; never manufacture an increment locally.
  if (!validTotal(lastTotal) || !Number.isFinite(Date.parse(lastUpdated))) return;

  // Preview traffic must never enter the public website's visitor count.
  if (window.location.hostname !== counter.getAttribute('data-counter-host') || window.location.port) {
    showSaved(true);
    return;
  }

  // Storage can be denied in private browsing or by privacy settings.
  try {
    var saved = JSON.parse(window.localStorage.getItem(storageKey));
    if (saved && validTotal(saved.total) && typeof saved.updated === 'string' && Number.isFinite(Date.parse(saved.updated))
        && saved.total >= lastTotal && Date.parse(saved.updated) >= Date.parse(lastUpdated)
        && Date.parse(saved.updated) <= Date.now()) {
      lastTotal = saved.total;
      lastUpdated = saved.updated;
    }
  } catch (error) { /* The published snapshot remains available. */ }
  showSaved(false);

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
      showSaved(false);
    }

    window[callback] = function (data) {
      if (finished) return;
      if (!data || !validTotal(data.site_pv)) {
        unavailable();
        return;
      }
      cleanup();
      // Responses can arrive out of order after rapid section navigation.
      if (data.site_pv >= lastTotal) {
        lastTotal = data.site_pv;
        lastUpdated = new Date().toISOString();
        value.textContent = lastTotal.toLocaleString('en-US');
        status.textContent = '';
        counter.title = 'Cumulative recorded page views, including repeat visits and section views';
        try {
          window.localStorage.setItem(storageKey, JSON.stringify({total: lastTotal, updated: lastUpdated}));
        } catch (error) { /* Counting still works without browser storage. */ }
      } else if (requestId === requestNumber) {
        showSaved(false);
      }
      if (requestId === requestNumber) {
        value.setAttribute('aria-busy', 'false');
      }
    };

    // Use the same persistent site_pv total as the official Busuanzi client.
    // One request both records this view and returns the site's running total.
    script.src = 'https://busuanzi.ibruce.info/busuanzi?jsonpCallback=' + callback;
    // Site totals only need the origin; Firefox rejects less restrictive policies.
    script.referrerPolicy = 'strict-origin-when-cross-origin';
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
