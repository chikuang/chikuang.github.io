(function () {
  var counter = document.getElementById('site-visitor-count');
  var value = document.getElementById('site-view-count');
  var status = document.getElementById('site-view-status');
  if (!counter || !value || !status) return;
  var endpoint = counter.getAttribute('data-counter-endpoint');
  if (!endpoint) return;
  var preview = window.location.hostname !== counter.getAttribute('data-counter-host') || !!window.location.port;
  var lastTotal = null;
  var lastSuccess = 0;
  var sequence = 0;
  var reading = false;

  function unavailable(id) {
    if (id < lastSuccess) return;
    status.textContent = lastTotal === null ? ' (temporarily unavailable)' : ' (offline)';
    counter.title = 'The live total could not be refreshed. No estimated views are added.';
    if (window.siteVisitorMap) window.siteVisitorMap.unavailable();
  }

  async function request(eventId) {
    var id = ++sequence;
    var controller = new AbortController();
    var timeout = window.setTimeout(function () { controller.abort(); }, 15000);
    try {
      var response = await fetch(endpoint, {
        method: eventId ? 'POST' : 'GET',
        headers: eventId ? { 'Content-Type': 'application/json' } : {},
        body: eventId ? JSON.stringify({ eventId: eventId, visitorMap: true }) : undefined,
        credentials: 'omit', cache: 'no-store', referrerPolicy: 'no-referrer', signal: controller.signal
      });
      if (!response.ok) throw new Error('Counter unavailable');
      var data = await response.json();
      if (!data || !Number.isSafeInteger(data.total) || data.total < 0 || (lastTotal !== null && data.total < lastTotal)) {
        throw new Error('Invalid total');
      }
      lastTotal = data.total;
      lastSuccess = Math.max(lastSuccess, id);
      value.textContent = lastTotal.toLocaleString('en-US');
      status.textContent = preview ? ' (preview)' : '';
      counter.title = 'Cumulative recorded views, refreshed ' + new Date().toLocaleTimeString()
        + (preview ? '. Preview visits are not counted.' : '. One visit per browser-tab session; internal navigation and reloads do not add views.');
      if (window.siteVisitorMap) window.siteVisitorMap.update(data.map);
      return true;
    } catch (error) {
      unavailable(id);
      return false;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  async function readTotal() {
    if (reading || document.hidden) return;
    reading = true;
    try { await request(); }
    finally { reading = false; }
  }

  function visitId() {
    var storageKey = 'ck-website-visit-v2';
    var storage;
    try {
      storage = window.sessionStorage;
      var saved = storage.getItem(storageKey);
      if (/^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i.test(saved)) return saved;
    } catch (error) { storage = null; }
    // When storage is blocked, prefer a missed repeat visit to overcounting
    // internal navigation. This fallback uses no cookies or persistent IDs.
    if (!storage) {
      try {
        var navigation = window.performance.getEntriesByType('navigation')[0];
        if (navigation && (navigation.type === 'reload' || navigation.type === 'back_forward')) return null;
      } catch (error) {}
      try {
        if (document.referrer && new URL(document.referrer).origin === window.location.origin) return null;
      } catch (error) {}
    }
    var bytes = new Uint8Array(16);
    window.crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 15) | 64;
    bytes[8] = (bytes[8] & 63) | 128;
    var hex = Array.from(bytes, function (byte) { return byte.toString(16).padStart(2, '0'); }).join('');
    var eventId = hex.slice(0, 8) + '-' + hex.slice(8, 12) + '-' + hex.slice(12, 16) + '-' + hex.slice(16, 20) + '-' + hex.slice(20);
    try { if (storage) storage.setItem(storageKey, eventId); }
    catch (error) {
      // Treat quota-denied writes like blocked storage on later page loads.
      try {
        var nav = window.performance.getEntriesByType('navigation')[0];
        if (nav && (nav.type === 'reload' || nav.type === 'back_forward')) return null;
        if (document.referrer && new URL(document.referrer).origin === window.location.origin) return null;
      } catch (ignored) {}
    }
    return eventId;
  }

  async function recordView() {
    if (preview) return;
    // Reuse the same ID across pages and retries in this tab. Deduplication is
    // atomic on the server, including if navigation aborts an earlier response.
    var eventId = visitId();
    if (!eventId) return readTotal();
    for (var attempt = 0; attempt < 3; attempt++) {
      if (await request(eventId)) return;
      if (attempt < 2) await new Promise(function (resolve) { window.setTimeout(resolve, (attempt + 1) * 2000); });
    }
  }

  value.textContent = '—';
  status.textContent = ' (loading)';
  document.addEventListener('site:pageview', readTotal);
  window.addEventListener('pageshow', function (event) { if (event.persisted) readTotal(); });
  document.addEventListener('visibilitychange', function () { if (!document.hidden) readTotal(); });
  window.addEventListener('online', readTotal);
  // Reading every 30 seconds never creates a view; background tabs skip reads.
  window.setInterval(readTotal, 30000);
  if (preview) readTotal();
  else recordView();
})();
