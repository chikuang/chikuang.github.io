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
  }

  async function request(eventId) {
    var id = ++sequence;
    var controller = new AbortController();
    var timeout = window.setTimeout(function () { controller.abort(); }, 15000);
    try {
      var response = await fetch(endpoint, {
        method: eventId ? 'POST' : 'GET',
        headers: eventId ? { 'Content-Type': 'application/json' } : {},
        body: eventId ? JSON.stringify({ eventId: eventId }) : undefined,
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
        + (preview ? '. Preview visits are not counted.' : '. Includes repeat visits and section views.');
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

  async function recordView() {
    if (preview) return;
    // A new random ID identifies this event only, not the visitor. The server
    // stores it atomically with the total so retrying cannot count twice.
    var bytes = new Uint8Array(16);
    window.crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 15) | 64;
    bytes[8] = (bytes[8] & 63) | 128;
    var hex = Array.from(bytes, function (byte) { return byte.toString(16).padStart(2, '0'); }).join('');
    var eventId = hex.slice(0, 8) + '-' + hex.slice(8, 12) + '-' + hex.slice(12, 16) + '-' + hex.slice(16, 20) + '-' + hex.slice(20);
    for (var attempt = 0; attempt < 3; attempt++) {
      if (await request(eventId)) return;
      if (attempt < 2) await new Promise(function (resolve) { window.setTimeout(resolve, (attempt + 1) * 2000); });
    }
  }

  value.textContent = '—';
  status.textContent = ' (loading)';
  document.addEventListener('site:pageview', recordView);
  window.addEventListener('pageshow', function (event) { if (event.persisted) recordView(); });
  document.addEventListener('visibilitychange', function () { if (!document.hidden) readTotal(); });
  window.addEventListener('online', readTotal);
  // Reading every 30 seconds never creates a view; background tabs skip reads.
  window.setInterval(readTotal, 30000);
  if (preview) readTotal();
  else recordView();
})();
