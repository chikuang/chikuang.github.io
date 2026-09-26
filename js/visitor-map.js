(function () {
  var map = document.getElementById('site-visitor-map');
  if (!map) return;
  var points = document.getElementById('visitor-map-points');
  var status = document.getElementById('visitor-map-status');
  var period = document.getElementById('visitor-map-period');
  var list = document.getElementById('visitor-map-countries');
  var summary = 'Loading visitor locations…';
  var lastSignature = '';
  var names;
  try { names = new Intl.DisplayNames(['en'], { type: 'region' }); } catch (error) {}
  function regionName(code) {
    try { return names ? names.of(code) : code; } catch (error) { return code; }
  }
  function count(n) { return n.toLocaleString('en-US'); }
  function validLocation(item) {
    return item && /^[A-Z]{2}$/.test(item.country) && item.country !== 'XX'
      && Number.isFinite(item.latitude) && Math.abs(item.latitude) <= 90
      && Number.isFinite(item.longitude) && Math.abs(item.longitude) <= 180
      && Number.isSafeInteger(item.visits) && item.visits > 0;
  }
  function showPoint(event) {
    var label = event.target.getAttribute && event.target.getAttribute('data-location-label');
    if (label) status.textContent = label;
  }
  points.addEventListener('pointerover', showPoint);
  points.addEventListener('focusin', showPoint);
  points.addEventListener('pointerout', function () { status.textContent = summary; });
  points.addEventListener('focusout', function () { status.textContent = summary; });
  window.siteVisitorMap = {
    unavailable: function () {
      status.textContent = lastSignature ? summary + ' (offline)' : 'Locations temporarily unavailable';
    },
    update: function (data) {
      if (!data || !/^\d{4}-\d{2}-\d{2}$/.test(data.since) || !Number.isSafeInteger(data.visits)
        || data.visits < 0 || !Array.isArray(data.locations) || !data.locations.every(validLocation)) {
        this.unavailable(); return;
      }
      var located = data.locations.reduce(function (sum, item) { return sum + item.visits; }, 0);
      if (!Number.isSafeInteger(located) || located > data.visits) { this.unavailable(); return; }
      var signature = JSON.stringify(data);
      if (signature === lastSignature) { status.textContent = summary; return; }
      lastSignature = signature;
      var countries = new Map();
      var fragment = document.createDocumentFragment();
      data.locations.forEach(function (item) {
        countries.set(item.country, (countries.get(item.country) || 0) + item.visits);
        var label = regionName(item.country) + ' · ' + count(item.visits) + (item.visits === 1 ? ' visit' : ' visits') + ' in this approximate region';
        var circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', (item.longitude + 180) * 2);
        circle.setAttribute('cy', (90 - item.latitude) * 2);
        circle.setAttribute('r', Math.min(12, 6 + Math.log2(item.visits + 1)));
        circle.setAttribute('class', 'visitor-map-point');
        circle.setAttribute('tabindex', '0');
        circle.setAttribute('role', 'img');
        circle.setAttribute('aria-label', label);
        circle.setAttribute('data-location-label', label);
        var title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = label;
        circle.appendChild(title);
        fragment.appendChild(circle);
      });
      points.replaceChildren(fragment);
      var items = document.createDocumentFragment();
      Array.from(countries).sort(function (a, b) { return b[1] - a[1] || a[0].localeCompare(b[0]); }).forEach(function (entry) {
        var li = document.createElement('li');
        li.textContent = regionName(entry[0]) + ': ' + count(entry[1]);
        items.appendChild(li);
      });
      if (data.visits > located) {
        var unknown = document.createElement('li');
        unknown.textContent = 'Location unavailable: ' + count(data.visits - located);
        items.appendChild(unknown);
      }
      list.replaceChildren(items);
      summary = located ? count(located) + ' mapped visits · ' + countries.size + (countries.size === 1 ? ' country/region' : ' countries/regions') : 'No mapped visits yet';
      status.textContent = summary;
      period.textContent = 'Location records since ' + data.since + '. Earlier views are included in Total views, but cannot be placed on this map.';
    }
  };
})();
