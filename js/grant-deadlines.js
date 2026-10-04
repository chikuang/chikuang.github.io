/* Public deadlines only. No login, user profile, proposal data, or external request. */
(function () {
  'use strict';

  function dayInZone(now, timeZone) {
    const parts = new Intl.DateTimeFormat('en-US', {
      timeZone, year: 'numeric', month: '2-digit', day: '2-digit'
    }).formatToParts(now);
    const part = name => parts.find(value => value.type === name).value;
    return `${part('year')}-${part('month')}-${part('day')}`;
  }

  function daysUntil(item, now) {
    // Compare calendar days in the funder's zone, including across DST changes.
    return Math.round((Date.parse(item.date + 'T12:00:00Z') -
      Date.parse(dayInZone(now, item.timezone) + 'T12:00:00Z')) / 86400000);
  }

  function selectUpcoming(items, now) {
    return items.filter(item => Date.parse(item.expiresAt) > now.getTime())
      .sort((a, b) => Date.parse(a.expiresAt) - Date.parse(b.expiresAt) || a.grantId.localeCompare(b.grantId))
      .slice(0, 3);
  }

  function refresh(widget, now) {
    const nodes = Array.from(widget.querySelectorAll('[data-grant-id]'));
    const next = selectUpcoming(nodes.map(node => node.dataset), now);
    const ids = new Set(next.map(item => item.grantId));
    for (const node of nodes) {
      node.hidden = !ids.has(node.dataset.grantId);
      if (!node.hidden) {
        const days = daysUntil(node.dataset, now);
        node.querySelector('[data-grant-countdown]').textContent = days === 0 ? 'Due today' : days === 1 ? 'Tomorrow' : `${days} days`;
        node.classList.toggle('is-soon', days <= 14);
      }
    }
    const list = widget.querySelector('.grant-deadlines__list');
    // Move only if the ordering changed; keep focused links stable on minute ticks.
    const ordered = next.map(item => nodes.find(node => node.dataset.grantId === item.grantId));
    const visible = nodes.filter(node => !node.hidden);
    if (ordered.some((node, i) => node !== visible[i])) {
      for (const node of ordered) list.appendChild(node);
    }
    widget.querySelector('[data-grant-empty]').hidden = next.length > 0;
    const first = next[0];
    const summary = widget.querySelector('[data-grant-summary]');
    if (first) {
      const label = new Date(first.date + 'T12:00:00Z').toLocaleDateString('en-US', {
        month:'short', day:'numeric', year:'numeric', timeZone:'UTC'
      });
      const days = daysUntil(first, now);
      const countdown = days === 0 ? 'due today' : days === 1 ? 'tomorrow' : `${days} days`;
      summary.textContent = `Next: ${first.agency} · ${label} · ${countdown}`;
    } else summary.textContent = 'No confirmed upcoming dates';
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { dayInZone, daysUntil, selectUpcoming, refresh };
  }
  if (typeof document === 'undefined') return;
  for (const widget of document.querySelectorAll('[data-grant-deadlines]')) {
    const update = () => refresh(widget, new Date());
    update();
    window.addEventListener('pageshow', update);
    document.addEventListener('visibilitychange', () => { if (!document.hidden) update(); });
    window.setInterval(() => { if (!document.hidden) update(); }, 60000);
  }
})();
