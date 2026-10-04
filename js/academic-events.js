/* Date-only event boundaries use Atlanta time, independent of the reader's timezone. */
(function () {
  'use strict';

  function todayInZone(now, timeZone) {
    const parts = new Intl.DateTimeFormat('en-US', {
      timeZone, year: 'numeric', month: '2-digit', day: '2-digit'
    }).formatToParts(now);
    const part = name => parts.find(value => value.type === name).value;
    return `${part('year')}-${part('month')}-${part('day')}`;
  }

  function bucketFor(event, today) {
    if (!event.end) return 'pending';
    return event.end < today ? 'past' : event.category;
  }

  function compareEvents(a, b, bucket) {
    const order = (a.start || '').localeCompare(b.start || '');
    return (bucket === 'past' ? -order : order) || a.title.localeCompare(b.title);
  }

  function refresh(directory, now) {
    const today = todayInZone(now, directory.dataset.timezone);
    // Avoid rebuilding lists while a reader is interacting with them on the same day.
    if (directory.dataset.currentDate === today) return;
    const lists = new Map(Array.from(directory.querySelectorAll('[data-event-list]'),
      list => [list.dataset.eventList, list]));
    const groups = new Map(Array.from(lists.keys(), key => [key, []]));
    for (const node of directory.querySelectorAll('[data-event-id]')) {
      const event = { ...node.dataset, title: node.querySelector('a').textContent };
      groups.get(bucketFor(event, today)).push({ node, event });
    }
    for (const [bucket, group] of groups) {
      group.sort((a, b) => compareEvents(a.event, b.event, bucket));
      for (const item of group) lists.get(bucket).appendChild(item.node);
      directory.querySelector(`[data-empty-for="${bucket}"]`).hidden = group.length > 0;
    }
    for (const deadline of directory.querySelectorAll('[data-deadline]')) {
      deadline.hidden = deadline.dataset.deadline < today;
    }
    directory.querySelector('[data-past-count]').textContent = `(${groups.get('past').length})`;
    directory.querySelector('[data-event-date-label]').textContent = `As of ${today} (Atlanta time)`;
    directory.dataset.currentDate = today;
  }

  // Expose only to the Node test runner; the browser creates no global API.
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { todayInZone, bucketFor, compareEvents, refresh };
  }
  if (typeof document === 'undefined') return;
  for (const directory of document.querySelectorAll('[data-event-directory]')) {
    const update = () => refresh(directory, new Date());
    update();
    window.addEventListener('pageshow', update);
    document.addEventListener('visibilitychange', () => { if (!document.hidden) update(); });
    // Also update tabs left open across midnight; no network request is needed.
    window.setInterval(() => { if (!document.hidden) update(); }, 60000);
  }
})();
