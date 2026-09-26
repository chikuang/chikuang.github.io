(function () {
  'use strict';

  var toc = document.getElementById('resource-toc');
  var body = document.getElementById('resource-body');
  if (!toc || !body) return;

  var headings = Array.from(body.querySelectorAll('h1, h2, h3, h4'));
  if (!headings.length) {
    toc.hidden = true;
    return;
  }

  var box = toc.querySelector('details');
  var summary = toc.querySelector('summary');
  var inner = toc.querySelector('.resource-toc__inner');
  var control = document.getElementById('resource-toc-mode');
  var wideScreen = window.matchMedia('(min-width: 1100px)');
  var storageKey = 'ck-resource-toc-mode';
  var preference = 'sidebar';
  try {
    var saved = localStorage.getItem(storageKey);
    if (saved === 'sidebar' || saved === 'floating') preference = saved;
  } catch (e) { /* The menu still works when storage is unavailable. */ }

  // R Markdown puts anchors on section wrappers; Markdown uses heading IDs.
  // Reuse both, without accidentally linking every heading to resource-body.
  var entries = headings.map(function (heading, index) {
    var section = heading.parentElement;
    var target = heading.id ? heading :
      (section.matches('.section[id]') ? section : heading);
    if (!target.id) {
      var base = 'resource-section-' + (index + 1);
      var id = base;
      var suffix = 2;
      while (document.getElementById(id)) id = base + '-' + suffix++;
      target.id = id;
    }
    return { heading: heading, target: target, depth: Number(heading.tagName.slice(1)) };
  });

  var minDepth = Math.min.apply(null, entries.map(function (entry) { return entry.depth; }));
  var nav = document.createElement('nav');
  nav.id = 'TableOfContents';
  nav.setAttribute('aria-label', 'Table of contents');
  var list = document.createElement('ul');
  entries.forEach(function (entry) {
    var item = document.createElement('li');
    item.className = 'resource-toc-entry';
    item.style.setProperty('--toc-depth', entry.depth - minDepth);
    var link = document.createElement('a');
    link.href = '#' + encodeURIComponent(entry.target.id);
    link.textContent = entry.heading.textContent.trim();
    entry.link = link;
    link.addEventListener('click', function (event) {
      if (event.button || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      // Reveal headings inside folded examples before the native anchor jump.
      var ancestor = entry.heading.parentElement;
      while (ancestor && ancestor !== body) {
        if (ancestor.tagName === 'DETAILS') ancestor.open = true;
        ancestor = ancestor.parentElement;
      }
      if (document.body.dataset.tocMode === 'floating') box.open = false;
      entry.heading.setAttribute('tabindex', '-1');
      entry.heading.focus({ preventScroll: true });
    });
    item.appendChild(link);
    list.appendChild(item);
  });
  nav.appendChild(list);
  inner.replaceChildren(nav);
  toc.querySelector('.resource-toc__controls').hidden = false;

  function applyMode(keepOpen) {
    var mode = wideScreen.matches ? preference : 'floating';
    document.body.dataset.tocMode = mode;
    control.value = preference;
    box.open = mode === 'sidebar' || Boolean(keepOpen);
    queueUpdate();
  }

  control.addEventListener('change', function () {
    var readingTop = activeEntry && activeEntry.heading.getBoundingClientRect().top;
    preference = control.value;
    try { localStorage.setItem(storageKey, preference); } catch (e) {}
    applyMode(true);
    if (readingTop !== undefined && readingTop <= 110) {
      window.scrollBy(0, activeEntry.heading.getBoundingClientRect().top - readingTop);
    }
  });
  wideScreen.addEventListener('change', function () { applyMode(false); });

  toc.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && document.body.dataset.tocMode === 'floating' && box.open) {
      box.open = false;
      summary.focus();
      event.preventDefault();
    }
  });
  document.addEventListener('click', function (event) {
    if (document.body.dataset.tocMode === 'floating' && !toc.contains(event.target)) box.open = false;
  });

  var activeEntry;
  var queued = false;
  function revealActive() {
    if (!box.open || !activeEntry) return;
    var linkRect = activeEntry.link.getBoundingClientRect();
    var viewRect = inner.getBoundingClientRect();
    if (linkRect.top < viewRect.top) inner.scrollTop -= viewRect.top - linkRect.top + 8;
    else if (linkRect.bottom > viewRect.bottom) inner.scrollTop += linkRect.bottom - viewRect.bottom + 8;
  }
  function updateActive() {
    queued = false;
    var current = entries[0];
    entries.forEach(function (entry) {
      var rect = entry.heading.getBoundingClientRect();
      if (rect.height && rect.top <= 110) current = entry;
    });
    if (current === activeEntry) return;
    if (activeEntry) activeEntry.link.removeAttribute('aria-current');
    activeEntry = current;
    activeEntry.link.setAttribute('aria-current', 'location');
    // Do not move the list while someone is browsing it with a mouse/keyboard.
    if (!toc.matches(':hover') && !toc.contains(document.activeElement)) revealActive();
  }
  function queueUpdate() {
    if (queued) return;
    queued = true;
    window.requestAnimationFrame(updateActive);
  }
  window.addEventListener('scroll', queueUpdate, { passive: true });
  window.addEventListener('resize', queueUpdate, { passive: true });
  window.addEventListener('hashchange', queueUpdate);
  window.addEventListener('load', queueUpdate);
  box.addEventListener('toggle', revealActive);
  applyMode(false);
})();
