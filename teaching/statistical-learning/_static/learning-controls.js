// Reading preference applies to source code only; outputs and answers stay distinct.
function initializeStudyControls() {
  if (document.querySelector('.course-learning-controls')) return;
  const blocks = [...document.querySelectorAll('details.course-code-fold')];
  if (!blocks.length) return;
  const controls = document.createElement('div');
  controls.className = 'course-learning-controls';
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'btn btn-sm';
  let expanded = false;
  try { expanded = localStorage.getItem('tsl-show-code') === 'true'; } catch (_) {}
  function apply() {
    blocks.forEach(block => { block.open = expanded; });
    button.textContent = expanded ? 'Collapse all code' : 'Show all code';
    button.setAttribute('aria-expanded', String(expanded));
  }
  button.addEventListener('click', () => {
    expanded = !expanded;
    try { localStorage.setItem('tsl-show-code', String(expanded)); } catch (_) {}
    apply();
  });
  const note = document.createElement('span');
  note.textContent = ' Study mode: expand code as you read; try exercises before revealing solutions.';
  controls.append(button, note);
  const heading = document.querySelector('article h1');
  if (heading) heading.after(controls);
  apply();
}
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeStudyControls, {once:true});
} else {
  initializeStudyControls();
}

