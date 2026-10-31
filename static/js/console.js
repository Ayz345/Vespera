// The console: Classify / Compare tabs, upload, call the API, render results.
import { renderMap } from './map3d.js';

export function setupConsole({ reducedMotion = false } = {}) {
  const form = document.getElementById('run-form');
  const tabs = [...document.querySelectorAll('.tab')];
  const ink = document.querySelector('.tabs__ink');
  const panelClassify = document.getElementById('mode-classify');
  const panelCompare = document.getElementById('mode-compare');
  const fileInput = document.getElementById('file');
  const drop = document.getElementById('drop');
  const dropName = document.getElementById('drop-name');
  const dropHint = document.getElementById('drop-hint');
  const runBtn = document.getElementById('run-btn');
  const runLabel = runBtn.querySelector('.btn__label');
  const status = document.getElementById('run-status');
  const scan = document.getElementById('scan');
  const results = document.getElementById('results');
  const resultsSub = document.getElementById('results-sub');
  const gridClassify = document.getElementById('results-classify');
  const gridCompare = document.getElementById('results-compare');
  const map = document.getElementById('map');
  let mode = 'classify';

  /* Tabs */
  function moveInk(tab) {
    ink.style.left = tab.offsetLeft + 'px';
    ink.style.width = tab.offsetWidth + 'px';
  }
  function setMode(next) {
    mode = next;
    tabs.forEach((t) => t.setAttribute('aria-selected', String(t.id === 'tab-' + next)));
    panelClassify.hidden = next !== 'classify';
    panelCompare.hidden = next !== 'compare';
    runLabel.textContent = next === 'classify' ? 'Run the model' : 'Compare the four models';
    moveInk(tabs.find((t) => t.id === 'tab-' + next));
  }
  tabs.forEach((t) => t.addEventListener('click', () => setMode(t.id.replace('tab-', ''))));
  requestAnimationFrame(() => moveInk(tabs[0]));
  window.addEventListener('resize', () => moveInk(tabs.find((t) => t.getAttribute('aria-selected') === 'true')));

  /* File field */
  const emptyName = dropName.textContent, emptyHint = dropHint.textContent;
  function updateFile() {
    const f = fileInput.files && fileInput.files[0];
    if (f) {
      const kb = f.size / 1024;
      dropName.textContent = f.name;
      dropHint.textContent = (kb > 1024 ? (kb / 1024).toFixed(1) + ' MB' : Math.round(kb) + ' KB') + ' · ready';
      drop.classList.add('has-file');
    } else {
      dropName.textContent = emptyName; dropHint.textContent = emptyHint; drop.classList.remove('has-file');
    }
  }
  fileInput.addEventListener('change', updateFile);
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('is-over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('is-over'); }));
  drop.addEventListener('drop', (e) => { if (e.dataTransfer?.files?.length) { fileInput.files = e.dataTransfer.files; updateFile(); } });

  /* Run */
  function setBusy(on, text) {
    runBtn.disabled = on;
    scan.hidden = !on;
    status.classList.remove('is-error');
    status.textContent = text || '';
    if (on) runLabel.textContent = text; else setMode(mode);
  }
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput.files?.length) {
      status.textContent = 'Choose a CSV first.'; status.classList.add('is-error'); drop.classList.add('is-over');
      setTimeout(() => drop.classList.remove('is-over'), 900);
      return;
    }
    const data = new FormData();
    data.append('dataset', form.querySelector('input[name=dataset]:checked').value);
    data.append('file', fileInput.files[0]);
    let url = '/api/compare';
    if (mode === 'classify') { url = '/api/analyze'; data.append('model_choice', form.model_choice.value); }
    setBusy(true, mode === 'classify' ? 'Training…' : 'Training four models…');
    try {
      const res = await fetch(url, { method: 'POST', body: data });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(json.error || 'The server could not run the model.');
      setBusy(false, '');
      mode === 'classify' ? showClassify(json) : showCompare(json);
    } catch (err) {
      setBusy(false, '');
      status.textContent = err.message; status.classList.add('is-error');
    }
  });
  /* Results */
  const dlIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 4v12m0 0l-4-4m4 4l4-4"/><path d="M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3"/></svg>';

  function count(el, to, { decimals = 0, duration = 1.2 } = {}) {
    if (reducedMotion || !window.gsap) { el.textContent = to.toFixed(decimals); return; }
    const o = { v: 0 };
    gsap.to(o, { v: to, duration, ease: 'power3.out', onUpdate: () => { el.textContent = o.v.toFixed(decimals); } });
  }

  function reveal() {
    results.hidden = false;
    if (window.gsap && !reducedMotion) {
      gsap.fromTo(results, { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.9, ease: 'expo.out' });
      gsap.fromTo(results.querySelectorAll('.stat, .matrix, .downloads a, .rank'), { opacity: 0, y: 18 }, { opacity: 1, y: 0, duration: 0.8, ease: 'expo.out', stagger: 0.05, delay: 0.1 });
    }
    results.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'start' });
    window.ScrollTrigger && ScrollTrigger.refresh();
  }

  function showClassify(r) {
    gridClassify.hidden = false; gridCompare.hidden = true;
    resultsSub.textContent = `${r.model_name} on ${r.dataset.toUpperCase()}`;
    count(document.getElementById('r-acc'), r.accuracy * 100, { decimals: 1 });
    count(document.getElementById('r-cand'), r.counts.candidates);
    count(document.getElementById('r-planets'), r.counts.predicted_planets);
    count(document.getElementById('r-hab'), r.counts.habitable);
    const m = r.confusion;
    [['m00', m[0][0]], ['m01', m[0][1]], ['m10', m[1][0]], ['m11', m[1][1]]].forEach(([id, v]) => count(document.getElementById(id), v));
    document.getElementById('downloads').innerHTML = [
      ['Candidate predictions', 'CSV, one row per candidate', r.files.predictions],
      ['Prediction probabilities', 'CSV, model confidence per candidate', r.files.probabilities],
      ['Performance report', 'TXT, confusion matrix and class report', r.files.report],
    ].map(([t, s, f]) => `<li><a href="/results/${encodeURIComponent(f)}" download>${dlIcon}<span><strong>${t}</strong><small>${s}</small></span></a></li>`).join('');
    map.hidden = false;
    reveal();
    renderMap(document.getElementById('map-view'), r, { reducedMotion });
  }

  function showCompare(r) {
    gridClassify.hidden = true; gridCompare.hidden = false; map.hidden = true;
    resultsSub.textContent = `four models on ${r.dataset.toUpperCase()}`;
    const best = Math.max(...r.results.map((x) => x.accuracy));
    const list = document.getElementById('ranking');
    list.innerHTML = [...r.results].sort((a, b) => b.accuracy - a.accuracy).map((x) =>
      `<li class="rank${x.accuracy === best ? ' is-best' : ''}"><span class="rank__name">${x.name}</span><span class="rank__bar"><span data-w="${(x.accuracy * 100).toFixed(1)}"></span></span><span class="rank__v">${(x.accuracy * 100).toFixed(1)}%</span></li>`).join('');
    reveal();
    requestAnimationFrame(() => list.querySelectorAll('.rank__bar span').forEach((b) => {
      if (window.gsap && !reducedMotion) gsap.to(b, { width: b.dataset.w + '%', duration: 1.2, ease: 'expo.out', delay: 0.2 });
      else b.style.width = b.dataset.w + '%';
    }));
    // Pre-select the best model for the next classify run
    const bestKey = r.results.find((x) => x.accuracy === best).model;
    form.model_choice.value = bestKey;
  }
}
