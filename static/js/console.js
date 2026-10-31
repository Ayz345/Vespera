// The launch sequence: pick a mission, pick a model (or all four), drop a table, run.

export function setupConsole({ reducedMotion = false, world = null } = {}) {
  const form = document.getElementById('run-form');
  const fileInput = document.getElementById('file');
  const drop = document.getElementById('drop');
  const dropName = document.getElementById('drop-name');
  const dropHint = document.getElementById('drop-hint');
  const runBtn = document.getElementById('run-btn');
  const runLabel = document.getElementById('run-label');
  const runFill = document.getElementById('run-fill');
  const status = document.getElementById('run-status');
  const results = document.getElementById('results');
  const resultsSub = document.getElementById('results-sub');
  const gridClassify = document.getElementById('results-classify');
  const gridCompare = document.getElementById('results-compare');
  const map = document.getElementById('map');
  const emptyName = dropName.textContent, emptyHint = dropHint.textContent, idleStatus = status.textContent;

  const model = () => form.querySelector('input[name=model_choice]:checked').value;
  const mission = () => form.querySelector('input[name=dataset]:checked').value;
  const MODEL_NAMES = { logreg: 'logistic regression', rf: 'random forest', svm: 'the SVM', xgb: 'XGBoost', all: 'all four models' };

  function syncLabel() {
    runLabel.textContent = model() === 'all' ? 'Compare the four models' : 'Run the model';
    if (!runBtn.disabled) status.textContent = idleStatus;
  }
  form.querySelectorAll('input[name=model_choice], input[name=dataset]').forEach((i) => i.addEventListener('change', syncLabel));

  /* File: show the name, size and a quick row count read in the browser */
  let rowCount = null;
  function updateFile() {
    const f = fileInput.files && fileInput.files[0];
    if (!f) { dropName.textContent = emptyName; dropHint.textContent = emptyHint; drop.classList.remove('has-file'); rowCount = null; return; }
    const kb = f.size / 1024;
    dropName.textContent = f.name;
    dropHint.textContent = (kb > 1024 ? (kb / 1024).toFixed(1) + ' MB' : Math.round(kb) + ' KB') + ' · counting rows…';
    drop.classList.add('has-file');
    f.slice(0, 8 * 1024 * 1024).text().then((txt) => {
      const lines = txt.split('\n').filter((l) => l.trim() && !l.startsWith('#'));
      rowCount = Math.max(0, lines.length - 1);
      dropHint.textContent = (kb > 1024 ? (kb / 1024).toFixed(1) + ' MB' : Math.round(kb) + ' KB') + ' · ' + rowCount.toLocaleString('en') + (f.size > 8 * 1024 * 1024 ? '+' : '') + ' rows · ready';
    }).catch(() => {});
  }
  fileInput.addEventListener('change', updateFile);
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('is-over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('is-over'); }));
  drop.addEventListener('drop', (e) => { if (e.dataTransfer?.files?.length) { fileInput.files = e.dataTransfer.files; updateFile(); } });

  /* Run */
  let fillTween = null;
  function setBusy(on, text) {
    runBtn.disabled = on;
    runBtn.classList.toggle('is-busy', on);
    form.classList.toggle('is-busy', on);
    status.classList.remove('is-error');
    if (on) {
      runLabel.textContent = text;
      status.textContent = `${MODEL_NAMES[model()]} on ${mission().toUpperCase()}${rowCount ? ' · ' + rowCount.toLocaleString('en') + ' rows' : ''}`;
      // The fill creeps toward 90% while we wait, then snaps to 100% on return
      if (window.gsap && !reducedMotion) { gsap.set(runFill, { scaleX: 0 }); fillTween = gsap.to(runFill, { scaleX: 0.9, duration: 12, ease: 'power1.out' }); }
      else runFill.style.transform = 'scaleX(0.5)';
    } else {
      fillTween && fillTween.kill();
      if (window.gsap && !reducedMotion) gsap.to(runFill, { scaleX: 1, duration: 0.4, ease: 'power3.out', onComplete: () => gsap.to(runFill, { scaleX: 0, duration: 0.6, delay: 0.4, transformOrigin: 'right center' }) });
      else runFill.style.transform = 'scaleX(0)';
      syncLabel();
    }
  }
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput.files?.length) {
      status.textContent = 'drop a CSV first'; status.classList.add('is-error'); drop.classList.add('is-over');
      drop.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'center' });
      setTimeout(() => drop.classList.remove('is-over'), 900);
      return;
    }
    const compare = model() === 'all';
    const data = new FormData();
    data.append('dataset', mission());
    data.append('file', fileInput.files[0]);
    if (!compare) data.append('model_choice', model());
    setBusy(true, compare ? 'Training four models' : 'Training');
    try {
      const res = await fetch(compare ? '/api/compare' : '/api/analyze', { method: 'POST', body: data });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(json.error || 'The server could not run the model.');
      setBusy(false);
      compare ? showCompare(json) : showClassify(json);
    } catch (err) {
      setBusy(false);
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
    document.getElementById('results-scene').hidden = false;
    if (window.gsap && !reducedMotion) {
      gsap.fromTo(results, { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.9, ease: 'expo.out' });
      gsap.fromTo(results.querySelectorAll('.stat, .matrix, .downloads a, .rank'), { opacity: 0, y: 18 }, { opacity: 1, y: 0, duration: 0.8, ease: 'expo.out', stagger: 0.05, delay: 0.1 });
    }
    requestAnimationFrame(() => {
      const start = window.__vesperaResultsReady ? window.__vesperaResultsReady() : results.getBoundingClientRect().top + window.scrollY;
      window.scrollTo({ top: start + 2, behavior: reducedMotion ? 'auto' : 'smooth' });
      if (world) world.arriveResults();
    });
  }
  function showClassify(r) {
    gridClassify.hidden = false; gridCompare.hidden = true;
    resultsSub.textContent = `${r.model_name} · ${r.dataset.toUpperCase()}`;
    count(document.getElementById('r-acc'), r.accuracy * 100, { decimals: 1 });
    count(document.getElementById('r-cand'), r.counts.candidates);
    count(document.getElementById('r-planets'), r.counts.predicted_planets);
    count(document.getElementById('r-hab'), r.counts.habitable);
    const m = r.confusion;
    [['m00', m[0][0]], ['m01', m[0][1]], ['m10', m[1][0]], ['m11', m[1][1]]].forEach(([id, v]) => count(document.getElementById(id), v));
    document.getElementById('downloads').innerHTML = [
      ['Candidate predictions', 'CSV · one row per candidate', r.files.predictions],
      ['Prediction probabilities', 'CSV · confidence per candidate', r.files.probabilities],
      ['Performance report', 'TXT · matrix and class report', r.files.report],
    ].map(([t, s, f]) => `<li><a href="/results/${encodeURIComponent(f)}" download>${dlIcon}<span><strong>${t}</strong><small>${s}</small></span></a></li>`).join('');
    map.hidden = false;
    const info = world ? world.setResults(r) : null;
    const note = document.getElementById('map-note');
    if (note) note.textContent = (info && !info.wide ? `the field is opened ${info.angleScale.toFixed(1)}× wider than it is on the sky · ` : '') + 'distance from the Sun on a log scale · drag to turn · wheel to zoom';
    reveal();
  }
  function showCompare(r) {
    gridClassify.hidden = true; gridCompare.hidden = false; map.hidden = true;
    resultsSub.textContent = `four models · ${r.dataset.toUpperCase()}`;
    const best = Math.max(...r.results.map((x) => x.accuracy));
    const list = document.getElementById('ranking');
    list.innerHTML = [...r.results].sort((a, b) => b.accuracy - a.accuracy).map((x) =>
      `<li class="rank${x.accuracy === best ? ' is-best' : ''}"><span class="rank__name">${x.name}</span><span class="rank__bar"><span data-w="${(x.accuracy * 100).toFixed(1)}"></span></span><span class="rank__v">${(x.accuracy * 100).toFixed(1)}%</span></li>`).join('');
    reveal();
    requestAnimationFrame(() => list.querySelectorAll('.rank__bar span').forEach((b) => {
      if (window.gsap && !reducedMotion) gsap.to(b, { width: b.dataset.w + '%', duration: 1.2, ease: 'expo.out', delay: 0.2 });
      else b.style.width = b.dataset.w + '%';
    }));
    // Pre-select the best model for the next run
    const bestKey = r.results.find((x) => x.accuracy === best).model;
    const input = form.querySelector(`input[name=model_choice][value="${bestKey}"]`);
    if (input) { input.checked = true; syncLabel(); }
  }
  syncLabel();
}
