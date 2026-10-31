// Scroll drives the world through one director: the scroll position decides exactly one
// scene and one 0–1 progress, in both directions, so no two triggers ever fight over the camera.

const smoothstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

export function setupScenes({ world, reducedMotion = false } = {}) {
  if (!window.gsap || !window.ScrollTrigger) return;
  gsap.registerPlugin(ScrollTrigger);
  const scrub = reducedMotion ? true : 0.6;
  const blackout = document.getElementById('blackout');
  const setDark = (d) => { blackout.style.opacity = d.toFixed(3); world && world.setDark(d); };
  const show = (el, on) => el && el.classList.toggle('is-in', on);

  /* ----- Pinned scenes, in page order. Each gets a handler(progress). ----- */
  const scenes = [];
  function pin(id, length, handler) {
    const section = document.getElementById(id);
    const stage = section.querySelector('.scene__stage');
    const st = ScrollTrigger.create({ trigger: section, start: 'top top', end: '+=' + length, pin: stage, scrub, anticipatePin: 1 });
    const entry = { id, st, stage, handler };
    scenes.push(entry);
    return entry;
  }

  /* Hero: the eclipse over the first 84% of the pin, then the title card at a fixed speed */
  {
    const dark = document.getElementById('dark');
    const note = document.getElementById('hero-note');
    const copy = document.querySelector('.hero__copy');
    pin('hero', '165%', (p) => {
      world && world.set('hero', p);
      gsap.set(copy, { opacity: Math.max(0, 1 - p * 3.5), y: p * -80 });
      note.classList.toggle('is-on', p > 0.08 && p < 0.26);
      setDark(smoothstep(0.43, 0.47, p));          // totality lands at 0.44
      dark.classList.remove('is-out');
      dark.classList.toggle('is-on', p > 0.48);      // the card plays once, by CSS, at its own speed
    });
    gsap.from('.hero__title .line > span', { yPercent: 110, duration: 1.3, ease: 'expo.out', stagger: 0.09, delay: 0.2 });
    gsap.from(['.hero__tag', '.hero__lede', '.hero__actions'], { y: 24, opacity: 0, duration: 1, ease: 'expo.out', stagger: 0.1, delay: 0.5 });
  }

  /* Method: the light returns fast, then the sweep */
  {
    const dark = document.getElementById('dark');
    const title = document.getElementById('method-title');
    const caps = [...document.querySelectorAll('#method .cap')];
    const readout = document.getElementById('sweep-readout');
    pin('method', '240%', (p) => {
      dark.classList.toggle('is-out', p > 0.01);
      setDark(1 - smoothstep(0.02, 0.1, p));
      world && world.set('method', p);
      show(title, p > 0.04);
      caps.forEach((c, i) => c.classList.toggle('is-on', p >= [0.08, 0.34, 0.74][i] && p < [0.34, 0.74, 1.01][i]));
      readout.classList.toggle('is-on', p > 0.2 && p < 0.96);
    });
  }

  /* Missions */
  {
    const title = document.getElementById('missions-title');
    const caps = [...document.querySelectorAll('#missions .mission')];
    const links = document.querySelector('#missions .mission__links');
    pin('missions', '220%', (p) => {
      world && world.set('missions', p);
      show(title, p > 0.03);
      const third = p * 3;
      caps.forEach((c, i) => c.classList.toggle('is-on', third >= i && third < i + 1));
      links && links.classList.toggle('is-in', p > 0.05);
    });
  }

  /* Zone */
  {
    const section = document.getElementById('zone');
    const title = document.getElementById('zone-title');
    const gauges = section.querySelector('.gauges');
    const verdictEl = document.getElementById('zone-verdict');
    const readout = section.querySelector('.zone__readout');
    const auOut = document.getElementById('zone-au');
    const g = {
      period: { out: document.getElementById('g-period'), pin: document.getElementById('p-period'), el: section.querySelector('[data-gauge="period"]'), lo: 50, hi: 500, min: 0, max: 1000 },
      teq: { out: document.getElementById('g-teq'), pin: document.getElementById('p-teq'), el: section.querySelector('[data-gauge="teq"]'), lo: 180, hi: 310, min: 0, max: 600 },
    };
    section.querySelector('[data-gauge="radius"]').classList.add('is-ok');
    section.querySelector('[data-gauge="teff"]').classList.add('is-ok');
    function render(t) {
      const au = 0.2 + 2.4 * t;
      const period = Math.pow(au, 1.5) * 365.25;
      const teq = 255 / Math.sqrt(au);
      auOut.textContent = au.toFixed(2);
      g.period.out.textContent = Math.round(period).toLocaleString('en') + ' d';
      g.period.pin.style.left = Math.min(100, Math.max(0, (period - g.period.min) / (g.period.max - g.period.min) * 100)) + '%';
      const okP = period >= g.period.lo && period <= g.period.hi;
      g.period.el.classList.toggle('is-ok', okP);
      g.teq.out.textContent = Math.round(teq).toLocaleString('en') + ' K';
      g.teq.pin.style.left = Math.min(100, Math.max(0, (teq - g.teq.min) / (g.teq.max - g.teq.min) * 100)) + '%';
      const okT = teq >= g.teq.lo && teq <= g.teq.hi;
      g.teq.el.classList.toggle('is-ok', okT);
      const ok = okP && okT;
      verdictEl.classList.toggle('is-ok', ok);
      verdictEl.textContent = ok ? 'Inside the window.' : teq > 310 ? 'Too close.' : period > 500 ? 'Temperate, but the year is too long.' : 'Too far.';
    }
    pin('zone', '220%', (p) => {
      world && world.set('zone', p);
      show(title, p > 0.06);
      show(gauges, p > 0.1); show(verdictEl, p > 0.12); show(readout, p > 0.12);
      const t = smoothstep(0.14, 1, p);
      world && world.setOrbit(t);
      render(t);
    });
    render(0);
  }

  /* Console */
  {
    const title = document.getElementById('console-title');
    const grid = document.querySelector('.console__grid');
    const top = document.querySelector('.console__top');
    pin('console', '100%', (p) => {
      world && world.set('console', p);
      show(title, p > 0.06);
      grid.classList.toggle('is-in', p > 0.1);
      top.classList.toggle('is-in', p > 0.06);
    });
  }

  /* Results: pinned once it exists (console.js calls window.__vesperaResultsReady) */
  const resultsSection = document.getElementById('results-scene');
  let resultsEntry = null;
  function armResults() {
    if (resultsEntry) { ScrollTrigger.refresh(); return resultsEntry; }
    resultsEntry = pin('results-scene', '80%', (p) => { world && world.set('results', p); });
    ScrollTrigger.refresh();
    return resultsEntry;
  }
  window.__vesperaResultsReady = () => { armResults(); return resultsEntry.st.start; };

  /* Team: a plain section after the last pin */
  const teamST = ScrollTrigger.create({ trigger: '#team', start: 'top bottom', end: 'bottom top' });

  /* ----- The director ----- */
  let current = null;
  function direct(y) {
    const list = scenes.filter((s) => !s.st.disabled && s.st.end > s.st.start).sort((a, b) => a.st.start - b.st.start);
    let hit = null;
    for (const s of list) { if (y >= s.st.start && y <= s.st.end) { hit = s; break; } }
    if (!hit) {
      const last = list.filter((s) => s.st.end < y).pop();
      if (last && teamST.start <= y) { world && world.set('team', teamST.progress); setTeamDone(last); return; }
      hit = last || list[0];
    }
    const p = Math.min(1, Math.max(0, (y - hit.st.start) / (hit.st.end - hit.st.start)));
    hit.handler(p);
    // Fade a stage's HTML out when its scene is done, so the next pinned stage can cut in cleanly
    list.forEach((s) => s.stage.classList.toggle('is-done', s !== hit));        // only the live scene shows its HTML
    current = hit;
  }
  function setTeamDone(last) { scenes.forEach((s) => s.stage.classList.toggle('is-done', true)); }
  ScrollTrigger.create({ start: 0, end: 'max', onUpdate: (self) => direct(self.scroll()), onRefresh: (self) => direct(self.scroll()) });
  direct(window.scrollY);

  /* ----- Results interaction: drag turns, wheel zooms, in the empty half only ----- */
  {
    const res = resultsSection;
    const hud = res.querySelector('.results__hud');
    const inHud = (e) => e.target.closest('.results__hud, a, button, .map');
    let dragging = false, lx = 0, ly = 0;
    res.addEventListener('pointerdown', (e) => { if (inHud(e)) return; e.preventDefault(); dragging = true; lx = e.clientX; ly = e.clientY; res.classList.add('is-dragging'); });
    addEventListener('pointermove', (e) => { if (!dragging || !world) return; world.turnResults((e.clientX - lx) * 0.0035, (e.clientY - ly) * 0.0025); lx = e.clientX; ly = e.clientY; });
    addEventListener('pointerup', () => { dragging = false; res.classList.remove('is-dragging'); });
    res.addEventListener('wheel', (e) => {
      if (inHud(e) || !resultsEntry || !world) return;
      const y = window.scrollY;
      if (y < resultsEntry.st.start - 5 || y > resultsEntry.st.end + 5) return;      // only while the scene is pinned; the panel side still scrolls
      e.preventDefault();
      world.zoomResults(Math.exp(e.deltaY * 0.0012));     // wheel up = closer
    }, { passive: false });
  }

  /* ----- Headings and reveals outside the pinned scenes ----- */
  ScrollTrigger.batch('#team-title', { start: 'top 85%', onEnter: (els) => els.forEach((el) => el.classList.add('is-in')), once: true });
  document.querySelectorAll('.team__copy').forEach((el) => el.setAttribute('data-reveal', ''));
  ScrollTrigger.batch('[data-reveal]', { start: 'top 88%', onEnter: (els) => els.forEach((el, i) => setTimeout(() => el.classList.add('is-in'), i * 90)), once: true });
  ScrollTrigger.create({ trigger: '#footer-word', start: 'top 95%', onEnter: () => document.getElementById('footer-word').classList.add('is-in'), once: true });

  /* ----- Progress bar, nav state ----- */
  const bar = document.getElementById('progress-bar');
  const nav = document.getElementById('nav');
  let last = 0;
  ScrollTrigger.create({
    start: 0, end: 'max',
    onUpdate: (self) => {
      bar.style.width = (self.progress * 100).toFixed(2) + '%';
      const y = self.scroll();
      nav.classList.toggle('is-hidden', y > 120 && y > last && self.direction === 1);
      last = y;
    },
  });
  nav.querySelectorAll('.nav__links a').forEach((a) => {
    const target = document.querySelector(a.getAttribute('href'));
    if (!target) return;
    ScrollTrigger.create({ trigger: target, start: 'top 50%', end: 'bottom 50%', onToggle: (self) => a.classList.toggle('is-active', self.isActive) });
  });

  /* ----- Magnetic buttons (desktop) ----- */
  if (window.matchMedia('(pointer: fine)').matches && !reducedMotion) {
    document.querySelectorAll('[data-magnetic]').forEach((el) => {
      el.addEventListener('pointermove', (e) => {
        const r = el.getBoundingClientRect();
        gsap.to(el, { x: (e.clientX - r.left - r.width / 2) * 0.25, y: (e.clientY - r.top - r.height / 2) * 0.35, duration: 0.5, ease: 'power3.out' });
      });
      el.addEventListener('pointerleave', () => gsap.to(el, { x: 0, y: 0, duration: 0.7, ease: 'elastic.out(1, 0.5)' }));
    });
  }
}
