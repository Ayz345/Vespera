// Scroll scenes: the pipeline, the missions track, the habitable-zone orbit.
// Each pins its stage and scrubs a timeline against the scroll.

export function setupScenes({ reducedMotion = false } = {}) {
  if (!window.gsap || !window.ScrollTrigger) return;
  gsap.registerPlugin(ScrollTrigger);
  const scrub = reducedMotion ? true : 0.8;
  const mm = gsap.matchMedia();

  /* ---------- Method: archive row → model → verdict ---------- */
  {
    const section = document.getElementById('method');
    const stage = section.querySelector('.method__stage');
    const stages = section.querySelectorAll('.pipe__stage');
    const rows = section.querySelectorAll('.row');
    const chips = section.querySelectorAll('.chips li');
    const models = section.querySelectorAll('.model');
    const packets = section.querySelectorAll('.pipe__packet');
    const fill = section.querySelector('.verdict__fill');
    const pEl = section.querySelector('.verdict__p');

    const tl = gsap.timeline({
      defaults: { ease: 'none' },
      scrollTrigger: { trigger: section, start: 'top top', end: '+=260%', pin: stage, scrub },
    });
    // Timeline: 0–0.9 rows light up (classes below), packet 1, chips and models, packet 2, verdict
    tl.to({}, { duration: 0.9 });
    tl.set(packets[0], { opacity: 1 }).fromTo(packets[0], { xPercent: -100, yPercent: -100 }, { xPercent: 200, yPercent: 100, duration: 0.5 }).set(packets[0], { opacity: 0 });
    tl.to({}, { duration: 0.9 });
    tl.set(packets[1], { opacity: 1 }).fromTo(packets[1], { xPercent: -100, yPercent: -100 }, { xPercent: 200, yPercent: 100, duration: 0.5 }).set(packets[1], { opacity: 0 });
    tl.to(fill, { width: '93%', duration: 0.8, ease: 'power2.out' });
    const counter = { v: 0 };
    tl.to(counter, { v: 0.93, duration: 0.8, onUpdate: () => { pEl.textContent = counter.v.toFixed(2); } }, '<');
    tl.to({}, { duration: 0.6 });

    // Class toggles keyed to progress so they reverse cleanly when scrolling back up
    ScrollTrigger.create({
      trigger: section, start: 'top top', end: '+=260%',
      onUpdate: (self) => {
        const p = self.progress;
        stages[0].classList.toggle('is-lit', p > 0.02);
        stages[1].classList.toggle('is-lit', p > 0.34);
        stages[2].classList.toggle('is-lit', p > 0.7);
        rows.forEach((r, i) => r.classList.toggle('is-on', p > 0.03 + i * 0.035));
        chips.forEach((c, i) => c.classList.toggle('is-on', p > 0.36 + i * 0.015));
        const mi = Math.floor((p - 0.5) / 0.045);
        models.forEach((m, i) => m.classList.toggle('is-on', p > 0.5 && i === Math.min(3, Math.max(0, mi))));
      },
    });
  }

  /* ---------- Missions: pinned horizontal track on wide screens ---------- */
  mm.add('(min-width: 901px)', () => {
    const section = document.getElementById('missions');
    const stage = section.querySelector('.missions__stage');
    const track = document.getElementById('missions-track');
    const tween = gsap.to(track, {
      x: () => -(track.scrollWidth - window.innerWidth),
      ease: 'none',
      scrollTrigger: {
        trigger: section, start: 'top top', pin: stage, scrub,
        end: () => '+=' + (track.scrollWidth - window.innerWidth + window.innerHeight * 0.4),
        invalidateOnRefresh: true,
      },
    });
    section.querySelectorAll('.mission').forEach((card) => {
      gsap.from(card, {
        y: 60, opacity: 0, duration: 0.8, ease: 'expo.out',
        scrollTrigger: { containerAnimation: tween, trigger: card, start: 'left 90%', toggleActions: 'play none none reverse' },
      });
    });
    return () => {};
  });
  mm.add('(max-width: 900px)', () => {
    ScrollTrigger.batch('#missions .mission', {
      start: 'top 85%',
      onEnter: (els) => gsap.fromTo(els, { y: 40, opacity: 0 }, { y: 0, opacity: 1, duration: 0.8, ease: 'expo.out', stagger: 0.1 }),
    });
  });

  /* ---------- Habitable zone: push a planet outward ---------- */
  {
    const section = document.getElementById('zone');
    const stage = section.querySelector('.zone__stage');
    const planet = document.getElementById('orbit-planet');
    const path = document.getElementById('orbit-path');
    const auOut = document.getElementById('orbit-au');
    const verdict = document.getElementById('zone-verdict');
    const g = {
      period: { out: document.getElementById('g-period'), pin: document.getElementById('p-period'), el: section.querySelector('[data-gauge="period"]'), lo: 50, hi: 500, min: 0, max: 1000 },
      teq: { out: document.getElementById('g-teq'), pin: document.getElementById('p-teq'), el: section.querySelector('[data-gauge="teq"]'), lo: 180, hi: 310, min: 0, max: 600 },
    };
    section.querySelector('[data-gauge="radius"]').classList.add('is-ok');
    section.querySelector('[data-gauge="teff"]').classList.add('is-ok');

    const state = { au: 0.2 };
    const PX_PER_AU = 110; // 1 AU = 110 px in the 600-box
    function render() {
      const au = state.au;
      const period = Math.pow(au, 1.5) * 365.25;          // Kepler's third law, Sun-mass star
      const teq = 255 / Math.sqrt(au);                      // equilibrium temperature, Earth albedo
      const r = 48 + au * PX_PER_AU;
      path.setAttribute('r', r);
      planet.setAttribute('transform', `translate(${300 + r * Math.cos(-0.6)}, ${300 + r * Math.sin(-0.6)})`);
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
      verdict.classList.toggle('is-ok', ok);
      verdict.textContent = ok ? 'Inside the window. Vespera would flag this one.'
        : teq > 310 ? 'Too close. Rock melts, water boils away.'
        : period > 500 ? 'Temperate, but a year this long falls outside the window the classifier trusts.'
        : 'Too far. Frozen, and a year too long to confirm quickly.';
    }
    gsap.to(state, {
      au: 2.6, ease: 'none',
      scrollTrigger: { trigger: section, start: 'top top', end: '+=200%', pin: stage, scrub },
      onUpdate: render,
    });
    render();
  }

  /* ---------- Entrances for everything else ---------- */
  document.querySelectorAll('.console__head, .panel, .team__copy, .team__photo, .section-title, .section-lede').forEach((el) => {
    if (el.closest('.hero')) return;
    el.setAttribute('data-reveal', '');
  });
  ScrollTrigger.batch('[data-reveal]', {
    start: 'top 88%',
    onEnter: (els) => els.forEach((el, i) => setTimeout(() => el.classList.add('is-in'), i * 80)),
  });

  /* ---------- Nav: hide on scroll down, show on scroll up; highlight section ---------- */
  const nav = document.getElementById('nav');
  let last = 0;
  ScrollTrigger.create({
    start: 0, end: 'max',
    onUpdate: (self) => {
      const y = self.scroll();
      nav.classList.toggle('is-hidden', y > 120 && y > last && self.direction === 1);
      last = y;
    },
  });
  const links = [...nav.querySelectorAll('.nav__links a')];
  links.forEach((a) => {
    const target = document.querySelector(a.getAttribute('href'));
    if (!target) return;
    ScrollTrigger.create({
      trigger: target, start: 'top 50%', end: 'bottom 50%',
      onToggle: (self) => a.classList.toggle('is-active', self.isActive),
    });
  });
}
