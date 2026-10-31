// Scroll drives the world. Each pinned section hands the world its progress;
// the HTML on top reveals captions at the moments the camera arrives.

const smoothstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

export function setupScenes({ world, reducedMotion = false } = {}) {
  if (!window.gsap || !window.ScrollTrigger) return;
  gsap.registerPlugin(ScrollTrigger);
  const scrub = reducedMotion ? true : 0.6;
  const blackout = document.getElementById('blackout');
  const setDark = (d) => { blackout.style.opacity = d.toFixed(3); world && world.setDark(d); };
  const pinned = (id, length, onUpdate) => {
    const section = document.getElementById(id);
    const stage = section.querySelector('.scene__stage');
    return ScrollTrigger.create({ trigger: section, start: 'top top', end: '+=' + length, pin: stage, scrub, onUpdate });
  };
  const show = (el, on) => el && el.classList.toggle('is-in', on);

  /* ----- Hero: the transit pulls the screen into darkness ----- */
  {
    const dark = document.getElementById('dark');
    const note = document.getElementById('hero-note');
    const copy = document.querySelector('.hero__copy');
    pinned('hero', '230%', (self) => {
      const p = self.progress;
      world && world.set('hero', p);                                // the planet sits in front of the star around 45%
      gsap.set(copy, { opacity: Math.max(0, 1 - p * 3), y: p * -80 });
      note.classList.toggle('is-on', p > 0.14 && p < 0.4);
      const d = smoothstep(0.495, 0.535, p);                        // totality at 0.496, the frame is full by 0.52
      setDark(d);
      dark.style.opacity = smoothstep(0.55, 0.68, p).toFixed(3);    // the strip surfaces in the dark
      dark.classList.toggle('is-on', p > 0.55);
    });
    gsap.from('.hero__title .line > span', { yPercent: 110, duration: 1.3, ease: 'expo.out', stagger: 0.09, delay: 0.2 });
    gsap.from(['.hero__tag', '.hero__lede', '.hero__actions'], { y: 24, opacity: 0, duration: 1, ease: 'expo.out', stagger: 0.1, delay: 0.5 });
  }

  /* ----- Method: light returns with the heading, then the sweep ----- */
  {
    const title = document.getElementById('method-title');
    const caps = [...document.querySelectorAll('#method .cap')];
    const readout = document.getElementById('sweep-readout');
    pinned('method', '260%', (self) => {
      const p = self.progress;
      setDark(1 - smoothstep(0.02, 0.22, p));
      world && world.set('method', p);
      show(title, p > 0.06);
      caps.forEach((c, i) => c.classList.toggle('is-on', p >= [0.1, 0.34, 0.74][i] && p < [0.34, 0.74, 1.01][i]));
      readout.classList.toggle('is-on', p > 0.2 && p < 0.96);
    });
  }

  /* ----- Missions: three fields on the sky ----- */
  {
    const title = document.getElementById('missions-title');
    const caps = [...document.querySelectorAll('#missions .mission')];
    const links = document.querySelector('#missions .mission__links');
    pinned('missions', '240%', (self) => {
      const p = self.progress;
      world && world.set('missions', p);
      show(title, p > 0.03);
      const third = p * 3;
      caps.forEach((c, i) => c.classList.toggle('is-on', third >= i && third < i + 1));
      links && links.classList.toggle('is-on', p > 0.05);
    });
  }

  /* ----- Zone: the camera arrives, then the planet is pushed outward ----- */
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
    function render(p) {
      const au = 0.2 + 2.4 * p;
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
    pinned('zone', '240%', (self) => {
      const p = self.progress;
      world && world.set('zone', p);
      show(title, p > 0.06);                                   // the camera dashes in over the first 10%
      show(gauges, p > 0.1); show(verdictEl, p > 0.12); show(readout, p > 0.12);
      const t = smoothstep(0.14, 1, p);                        // the planet moves only once we are there
      world && world.setOrbit(t);
      render(t);
    });
    render(0);
  }

  /* ----- Console and team: the world recedes ----- */
  {
    const title = document.getElementById('console-title');
    const grid = document.querySelector('.console__grid');
    const top = document.querySelector('.console__top');
    pinned('console', '120%', (self) => {
      const p = self.progress;
      world && world.set('console', p);
      show(title, p > 0.08);
      grid.classList.toggle('is-in', p > 0.12);
      top.classList.toggle('is-in', p > 0.08);
    });
  }
  ScrollTrigger.create({ trigger: '#results', start: 'top 60%', end: 'bottom 40%', onUpdate: (self) => { if (world && !document.getElementById('results').hidden) world.set('results', self.progress); } });
  ScrollTrigger.create({ trigger: '#team', start: 'top bottom', end: 'bottom top', onUpdate: (self) => world && world.set('team', self.progress) });

  /* ----- Headings and reveals outside the pinned scenes ----- */
  ScrollTrigger.batch('#team-title', { start: 'top 85%', onEnter: (els) => els.forEach((el) => el.classList.add('is-in')), once: true });
  document.querySelectorAll('.team__copy').forEach((el) => el.setAttribute('data-reveal', ''));
  ScrollTrigger.batch('[data-reveal]', { start: 'top 88%', onEnter: (els) => els.forEach((el, i) => setTimeout(() => el.classList.add('is-in'), i * 90)), once: true });
  ScrollTrigger.create({ trigger: '#footer-word', start: 'top 95%', onEnter: () => document.getElementById('footer-word').classList.add('is-in'), once: true });

  /* ----- The dark strip: three rows, each at its own speed, nudged by scroll velocity ----- */
  {
    const tracks = [...document.querySelectorAll('#dark .dark__track')].map((el) => ({ el, x: 0, speed: parseFloat(el.dataset.speed || '1') }));
    let vel = 0;
    ScrollTrigger.create({ start: 0, end: 'max', onUpdate: (self) => { vel = self.getVelocity() / 500; } });
    (function tick() {
      const kick = Math.min(10, Math.abs(vel)) * Math.sign(vel || 1);
      tracks.forEach((t) => {
        const half = t.el.scrollWidth / 2 || 1;
        t.x -= (reducedMotion ? 0 : t.speed) + kick * Math.sign(t.speed);
        if (t.x < -half) t.x += half; if (t.x > 0) t.x -= half;
        t.el.style.transform = `translate3d(${t.x}px,0,0)`;
      });
      vel *= 0.92;
      requestAnimationFrame(tick);
    })();
  }

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
