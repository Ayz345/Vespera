// Scroll drives the world. Each pinned section hands the world its progress;
// the HTML on top only reveals captions at the right moments.

export function setupScenes({ world, reducedMotion = false } = {}) {
  if (!window.gsap || !window.ScrollTrigger) return;
  gsap.registerPlugin(ScrollTrigger);
  const scrub = reducedMotion ? true : 0.6;
  const drive = (name) => (self) => world && world.set(name, self.progress);
  const pinned = (id, length, name, extra = {}) => {
    const section = document.getElementById(id);
    const stage = section.querySelector('.scene__stage');
    return ScrollTrigger.create({ trigger: section, start: 'top top', end: '+=' + length, pin: stage, scrub, onUpdate: drive(name), ...extra });
  };

  /* ----- Hero: the transit ----- */
  pinned('hero', '170%', 'hero', {
    onUpdate: (self) => {
      world && world.set('hero', self.progress);
      const fade = Math.max(0, 1 - self.progress * 2.2);
      gsap.set('.hero__copy', { opacity: fade, y: self.progress * -60 });
      document.getElementById('hero-note').classList.toggle('is-on', self.progress > 0.45);
    },
  });
  gsap.from('.hero__title .line > span', { yPercent: 110, duration: 1.3, ease: 'expo.out', stagger: 0.09, delay: 0.2 });
  gsap.from(['.hero__tag', '.hero__lede', '.hero__actions'], { y: 24, opacity: 0, duration: 1, ease: 'expo.out', stagger: 0.1, delay: 0.5 });

  /* ----- Method: the sweep ----- */
  {
    const caps = [...document.querySelectorAll('#method .cap')];
    pinned('method', '240%', 'method', {
      onUpdate: (self) => {
        world && world.set('method', self.progress);
        const p = self.progress;
        caps.forEach((c, i) => c.classList.toggle('is-on', p >= [0.0, 0.3, 0.72][i] && p < [0.3, 0.72, 1.01][i]));
        document.getElementById('sweep-readout').classList.toggle('is-on', p > 0.14 && p < 0.95);
      },
    });
  }

  /* ----- Missions: three fields on the sky ----- */
  {
    const caps = [...document.querySelectorAll('#missions .mission')];
    pinned('missions', '240%', 'missions', {
      onUpdate: (self) => {
        world && world.set('missions', self.progress);
        const third = self.progress * 3;
        caps.forEach((c, i) => c.classList.toggle('is-on', third >= i && third < i + 1));
      },
    });
  }

  /* ----- Zone: push the planet outward ----- */
  {
    const section = document.getElementById('zone');
    const auOut = document.getElementById('zone-au');
    const verdict = document.getElementById('zone-verdict');
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
      verdict.classList.toggle('is-ok', ok);
      verdict.textContent = ok ? 'Inside the window.' : teq > 310 ? 'Too close.' : period > 500 ? 'Temperate, but the year is too long.' : 'Too far.';
    }
    pinned('zone', '220%', 'zone', { onUpdate: (self) => { world && world.set('zone', self.progress); render(self.progress); } });
    render(0);
  }

  /* ----- Console and team: the world recedes ----- */
  ScrollTrigger.create({ trigger: '#console', start: 'top bottom', end: 'bottom top', onUpdate: drive('console') });
  ScrollTrigger.create({ trigger: '#team', start: 'top bottom', end: 'bottom top', onUpdate: drive('team') });

  /* ----- Split headings and reveals ----- */
  ScrollTrigger.batch('[data-split]:not(.hero__title)', { start: 'top 85%', onEnter: (els) => els.forEach((el) => el.classList.add('is-in')), once: true });
  document.querySelectorAll('.console__head, .panel, .team__copy, .team__photo').forEach((el) => el.setAttribute('data-reveal', ''));
  ScrollTrigger.batch('[data-reveal]', { start: 'top 88%', onEnter: (els) => els.forEach((el, i) => setTimeout(() => el.classList.add('is-in'), i * 80)), once: true });
  ScrollTrigger.create({ trigger: '#footer-word', start: 'top 95%', onEnter: () => document.getElementById('footer-word').classList.add('is-in'), once: true });

  /* ----- Marquee follows scroll velocity ----- */
  {
    const track = document.getElementById('marquee');
    if (track) {
      let x = 0, vel = 0, base = reducedMotion ? 0 : 0.6;
      const half = () => track.scrollWidth / 2;
      ScrollTrigger.create({ start: 0, end: 'max', onUpdate: (self) => { vel = self.getVelocity() / 400; } });
      (function tick() {
        x -= base + Math.min(14, Math.abs(vel)) * Math.sign(vel || 1);
        vel *= 0.92;
        const h = half();
        if (x < -h) x += h; if (x > 0) x -= h;
        track.style.transform = `translate3d(${x}px,0,0)`;
        requestAnimationFrame(tick);
      })();
    }
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
