// The hero transit. Scroll scrubs the planet across the star; the light curve
// is drawn from the real geometry: the overlap of two discs over the star's area.

const SUN_R_EARTH = 109.2; // solar radii in Earth radii

function overlapArea(R, r, d) {
  if (d >= R + r) return 0;
  if (d <= Math.abs(R - r)) return Math.PI * Math.min(R, r) ** 2;
  const a = r * r * Math.acos((d * d + r * r - R * R) / (2 * d * r));
  const b = R * R * Math.acos((d * d + R * R - r * r) / (2 * d * R));
  const c = 0.5 * Math.sqrt((-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R));
  return a + b - c;
}

export function setupTransit({ reducedMotion = false } = {}) {
  const section = document.getElementById('hero');
  const stage = section.querySelector('.hero__stage');
  const star = document.getElementById('t-star');
  const planet = document.getElementById('t-planet');
  const canvas = document.getElementById('t-curve');
  const ctx = canvas.getContext('2d');
  const radiusInput = document.getElementById('t-radius');
  const hud = {
    phase: document.getElementById('hud-phase'),
    flux: document.getElementById('hud-flux'),
    depth: document.getElementById('hud-depth'),
    radius: document.getElementById('hud-radius'),
  };

  const state = { progress: 0, radiusEarth: 1 };
  const small = window.matchMedia('(max-width: 900px)').matches;
  const STAR_PX = small ? 60 : 90;   // star radius in px (matches the CSS diameter)
  const TRAVEL = small ? 170 : 280;  // planet travel half-width in px

  function planetPx() {
    // Visual radius: real ratio, but never smaller than a visible dot
    const real = (state.radiusEarth / SUN_R_EARTH) * STAR_PX;
    return Math.max(5, real * 2.6);
  }

  function flux(progress) {
    const dPx = (progress * 2 - 1) * TRAVEL;
    const r = state.radiusEarth / SUN_R_EARTH; // in star radii
    const d = Math.abs(dPx) / STAR_PX;         // in star radii
    const blocked = overlapArea(1, r, d) / Math.PI;
    return 1 - blocked;
  }

  const css = getComputedStyle(document.documentElement);
  const colors = {
    line: css.getPropertyValue('--line-2').trim() || 'rgba(163,179,207,.3)',
    ink3: css.getPropertyValue('--ink-3').trim() || '#6b7d9b',
    cyan: css.getPropertyValue('--cyan').trim() || '#22d3ee',
  };

  function draw() {
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const pad = { l: 70, r: 30, t: 30, b: 50 };
    const depth = (state.radiusEarth / SUN_R_EARTH) ** 2;
    // Vertical scale follows the dip, so an Earth-sized transit fills the chart
    // and the axis label shows how small the signal really is.
    const span = Math.max(depth * 1.4, 1e-5);
    const y = (f) => pad.t + ((1 - f) / span) * (H - pad.t - pad.b);
    const x = (p) => pad.l + p * (W - pad.l - pad.r);

    // Grid
    ctx.strokeStyle = colors.line; ctx.lineWidth = 1; ctx.setLineDash([3, 6]);
    [0, 0.5, 1].forEach((k) => { const yy = pad.t + k * (H - pad.t - pad.b); ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(W - pad.r, yy); ctx.stroke(); });
    ctx.setLineDash([]);
    ctx.font = '500 13px "JetBrains Mono", monospace'; ctx.fillStyle = colors.ink3; ctx.textAlign = 'right';
    const digits = span < 0.001 ? 5 : 4;
    ctx.fillText((1).toFixed(digits), pad.l - 10, pad.t + 5);
    ctx.fillText((1 - span).toFixed(digits), pad.l - 10, H - pad.b + 5);
    ctx.textAlign = 'left'; ctx.fillText('relative flux', pad.l, H - 16);
    ctx.textAlign = 'right'; ctx.fillText('time →', W - pad.r, H - 16);

    // Curve, drawn up to current progress bright, the rest faint
    const N = 400;
    ctx.lineWidth = 2.5; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.beginPath();
    for (let i = 0; i <= N; i++) { const p = i / N; const px = x(p), py = y(flux(p)); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.strokeStyle = 'rgba(163,179,207,0.22)'; ctx.stroke();
    ctx.beginPath();
    const upto = Math.round(state.progress * N);
    for (let i = 0; i <= upto; i++) { const p = i / N; const px = x(p), py = y(flux(p)); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.strokeStyle = colors.cyan; ctx.shadowColor = colors.cyan; ctx.shadowBlur = 12; ctx.stroke(); ctx.shadowBlur = 0;

    // Current point
    const f = flux(state.progress);
    ctx.beginPath(); ctx.arc(x(state.progress), y(f), 5, 0, Math.PI * 2);
    ctx.fillStyle = '#fff'; ctx.shadowColor = colors.cyan; ctx.shadowBlur = 16; ctx.fill(); ctx.shadowBlur = 0;

    hud.phase.textContent = (state.progress * 2 - 1).toFixed(3);
    hud.flux.textContent = f.toFixed(5);
    const ppm = depth * 1e6;
    hud.depth.textContent = ppm >= 10000 ? (depth * 100).toFixed(2) + ' %' : Math.round(ppm).toLocaleString('en') + ' ppm';
  }

  function place() {
    const dPx = (state.progress * 2 - 1) * TRAVEL;
    const size = planetPx() * 2;
    planet.style.width = planet.style.height = size + 'px';
    planet.style.margin = `${-size / 2}px 0 0 ${-size / 2}px`;
    planet.style.transform = `translateX(${dPx}px)`;
    // The star dims with the blocked light, exaggerated so it reads
    const f = flux(state.progress);
    star.style.filter = `brightness(${1 - (1 - f) * 25})`;
    draw();
  }

  radiusInput.addEventListener('input', () => {
    state.radiusEarth = parseFloat(radiusInput.value);
    hud.radius.textContent = state.radiusEarth.toFixed(1);
    place();
  });

  if (window.gsap && window.ScrollTrigger) {
    gsap.registerPlugin(ScrollTrigger);
    ScrollTrigger.create({
      trigger: section,
      start: 'top top',
      end: '+=180%',
      pin: stage,
      scrub: reducedMotion ? true : 0.6,
      onUpdate: (self) => { state.progress = self.progress; place(); },
    });
    // Headline entrance, once
    gsap.from('.hero__title .line', { yPercent: 110, opacity: 0, duration: 1.2, ease: 'expo.out', stagger: 0.08, delay: 0.1 });
    gsap.from(['.hero__lede', '.hero__actions', '.hero__hint'], { y: 24, opacity: 0, duration: 1, ease: 'expo.out', stagger: 0.08, delay: 0.45 });
    gsap.from('.transit', { opacity: 0, scale: 0.96, duration: 1.4, ease: 'expo.out', delay: 0.3 });
  } else {
    state.progress = 0.5;
  }
  place();
}
