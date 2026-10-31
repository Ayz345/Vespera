import { startStarfield } from './starfield.js';
import { setupTransit } from './transit.js';
import { setupScenes } from './scenes.js';
import { setupConsole } from './console.js';

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// Cursor glow follows the pointer (desktop only)
if (window.matchMedia('(pointer: fine)').matches) {
  document.body.classList.add('has-pointer');
  const glow = document.querySelector('.cursor-glow');
  let x = innerWidth / 2, y = innerHeight / 2, tx = x, ty = y;
  addEventListener('pointermove', (e) => { tx = e.clientX; ty = e.clientY; }, { passive: true });
  (function tick() { x += (tx - x) * 0.12; y += (ty - y) * 0.12; glow.style.transform = `translate(${x}px, ${y}px)`; requestAnimationFrame(tick); })();
}

try { startStarfield(document.getElementById('stars'), { reducedMotion }); } catch (e) { console.warn('Starfield unavailable', e); }

setupTransit({ reducedMotion });
setupScenes({ reducedMotion });
setupConsole({ reducedMotion });

// Fonts change layout; recompute pins once they are in
document.fonts?.ready.then(() => window.ScrollTrigger && ScrollTrigger.refresh());
