import { createWorld } from './world.js';
import { setupScenes } from './scenes.js';
import { setupConsole } from './console.js';

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

let world = null;
try {
  world = createWorld(document.getElementById('world'), {
    reducedMotion,
    onSweep: (found, total) => {
      const el = document.getElementById('sweep-count');
      if (el) el.textContent = found.toLocaleString('en');
      const tot = document.getElementById('sweep-total');
      if (tot) tot.textContent = total.toLocaleString('en');
    },
  });
} catch (e) {
  console.warn('WebGL unavailable', e);
  document.documentElement.classList.add('no-webgl');
}

setupScenes({ world, reducedMotion });
setupConsole({ reducedMotion, world });

document.fonts?.ready.then(() => window.ScrollTrigger && ScrollTrigger.refresh());
