// A field of stars behind everything. Three.js points, soft sprites, slow drift,
// parallax toward the pointer, and a gentle push forward as the page scrolls.
import * as THREE from 'three';

export function startStarfield(canvas, { reducedMotion = false } = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true, powerPreference: 'low-power' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 400);
  camera.position.z = 60;

  // Soft round sprite drawn once on a canvas
  const spriteCanvas = document.createElement('canvas');
  spriteCanvas.width = spriteCanvas.height = 64;
  const g = spriteCanvas.getContext('2d');
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.3, 'rgba(255,255,255,0.8)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  const sprite = new THREE.CanvasTexture(spriteCanvas);

  function layer(count, spread, size, color, opacity) {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * spread;
      pos[i * 3 + 1] = (Math.random() - 0.5) * spread;
      pos[i * 3 + 2] = (Math.random() - 0.5) * spread;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({ size, map: sprite, color, transparent: true, opacity, depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true });
    const pts = new THREE.Points(geo, mat);
    scene.add(pts);
    return pts;
  }
  const far = layer(2600, 260, 0.8, 0xdbe7ff, 0.5);
  const mid = layer(700, 200, 1.3, 0xbfe9ff, 0.6);
  const near = layer(70, 160, 2.2, 0x7dd3fc, 0.55);

  const target = { x: 0, y: 0 };
  const current = { x: 0, y: 0 };
  let scrollZ = 0;
  window.addEventListener('pointermove', (e) => {
    target.x = (e.clientX / window.innerWidth - 0.5) * 2;
    target.y = (e.clientY / window.innerHeight - 0.5) * 2;
  }, { passive: true });
  window.addEventListener('scroll', () => { scrollZ = window.scrollY; }, { passive: true });

  function resize() {
    const w = canvas.clientWidth || window.innerWidth;
    const h = canvas.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  window.addEventListener('resize', resize);

  let visible = true;
  document.addEventListener('visibilitychange', () => { visible = !document.hidden; });

  const clock = new THREE.Clock();
  function frame() {
    requestAnimationFrame(frame);
    if (!visible) return;
    const t = clock.getElapsedTime();
    current.x += (target.x - current.x) * 0.04;
    current.y += (target.y - current.y) * 0.04;
    const drift = reducedMotion ? 0 : t * 0.01;
    far.rotation.y = drift * 0.5 + current.x * 0.03;
    far.rotation.x = current.y * 0.02;
    mid.rotation.y = drift + current.x * 0.06;
    mid.rotation.x = current.y * 0.04;
    near.rotation.y = drift * 1.6 + current.x * 0.1;
    near.rotation.x = current.y * 0.07;
    // Scrolling moves the camera forward through the field
    const depth = reducedMotion ? 0 : Math.min(scrollZ / 6000, 1) * 40;
    camera.position.z += (60 - depth - camera.position.z) * 0.06;
    renderer.render(scene, camera);
  }
  frame();
}
