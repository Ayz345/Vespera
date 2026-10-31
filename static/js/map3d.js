// The 3D map of results: four point groups around the Sun, orbit controls, slow auto-rotate.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let current = null;

export function renderMap(container, { pointscand = [], pointsconf = [] }, { reducedMotion = false } = {}) {
  if (current) { current.dispose(); current = null; }
  container.innerHTML = '';

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 100000);

  // Scale: fit the bulk of the points (90th percentile distance) into ~40 units
  const all = [...pointscand, ...pointsconf];
  const dists = all.map((p) => Math.hypot(p.x, p.y, p.z)).sort((a, b) => a - b);
  const p90 = dists.length ? dists[Math.floor(dists.length * 0.9)] || 1 : 1;
  const k = 40 / p90;

  const sprite = makeSprite();
  function group(points, color, size) {
    if (!points.length) return;
    const pos = new Float32Array(points.length * 3);
    points.forEach((p, i) => { pos[i * 3] = p.x * k; pos[i * 3 + 1] = p.z * k; pos[i * 3 + 2] = -p.y * k; });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({ size, color, map: sprite, transparent: true, opacity: 0.95, depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true });
    scene.add(new THREE.Points(geo, mat));
  }
  const split = (pts) => ({ hab: pts.filter((p) => p.Habitability === 'Habitable'), not: pts.filter((p) => p.Habitability !== 'Habitable') });
  const c = split(pointscand), f = split(pointsconf);
  group(f.not, 0x8ea0bc, 1.1);
  group(f.hab, 0xf59e0b, 2.2);
  group(c.not, 0x60a5fa, 1.4);
  group(c.hab, 0x22d3ee, 2.6);

  // The Sun at the origin, and faint reference rings at 100, 500, 1000 pc
  const sun = new THREE.Mesh(new THREE.SphereGeometry(0.6, 16, 16), new THREE.MeshBasicMaterial({ color: 0xffc857 }));
  scene.add(sun);
  [100, 500, 1000, 2000].forEach((pc) => {
    const r = pc * k;
    if (r > 400) return;
    const ring = new THREE.Mesh(new THREE.RingGeometry(r - 0.05, r + 0.05, 128), new THREE.MeshBasicMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.25, side: THREE.DoubleSide }));
    ring.rotation.x = Math.PI / 2;
    scene.add(ring);
  });

  camera.position.set(55, 32, 55);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.06;
  controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.5;
  controls.minDistance = 5; controls.maxDistance = 400;

  let alive = true;
  function frame() { if (!alive) return; requestAnimationFrame(frame); controls.update(); renderer.render(scene, camera); }
  frame();
  function onResize() {
    const w = container.clientWidth, h = container.clientHeight;
    renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', onResize);
  current = { dispose() { alive = false; window.removeEventListener('resize', onResize); renderer.dispose(); controls.dispose(); } };
}

function makeSprite() {
  const cv = document.createElement('canvas'); cv.width = cv.height = 64;
  const g = cv.getContext('2d');
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, 'rgba(255,255,255,1)'); grad.addColorStop(0.35, 'rgba(255,255,255,0.7)'); grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad; g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(cv);
}
