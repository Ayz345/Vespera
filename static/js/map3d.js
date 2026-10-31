// The results map. Each world sits in the direction it really is on the sky; its distance from
// the Sun is drawn on a log scale so the near and the far both get room. A narrow field such as
// Kepler's is opened up angularly (and the legend says so) so the points do not pile into a spike.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let current = null;
const D2R = Math.PI / 180;

export function renderMap(container, { pointscand = [], pointsconf = [] }, { reducedMotion = false } = {}) {
  if (current) { current.dispose(); current = null; }
  container.innerHTML = '';
  const all = [...pointsconf.map((p) => ({ ...p, kind: 'conf' })), ...pointscand.map((p) => ({ ...p, kind: 'cand' }))].filter((p) => p.dist > 0);
  if (!all.length) return;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x070707, 40, 140);
  const camera = new THREE.PerspectiveCamera(42, container.clientWidth / container.clientHeight, 0.1, 1000);

  // Mean direction of the field and its angular extent
  let mx = 0, my = 0, mz = 0;
  all.forEach((p) => { const ra = p.ra * D2R, dec = p.dec * D2R; mx += Math.cos(dec) * Math.cos(ra); my += Math.cos(dec) * Math.sin(ra); mz += Math.sin(dec); });
  const n = Math.hypot(mx, my, mz) || 1; mx /= n; my /= n; mz /= n;
  const dec0 = Math.asin(mz), ra0 = Math.atan2(my, mx);
  let maxOff = 0;
  const offs = all.map((p) => {
    let dra = (p.ra * D2R - ra0); while (dra > Math.PI) dra -= 2 * Math.PI; while (dra < -Math.PI) dra += 2 * Math.PI;
    const u = dra * Math.cos(p.dec * D2R), v = p.dec * D2R - dec0;
    maxOff = Math.max(maxOff, Math.hypot(u, v));
    return [u, v];
  });
  const wide = maxOff > 0.6;                                   // a whole-sky catalog keeps true angles
  const angleScale = wide ? 1 : Math.min(8, 0.55 / Math.max(maxOff, 0.01));
  const radius = (d) => 6 + Math.log10(1 + d / 10) * 14;     // 10 pc → 10, 100 pc → 20.6, 1000 pc → 34

  const groups = { 'conf-no': { color: 0x6e6a64, size: 1.0, pts: [] }, 'conf-hab': { color: 0xf2f0ec, size: 1.8, pts: [] }, 'cand-no': { color: 0xff3b4e, size: 1.4, pts: [] }, 'cand-hab': { color: 0xc8ff3d, size: 2.2, pts: [] } };
  all.forEach((p, i) => {
    const [u, v] = offs[i];
    const r = radius(p.dist);
    const ang = Math.hypot(u, v) * angleScale, phi = Math.atan2(v, u);
    const x = r * Math.sin(ang) * Math.cos(phi), y = r * Math.sin(ang) * Math.sin(phi), z = -r * Math.cos(ang);
    groups[p.kind + (p.Habitability === 'Habitable' ? '-hab' : '-no')].pts.push(x, y, z);
  });
  const sprite = makeSprite();
  Object.values(groups).forEach((g) => {
    if (!g.pts.length) return;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(g.pts), 3));
    const mat = new THREE.PointsMaterial({ size: g.size, color: g.color, map: sprite, transparent: true, opacity: 0.92, depthWrite: false, alphaTest: 0.05, sizeAttenuation: true });
    scene.add(new THREE.Points(geo, mat));
  });

  // The Sun, and the line of sight with distance ticks at 10, 100 and 1 000 pc
  const sun = new THREE.Mesh(new THREE.SphereGeometry(0.3, 24, 24), new THREE.MeshBasicMaterial({ color: 0xff6a4a }));
  scene.add(sun);
  const far = radius(5000);
  const los = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -far)]), new THREE.LineBasicMaterial({ color: 0xf2f0ec, transparent: true, opacity: 0.18 }));
  scene.add(los);
  [10, 100, 1000].forEach((pc) => {
    const r = radius(pc);
    const tick = new THREE.Mesh(new THREE.RingGeometry(0.9, 1.05, 64), new THREE.MeshBasicMaterial({ color: 0xc8ff3d, transparent: true, opacity: 0.6, side: THREE.DoubleSide }));
    tick.position.set(0, 0, -r); scene.add(tick);
  });

  // Camera: look at the cloud from the side, slightly above, centred on its middle
  const mid = radius(all.map((p) => p.dist).sort((a, b) => a - b)[Math.floor(all.length / 2)]);
  camera.position.set(mid * 1.3, mid * 0.55, -mid * 0.2);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, -mid);
  controls.enableDamping = true; controls.dampingFactor = 0.06;
  controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.35;
  controls.minDistance = 4; controls.maxDistance = 160;
  controls.update();

  let alive = true;
  function frame() { if (!alive) return; requestAnimationFrame(frame); controls.update(); renderer.render(scene, camera); }
  frame();
  function onResize() { const w = container.clientWidth, h = container.clientHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
  window.addEventListener('resize', onResize);
  current = { dispose() { alive = false; window.removeEventListener('resize', onResize); renderer.dispose(); controls.dispose(); } };
  return { angleScale, wide };
}

function makeSprite() {
  const cv = document.createElement('canvas'); cv.width = cv.height = 64;
  const g = cv.getContext('2d');
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, 'rgba(255,255,255,1)'); grad.addColorStop(0.5, 'rgba(255,255,255,0.9)'); grad.addColorStop(0.7, 'rgba(255,255,255,0.25)'); grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad; g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(cv);
}
