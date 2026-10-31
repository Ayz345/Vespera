// The world: one full-screen Three.js scene that the scroll flies the camera through.
// Sections hand it a name and a 0–1 progress; it moves the camera and plays the scene.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

const AU = 3.2;               // scene units per astronomical unit in the zone scene
const SUN_R = 2.0;

const NOISE = `
float hash(vec3 p){ p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3)); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float noise(vec3 x){ vec3 i = floor(x); vec3 f = fract(x); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x), mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x), mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z); }
float fbm(vec3 p){ float v = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { v += a * noise(p); p = p * 2.03 + 11.3; a *= 0.5; } return v; }
`;

function sunMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uDark: { value: 0 } },
    vertexShader: `
      varying vec3 vN; varying vec3 vP; varying vec3 vView;
      void main(){ vN = normalize(normalMatrix * normal); vP = normalize(position);
        vec4 mv = modelViewMatrix * vec4(position, 1.0); vView = -mv.xyz; gl_Position = projectionMatrix * mv; }`,
    fragmentShader: NOISE + `
      uniform float uTime; uniform float uDark; varying vec3 vN; varying vec3 vP; varying vec3 vView;
      void main(){
        float n1 = fbm(vP * 2.6 + vec3(0.0, 0.0, uTime * 0.08));
        float n2 = fbm(vP * 7.0 - vec3(uTime * 0.05, 0.0, 0.0));
        float t = n1 * 0.65 + n2 * 0.45;
        vec3 c = mix(vec3(0.22, 0.005, 0.015), vec3(0.95, 0.12, 0.06), smoothstep(0.30, 0.68, t));
        c = mix(c, vec3(1.0, 0.55, 0.28), smoothstep(0.74, 0.95, t));
        float fres = pow(1.0 - max(dot(normalize(vView), vN), 0.0), 2.2);
        c += vec3(1.0, 0.18, 0.12) * fres * 1.4;
        gl_FragColor = vec4(c * 1.35 * (1.0 - uDark * 0.97), 1.0);
      }`,
  });
}

function rimMaterial(color) {
  return new THREE.ShaderMaterial({
    uniforms: { uColor: { value: new THREE.Color(color) } },
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.BackSide,
    vertexShader: `varying vec3 vN; varying vec3 vView; void main(){ vN = normalize(normalMatrix * normal); vec4 mv = modelViewMatrix * vec4(position, 1.0); vView = -mv.xyz; gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `uniform vec3 uColor; varying vec3 vN; varying vec3 vView; void main(){ float f = pow(1.0 - abs(dot(normalize(vView), vN)), 3.0); gl_FragColor = vec4(uColor * f * 1.6, f); }`,
  });
}

function spriteTexture(inner, outer) {
  const cv = document.createElement('canvas'); cv.width = cv.height = 256;
  const g = cv.getContext('2d');
  const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, inner); grad.addColorStop(0.35, outer); grad.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = grad; g.fillRect(0, 0, 256, 256);
  return new THREE.CanvasTexture(cv);
}

function coronaTexture(rgb) {
  // Radial streaks like a real corona photograph, plus a soft core
  const S = 1024, cv = document.createElement('canvas'); cv.width = cv.height = S;
  const g = cv.getContext('2d'); const c = S / 2;
  const soft = g.createRadialGradient(c, c, 0, c, c, c);
  soft.addColorStop(0, `rgba(${rgb},0.5)`); soft.addColorStop(0.16, `rgba(${rgb},0.16)`); soft.addColorStop(0.4, `rgba(${rgb},0.03)`); soft.addColorStop(1, `rgba(${rgb},0)`);
  g.fillStyle = soft; g.fillRect(0, 0, S, S);
  g.globalCompositeOperation = 'lighter';
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < 150; i++) {
    const a = rnd() * Math.PI * 2, len = c * (0.3 + Math.pow(rnd(), 2.6) * 0.65), w = 0.7 + rnd() * 1.4, op = 0.05 + rnd() * 0.2;
    const grad = g.createLinearGradient(c, c, c + Math.cos(a) * len, c + Math.sin(a) * len);
    grad.addColorStop(0, `rgba(${rgb},${op})`); grad.addColorStop(0.5, `rgba(${rgb},${op * 0.5})`); grad.addColorStop(1, `rgba(${rgb},0)`);
    g.strokeStyle = grad; g.lineWidth = w; g.lineCap = 'round';
    g.beginPath(); g.moveTo(c + Math.cos(a) * c * 0.17, c + Math.sin(a) * c * 0.17); g.lineTo(c + Math.cos(a) * len, c + Math.sin(a) * len); g.stroke();
  }
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return t;
}
function ringTexture(rgb, inner, width) {
  const S = 512, cv = document.createElement('canvas'); cv.width = cv.height = S;
  const g = cv.getContext('2d'); const c = S / 2;
  const grad = g.createRadialGradient(c, c, 0, c, c, c);
  grad.addColorStop(0, `rgba(${rgb},0)`); grad.addColorStop(inner - 0.01, `rgba(${rgb},0)`); grad.addColorStop(inner, `rgba(${rgb},1)`);
  grad.addColorStop(inner + width, `rgba(${rgb},0.35)`); grad.addColorStop(Math.min(1, inner + width * 2.5), `rgba(${rgb},0)`); grad.addColorStop(1, `rgba(${rgb},0)`);
  g.fillStyle = grad; g.fillRect(0, 0, S, S);
  return new THREE.CanvasTexture(cv);
}
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (t) => t * t * (3 - 2 * t);
const clamp01 = (t) => Math.min(1, Math.max(0, t));

export function createWorld(canvas, { reducedMotion = false, onSweep = () => {} } = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.setClearColor(0x000000, 1);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 500);
  const small = window.matchMedia('(max-width: 900px)').matches;

  /* ----- Sun ----- */
  const sunMat = sunMaterial();
  const sun = new THREE.Mesh(new THREE.SphereGeometry(SUN_R, 96, 96), sunMat);
  scene.add(sun);
  const corona = new THREE.Sprite(new THREE.SpriteMaterial({ map: spriteTexture('rgba(255,120,80,0.9)', 'rgba(229,16,43,0.35)'), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  corona.scale.setScalar(SUN_R * 4.4);
  scene.add(corona);
  const light = new THREE.PointLight(0xff6a4a, 60, 0, 1.6); scene.add(light);
  scene.add(new THREE.AmbientLight(0x1a0a0a, 0.6));

  /* ----- Transit planet (hero) ----- */
  const planet = new THREE.Group();
  planet.add(new THREE.Mesh(new THREE.SphereGeometry(0.34, 48, 48), new THREE.MeshStandardMaterial({ color: 0x060606, roughness: 0.95, metalness: 0.05 })));
  planet.add(new THREE.Mesh(new THREE.SphereGeometry(0.352, 48, 48), rimMaterial(0xff3b2e)));
  scene.add(planet);

  /* ----- Eclipse props ----- */
  const LOOK = new URLSearchParams(location.search).get('eclipse') || 'red';     // red | white | bare
  const coronaRGB = LOOK === 'red' ? '255,90,80' : '255,236,228';
  const eCorona = new THREE.Sprite(new THREE.SpriteMaterial({ map: coronaTexture(coronaRGB), blending: THREE.AdditiveBlending, depthWrite: false, depthTest: true, transparent: true, opacity: 0 }));
  eCorona.scale.setScalar(SUN_R * 7); eCorona.position.set(0, 0, -0.3); scene.add(eCorona);
  const chromo = new THREE.Sprite(new THREE.SpriteMaterial({ map: ringTexture('255,120,90', 0.47, 0.025), blending: THREE.AdditiveBlending, depthWrite: false, depthTest: true, transparent: true, opacity: 0 }));
  chromo.scale.setScalar(SUN_R * 2.3); chromo.position.set(0, 0, -0.2); scene.add(chromo);
  const diamond = new THREE.Sprite(new THREE.SpriteMaterial({ map: spriteTexture('rgba(255,250,240,1)', 'rgba(255,200,170,0.5)'), blending: THREE.AdditiveBlending, depthWrite: false, depthTest: true, transparent: true, opacity: 0 }));
  diamond.scale.setScalar(SUN_R * 0.9); scene.add(diamond);
  if (LOOK !== 'bare') { planet.children[1].material.uniforms.uColor.value.set(0x2a0608); }

  /* ----- Catalog: a disk of candidate rows (method) ----- */
  const N = small ? 2200 : 4800;
  const pos = new Float32Array(N * 3), col = new Float32Array(N * 3), meta = new Float32Array(N * 2); // r, label
  for (let i = 0; i < N; i++) {
    const r = 6 + Math.pow(Math.random(), 0.7) * 11, a = Math.random() * Math.PI * 2;
    pos[i * 3] = Math.cos(a) * r; pos[i * 3 + 1] = (Math.random() - 0.5) * 0.9 * (r / 10); pos[i * 3 + 2] = Math.sin(a) * r;
    meta[i * 2] = r; meta[i * 2 + 1] = Math.random() < 0.62 ? 1 : 0;
    col[i * 3] = col[i * 3 + 1] = col[i * 3 + 2] = 0.28;
  }
  const catGeo = new THREE.BufferGeometry();
  catGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  catGeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const dot = spriteTexture('rgba(255,255,255,1)', 'rgba(255,255,255,0.6)');
  const catMat = new THREE.PointsMaterial({ size: 0.16, map: dot, vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true });
  const catalog = new THREE.Points(catGeo, catMat);
  scene.add(catalog);
  const LIME = new THREE.Color(0xc8ff3d), RED = new THREE.Color(0xff3b4e), GREY = new THREE.Color(0x474747);
  const sweep = new THREE.Mesh(new THREE.RingGeometry(5.9, 6.1, 160), new THREE.MeshBasicMaterial({ color: 0xc8ff3d, transparent: true, opacity: 0, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false }));
  sweep.rotation.x = Math.PI / 2; scene.add(sweep);
  let sweepState = { r: 0, found: 0, lastR: -1 };

  /* ----- Sky sphere and mission fields (missions) ----- */
  const sky = new THREE.Group(); scene.add(sky);
  const skySphere = new THREE.Mesh(new THREE.SphereGeometry(30, 40, 28), new THREE.MeshBasicMaterial({ color: 0xff3b4e, wireframe: true, transparent: true, opacity: 0 }));
  sky.add(skySphere);
  const fieldMat = () => new THREE.MeshBasicMaterial({ color: 0xff2a3c, transparent: true, opacity: 0, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false });
  function patch(w, h, lat, lon, roll = 0) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), fieldMat());
    const r = 29.6;
    m.position.set(r * Math.cos(lat) * Math.cos(lon), r * Math.sin(lat), r * Math.cos(lat) * Math.sin(lon));
    m.lookAt(0, 0, 0); m.rotateZ(roll);
    return m;
  }
  const fields = { kepler: new THREE.Group(), k2: new THREE.Group(), tess: new THREE.Group() };
  fields.kepler.add(patch(7, 7, 0.75, 0.9));
  for (let i = 0; i < 6; i++) fields.k2.add(patch(3.4, 3.4, Math.sin(i * 1.05) * 0.42, i * 1.05 + 0.4));
  for (let i = 0; i < 13; i++) fields.tess.add(patch(2.4, 24, 0.6, i * (Math.PI * 2 / 13), 0));
  for (let i = 0; i < 13; i++) fields.tess.add(patch(2.4, 24, -0.6, i * (Math.PI * 2 / 13) + 0.24, 0));
  Object.values(fields).forEach((g) => sky.add(g));

  /* ----- Orbit rings and the habitable annulus (zone) ----- */
  const zone = new THREE.Group(); scene.add(zone);
  const ringMat = () => new THREE.MeshBasicMaterial({ color: 0xf2f0ec, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false });
  const rings = [0.5, 1, 1.5, 2, 2.5].map((au) => { const m = new THREE.Mesh(new THREE.RingGeometry(au * AU - 0.012, au * AU + 0.012, 180), ringMat()); m.rotation.x = Math.PI / 2; zone.add(m); return m; });
  const annulus = new THREE.Mesh(new THREE.RingGeometry(0.68 * AU, 1.23 * AU, 180), new THREE.MeshBasicMaterial({ color: 0xc8ff3d, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
  annulus.rotation.x = Math.PI / 2; zone.add(annulus);
  const orbitPath = new THREE.Mesh(new THREE.RingGeometry(1, 1.02, 180), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
  orbitPath.rotation.x = Math.PI / 2; zone.add(orbitPath);
  const earth = new THREE.Group();
  earth.add(new THREE.Mesh(new THREE.SphereGeometry(0.2, 32, 32), new THREE.MeshStandardMaterial({ color: 0xbfc4cc, roughness: 0.8 })));
  earth.add(new THREE.Mesh(new THREE.SphereGeometry(0.22, 32, 32), rimMaterial(0xc8ff3d)));
  earth.visible = false; zone.add(earth);

  /* ----- Post: bloom ----- */
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 1.0, 0.6, 0.55);
  composer.addPass(bloom);

  /* ----- Camera targets ----- */
  const cam = { pos: new THREE.Vector3(0.6, 0.5, 8.6), look: new THREE.Vector3(0, 0.2, 0), fov: 42 };
  const target = { pos: cam.pos.clone(), look: cam.look.clone(), fov: 42 };
  let section = 'hero', progress = 0, dim = 0, dark = 0, orbitT = 0, orbitTime = 0;

  // Camera keyframes. Each section's last key equals the next section's first key,
  // so the camera never moves on its own between pinned scenes.
  const Z = small ? 1.5 : 1;
  const KF = {
    hero: [
      { t: 0, pos: [0.6, 0.5, 8.6 * Z], look: [0, 0.2, 0], fov: 42, dim: 0 },
      { t: 0.8, pos: [-1.4, 0.9, 7.6 * Z], look: [0, 0.2, 0], fov: 42, dim: 0 },
      { t: 1, pos: [-2, 7, 15], look: [0, 0, 0], fov: 46, dim: 0 },
    ],
    method: [
      { t: 0, pos: [-2, 7, 15], look: [0, 0, 0], fov: 46, dim: 0 },
      { t: 0.85, pos: [2, 15, 25], look: [0, 0, 0], fov: 48, dim: 0 },
      { t: 1, pos: [17.9, 14, 42.4], look: [0, 0, 0], fov: 50, dim: 0 },
    ],
    zone: [
      { t: 0, pos: [-45.9, 6, -3.3], look: [0, 0, 0], fov: 50, dim: 0 },
      { t: 0.1, pos: [-1, 11, 16], look: [0, 0, 0], fov: 44, dim: 0 },
      { t: 1, pos: [5, 7, 14], look: [0, 0, 0], fov: 44, dim: 0 },
    ],
    console: [
      { t: 0, pos: [5, 7, 14], look: [0, 0, 0], fov: 44, dim: 0 },
      { t: 0.35, pos: [0, 5.5, 9.5], look: [0, 0, 0], fov: 42, dim: 0.4 },
      { t: 1, pos: [0, 5.5, 9.5], look: [0, 0, 0], fov: 42, dim: 0.4 },
    ],
    team: [
      { t: 0, pos: [0, 5.5, 9.5], look: [0, 0, 0], fov: 44, dim: 0.65 },
      { t: 1, pos: [3, 3, 48], look: [0, 0, 0], fov: 44, dim: 0.7 },
    ],
  };
  function keyAt(keys, p) {
    let i = 0;
    while (i < keys.length - 2 && p > keys[i + 1].t) i++;
    const a = keys[i], b = keys[i + 1];
    const u = smooth(clamp01((p - a.t) / (b.t - a.t || 1)));
    target.pos.set(lerp(a.pos[0], b.pos[0], u), lerp(a.pos[1], b.pos[1], u), lerp(a.pos[2], b.pos[2], u));
    target.look.set(lerp(a.look[0], b.look[0], u), lerp(a.look[1], b.look[1], u), lerp(a.look[2], b.look[2], u));
    target.fov = lerp(a.fov, b.fov, u); dim = lerp(a.dim, b.dim, u);
  }
  const rp = { pos: new THREE.Vector3(), look: new THREE.Vector3() };
  function resultsPose(pos, look) {
    const a = 0.7, R = results.radius * 2.4 * results.zoom;
    pos.set(results.center.x + Math.sin(a) * R, results.center.y + R * 0.3, results.center.z + Math.cos(a) * R);
    fwd.copy(results.center).sub(pos).normalize(); right.crossVectors(fwd, UP).normalize();
    look.copy(results.center).addScaledVector(right, -results.radius * 0.95 * results.zoom);
  }
  function set(name, p) {
    section = name; progress = clamp01(p);
    if (name === 'results') {
      // Arrive from the console pose over the first third, then hold. Drag turns the cloud, wheel zooms.
      resultsPose(rp.pos, rp.look);
      const u = smooth(clamp01(results.arrive));                      // arrival plays in time, not scroll
      target.pos.set(lerp(0, rp.pos.x, u), lerp(5.5, rp.pos.y, u), lerp(9.5, rp.pos.z, u));
      target.look.set(lerp(0, rp.look.x, u), lerp(0, rp.look.y, u), lerp(0, rp.look.z, u));
      target.fov = 44; dim = 0;
      return;
    }
    if (name === 'team') {
      // Leave from wherever the previous scene left the camera
      if (resultPts) resultsPose(rp.pos, rp.look); else { rp.pos.set(0, 5.5, 9.5); rp.look.set(0, 0, 0); }
      const u = smooth(progress);
      target.pos.set(lerp(rp.pos.x, 3, u), lerp(rp.pos.y, 3, u), lerp(rp.pos.z, 48, u));
      target.look.set(lerp(rp.look.x, 0, u), lerp(rp.look.y, 0, u), lerp(rp.look.z, 0, u));
      target.fov = 44; dim = lerp(0.4, 0.7, u);
      return;
    }
    if (name === 'missions') {
      // Orbit the sky sphere. Starts where method ends, ends where zone starts.
      const t = smooth(progress);
      const a = t * Math.PI * 1.35 + 0.4;
      target.pos.set(Math.sin(a) * 46, lerp(14, 6, t), Math.cos(a) * 46);
      target.look.set(0, 0, 0); target.fov = 50; dim = 0;
      return;
    }
    keyAt(KF[name] || KF.hero, progress);
  }

  const results = { center: new THREE.Vector3(0, 0, -20), radius: 20, wide: true, angleScale: 1, yaw: 0, pitch: 0, zoom: 1, arrive: 0 };
  const resultGroup = new THREE.Group(); resultGroup.visible = false; scene.add(resultGroup);
  const resultMat = new THREE.PointsMaterial({ size: 0.62, map: dot, vertexColors: true, transparent: true, opacity: 1, depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true });
  let resultPts = null;
  function setResults({ pointscand = [], pointsconf = [] }) {
    const D2R = Math.PI / 180;
    const all = [...pointsconf.map((p) => ({ ...p, kind: 'conf' })), ...pointscand.map((p) => ({ ...p, kind: 'cand' }))].filter((p) => p.dist > 0);
    if (!all.length) return results;
    // Mean direction of the field and its angular extent
    let mx = 0, my = 0, mz = 0;
    all.forEach((p) => { const ra = p.ra * D2R, dec = p.dec * D2R; mx += Math.cos(dec) * Math.cos(ra); my += Math.cos(dec) * Math.sin(ra); mz += Math.sin(dec); });
    const nn = Math.hypot(mx, my, mz) || 1; mx /= nn; my /= nn; mz /= nn;
    const dec0 = Math.asin(mz), ra0 = Math.atan2(my, mx);
    let maxOff = 0;
    const offs = all.map((p) => {
      let dra = p.ra * D2R - ra0; while (dra > Math.PI) dra -= 2 * Math.PI; while (dra < -Math.PI) dra += 2 * Math.PI;
      const u = dra * Math.cos(p.dec * D2R), v = p.dec * D2R - dec0;
      maxOff = Math.max(maxOff, Math.hypot(u, v)); return [u, v];
    });
    const wide = maxOff > 0.6;
    const angleScale = wide ? 1 : Math.min(8, 0.55 / Math.max(maxOff, 0.01));
    const radius = (d) => 6 + Math.log10(1 + d / 10) * 14;
    const n = all.length, ps = new Float32Array(n * 3), cs = new Float32Array(n * 3);
    const c = new THREE.Vector3(); let far = 0;
    all.forEach((p, i) => {
      const [u, v] = offs[i];
      const r = radius(p.dist), ang = Math.hypot(u, v) * angleScale, phi = Math.atan2(v, u);
      const x = r * Math.sin(ang) * Math.cos(phi), y = r * Math.sin(ang) * Math.sin(phi), z = -r * Math.cos(ang);
      ps[i * 3] = x; ps[i * 3 + 1] = y; ps[i * 3 + 2] = z; c.x += x; c.y += y; c.z += z; far = Math.max(far, r);
      const hab = p.Habitability === 'Habitable';
      const col = p.kind === 'cand' ? (hab ? LIME : RED) : (hab ? new THREE.Color(0xf2f0ec) : GREY);
      cs[i * 3] = col.r; cs[i * 3 + 1] = col.g; cs[i * 3 + 2] = col.b;
    });
    c.divideScalar(n);
    // Centre the cloud on its own origin so it can be rotated in place
    for (let i = 0; i < n; i++) { ps[i * 3] -= c.x; ps[i * 3 + 1] -= c.y; ps[i * 3 + 2] -= c.z; }
    if (resultPts) { resultGroup.remove(resultPts); resultPts.geometry.dispose(); }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(ps, 3));
    g.setAttribute('color', new THREE.BufferAttribute(cs, 3));
    resultPts = new THREE.Points(g, resultMat); resultGroup.add(resultPts);
    resultGroup.position.copy(c);
    results.center.copy(c); results.radius = Math.max(8, far * 0.75); results.wide = wide; results.angleScale = angleScale;
    results.yaw = 0; results.pitch = 0; results.zoom = 1; resultGroup.rotation.set(0, 0, 0);
    return results;
  }

  function resize() {
    const w = canvas.clientWidth || innerWidth, h = canvas.clientHeight || innerHeight;
    renderer.setSize(w, h, false); composer.setSize(w, h);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  resize(); addEventListener('resize', resize);

  let visible = true;
  document.addEventListener('visibilitychange', () => { visible = !document.hidden; });
  const clock = new THREE.Clock();
  const tmp = new THREE.Vector3(), fwd = new THREE.Vector3(), right = new THREE.Vector3(), UP = new THREE.Vector3(0, 1, 0);

  function frame() {
    requestAnimationFrame(frame);
    if (!visible) return;
    const time = clock.getElapsedTime();
    const showRes = !!resultPts && (section === 'results' || (section === 'team' && progress < 0.4));
    resultGroup.visible = showRes;
    resultMat.opacity = section === 'results' ? 1 : section === 'team' ? Math.max(0, 1 - progress / 0.35) : 0;
    if (resultPts && showRes) { resultGroup.rotation.y = results.yaw + (reducedMotion ? 0 : time * 0.03); resultGroup.rotation.x = results.pitch; }
    sunMat.uniforms.uTime.value = reducedMotion ? 0 : time;
    sun.rotation.y = reducedMotion ? 0 : time * 0.02;

    // Camera easing
    cam.pos.lerp(target.pos, 0.09); cam.look.lerp(target.look, 0.09); cam.fov += (target.fov - cam.fov) * 0.09;
    camera.position.copy(cam.pos); camera.lookAt(cam.look); camera.fov = cam.fov; camera.updateProjectionMatrix();

    // Hero: the planet crosses in front of the star, between the camera and the sun
    // The planet comes in from the left, crosses the star, then flies at the camera until it fills the view.
    // Stage A (p 0.06–0.30): a small transit, left to right across the star, then gone.
    // Gap (0.30–0.36): nothing. Stage B (0.36–0.52): the same world comes back huge, just in
    // front of the lens, and slides in from the left edge until it covers the star exactly.
    const p = section === 'hero' ? progress / 0.84 : 0;
    const sA = clamp01((p - 0.06) / 0.24), sB = clamp01((p - 0.36) / 0.16);
    let covered = 0;
    if (section === 'hero' && p < 0.32) {
      const ea = sA * (2 - sA);
      planet.position.set(lerp(-6.5, 6.5, ea), 0.1, 2.6);
      planet.visible = sA > 0 && sA < 1;
    } else if (section === 'hero' && p >= 0.36) {
      const eb = sB * sB * (3 - 2 * sB);
      fwd.copy(cam.pos).negate().normalize();                            // lens → star centre
      right.crossVectors(fwd, UP).normalize();
      const dp = 0.152 * cam.pos.length();                               // apparent size ≈ 1.1× the star
      const off = lerp(-1.15, 0, eb);                                    // lateral offset, left edge to centre
      planet.position.copy(cam.pos).addScaledVector(fwd, dp).addScaledVector(right, off);
      planet.visible = true;
      covered = clamp01(1 - Math.abs(off) / 0.35);                       // the corona flares as it closes
      if (LOOK !== 'bare') {
        const near = clamp01(1 - Math.abs(off) / 0.22);                  // last fifth of the slide
        eCorona.material.opacity = near * near * 0.8;
        eCorona.material.rotation = time * 0.01;
        eCorona.scale.setScalar(SUN_R * lerp(5, 7.5, near));
        chromo.material.opacity = clamp01(1 - Math.abs(off) / 0.06);
        // Baily's bead at the last sliver, on the side the planet has not yet reached
        const bead = clamp01(1 - Math.abs(off) / 0.09) * clamp01(Math.abs(off) / 0.02);
        diamond.material.opacity = bead;
        diamond.position.copy(right).multiplyScalar(SUN_R * 0.96 * Math.sign(-off || 1));
        diamond.scale.setScalar(SUN_R * lerp(0.4, 1.0, bead));
      }
    } else {
      planet.visible = false;
      eCorona.material.opacity = chromo.material.opacity = diamond.material.opacity = 0;
    }
    bloom.strength = lerp(1.0, LOOK === 'bare' ? 1.3 : 1.15, covered) * (1 - dim * 0.5) * (1 - dark);

    // Method: the sweep classifies the catalog
    const sweepTarget = section === 'method' ? lerp(5.5, 17.5, clamp01((progress - 0.12) / 0.75)) : (section === 'hero' ? 0 : 17.5);
    if (!catalog.userData.real && Math.abs(sweepTarget - sweepState.lastR) > 0.01) {
      sweepState.lastR = sweepTarget;
      const c = catGeo.attributes.color.array; let found = 0;
      for (let i = 0; i < N; i++) {
        const r = meta[i * 2], lab = meta[i * 2 + 1];
        const on = r < sweepTarget;
        const color = !on ? GREY : lab ? LIME : RED;
        if (on && lab) found++;
        c[i * 3] = color.r; c[i * 3 + 1] = color.g; c[i * 3 + 2] = color.b;
      }
      catGeo.attributes.color.needsUpdate = true;
      sweepState.found = found; onSweep(found, N);
    }
    sweep.scale.setScalar(Math.max(0.01, sweepTarget / 6));
    sweep.material.opacity = section === 'method' && progress > 0.1 && progress < 0.9 ? 0.6 : 0;
    catalog.rotation.y = reducedMotion ? 0 : time * 0.02;
    catalog.visible = section !== 'results' && !(section === 'team' && resultPts);
    catMat.opacity = (section === 'hero' ? 0.35 : section === 'console' ? 0.45 : (1 - dim * 0.6)) * (1 - dark);

    // Missions: the sky and its fields
    const inSky = section === 'missions';
    skySphere.material.opacity += ((inSky ? 0.035 : 0) - skySphere.material.opacity) * 0.08;
    const third = inSky ? progress * 3 : -1;
    const fade = (idx) => Math.max(0, 1 - Math.abs(third - (idx + 0.5)) * 1.6);
    [fields.kepler, fields.k2, fields.tess].forEach((g, i) => g.children.forEach((m) => { m.material.opacity += (fade(i) * 0.7 - m.material.opacity) * (inSky ? 0.1 : 0.3); }));
    sky.rotation.y = reducedMotion ? 0 : time * 0.01;

    // Zone: rings, annulus and the pushed planet
    const inZone = section === 'zone';
    const zoneOp = inZone ? 1 : 0;
    rings.forEach((r) => { r.material.opacity += (zoneOp * 0.22 - r.material.opacity) * 0.08; });
    annulus.material.opacity += (zoneOp * 0.16 - annulus.material.opacity) * 0.08;
    orbitPath.material.opacity += (zoneOp * 0.5 - orbitPath.material.opacity) * 0.08;
    earth.visible = annulus.material.opacity > 0.01;
    const au = 0.2 + 2.4 * (inZone ? orbitT : 1);
    orbitPath.scale.setScalar(au * AU);
    earth.position.set(Math.cos(-0.7) * au * AU, 0, Math.sin(-0.7) * au * AU);

    corona.material.opacity = (0.8 - dim * 0.4) * (1 - dark);
    sunMat.uniforms.uDark.value = dark;
    composer.render();
  }
  frame();

  const setDark = (d) => { dark = clamp01(d); };
  const setOrbit = (t) => { orbitT = clamp01(t); };
  const turnResults = (dx, dy) => { results.yaw += dx; results.pitch = Math.max(-1.2, Math.min(1.2, results.pitch + dy)); };
  const arriveResults = () => {
    results.arrive = 0;
    const o = { v: 0 };
    if (window.gsap && !reducedMotion) window.gsap.to(o, { v: 1, duration: 1.4, ease: 'power3.inOut', onUpdate: () => { results.arrive = o.v; if (section === 'results') set('results', progress); } });
    else results.arrive = 1;
  };
  const zoomResults = (f) => { results.zoom = Math.max(0.3, Math.min(2.2, results.zoom * f)); if (section === 'results') set('results', progress); };
  return { set, setDark, setOrbit, setResults, turnResults, zoomResults, arriveResults, resize };
}
