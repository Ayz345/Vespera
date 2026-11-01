# Vespera design notes

One page, one night sky. The site is a scroll-driven story that ends in the tool. No build step: ES modules
served by Flask, GSAP + ScrollTrigger from cdnjs, Three.js from jsDelivr through an import map.

## Files

| Path | Holds |
|---|---|
| `templates/index.html` | The whole page: hero, method, missions, habitable zone, console, team, footer |
| `static/css/app.css` | Tokens and every style; the border-beam, halo button and HUD readouts live here |
| `static/js/app.js` | Entry: cursor glow, starts the modules below |
| `static/js/starfield.js` | Three.js star layers behind the page; parallax to the pointer, camera moves forward on scroll |
| `static/js/transit.js` | Hero: scroll scrubs a planet across the star, light curve drawn from real disc-overlap geometry |
| `static/js/scenes.js` | Pinned scroll scenes (pipeline, horizontal missions track, habitable-zone orbit), reveals, nav state |
| `static/js/console.js` | Classify / Compare tabs, upload, `fetch` to `/api/analyze` and `/api/compare`, results |
| `static/js/map3d.js` | Three.js point cloud of results around the Sun with orbit controls |

## Palette

| Token | Value | Use |
|---|---|---|
| `--bg` | `#04070f` | Page ground |
| `--ink` / `--ink-2` / `--ink-3` | `#e8eef9` / `#a3b3cf` / `#6b7d9b` | Text, secondary text, labels |
| `--cyan` | `#22d3ee` | The one accent: actions, glows, "habitable", the best model |
| `--blue` | `#3b82f6` | Gradient partner for cyan; plain candidates on the map |
| `--amber` | `#f59e0b` | Confirmed planets in the habitable window; the star |
| `--rose` | `#fb7185` | False positives, errors |

## Type

Unbounded (display, 600–800), Instrument Sans (text), JetBrains Mono for HUD readouts, column names and
gauge labels, which are data, not decoration.

## Motion

Scroll-linked: the hero transit (pinned 180%), the pipeline (pinned 260%, packets travel between stages),
the missions track (pinned, horizontal on screens wider than 900px, vertical stack below), the orbit scene
(pinned 200%, planet pushed from 0.2 to 2.6 AU with live period and temperature). Ambient: star drift and
cursor parallax, the border beam on cards, the halo button. Feedback: hover lift on buttons, the scan line
while training, counters and bars when results land. `prefers-reduced-motion` removes the ambient loops and
auto-rotation; scroll scrubbing stays because the visitor drives it.

## Physics shown on the page

- Transit depth = (R_planet / R_star)². Earth over the Sun: 84 ppm.
- Orbital period from Kepler's third law for a solar-mass star: P = a^1.5 years.
- Equilibrium temperature for Earth's albedo: T = 255 K / sqrt(a in AU).
- The habitable window is the one in `classify_habitability()`: period 50–500 d, radius 0.5–2.5 R⊕,
  star 4 800–6 300 K, planet 180–310 K.
