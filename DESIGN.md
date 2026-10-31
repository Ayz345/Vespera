# Vespera design notes

Short reference for anyone editing the templates. All visual decisions live in `static/css/site.css`; pages extend `templates/base.html`.

## Idea

Vespera is a scientific tool, so the site reads like a clean instrument panel, not a space poster. The one expressive element is the hero figure on the home page: a planet transiting its star with the matching light-curve dip, which is literally how Kepler and TESS find planets.

## Palette

| Token | Value | Use |
|---|---|---|
| `--paper` | `#F4F7FB` | Page background |
| `--surface` | `#FFFFFF` | Panels, footer |
| `--ink` | `#0A1E3D` | Text, selected controls |
| `--ink-2` / `--ink-3` | `#3F5478` / `#6B7D9B` | Secondary text, hints |
| `--brand` | `#0B5FC4` | Links, primary button, candidate marks (from the logo) |
| `--sky` | `#19B8E0` | Reserved for "habitable" data marks and the best model |
| `--amber` | `#D98A1A` | Confirmed planets in the 3D map |
| `--danger` | `#B42318` | Errors |

No gradients, no purple, no dark theme. Cyan and amber are data colours, not decoration.

## Type

One family: Bricolage Grotesque (variable, loaded from Google Fonts). Headings use the display optical size and a slightly narrow width; body is 17px at 1.55 line height. Numbers in tables use tabular figures.

## Layout

Content is left aligned inside a 1120px container with 16px side gutters on phones. Sections are separated by a hairline and generous vertical space, not by cards. Cards appear only around forms and download links.

## Motion

One authored loop: the hero transit (planet crosses, the dot traces the curve, the depth label appears). Everything else is a short hover or state transition. `prefers-reduced-motion` freezes the hero mid-transit.

## States

Forms show the chosen file name, accept drag and drop, lock the button and show a spinner while training, and render a red alert with a specific reason on failure (wrong mission, unreadable CSV, too few rows).
