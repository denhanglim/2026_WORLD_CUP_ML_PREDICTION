# Web Dashboard — Design Brief: "THE ALBUM '26"

Awwwards-grade, **light**, playful, football-flavoured single-page dashboard for the 2026 World
Cup title-probability model. Concept = **a World Cup sticker album come to life** (Panini
nostalgia). three.js hero + GSAP motion. No dark mode.

## Visual thesis (one line)

A bright, nostalgic sticker-album come to life — warm pastel paper stock, gold-foil shine on the
contenders, halftone print texture, glossy peelable team stickers, and a spinnable 3D foil ball.
Tactile, collectible-joy, summery.

## Tech (no build step — must run via `python -m http.server` from `web/`)

- Plain static: `web/index.html`, `web/css/`, `web/js/` (ES modules).
- **three.js** + **GSAP** (+ ScrollTrigger) via CDN importmap (jsdelivr/unpkg ESM). No bundler.
- Data: `fetch('./data/predictions.json')` (already generated; schema below). Do NOT hardcode the
  numbers — read them from JSON so a re-run updates the site.
- Keep files focused: e.g. `js/main.js`, `js/ball.js` (three.js), `js/album.js` (leaderboard),
  `js/detail.js` (team panel), `js/confetti.js`.

## Data schema (`web/data/predictions.json`)

```jsonc
{
  "meta": { "title", "method", "as_of", "n_sims", "n_teams", "favourite", "favourite_p", "caveat" },
  "groups": { "A": ["Algeria","Argentina","Austria","Jordan"], ... 12 groups },
  "teams": [
    { "rank":1, "team":"Argentina", "code":"AR", "flag":"🇦🇷", "group":"A",
      "p_win":0.2034, "p_final":0.3090, "p_sf":0.4791, "p_qf":0.6153,
      "p_r16":0.7789, "p_qualify":0.9736 }, ... 48 teams sorted by p_win desc
  ]
}
```
Render probabilities as percentages (`p_win 0.2034 → 20.3%`).

## Palette (CSS variables — commit to these)

```css
--paper:#FAF3E0;  --paper-2:#F2E8CF;  --ink:#2B2A26;  --ink-soft:#6B6557;
--pitch:#3DA35D;  --pitch-deep:#2E7D46;            /* primary accent */
--coral:#FF7A59;                                    /* playful pop / CTA */
--sky:#BFE3F0;  --mint:#CDECD8;  --peach:#FBD9C0;   /* pastel sticker tints */
--gold:#E8C66B;                                     /* foil base */
--foil: linear-gradient(120deg,#F7E98A,#E8C66B,#C9972B,#F7E98A,#E8C66B); /* animate bg-position for shimmer */
--line:#FFFFFF;                                     /* pitch lines */
```
Dominant = warm cream paper; pitch-green primary; gold-foil reserved for the top contenders;
coral for one CTA / interactive pop. Add a faint halftone-dot texture overlay (CSS radial-gradient
or tiny SVG) at low opacity for the "print" feel.

## Type (Google Fonts)

- **Display:** `Bricolage Grotesque` (700/800) — headlines, team names, big %.
- **Body:** `Fraunces` (soft, warm) — descriptive copy, the honesty panel.
- **Stats accent:** `Space Mono` — small labels / round-by-round numerals (the "stat sheet" feel).
- Do NOT use Inter, Roboto, Arial, Space Grotesk, or system defaults.

## Sections (top → bottom)

1. **Hero** (full viewport). Cream paper + faint halftone + subtle pitch-line markings. Album
   masthead: small kicker "OFFICIAL STICKER ALBUM", giant headline **"WHO LIFTS IT?"** + "FIFA
   WORLD CUP 2026", one-line subhead naming the method. Feature the favourite as a **rare
   gold-foil sticker** (Argentina, 20.3%) with shimmer. Centre/right: the **3D foil ball** (see
   below). Bouncing-ball scroll cue.
2. **The Album** — section title e.g. "48 STICKERS · ONE TROPHY". Responsive grid of **48 sticker
   cards** ranked by `p_win`: flag, team name, group chip (e.g. "GRP A"), big `p_win` %, a thin
   win-probability bar. **Top ~6 get the gold-foil treatment** (shimmer border) as "rare"
   stickers; the rest use rotating pastel tints. This is the core dashboard.
3. **Team detail** — clicking a sticker opens a panel/modal that **flips in** showing that team's
   full ladder: Qualify → R16 → QF → SF → Final → **Win** as animated horizontal bars with the %
   on each ("Road to the final"). Close button + ESC + backdrop click.
4. **How it works (honesty panel)** — playful "ALBUM RULES" card. Explain in plain language:
   self-computed Elo + a Bayesian goal model (M1) → 20,000 Monte Carlo tournament simulations.
   **Prominently** surface `meta.caveat` (~1-in-5 favourite = ~4-in-5 against; variance dominates).
   This honesty is a feature, not fine print.
5. **Footer** — `meta.method`, `meta.as_of`, and a clear line: "A personal model. Not affiliated
   with FIFA or Panini." Credit.

## 3D ball (three.js)

- A football: icosphere or a Telstar-style sphere (white base with subtle pentagon/hex hint is
  enough — don't over-engineer geometry). Glossy, slightly foil/specular material catching a soft
  light so it glints. **Drag to spin** (OrbitControls, rotate only — disable zoom/pan), gentle
  auto-rotate, subtle vertical float (sin bob), light cursor parallax. Transparent canvas over the
  paper background. Lightweight; dispose properly; respect `prefers-reduced-motion` (freeze spin).

## Motion (GSAP — ship these)

1. **Sticker press-down entrance** (ScrollTrigger, staggered): cards start slightly scaled-down +
   rotated + transparent, then "press onto the page" (scale→1, rotation→0, opacity→1) in a stagger
   as the Album scrolls in. Feels like sticking stickers in.
2. **Foil hover**: on sticker hover — 3D tilt toward cursor + a foil shine sweep (animate the
   `--foil` gradient background-position) + a slight corner "peel" (transform/clip).
3. **Win-count-up + bars**: percentages count up and bars grow when revealed (hero favourite + team
   detail ladder).
4. **Winner confetti**: a tasteful confetti / shine burst on the favourite sticker (hero load or
   first reveal). Subtle, not a constant loop.
Respect `prefers-reduced-motion`: provide static fallbacks (no parallax/auto-spin/confetti).

## Quality floor (non-negotiable)

- Semantic HTML (`header/main/section/nav/button`), keyboard focus states, WCAG-AA contrast (warm
  ink on cream passes — verify the gold/coral text cases).
- Responsive: looks great desktop AND mobile (grid reflows; ball scales/hides gracefully on small
  screens; no horizontal scroll).
- No AI-commentary copy. Write like a playful football product, not a prompt.
- 60fps-smooth motion; lazy-init three.js; clean console (no errors).

## What "done" looks like

Open `web/index.html` (served), see the sticker-album hero with the spinning foil ball, scroll to
the 48-sticker album that presses in, click a sticker to flip open its road-to-the-final, read the
honest "album rules". Verified via desktop + mobile screenshots, no console errors.
