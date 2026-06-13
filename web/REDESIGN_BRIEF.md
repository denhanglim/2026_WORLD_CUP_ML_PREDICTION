# Redesign Brief — Kill the "AI look", hit Awwwards standard

The current dashboard works but reads **generic / template-y**. This is an art-direction rebuild of
`web/index.html` (+ CSS/JS), keeping the data wiring, sections, and concept (light, playful, football
**sticker album**), but executing at a genuinely distinctive, award-worthy level. Light, NOT dark.
Carry the same elevated styling consistently to `web/math.html`.

## The exact "AI slop" tells to eliminate (diagnosed from the current build)

1. **The 3D ball is a dull grey default sphere.** Biggest tell. It looks like an untextured
   three.js demo. → Must become a real, beautiful object (see below).
2. **The album is a uniform rounded-card grid** — the textbook "generic SaaS card grid". → Must
   become a real *sticker-album page*, not a dashboard grid.
3. **Timid centered two-column hero** with lots of dead beige space and stock coral/white pill
   buttons. → Bold, asymmetric, art-directed composition with texture and depth.
4. **Washed-out, flat, textureless surfaces.** → Paper grain, halftone, print marks, real shadows.
5. **No motion personality** in stills (no tactility, no shine, no peel). → Tactile micro-interactions.

## Non-negotiable art-direction moves

### The ball (three.js) — make it a centerpiece
- A **glossy real football**: classic white body + black pentagons with proper material (clearcoat
  / subtle metalness / environment reflections so it actually shines), OR a **holographic-foil /
  chrome** ball with iridescent shimmer — pick the more striking. Add soft **bloom/glow** and a
  **soft contact shadow on the paper** so it sits in the scene.
- Bigger, off-center, allowed to **overlap the headline**. Cursor-reactive (parallax + spin),
  gentle float. Dispose properly; freeze under `prefers-reduced-motion`.

### Hero composition — break the grid
- Asymmetric, editorial. Oversized headline with **dramatic type scale contrast**; let a giant
  ghosted **"26" / "2026"** bleed off an edge as a background graphic. Add **print registration
  crosshairs** in corners, **dotted die-cut** line motifs, a faint **pitch-line** baseline grid.
- Feature the **#1 sticker (Argentina)** large, **tilted**, with an **animated foil shimmer sweep**,
  a "★ RARE" foil stamp, a **peeling corner**, and a real drop shadow — like a sticker pressed onto
  the page, not a floating card.
- Replace stock pills with **crafted** controls: a chunky magnetic "Open the album" button (hover
  fill / slight magnet), an underline-draw text link. Consider a **custom cursor** (small ball or
  crosshair).
- Layer **paper grain + halftone dots** over everything (SVG noise / CSS), low opacity.

### The album — a real sticker-album spread (this is the worst offender, fix it hardest)
- Stop the rigid identical grid. Make it feel like an **open Panini album page**: numbered slots
  (**#1…#48**), **dotted die-cut sticker borders**, slight **random rotations (±2–4°)** and organic
  scatter so it breathes, a faint **binding/spine** down the middle of the spread, page texture, a
  page-title treatment with a page number.
- **Top contenders = big glossy FOIL stickers** with an animated **holographic shimmer sweep**
  (not flat gold); lower teams = matte paper stickers. Clear visual tiering beyond just color.
- **Oversized condensed numerals** for the win %, the "%" set smaller — sports-broadcast energy.
- **Hover**: the sticker **peels up at a corner + tilts toward the cursor + a shine sweep crosses
  it + the shadow lifts**. Genuinely tactile.
- **Entrance**: stickers **slap onto the page** (scale + rotation settle + a quick shine), staggered
  on scroll.

### Global craft
- **Texture is mandatory**: paper grain, halftone, registration marks — the "print" feel.
- **Richer palette**: deepen the cream so it's warm (not washed-out); use pitch-green + coral
  **boldly**; gold foil as an **animated** gradient; one extra playful accent (electric sky-blue)
  used sparingly. Still light overall.
- **Typography**: push it — a characterful display at large scale with strong weight/size contrast,
  italic accents, tabular oversized numerals. Avoid anything that reads as a default. (Keep within
  Bricolage Grotesque / Fraunces / Space Mono unless a more characterful display clearly helps —
  if you add one, it must be distinctive, not Inter/Roboto/Arial/Space Grotesk.)
- **Motion everywhere, tasteful**: cursor-reactive ball, magnetic button, scroll parallax layers,
  count-ups, a **foil/confetti burst** on the winner, shine sweeps. 60fps; full `prefers-reduced-
  motion` fallback (static, legible, no auto-spin/parallax/confetti).
- Keep the **"How it works" honesty panel** + footer, but craft them (e.g. an "ALBUM RULES" rubber-
  stamp / ticket motif). Keep the **"The Maths" link** + hero CTA → `math.html` working.

## Constraints (unchanged)
- Static, no build step; runs via `python3 -m http.server` from `web/`. three.js + GSAP via CDN
  importmap; KaTeX on the math page.
- Read data from `web/data/predictions.json` (don't hardcode). 48 teams, ranked, groups, flags.
- Semantic HTML, keyboard focus states, WCAG-AA contrast, responsive (no horizontal scroll on
  mobile), clean console.

## The bar (how to know it's done)
Iterate: **screenshot → critique → improve**, at least **3 cycles** for the hero and the album.
Each cycle ask honestly: *"Would this get featured on Awwwards? Is it distinctive, textured,
art-directed, tactile, motion-rich — or does it still look like a generic AI template?"* Keep going
until the answer is clearly the former. The grey-sphere and uniform-card-grid looks must be gone.
