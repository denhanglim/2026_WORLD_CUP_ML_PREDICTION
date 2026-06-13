# Math Page Redesign Brief — from "AI docs template" to a designed scientific zine

The maths page (`web/math.html` + `web/css/math.css`) currently reads **generic / template-y**: a
stacked column of big rounded beige cards (a TOC card, then one rounded card per section) with
green numbered-circle bullets. That's the Notion/docs-template "AI" look. Rebuild it as a genuinely
distinctive, art-directed **engineering-notebook / scientific zine** — while keeping every equation
correct (KaTeX) and matching the project's light, playful, football sticker-album DNA. NOT dark.

## The "AI slop" tells to eliminate
1. **Stacked rounded-card layout** (TOC-in-a-card, section-in-a-card). → Editorial spread with a
   sticky rail + a real content column; equations as framed exhibits, not cards.
2. **Flat beige emptiness, no texture.** → Graph/ruled-paper grid + grain ("engineering notebook").
3. **Generic green numbered circles + plain TOC grid.** → Big ghost section numerals, a sticky
   progress rail, crafted dividers.
4. **Wall of LaTeX with no visuals.** → Small on-brand SVG diagrams + color-coded variables.

## Required art-direction moves

### Layout — kill the card stack
- **Sticky left rail** (desktop): the section list (01–08) with the **active section highlighted**
  and a **scroll-progress** indicator. Collapses to a top progress bar on mobile.
- **Wide reading column** with strong typographic hierarchy. No more "everything in a rounded box."
- **Giant ghosted section numerals** ("01"…"08") bleeding behind each section header (editorial).
- Section dividers crafted from the project's vocabulary (pitch-line, dotted die-cut, registration
  crosshairs) — not plain rules.

### Texture — engineering notebook
- A subtle **graph-paper / ruled grid** background (faint CSS lines) + the existing paper grain &
  halftone. It should feel like worked solutions in a designed notebook, on warm cream. Light.

### The equations — present them, don't dump them
- Each key equation is a **framed "exhibit" / plate** with a small label (e.g. `EQ. 3.1 — goal
  rate`) and, where useful, a one-line caption. Featured equations get more space + emphasis.
- **Color-code the variables** consistently and subtly (and show a tiny legend): attack `α`
  pitch-green, defense `β` coral, rate `λ` ink, correction `ρ` gold. Use KaTeX `\color{}` or wrap
  rendered terms. Readable on cream (AA contrast); not a rainbow — restrained.
- **Margin annotations**: short notes set in the margin beside key equations, like a marked-up
  exercise book — e.g. "← partial pooling shrinks minnows toward the mean", "← one draw per sim =
  correlated outcomes". An italic/handwriting-flavoured accent is welcome (a tasteful Google Fonts
  handwriting face is allowed here as a 4th, clearly-purposed accent; keep it legible & sparse).

### Break the LaTeX wall with small SVG diagrams (on-brand, hand-drawn-ish)
At least **3** of these, inline where they belong:
- **The pipeline** as a real flow diagram (Data → Elo/features → 3 models → meta → Monte Carlo →
  probabilities) — replace the current code-block arrow line.
- A small **Poisson bar mini-chart** (P(0),P(1),P(2)… goals) next to the M1 section.
- The **bracket funnel** (48 → 32 → 16 → 8 → 4 → 2 → 1) next to the Monte Carlo section.
- Optional: the **Elo expected-score S-curve** next to the Elo section.
Keep them simple, cream/green/coral, slightly imperfect/sketchy to match the playful tone.

### Type & detail
- Push hierarchy: oversized section titles, a strong **lead paragraph / drop-cap** on the intro,
  a **pull-quote** for the "honest limits" section styled as a rubber-stamp / ticket (reuse the
  album's stamp motif). Bricolage Grotesque + Fraunces + Space Mono (+ optional handwriting accent).
- Keep the headline treatment ("The *Maths*") and the ghosted Σ if it still works, but make the
  whole page feel composed, not stacked.

### Motion (GSAP) — calm, this is a reading page
- Scroll-progress on the rail; sections + exhibits **reveal on scroll**; equation plates fade/slide
  in; active TOC item updates on scroll (scrollspy); subtle hover on margin notes / diagrams.
- Full `prefers-reduced-motion` fallback (everything visible, static, legible).

## Content (unchanged — keep all of it, formulas verbatim from the existing math.html)
All 8 sections + "Honest limits". Do not alter the mathematics; only the presentation. The exact
LaTeX already in `web/math.html` (and `web/MATH_BRIEF.md`) is correct — reuse it.

## Constraints
- Static, no build; KaTeX via CDN; GSAP via CDN importmap. Reuse `styles.css` palette/vars.
- Equations must render (no raw `$$`); display math must not overflow on mobile (scroll inside its
  own container); no page horizontal scroll. Semantic HTML, focus states, AA contrast, clean
  console. Keep the `index.html` ↔ `math.html` cross-links working.

## The bar
Iterate: **screenshot → critique → improve**, at least **3 cycles**. Each time ask: *"Does this
look like a designed scientific zine someone would feature — or still a docs template?"* Keep going
until the card-stack look is gone and it's genuinely distinctive. Re-capture
`docs/screenshots/math-desktop.png` and `docs/screenshots/math-mobile.png` when done.
```
