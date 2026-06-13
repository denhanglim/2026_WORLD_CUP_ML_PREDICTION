# Math Page — Brief: "THE MATHS" (`web/math.html`)

A second page for the dashboard that lays out the **full mathematics** behind the model.
Same project, same light playful sticker-album design language, but tuned for dense math: think
**a tidy exercise-book / chalk-on-cream worked-solution** feel — readable, generous whitespace,
real typeset equations. Light, never dark.

## Tech
- New static page `web/math.html` (+ `web/css/math.css` if needed; may reuse `web/css/styles.css`
  variables). No build step; served by the same `python3 -m http.server` from `web/`.
- **KaTeX** via CDN (CSS + JS + auto-render) for equations. (KaTeX over MathJax: faster, lighter.)
- Reuse the palette + fonts from `web/DESIGN_BRIEF.md` (cream paper, pitch-green, gold-foil, coral;
  Bricolage Grotesque display, Fraunces body, Space Mono for inline symbols/numbers).
- Optional light GSAP scroll reveals for sections — keep it calm, this is a reading page. Respect
  `prefers-reduced-motion`.

## Linking (both directions)
- Add a header nav / link on `web/index.html` → `math.html` (e.g. a "The Maths" link, and wire the
  hero's "How it works" CTA to it).
- `math.html` has a clear "← Back to the album" link to `index.html`.

## Page structure & EXACT content (use these formulas verbatim — they match the code)

Title block: "The Maths" + one-line framing: *"From 150 years of results to one number per team —
every layer, written out."* Then a short table-of-contents (anchor links) to the sections below.

### 1. The pipeline (overview)
One sentence + a simple flow: **Data → Elo & features → match models (Bayesian Poisson · gradient
boosting · market odds) → meta-learner → Monte Carlo tournament simulation → win probabilities.**

### 2. Elo ratings (the baseline strength signal)
Expected score (home), with home advantage $H$ (zero on neutral ground):
$$E_{\text{home}} = \frac{1}{1 + 10^{-\big((R_{\text{home}} + H) - R_{\text{away}}\big)/400}}$$
Update after a match ($W\in\{1,\tfrac12,0\}$ for win/draw/loss, zero-sum):
$$R' = R + K \cdot G \cdot (W - E)$$
Margin-of-victory multiplier $G$ (goal difference $d$):
$$G = \begin{cases} 1 & |d| \le 1 \\ 1.5 & |d| = 2 \\ \dfrac{11 + |d|}{8} & |d| \ge 3 \end{cases}$$
Note: $K$ scales with match importance (World Cup $=60$, continental $=50$, qualifiers $=40$,
friendlies $=20$). Computed from results alone — fully reproducible.

### 3. M1 — dynamic hierarchical Poisson goal model (the flagship)
Each team $i$ has latent **attack** $\alpha_{i,t}$ and **defense** $\beta_{i,t}$ at period $t$.
Expected goals:
$$\log \lambda_{\text{home}} = \mu + h\,\mathbb{1}[\text{not neutral}] + \alpha_{\text{home},t} - \beta_{\text{away},t}$$
$$\log \lambda_{\text{away}} = \mu + \alpha_{\text{away},t} - \beta_{\text{home},t}$$
**Dynamic state-space** — strengths evolve as a Gaussian random walk across yearly periods:
$$\alpha_{i,t} = \alpha_{i,t-1} + \varepsilon_{i,t}, \qquad \varepsilon_{i,t} \sim \mathcal{N}(0, \sigma_{\text{evo}}^2)$$
**Hierarchical priors** (partial pooling shrinks small-sample teams toward the mean):
$$\alpha_{i,0},\,\beta_{i,0} \sim \mathcal{N}(0, \sigma^2), \quad \sigma_{\text{att}},\sigma_{\text{def}} \sim \text{HalfNormal}(1), \quad \sigma_{\text{evo}} \sim \text{HalfNormal}(0.15)$$
$$\mu \sim \mathcal{N}(0,1), \quad h \sim \mathcal{N}(0.25, 0.5), \quad \rho \sim \mathcal{N}(0, 0.1)$$
**Likelihood** — independent Poisson for the two goal counts:
$$\text{home goals} \sim \text{Poisson}(\lambda_{\text{home}}), \qquad \text{away goals} \sim \text{Poisson}(\lambda_{\text{away}})$$
**Dixon–Coles low-score correction** $\tau$ (fixes the false independence on 0–0/1–0/0–1/1–1),
applied to the joint pmf:
$$\tau(x,y) = \begin{cases} 1 - \lambda_{\text{home}}\lambda_{\text{away}}\rho & (x,y)=(0,0) \\ 1 + \lambda_{\text{home}}\rho & (x,y)=(0,1) \\ 1 + \lambda_{\text{away}}\rho & (x,y)=(1,0) \\ 1 - \rho & (x,y)=(1,1) \\ 1 & \text{otherwise} \end{cases}$$
Full joint scoreline probability:
$$P(X=x, Y=y) = \tau(x,y)\cdot \frac{\lambda_{\text{home}}^{x} e^{-\lambda_{\text{home}}}}{x!}\cdot \frac{\lambda_{\text{away}}^{y} e^{-\lambda_{\text{away}}}}{y!}$$
The posterior over all $\{\alpha,\beta,\mu,h,\rho,\sigma\}$ is sampled with **NUTS** (Hamiltonian
Monte Carlo). Match outcome probabilities come from summing the (normalised) score grid:
$$P(\text{home win}) = \sum_{x>y} P(x,y),\quad P(\text{draw}) = \sum_{x=y} P(x,y),\quad P(\text{away win}) = \sum_{x<y} P(x,y)$$

### 4. M2 & M3 — gradient boosting and the market
- **M2:** a gradient-boosted classifier on engineered features (Elo diff, form, rest, rolling
  goals…) → calibrated $P(\text{W}/\text{D}/\text{L})$. (One line; no heavy formula needed.)
- **M3 — de-vigging the odds:** strip the bookmaker margin from decimal odds $o_i$:
$$p_i = \frac{1/o_i}{\sum_{j} 1/o_j}$$

### 5. The meta-learner (stacking)
Stack the base models' probability vectors and combine with multinomial logistic regression
(softmax over a linear map of the concatenated base probabilities):
$$\mathbf{x} = \big[\,\mathbf{p}^{(\text{M1})} \,\|\, \mathbf{p}^{(\text{M2})} \,\|\, \mathbf{p}^{(\text{odds})}\,\big], \qquad P(\text{class}=k) = \frac{e^{\mathbf{w}_k^\top \mathbf{x} + b_k}}{\sum_{c} e^{\mathbf{w}_c^\top \mathbf{x} + b_c}}$$
Trained walk-forward on out-of-sample base predictions (no leakage).

### 6. How we score it (proper scoring rules)
Outcomes are ordinal ($\text{H} \prec \text{D} \prec \text{A}$). **Ranked Probability Score** (primary):
$$\text{RPS} = \frac{1}{r-1}\sum_{i=1}^{r-1}\left(\sum_{j=1}^{i} p_j - \sum_{j=1}^{i} o_j\right)^2, \quad r = 3$$
Also **Brier** $\sum_k (p_k - o_k)^2$ and **log-loss** $-\log p_{\text{true}}$. **Calibration** via
Expected Calibration Error over $B$ bins:
$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N}\,\big|\,\text{acc}_b - \text{conf}_b\,\big|$$
(Backtest headline: the meta-learner reached RPS $\approx 0.1999$ on WC 2010–2022, beating the
Elo baseline $\approx 0.2022$.)

### 7. Monte Carlo — turning match models into a champion
The crux: **posterior-propagated, correlated** simulation. For each simulation $s = 1,\dots,N$
($N = 20{,}000$), draw ONE posterior sample of every team's strengths
$\theta^{(s)} = \{\alpha^{(s)}, \beta^{(s)}, \dots\}$, then play the **entire** 48-team tournament
under that single draw (group stage with FIFA tiebreakers → 8 best third-placed → knockout with
extra time + penalties), sampling scorelines from the Poisson rates:
$$\theta^{(s)} \sim \text{posterior}, \qquad \text{champion}_s = \text{Simulate}\big(\text{bracket} \mid \theta^{(s)}\big)$$
Because one $\theta^{(s)}$ governs all of a team's matches in sim $s$, outcomes are **correlated**
(a secretly-strong team is strong in every match that sim) — capturing tail behaviour that naive
per-match sampling misses. The title probability is the Monte Carlo average:
$$P(\text{team wins}) = \frac{1}{N}\sum_{s=1}^{N}\mathbb{1}\big[\text{champion}_s = \text{team}\big]$$

### 8. Honest limits (keep this section — it matters)
Short prose: single-elimination is high variance (the favourite is only ~1-in-5); the model is
calibrated and uncertainty-aware, not an oracle; two documented approximations (group tiebreakers
beyond GD/GF use a random draw; the 8 best third-placed teams fill R32 slots by a fixed rule, not
FIFA's exact combination table). Match-model trains on history; the 48-team bracket is rule-encoded
(no precedent).

## Quality floor
KaTeX renders cleanly (no raw `$$` leaking), equations are readable on cream (dark ink), responsive
(equations scroll horizontally on mobile rather than overflowing the viewport), semantic HTML,
focus states, clean console, back-link works. It should feel like the same product as the album —
not a bland docs page.
