// Orchestrator: fetch model data, wire hero / album / detail, respect reduced motion.

import { pct } from "./util.js";
import { renderAlbum } from "./album.js";
import { createDetail } from "./detail.js";
import { burst } from "./confetti.js";

const reducedMotion =
  window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Lazy-load motion + 3D libs only when we'll actually use them.
async function loadGsap() {
  if (reducedMotion) return { gsap: null, ScrollTrigger: null };
  try {
    const [{ gsap }, stMod] = await Promise.all([
      import("gsap"),
      import("gsap/ScrollTrigger"),
    ]);
    const ScrollTrigger = stMod.ScrollTrigger || stMod.default;
    return { gsap, ScrollTrigger };
  } catch (err) {
    console.warn("GSAP failed to load; continuing with static UI.", err);
    return { gsap: null, ScrollTrigger: null };
  }
}

async function main() {
  let data;
  try {
    const res = await fetch("./data/predictions.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    console.error("Could not load predictions.json", err);
    document.getElementById("hero-method").textContent =
      "Could not load the model data. Serve this folder and reload.";
    return;
  }

  const { meta, teams } = data;
  const fav = teams[0];

  // ---- Hero copy ----
  document.getElementById("hero-method").textContent = meta.method;

  // ---- Honesty panel + footer ----
  document.getElementById("caveat-text").textContent = meta.caveat;
  document.getElementById("foot-method").textContent = meta.method;
  document.getElementById("foot-asof").textContent = `Model run: ${meta.as_of} · ${meta.n_sims.toLocaleString()} simulations · ${meta.n_teams} teams`;

  // ---- Hero favourite foil sticker ----
  document.getElementById("ff-flag").textContent = fav.flag;
  document.getElementById("ff-name").textContent = fav.team;
  const ffPct = document.getElementById("ff-pct");
  const ffNum = ffPct.querySelector(".ff-pct-num");
  ffPct.dataset.target = String(fav.p_win);
  const setFavPct = (p) => { ffNum.textContent = pct(p).replace("%", ""); };

  const { gsap, ScrollTrigger } = await loadGsap();

  // Motion 3: count-up the hero favourite percentage.
  if (!reducedMotion && gsap) {
    const obj = { v: 0 };
    gsap.to(obj, {
      v: fav.p_win, duration: 1.2, ease: "power2.out", delay: 0.4,
      onUpdate: () => { setFavPct(obj.v); },
    });
  } else {
    setFavPct(fav.p_win);
  }

  // ---- Team detail modal ----
  const detail = createDetail({ gsap, reducedMotion });

  // ---- The Album ----
  renderAlbum(document.getElementById("sticker-grid"), teams, {
    onSelect: (t) => detail.open(t),
    gsap, ScrollTrigger, reducedMotion,
  });

  // ---- Motion 4: winner confetti on the hero favourite (one-shot) ----
  if (!reducedMotion) {
    const feature = document.getElementById("foil-feature");
    setTimeout(() => burst(feature, { reducedMotion }), 900);
  }

  // ---- 3D ball: lazy-init after first paint so it never blocks content ----
  initBall();
}

function initBall() {
  const wrap = document.getElementById("ball-wrap");
  if (!wrap) return;
  // Defer to idle so the album paints first.
  const start = () => {
    import("./ball.js")
      .then(({ mountBall }) => { mountBall(wrap, { reducedMotion }); })
      .catch((err) => {
        console.warn("3D ball unavailable; hero remains static.", err);
        wrap.style.display = "none";
      });
  };
  if ("requestIdleCallback" in window) requestIdleCallback(start, { timeout: 1200 });
  else setTimeout(start, 300);
}

main();
