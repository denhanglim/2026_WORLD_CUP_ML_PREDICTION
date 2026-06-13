// The Album: render 48 ranked stickers into numbered die-cut album slots.
// Top contenders get the foil treatment; the rest are matte paper. Stickers
// sit at slight random angles inside their slots (Panini scatter), slap onto
// the page on scroll, and peel + tilt + shine toward the cursor on hover.

import { pct, barWidth, isFoil, tintFor, groupChip, maxWin } from "./util.js";

// A stable pseudo-random rotation per rank so the scatter is the same each load
// (looks intentional, not jittery) but still organic. Range ~ ±4.5deg, biased
// away from zero so very few stickers sit perfectly straight.
function scatterAngle(rank) {
  const s = Math.sin(rank * 12.9898) * 43758.5453;
  const f = s - Math.floor(s); // 0..1
  const signed = f * 2 - 1; // -1..1
  const mag = 1.6 + Math.abs(signed) * 2.9; // 1.6..4.5
  return (signed < 0 ? -1 : 1) * mag;
}

export function renderAlbum(gridEl, teams, { onSelect, gsap, ScrollTrigger, reducedMotion }) {
  if (!gridEl) return;
  const max = maxWin(teams);
  gridEl.innerHTML = "";

  teams.forEach((t) => {
    const foil = isFoil(t.rank);
    const angle = scatterAngle(t.rank);

    const li = document.createElement("li");
    li.className = "slot" + (foil ? " slot-foil" : "");
    li.setAttribute("role", "listitem");
    li.style.setProperty("--angle", angle.toFixed(2) + "deg");

    const slotNo = String(t.rank).padStart(2, "0");

    const card = document.createElement("button");
    card.type = "button";
    card.className = "sticker" + (foil ? " foil" : ` tint-${tintFor(t.rank)}`);
    card.setAttribute("aria-label",
      `Number ${t.rank}, ${t.team}, ${pct(t.p_win)} chance to win. View road to the final.`);

    const w = barWidth(t.p_win, max);
    const [whole, frac] = pct(t.p_win).replace("%", "").split(".");
    card.innerHTML = `
      ${foil ? '<span class="foil-shine" aria-hidden="true"></span>' : ""}
      <span class="shine-sweep" aria-hidden="true"></span>
      <div class="s-head">
        <span class="s-flag" aria-hidden="true">${t.flag}</span>
        <span class="s-group">${groupChip(t.group)}</span>
      </div>
      <div class="s-name">${t.team}</div>
      <div class="s-stat">
        <div class="s-pct">${whole}<span class="pct-frac">${frac ? '.' + frac : ''}</span><span class="pct-unit">%</span></div>
        <span class="s-pct-label">to lift it</span>
      </div>
      <div class="s-bar"><span data-w="${w}"></span></div>
    `;

    card.addEventListener("click", () => onSelect(t));
    wireHover(li, card, foil, gsap, reducedMotion);

    // Die-cut slot furniture: slot number + dotted outline live on the <li>.
    const num = document.createElement("span");
    num.className = "slot-no";
    num.setAttribute("aria-hidden", "true");
    num.textContent = "#" + slotNo;

    li.appendChild(num);
    li.appendChild(card);
    gridEl.appendChild(li);
  });

  const slots = Array.from(gridEl.querySelectorAll(".slot"));
  const cards = Array.from(gridEl.querySelectorAll(".sticker"));
  const bars = Array.from(gridEl.querySelectorAll(".s-bar > span"));

  if (reducedMotion || !gsap) {
    // Static fallback: slots settled at their scatter angle (CSS --angle), bars filled.
    bars.forEach((b) => { b.style.width = b.dataset.w + "%"; });
    return;
  }

  if (ScrollTrigger) gsap.registerPlugin(ScrollTrigger);

  // Motion 1: "slap onto the page" — slots drop in from above with scale + a
  // rotation overshoot that settles to their scatter angle, staggered on scroll.
  slots.forEach((slot, i) => {
    const angle = scatterAngle(i + 1);
    gsap.set(slot, { opacity: 0, scale: 1.16, rotation: angle - 9, y: -26, transformOrigin: "50% 30%" });
  });

  const fillBars = () =>
    gsap.to(bars, { width: (i) => bars[i].dataset.w + "%", duration: 0.8, ease: "power2.out", stagger: 0.008 });

  gsap.to(slots, {
    opacity: 1,
    scale: 1,
    rotation: (i) => scatterAngle(i + 1),
    y: 0,
    duration: 0.55,
    ease: "back.out(2.3)",
    stagger: { each: 0.026, from: "start", grid: "auto" },
    scrollTrigger: ScrollTrigger ? { trigger: gridEl, start: "top 88%", once: true } : undefined,
    onComplete: () => {
      // a quick shine sweeps across each sticker as it lands
      gsap.fromTo(gridEl.querySelectorAll(".shine-sweep"),
        { opacity: 0.9, backgroundPosition: "140% 0" },
        { opacity: 0, backgroundPosition: "-40% 0", duration: 0.6, stagger: 0.01 });
      fillBars();
    },
  });

  // Safety net: content must never get stuck invisible (e.g. a layout where the
  // grid is fully on-screen at load, or a no-scroll capture). Reveal + fill if
  // anything is still hidden after a grace period.
  setTimeout(() => {
    let stuck = false;
    slots.forEach((s, i) => {
      if (Number(getComputedStyle(s).opacity) < 0.99) {
        stuck = true;
        gsap.set(s, { opacity: 1, scale: 1, y: 0, rotation: scatterAngle(i + 1) });
      }
    });
    if (stuck) fillBars();
  }, 1600);
}

// Motion 2: hover — the sticker tilts toward the cursor in 3D, a shine sweeps
// across it, the slot peels a corner and the shadow lifts. The slot owns the
// scatter rotation; the sticker only adds a 3D tilt + a slight pop scale.
function wireHover(slot, card, foil, gsap, reducedMotion) {
  if (reducedMotion) return;

  function onMove(e) {
    const r = card.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width - 0.5;
    const py = (e.clientY - r.top) / r.height - 0.5;
    if (gsap) {
      gsap.to(card, {
        rotationY: px * 16,
        rotationX: -py * 16,
        scale: 1.04,
        transformPerspective: 700,
        duration: 0.3,
        ease: "power2.out",
      });
    }
    const shine = card.querySelector(".shine-sweep");
    if (shine) {
      shine.style.opacity = "0.85";
      shine.style.backgroundPosition = `${(px + 0.5) * 220 - 60}% 0`;
    }
    slot.classList.add("peeling");
  }
  function onLeave() {
    if (gsap) gsap.to(card, { rotationY: 0, rotationX: 0, scale: 1, duration: 0.5, ease: "power3.out" });
    const shine = card.querySelector(".shine-sweep");
    if (shine) shine.style.opacity = "0";
    slot.classList.remove("peeling");
  }
  card.addEventListener("pointerenter", onMove);
  card.addEventListener("pointermove", onMove);
  card.addEventListener("pointerleave", onLeave);
}
