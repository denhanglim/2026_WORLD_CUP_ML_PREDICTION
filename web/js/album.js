// The Album: render 48 ranked sticker cards from JSON, wire press-down entrance,
// foil hover tilt + shine, and open team detail on click.

import { pct, barWidth, isFoil, tintFor, groupChip, maxWin } from "./util.js";

export function renderAlbum(gridEl, teams, { onSelect, gsap, ScrollTrigger, reducedMotion }) {
  if (!gridEl) return;
  const max = maxWin(teams);
  gridEl.innerHTML = "";

  teams.forEach((t) => {
    const li = document.createElement("li");
    li.setAttribute("role", "listitem");

    const card = document.createElement("button");
    card.type = "button";
    const foil = isFoil(t.rank);
    card.className = "sticker" + (foil ? " foil" : ` tint-${tintFor(t.rank)}`);
    card.setAttribute("aria-label",
      `${t.team}, ranked ${t.rank}, ${pct(t.p_win)} chance to win. View road to the final.`);

    const w = barWidth(t.p_win, max);
    const [whole, frac] = pct(t.p_win).replace("%", "").split(".");
    card.innerHTML = `
      ${foil ? '<span class="foil-shine" aria-hidden="true"></span>' : ""}
      <div class="s-head">
        <span class="s-rank">#${t.rank}</span>
        <span class="s-flag" aria-hidden="true">${t.flag}</span>
        <span class="s-group">${groupChip(t.group)}</span>
      </div>
      <div class="s-name">${t.team}</div>
      <span class="s-pct-label">Win probability</span>
      <div class="s-pct">${whole}${frac ? '.' + frac : ''}<span class="pct-unit">%</span></div>
      <div class="s-bar"><span data-w="${w}"></span></div>
    `;

    card.addEventListener("click", () => onSelect(t));
    wireHover(card, foil, gsap, reducedMotion);

    li.appendChild(card);
    gridEl.appendChild(li);
  });

  const cards = Array.from(gridEl.querySelectorAll(".sticker"));
  const bars = Array.from(gridEl.querySelectorAll(".s-bar > span"));

  if (reducedMotion || !gsap) {
    // Static fallback: everything visible, bars filled.
    bars.forEach((b) => { b.style.width = b.dataset.w + "%"; });
    return;
  }

  // Motion 1: press-down stagger as the album scrolls in.
  gsap.set(cards, { opacity: 0, scale: 0.82, rotation: () => (Math.random() * 10 - 5), y: 24 });
  if (ScrollTrigger) gsap.registerPlugin(ScrollTrigger);

  gsap.to(cards, {
    opacity: 1, scale: 1, rotation: 0, y: 0,
    duration: 0.5, ease: "back.out(1.7)",
    stagger: { each: 0.035, from: "start" },
    scrollTrigger: ScrollTrigger ? { trigger: gridEl, start: "top 78%", once: true } : undefined,
    onComplete: () => {
      gsap.to(bars, { width: (i) => bars[i].dataset.w + "%", duration: 0.8, ease: "power2.out", stagger: 0.01 });
    },
  });
}

// Motion 2: foil hover — 3D tilt toward cursor + shine sweep + corner peel (via CSS ::after).
function wireHover(card, foil, gsap, reducedMotion) {
  if (reducedMotion) return;

  function onMove(e) {
    const r = card.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width - 0.5;
    const py = (e.clientY - r.top) / r.height - 0.5;
    if (gsap) {
      gsap.to(card, { rotationY: px * 12, rotationX: -py * 12, transformPerspective: 600, duration: 0.3, ease: "power2.out" });
    } else {
      card.style.transform = `perspective(600px) rotateY(${px * 12}deg) rotateX(${-py * 12}deg)`;
    }
    if (foil) {
      const shine = card.querySelector(".foil-shine");
      if (shine) shine.style.backgroundPosition = `${(px + 0.5) * 200 - 50}% 0`;
    }
  }
  function onLeave() {
    if (gsap) gsap.to(card, { rotationY: 0, rotationX: 0, duration: 0.45, ease: "power3.out" });
    else card.style.transform = "";
  }
  card.addEventListener("pointermove", onMove);
  card.addEventListener("pointerleave", onLeave);
}
