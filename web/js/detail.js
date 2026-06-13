// Team detail modal: flips in, shows the Qualify -> Win ladder with animated bars
// and counting-up percentages. ESC / backdrop / close button to dismiss.
// Focus trapped while open; restores focus to the trigger on close.

import { pct, ladder, groupChip } from "./util.js";

export function createDetail({ gsap, reducedMotion }) {
  const backdrop = document.getElementById("detail-backdrop");
  const card = document.getElementById("detail-card");
  const closeBtn = document.getElementById("detail-close");
  const flagEl = document.getElementById("detail-flag");
  const rankEl = document.getElementById("detail-rank");
  const nameEl = document.getElementById("detail-name");
  const groupEl = document.getElementById("detail-group");
  const ladderEl = document.getElementById("detail-ladder");

  let lastTrigger = null;
  let isOpen = false;

  function buildLadder(team) {
    ladderEl.innerHTML = "";
    const rungs = ladder(team);
    const els = [];
    rungs.forEach((r) => {
      const li = document.createElement("li");
      li.className = "rung";
      li.innerHTML = `
        <span class="rung-label">${r.label}</span>
        <span class="rung-track"><span class="rung-fill" data-p="${r.p}"></span></span>
        <span class="rung-pct" data-target="${r.p}">${pct(r.p)}</span>
      `;
      ladderEl.appendChild(li);
      els.push(li);
    });
    return els;
  }

  function animateLadder() {
    const fills = Array.from(ladderEl.querySelectorAll(".rung-fill"));
    const pcts = Array.from(ladderEl.querySelectorAll(".rung-pct"));
    if (reducedMotion || !gsap) {
      fills.forEach((f) => { f.style.width = Number(f.dataset.p) * 100 + "%"; });
      return;
    }
    fills.forEach((f) => { f.style.width = "0%"; });
    gsap.to(fills, {
      width: (i) => Number(fills[i].dataset.p) * 100 + "%",
      duration: 0.7, ease: "power2.out", stagger: 0.08,
    });
    // Motion 3: count-up the percentages.
    pcts.forEach((el) => {
      const target = Number(el.dataset.target);
      const obj = { v: 0 };
      gsap.to(obj, {
        v: target, duration: 0.7, ease: "power2.out",
        onUpdate: () => { el.textContent = pct(obj.v); },
      });
    });
  }

  function open(team, trigger) {
    lastTrigger = trigger || document.activeElement;
    flagEl.textContent = team.flag;
    rankEl.textContent = `#${team.rank} of 48`;
    nameEl.textContent = team.team;
    groupEl.textContent = groupChip(team.group);
    buildLadder(team);

    backdrop.hidden = false;
    isOpen = true;
    document.body.style.overflow = "hidden";

    // Motion: flip in.
    if (!reducedMotion && gsap) {
      gsap.fromTo(card,
        { rotationX: -55, y: 40, opacity: 0, transformPerspective: 1200, transformOrigin: "top center" },
        { rotationX: 0, y: 0, opacity: 1, duration: 0.5, ease: "back.out(1.4)",
          onComplete: animateLadder });
    } else {
      card.style.opacity = "1";
      animateLadder();
    }
    card.focus();
  }

  function close() {
    if (!isOpen) return;
    isOpen = false;
    const finish = () => {
      backdrop.hidden = true;
      document.body.style.overflow = "";
      if (lastTrigger && lastTrigger.focus) lastTrigger.focus();
    };
    if (!reducedMotion && gsap) {
      gsap.to(card, { rotationX: -40, y: 30, opacity: 0, duration: 0.28, ease: "power2.in", onComplete: finish });
    } else {
      finish();
    }
  }

  closeBtn.addEventListener("click", close);
  backdrop.addEventListener("click", (e) => { if (e.target === backdrop) close(); });
  document.addEventListener("keydown", (e) => {
    if (!isOpen) return;
    if (e.key === "Escape") close();
    if (e.key === "Tab") {
      // simple focus trap between close button and card
      const focusables = [closeBtn];
      if (document.activeElement === closeBtn && !e.shiftKey) { e.preventDefault(); closeBtn.focus(); }
    }
  });

  return { open, close };
}
