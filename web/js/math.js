/* ============================================================
   THE MATHS — reading-page interactions
   - scroll-progress (top bar + rail bar)
   - scrollspy (active section in the sticky rail)
   - calm GSAP reveals for sections / exhibits / diagrams
   - full prefers-reduced-motion fallback: everything static + visible
   ============================================================ */

const reduce =
  window.matchMedia &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const railBar = document.getElementById("rail-bar");
const readBar = document.getElementById("read-bar");
const spyLinks = Array.from(document.querySelectorAll(".rail-list a[data-spy]"));
const sections = spyLinks
  .map((a) => document.getElementById(a.dataset.spy))
  .filter(Boolean);

/* ---- scroll progress: cheap, rAF-throttled ---- */
let ticking = false;
function updateProgress() {
  const doc = document.documentElement;
  const max = doc.scrollHeight - doc.clientHeight;
  const pct = max > 0 ? Math.min(1, Math.max(0, doc.scrollTop / max)) : 0;
  if (readBar) readBar.style.transform = `scaleX(${pct})`;
  if (railBar) railBar.style.transform = `scaleY(${pct})`;
  ticking = false;
}
function onScroll() {
  if (!ticking) {
    ticking = true;
    requestAnimationFrame(updateProgress);
  }
}
window.addEventListener("scroll", onScroll, { passive: true });
updateProgress();

/* ---- scrollspy: highlight the section nearest the top third ---- */
function setActive(id) {
  spyLinks.forEach((a) =>
    a.classList.toggle("is-active", a.dataset.spy === id)
  );
}
if ("IntersectionObserver" in window && sections.length) {
  const seen = new Map();
  const spy = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => seen.set(e.target.id, e));
      // pick the topmost intersecting section
      let best = null;
      seen.forEach((e) => {
        if (!e.isIntersecting) return;
        if (!best || e.boundingClientRect.top < best.boundingClientRect.top)
          best = e;
      });
      if (best) setActive(best.target.id);
    },
    { rootMargin: "-15% 0px -70% 0px", threshold: 0 }
  );
  sections.forEach((s) => spy.observe(s));
  setActive(sections[0].id);
}

/* ---- reveals ---- */
const revealTargets = document.querySelectorAll(
  ".math-section, .diagram, .eq-plate"
);

if (reduce || !("IntersectionObserver" in window)) {
  // No motion: make sure everything is visible & static.
  revealTargets.forEach((el) => el.classList.add("is-in"));
} else {
  document.body.classList.add("reveal-ready");

  // Try GSAP for buttery staggered reveals; fall back to IO + CSS if the CDN
  // import fails (offline / blocked) so content is never stuck hidden.
  (async () => {
    let gsap, ScrollTrigger;
    try {
      ({ gsap } = await import("gsap"));
      ({ ScrollTrigger } = await import("gsap/ScrollTrigger"));
      gsap.registerPlugin(ScrollTrigger);
    } catch (err) {
      gsapFallback();
      return;
    }

    const allBits = [];
    document.querySelectorAll(".math-section").forEach((sec) => {
      const bits = sec.querySelectorAll(
        ".sec-head, .lead, p, h3, ul, .eq-plate, .diagram, .note, .limits-ticket"
      );
      bits.forEach((b) => allBits.push(b));
      gsap.set(sec, { opacity: 1 });
      gsap.from(bits, {
        opacity: 0,
        y: 22,
        duration: 0.55,
        ease: "power2.out",
        stagger: 0.06,
        clearProps: "opacity,transform",
        scrollTrigger: {
          trigger: sec,
          start: "top 88%",
          once: true,
        },
      });
    });

    // Safety net: nothing may stay hidden (covers off-screen content during
    // full-page capture, no-scroll, or a stalled ScrollTrigger). Force-clear
    // any inline opacity/transform left on animated bits.
    function unstick() {
      allBits.forEach((b) => {
        if (getComputedStyle(b).opacity !== "1") {
          b.style.opacity = "1";
          b.style.transform = "none";
        }
      });
      revealTargets.forEach((el) => el.classList.add("is-in"));
    }
    setTimeout(() => { ScrollTrigger.refresh(); }, 400);
    setTimeout(unstick, 1600);
    window.addEventListener("load", () => setTimeout(unstick, 200));
  })();

  function gsapFallback() {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("is-in");
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -8% 0px" }
    );
    revealTargets.forEach((el) => io.observe(el));
    setTimeout(() => {
      revealTargets.forEach((el) => {
        if (getComputedStyle(el).opacity !== "1") el.classList.add("is-in");
      });
    }, 1500);
  }
}
