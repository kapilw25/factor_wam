/* ============================================================
   Light interactivity · reveal animations & mermaid retry
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  // staggered reveal on scroll
  const revealTargets = document.querySelectorAll(
    ".section, .step, .contrib-card, .parallel-row, .mermaid-card, .lit-table-wrap, #graph-container, .timeline"
  );

  revealTargets.forEach((el) => {
    el.style.opacity = "0";
    el.style.transform = "translateY(18px)";
    el.style.transition = "opacity 0.7s ease, transform 0.7s ease";
  });

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry, i) => {
        if (entry.isIntersecting) {
          setTimeout(() => {
            entry.target.style.opacity = "1";
            entry.target.style.transform = "translateY(0)";
          }, i * 60);
          io.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.08, rootMargin: "0px 0px -40px 0px" }
  );

  revealTargets.forEach((el) => io.observe(el));

  // ensure mermaid re-runs after late layout
  setTimeout(() => {
    if (window.mermaid && typeof window.mermaid.run === "function") {
      try {
        window.mermaid.run({ querySelector: ".mermaid" });
      } catch (e) {
        /* mermaid auto-init handles this; ignore */
      }
    }
  }, 200);
});
