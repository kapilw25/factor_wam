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

    // --- Diagram lightbox: click to full-view ---
    // Create lightbox overlay (once)
    const overlay = document.createElement("div");
    overlay.id = "diagram-lightbox";
    overlay.innerHTML = `
      <div class="lightbox-content">
        <div class="lightbox-header">
          <span class="lightbox-title"></span>
          <button class="lightbox-close">&times;</button>
        </div>
        <div class="lightbox-body"></div>
      </div>
    `;
    document.body.appendChild(overlay);

    const lbTitle = overlay.querySelector(".lightbox-title");
    const lbBody = overlay.querySelector(".lightbox-body");
    const lbClose = overlay.querySelector(".lightbox-close");

    // Close on button, overlay click, or Escape
    function closeLightbox() {
      overlay.classList.remove("lb-open");
    }
    lbClose.addEventListener("click", closeLightbox);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) closeLightbox();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeLightbox();
    });

    // Add expand button to every mermaid-card
    document.querySelectorAll(".mermaid-card").forEach((card) => {
      // Add expand hint to header
      const head = card.querySelector(".mermaid-card-head");
      if (head && !head.querySelector(".expand-btn")) {
        const btn = document.createElement("button");
        btn.className = "expand-btn";
        btn.textContent = "⛶ expand";
        btn.title = "Click to full view";
        head.appendChild(btn);
      }

      // Make entire card clickable
      card.style.cursor = "pointer";
      card.addEventListener("click", (e) => {
        // Don't trigger if clicking a link inside
        if (e.target.closest("a")) return;

        const mermaidDiv = card.querySelector(".mermaid");
        const svg = mermaidDiv ? mermaidDiv.querySelector("svg") : null;
        const label = card.querySelector(".card-label");

        lbTitle.textContent = label ? label.textContent : "Diagram";
        lbBody.innerHTML = "";

        if (svg) {
          const clone = svg.cloneNode(true);
          clone.style.width = "100%";
          clone.style.height = "auto";
          clone.style.maxHeight = "80vh";
          lbBody.appendChild(clone);
        } else if (mermaidDiv) {
          lbBody.innerHTML = "<p style='color:var(--paper-faint);'>Diagram not yet rendered.</p>";
        }

        overlay.classList.add("lb-open");
      });
    });
  }, 600);
});
