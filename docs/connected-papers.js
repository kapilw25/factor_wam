/* ============================================================
   Connected Papers · D3.js v7 force graph
   21 nodes · 5 research families + ours at center
   ============================================================ */

(function () {
  // ------- DATA -------
  const nodes = [
    // OURS (center)
    {
      id: "factorwam",
      title: "FactorWAM (Ours)",
      family: "ours",
      year: 2027,
      venue: "CVPR / NeurIPS",
      desc:
        "World Action Models as Self-Supervised Reward Signals. Trajectory-consistency reward derived from a video JEPA world model. No robot, no VLM judge."
    },

    // WORLD MODELS
    { id: "vjepa", title: "V-JEPA", family: "world", year: 2024, venue: "CVPR",
      desc: "Joint-embedding predictive architecture for video. Backbone family for FactorWAM." },
    { id: "vjepa2", title: "V-JEPA 2.1", family: "world", year: 2025, venue: "arXiv",
      desc: "Successor with motion-aware features. Direct backbone for our encoder." },
    { id: "unisim", title: "UniSim", family: "world", year: 2024, venue: "ICLR",
      desc: "Universal video-prediction simulator. Pixel-level world model." },
    { id: "genie2", title: "Genie 2", family: "world", year: 2024, venue: "DeepMind",
      desc: "Foundation world model that generates playable environments from text/image." },
    { id: "gaia2", title: "GAIA-2", family: "world", year: 2025, venue: "Wayve",
      desc: "Driving-domain world model. Domain-specific cousin of FactorWAM." },
    { id: "cosmos", title: "Cosmos", family: "world", year: 2025, venue: "NVIDIA",
      desc: "Foundation world model for physical AI. Scale ancestor of WAM idea." },
    { id: "dreamerv3", title: "DreamerV3", family: "world", year: 2025, venue: "Nature",
      desc: "Generalist world-model RL. Closest ancestor for our reward formulation." },
    { id: "dinov2", title: "DINOv2", family: "world", year: 2024, venue: "Meta FAIR",
      desc: "SSL vision foundation. Co-author Vasu Sharma. Aesthetic ancestor." },

    // VLA / ACTION
    { id: "rt2", title: "RT-2", family: "vla", year: 2023, venue: "Google",
      desc: "Vision-Language-Action transformer. Jim Fan calls VLA family 'dead' post-2025." },
    { id: "openvla", title: "OpenVLA", family: "vla", year: 2024, venue: "Stanford",
      desc: "Open VLA used as our policy backbone in Step 4." },
    { id: "pizero", title: "Pi-Zero", family: "vla", year: 2024, venue: "Physical Intelligence",
      desc: "Generalist robot policy. Needs robot data — we don't." },
    { id: "octo", title: "Octo", family: "vla", year: 2024, venue: "Stanford",
      desc: "Cross-embodiment VLA. Trained on Open-X-Embodiment." },

    // REWARD MODELS
    { id: "roboreward", title: "RoboReward", family: "reward", year: 2026, venue: "NVIDIA",
      desc: "Vision-language reward model. Uses proprietary VLM judge — exactly what we replace." },
    { id: "vlp", title: "VLP", family: "reward", year: 2025, venue: "NeurIPS",
      desc: "Vision-Language Preference Learning. Preference-based, expensive labels." },
    { id: "sirl", title: "SIRL", family: "reward", year: 2025, venue: "NeurIPS",
      desc: "Solver-as-reward for LLMs. Text-only ancestor of our self-judging idea." },
    { id: "roboclip", title: "RoboCLIP", family: "reward", year: 2023, venue: "NeurIPS",
      desc: "CLIP-based reward for video imitation. No rollout, no world model." },
    { id: "vlrewardbench", title: "VL-RewardBench", family: "reward", year: 2025, venue: "CVPR",
      desc: "Benchmark for VL reward models. Eval target for our R1." },

    // DATASETS
    { id: "ego4d", title: "Ego4D", family: "dataset", year: 2022, venue: "Meta",
      desc: "First massive egocentric video corpus. Predecessor of EgoActionVideo-50K." },
    { id: "openx", title: "Open-X-Embodiment", family: "dataset", year: 2024, venue: "Google",
      desc: "Robot trajectories across 22 embodiments. Used in our Step 4." },
    { id: "egoscale", title: "EgoScale", family: "dataset", year: 2025, venue: "NVIDIA",
      desc: "Glove + teleop scaling. Direct inspiration for our annotation strategy." },

    // SIMULATORS
    { id: "dreamdojo", title: "DreamDojo", family: "sim", year: 2026, venue: "NVIDIA",
      desc: "Real2Sim2Real neural physics engine. We use its philosophy in Step 4." }
  ];

  // edges: from → to (which paper our paper directly cites or descends from)
  const links = [
    // ours connects to direct ancestors
    { source: "factorwam", target: "vjepa2", weight: 5 },
    { source: "factorwam", target: "vjepa", weight: 4 },
    { source: "factorwam", target: "dinov2", weight: 3 },
    { source: "factorwam", target: "unisim", weight: 2 },
    { source: "factorwam", target: "genie2", weight: 2 },
    { source: "factorwam", target: "cosmos", weight: 3 },
    { source: "factorwam", target: "dreamerv3", weight: 4 },
    { source: "factorwam", target: "openvla", weight: 4 },
    { source: "factorwam", target: "pizero", weight: 3 },
    { source: "factorwam", target: "octo", weight: 2 },
    { source: "factorwam", target: "roboreward", weight: 5 },
    { source: "factorwam", target: "vlp", weight: 3 },
    { source: "factorwam", target: "sirl", weight: 4 },
    { source: "factorwam", target: "roboclip", weight: 3 },
    { source: "factorwam", target: "vlrewardbench", weight: 2 },
    { source: "factorwam", target: "ego4d", weight: 2 },
    { source: "factorwam", target: "openx", weight: 4 },
    { source: "factorwam", target: "egoscale", weight: 4 },
    { source: "factorwam", target: "dreamdojo", weight: 4 },
    { source: "factorwam", target: "rt2", weight: 1 },
    { source: "factorwam", target: "gaia2", weight: 2 },

    // intra-family edges (existing literature topology)
    { source: "vjepa2", target: "vjepa", weight: 5 },
    { source: "vjepa", target: "dinov2", weight: 3 },
    { source: "genie2", target: "unisim", weight: 3 },
    { source: "cosmos", target: "genie2", weight: 2 },
    { source: "gaia2", target: "cosmos", weight: 2 },
    { source: "dreamerv3", target: "unisim", weight: 2 },

    { source: "openvla", target: "rt2", weight: 4 },
    { source: "octo", target: "rt2", weight: 3 },
    { source: "pizero", target: "openvla", weight: 3 },
    { source: "openvla", target: "openx", weight: 4 },
    { source: "octo", target: "openx", weight: 4 },

    { source: "roboreward", target: "roboclip", weight: 3 },
    { source: "vlp", target: "roboreward", weight: 3 },
    { source: "vlrewardbench", target: "roboreward", weight: 4 },
    { source: "sirl", target: "roboclip", weight: 1 },

    { source: "egoscale", target: "ego4d", weight: 3 },
    { source: "dreamdojo", target: "cosmos", weight: 3 },
    { source: "dreamdojo", target: "egoscale", weight: 2 }
  ];

  // ------- COLORS -------
  const family = {
    ours: "#e8b87a",
    world: "#6b9080",
    vla: "#7a8db8",
    reward: "#c77d49",
    dataset: "#d4a574",
    sim: "#b85c5c"
  };

  const familyLabel = {
    ours: "Ours",
    world: "World Model",
    vla: "VLA / Action",
    reward: "Reward Model",
    dataset: "Dataset",
    sim: "Simulator"
  };

  // ------- SETUP -------
  const container = document.getElementById("graph-container");
  const svgEl = document.getElementById("graph-svg");
  const tooltip = document.getElementById("graph-tooltip");

  let width = container.clientWidth;
  let height = container.clientHeight;

  const svg = d3.select("#graph-svg").attr("viewBox", `0 0 ${width} ${height}`);

  // soft glow for the center node
  const defs = svg.append("defs");
  const glow = defs
    .append("filter")
    .attr("id", "glow")
    .attr("x", "-50%")
    .attr("y", "-50%")
    .attr("width", "200%")
    .attr("height", "200%");
  glow.append("feGaussianBlur").attr("stdDeviation", "6").attr("result", "blur");
  const merge = glow.append("feMerge");
  merge.append("feMergeNode").attr("in", "blur");
  merge.append("feMergeNode").attr("in", "SourceGraphic");

  // zoom & pan
  const root = svg.append("g");
  const zoom = d3
    .zoom()
    .scaleExtent([0.4, 3])
    .on("zoom", (e) => root.attr("transform", e.transform));
  svg.call(zoom);

  // ------- FORCE SIMULATION -------
  const sim = d3
    .forceSimulation(nodes)
    .force(
      "link",
      d3
        .forceLink(links)
        .id((d) => d.id)
        .distance((d) => 90 + (6 - d.weight) * 18)
        .strength(0.35)
    )
    .force("charge", d3.forceManyBody().strength((d) => (d.id === "factorwam" ? -900 : -260)))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide((d) => (d.id === "factorwam" ? 50 : 28)));

  // ------- RENDER -------
  const link = root
    .append("g")
    .attr("class", "links")
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("stroke", "#3a3631")
    .attr("stroke-opacity", 0.55)
    .attr("stroke-width", (d) => Math.sqrt(d.weight) * 0.9);

  const nodeG = root
    .append("g")
    .attr("class", "nodes")
    .selectAll("g")
    .data(nodes)
    .join("g")
    .attr("data-family", (d) => d.family)
    .style("cursor", "grab")
    .call(
      d3
        .drag()
        .on("start", (e, d) => {
          if (!e.active) sim.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on("drag", (e, d) => {
          d.fx = e.x;
          d.fy = e.y;
        })
        .on("end", (e, d) => {
          if (!e.active) sim.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        })
    );

  // outer ring (for ours node)
  nodeG
    .filter((d) => d.id === "factorwam")
    .append("circle")
    .attr("r", 32)
    .attr("fill", "none")
    .attr("stroke", family.ours)
    .attr("stroke-width", 1)
    .attr("stroke-opacity", 0.4);

  nodeG
    .append("circle")
    .attr("r", (d) => (d.id === "factorwam" ? 22 : 11))
    .attr("fill", (d) => family[d.family])
    .attr("stroke", (d) => (d.id === "factorwam" ? "#0a0908" : "#0a0908"))
    .attr("stroke-width", 2)
    .attr("filter", (d) => (d.id === "factorwam" ? "url(#glow)" : null));

  // labels
  const labels = nodeG
    .append("text")
    .attr("class", "node-label")
    .attr("dy", (d) => (d.id === "factorwam" ? 38 : 22))
    .attr("text-anchor", "middle")
    .attr("font-family", '"IBM Plex Mono", monospace')
    .attr("font-size", (d) => (d.id === "factorwam" ? "13px" : "10.5px"))
    .attr("font-weight", (d) => (d.id === "factorwam" ? 600 : 400))
    .attr("fill", (d) => (d.id === "factorwam" ? "#e8b87a" : "#c9c2b3"))
    .attr("letter-spacing", "0.04em")
    .text((d) => d.title);

  // tooltip handlers
  nodeG
    .on("mouseenter", function (e, d) {
      d3.select(this).select("circle").transition().duration(120).attr("r", d.id === "factorwam" ? 26 : 14);
      tooltip.classList.add("visible");
      tooltip.innerHTML = `
        <span class="tt-title">${d.title}</span>
        <span class="tt-meta">${familyLabel[d.family]} · ${d.venue} · ${d.year}</span>
        <div class="tt-desc">${d.desc}</div>
      `;
    })
    .on("mousemove", function (e) {
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const tw = tooltip.offsetWidth;
      const th = tooltip.offsetHeight;
      const px = x + 18 + tw > rect.width ? x - tw - 18 : x + 18;
      const py = y + 18 + th > rect.height ? y - th - 18 : y + 18;
      tooltip.style.left = px + "px";
      tooltip.style.top = py + "px";
    })
    .on("mouseleave", function (e, d) {
      d3.select(this).select("circle").transition().duration(120).attr("r", d.id === "factorwam" ? 22 : 11);
      tooltip.classList.remove("visible");
    });

  // tick
  sim.on("tick", () => {
    link
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);

    nodeG.attr("transform", (d) => `translate(${d.x}, ${d.y})`);
  });

  // ------- CONTROLS -------
  document.getElementById("reset-zoom").addEventListener("click", () => {
    svg.transition().duration(450).call(zoom.transform, d3.zoomIdentity);
  });

  let labelsVisible = true;
  document.getElementById("toggle-labels").addEventListener("click", () => {
    labelsVisible = !labelsVisible;
    labels.attr("opacity", labelsVisible ? 1 : 0);
  });

  // legend hover dimming
  document.querySelectorAll(".legend-chip").forEach((chip) => {
    chip.addEventListener("mouseenter", () => {
      const f = chip.dataset.family;
      nodeG.style("opacity", (d) => (d.family === f ? 1 : 0.18));
      link.style("opacity", (d) =>
        d.source.family === f || d.target.family === f ? 0.7 : 0.08
      );
    });
    chip.addEventListener("mouseleave", () => {
      nodeG.style("opacity", 1);
      link.style("opacity", 0.55);
    });
  });

  // resize
  window.addEventListener("resize", () => {
    width = container.clientWidth;
    height = container.clientHeight;
    svg.attr("viewBox", `0 0 ${width} ${height}`);
    sim.force("center", d3.forceCenter(width / 2, height / 2));
    sim.alpha(0.3).restart();
  });
})();
