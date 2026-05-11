/* ============================================================
   Mermaid · custom dark theme matched to the proposal palette
   ============================================================ */

mermaid.initialize({
  startOnLoad: true,
  theme: "base",
  fontFamily: '"IBM Plex Sans", system-ui, sans-serif',
  themeVariables: {
    /* base */
    background: "#1d2024",
    primaryColor: "#1d2024",
    primaryTextColor: "#f5f1e8",
    primaryBorderColor: "#d4a574",

    /* nodes */
    secondaryColor: "#0f1114",
    tertiaryColor: "#0a0908",

    /* edges & lines */
    lineColor: "#756f63",
    textColor: "#f5f1e8",
    mainBkg: "#1d2024",
    secondBkg: "#0f1114",

    /* clusters */
    clusterBkg: "#0f1114",
    clusterBorder: "#3a3631",

    /* fontSize */
    fontSize: "15px"
  },
  flowchart: {
    htmlLabels: true,
    curve: "basis",
    padding: 20,
    nodeSpacing: 50,
    rankSpacing: 60
  },
  securityLevel: "loose"
});
