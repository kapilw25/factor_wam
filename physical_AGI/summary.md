# The End Game for Robotics — Jim Fan, NVIDIA

> Source: Jim Fan's Physical AGI talk at the Computer History Museum, 2026
> Timestamps from the original presentation below.

---

## 01:42 — The Great Parallel

The LLM recipe (Pre-Train → SFT → Reasoning RL) maps 1:1 onto the robotics stack (World Modeling → Action Fine-Tuning → Physical RL). Each stage is "simulating," "aligning," then "surpassing."

![LLM timeline: GPT-3 → InstructGPT → OpenAI o1 → AutoResearch 2026](02_the_great_parallel_llm_timeline.png)

![The full Great Parallel — LLM pipeline vs physical intelligence pipeline](02_the_great_parallel_full_diagram.png)

---

## 03:31 — Robotics, the Endgame

Two strategic axes: **Model Strategy** vs **Data Strategy**. The endgame requires getting both right simultaneously — a better model without better data hits a ceiling, and vice versa.

![Presenter framing the Model Strategy vs Data Strategy dichotomy](03_robotics_endgame_model_vs_data.png)

---

## 03:39 — Why VLAs Fall Short

Vision-Language-Action models (RT-2, Octo, OpenVLA) are saturating. The VLA paradigm lacks world modeling — it maps observations to actions without understanding physics. Jim declares VLAs dead; **World Action Models (WAM)** are the successor.

![VLA 2022–2025 RIP — WAM takes over](04_VLAs_fall_short_meme.png)

---

## 04:32 & 06:09 — Video World Models → World Action Models (WAM)

The "Aligning" stage (yellow box) is where world models acquire action-awareness via Action Fine-Tuning. WAM = world model + action latent prediction. The result: autonomous robot manipulation without task-specific programming.

![Great Parallel with "Aligning" stage highlighted — Action Fine-Tuning is the key step](06_WAM_aligning_stage_v1.png)

![Variant view emphasizing the Aligning ↔ Action Fine-Tuning correspondence](06_WAM_aligning_stage_v2.png)

![WAM in action — 8-panel montage of autonomous humanoid robot tasks at 4× speed](06_WAM_autonomous_robot_demo.png)

---

## 07:46 — Strategies for Robot Data Collection

The FSD-equivalent flywheel for manipulation data. Three tiers on the **Scalability vs Hardware Alignment** frontier: egocentric videos (10^7 hrs, cheap, low alignment), data wearables (10^5 hrs, mid), and teleoperation (10^3 hrs, expensive, high alignment). The thesis: **sensorized human data** replaces teleop.

![Egocentric hand manipulation data — 4-panel assembly and crafting views](07_data_collection_egocentric_hands.png)

![Dense Language Annotations on an egocentric dishwashing sequence](07_data_collection_dense_annotations.png)

![Scalability vs Hardware Alignment frontier — egocentric videos > wearables > teleop](07_data_collection_scalability_vs_alignment.png)

![Teleop 2022–2025 RIP — Sensorized Human Data is the new paradigm](07_data_collection_teleop_rip_meme.png)

---

## 11:06 — EgoScale & the Dexterity Scaling Law

**L = 0.024 − 0.003 · ln(D)** — action-prediction loss follows a log-linear scaling law with pretraining data size. 21K hours across 10K scenes and 43K objects for pretraining, then only 50 hrs glove + 4 hrs teleop for action fine-tuning. More data = better dexterity, predictably.

![Pretraining corpus: 21K hours, 10K scenes, 43K objects — egocentric video grid](08_egoscale_pretraining_data.png)

![Action Fine-Tuning data: 50 hrs glove-pose + 4 hrs teleoperation](08_egoscale_action_finetuning.png)

![Scaling curves — validation loss vs training steps at 1K/2K/4K/10K/20K hours](08_egoscale_scaling_curves.png)

![Dexterity Scaling Law: L = 0.024 − 0.003 · ln(D) — log-linear fit across data scales](08_egoscale_dexterity_scaling_law.png)

---

## 14:00 — Physical RL: Bridging the Last Mile

Three simulation fidelity tiers for massive parallel RL training: **Real World** (limited evals), **World Scan** via graphics engines (1K parallel envs), and **World Model** inference (100K parallel latent rollouts). Train cheap in latent space, validate in sim, deploy in real.

![Massive Parallel RL Training — Real World / World Scan / World Model tiers](09_physical_rl_parallel_training.png)

---

## 15:39 — DreamDojo: Real2Sim2Real

Neural physics engine for scaling RL in silico. **Digital cousins** (approximate scene reconstructions) replace exact digital twins — good enough physics at 1000× the throughput. Real scenes → sim variants → trained policies → real deployment.

![Real2Sim2Real — Real vs Digital Twins vs Digital Cousins side-by-side](10_dreamdojo_real2sim2real.png)

---

## 17:00 — Civilizational Technology Tree

Three branches of the physical AI tech tree, each building on the previous:

| Branch | Unlocks | Endgame |
|---|---|---|
| **Physical Turing Test** | The Great Parallel + The Great Flywheel | Energy in → Unit labor out |
| **Physical API** | Lights-out Factory + Self-Driving Science Lab | Energy in → Full objects out |
| **Physical AutoResearch** | Auto Materials + AutoBot | Energy in → Next robots out |

![Full technology tree — three branches of physical AI development](11_tech_tree_overview.png)

![Physical Turing Test — The Great Parallel + The Great Flywheel](11_tech_tree_physical_turing_test.png)

![Physical API — Lights-out Factory + Self-Driving Science Lab](11_tech_tree_physical_api.png)

![Physical AutoResearch — Auto Materials + AutoBot](11_tech_tree_physical_autoresearch.png)

---

*Screenshots captured from Jim Fan's presentation, May 9 2026.*
