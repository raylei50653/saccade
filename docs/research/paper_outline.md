# Technical Report / arXiv Outline (Skeleton)

**Status:** outline only — not a draft paper  
**Date:** 2026-07-09  
**Purpose:** structure claims, evidence, and limitations before writing prose  
**Living status snapshot:** [tracker-decision/status_2026-07-09.md](tracker-decision/status_2026-07-09.md)  
**Numbers ledger:** [evidence_ledger.md](evidence_ledger.md)

Do **not** invent metrics here. Pull every number from the ledger or linked
ablation reports. Update the outline when the production contract changes.

---

## Working title (placeholder)

**Saccade:** Real-time multi-object tracking on laptop GPUs with a pure-geometry
decision layer and whole-graph detection.

Alt focus titles:

- Geometry-first association for real-time MOT without ReID on the critical path  
- Engineering a closed-loop detect–track pipeline under CUDA-graph constraints  

---

## 1. Problem

**Real-time MOT on laptop-class GPUs** (e.g. RTX 50-series mobile), under:

- Strict latency / FPS budgets (target: multi-hundred FPS on MOT17-scale frames)
- No free lunch from large appearance models on the **critical path**
- Need for **auditable** association policy (match / birth / keep / relink)
- Production constraints: TensorRT engines, CUDA graphs, double-buffer overlap

Research questions (draft):

1. Can a **geometry + motion** decision layer match strong public-detection MOT
   quality without synchronous ReID on the hot path?
2. Which association knobs are truly **ACTIVE** vs historical debt (NO-GO/LATENT)?
3. How do **system engineering** choices (graphs, barriers, decode) interact with
   reported accuracy?

---

## 2. System

High-level stack:

```text
Public dets (MOT17 SDP) or live detector
  → YOLO26 backbone + Mamba detection head (whole-graph)
  → GPUByteTracker (CUDA): multiplicative cost, auction, OAO, occ_state
  → Geometry bridge relink (ReID off)
  → Simple interpolation post
```

Two capacity presets:

| Preset | Role |
|:--|:--|
| `mamba_whole_graph` (**s**) | Primary decision / throughput path |
| `mamba_whole_graph_m` (**m**) | Higher-capacity detector + motion/bridge deltas |

Point to: presets YAML, `docs/research/pipeline/mot17_mamba_whole_graph_m_sdp_double_buffer.md`,
module maps under `docs/modules/`.

---

## 3. Decision layer

Central technical contribution candidates:

| Component | Role (one line) |
|:--|:--|
| **Multiplicative cost** + λ | Soft association cost form; couples with stability reward |
| **Dual height stability** | Cost Π (`stability_cost_w`) + auction bid (`SACCADE_STABILITY_W`) — **both on** after P7 |
| **OAO duration-ramp** | Time-aware occlusion association (spatial OAO family = NO-GO) |
| **occ_state** | Explicit occlusion state in cost (production-on) |
| **Private continuation** | Expands det set; score-clamped so continue ≠ birth |
| **Bridge relink** | Geometry-only ID recovery; m looser gates / no dir bonus |
| **Kalman + GMC** | Motion prior; m higher `kalman_r_scale` (trust predict more) |

Contract narrative: small ACTIVE surface; large schema is LATENT/NO-GO-gated.  
Refs: `docs/research/tracker-decision/*`, `docs/reference/math_model.md`.

---

## 4. Engineering

Claims to support with measurement (not slogans):

- **CUDA graph** capture for detect / NMS / GMC / tracker update  
- **Double-buffer** detect(N+1) ∥ tracker(N) with event barrier  
- **Sync removal / CPU overhead** audits (see `docs/research/pipeline/`)  
- **Pinned memory / WSL2** footguns (eval notes)  
- Whole-graph vs eager / threaded path eligibility rules  

Separate “accuracy paper” vs “systems paper” if needed; this outline allows both
in one report with a clear section split.

---

## 5. Evidence

**Primary protocol (decision-layer):**

```text
MOT17 train-half, 7 sequences, SDP
presets: mamba_whole_graph and/or mamba_whole_graph_m
--double-buffer
metrics: IDF1, MOTA, HOTA, AssA, IDs, FP, FN, FPS / latency
host: document GPU / driver / SHA every table
```

Headline-style numbers (see ledger for SHA/protocol):

| Run class | Example source |
|:--|:--|
| Frozen baseline s/m | `docs/reference/mot17_default_config.md` |
| Dual stability A–D | `tracker-decision/audit/dual_stability_ablation_results_2026-07-09.md` |
| Pipeline / FPS attribution | `docs/research/pipeline/perf_attribution_whole_graph_m.md` |

**Always** report protocol (preset, detector, double-buffer, SHA, host).  
Do not mix short-seq or profiling-sync FPS with accuracy tables.

---

## 6. Ablations (planned paper sections)

| Ablation | Status in repo | Paper intent |
|:--|:--|:--|
| **Dual stability 4-way (A–D)** | **done** P7 | Justify keep both; note m cost near-noise |
| **ReID-free / bridge-only** | production `reid_mode=off` | Geometry-first claim |
| **Decode / pipeline drift** | pipeline + eval notes | Systems validity |
| **Jitter / noise floor** | protocol exists; use when Δ≲0.2 IDF1 | Honesty on small deltas |
| **s vs m capacity** | P7 §8 + baselines | Detector capacity vs decision knobs |
| **NO-GO knobs (fuse, NSA, …)** | registry + guardrails | Negative results catalog |

Do **not** open dual-stability behavior PR solely for the paper — evidence already
supports keep-both as default.

---

## 7. Limitations

Be explicit early (draft list):

1. **MOT17 train-half / SDP** — not full private-det MOT17 test-server numbers yet  
2. **Detector dependency** — tracker quality couples to YOLO/Mamba quality and SDP  
3. **No public MOTChallenge test submission** in this outline’s current evidence  
4. **Laptop GPU / driver stack** — FPS not portable without re-measure  
5. **Dual-stability m nuance** — cost additivity weak on m; global default still both-on from s  
6. **Schema surface size** — many LATENT/NO-GO knobs remain in code (guarded, not deleted)  
7. **Appearance** — offline / non-critical ReID paths not claimed as production identity  

---

## 8. Suggested section map (for later drafting)

```text
1 Introduction
2 Related work (MOT, ByteTrack-family, ReID, systems MOT)
3 System overview
4 Decision layer (math + contract)
5 Implementation (CUDA graphs, double-buffer, inject path)
6 Experiments
  6.1 Setup and protocol
  6.2 Main results (s/m)
  6.3 Dual stability ablation
  6.4 Negative results / NO-GO
  6.5 Latency and throughput
7 Limitations
8 Conclusion
Appendix: active contract C1–C9, knob tables, per-seq metrics
```

---

## 9. Non-goals for the outline phase

```text
✗ Full paper prose
✗ New experiments “for the outline”
✗ Default flips to make the story cleaner
✗ Tracker rewrite before narrative consolidation
```

Next drafting step (later PR): abstract + contribution list from this skeleton,
with every claim footnoted to [evidence_ledger.md](evidence_ledger.md).
