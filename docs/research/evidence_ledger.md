# Evidence Ledger

**Purpose:** single table for citable MOT17 / decision-layer results used in
README, PRs, and technical report drafting.  
**Rule:** every row must name **date, commit (or frozen tag), preset, detector,
protocol notes, and source doc**. Prefer linking over re-copying long tables.

Update when a new **decision-relevant** or **baseline** run lands. Do not dump
every probe; this is a ledger, not a lab notebook.

**Status snapshot:** [tracker-decision/status_2026-07-09.md](tracker-decision/status_2026-07-09.md)

---

## Protocol defaults (unless a row says otherwise)

| Item | Default |
|:--|:--|
| Benchmark | MOT17 **train-half**, 7 seq: 02, 04, 05, 09, 10, 11, 13 |
| Detector suffix | **SDP** |
| Scheduling | **`--double-buffer`** when noted |
| Metrics | IDF1 / MOTA / HOTA / AssA / IDs from project eval (TrackEval HOTA family) |
| FPS | Overall eval throughput from same run (host-dependent) |

Noise floor guidance for decision knobs: Δ IDF1 ≲ **0.2** ≈ near-noise without
jitter (see dual-stability protocol).

---

## Ledger

| Date | Commit / tag | Preset | Det | IDF1 | MOTA | HOTA | AssA | IDs | FPS | Conclusion | Source |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|:--|:--|
| 2026-06-21 | frozen_v2 (see source) | `mamba_whole_graph` | SDP | 78.2 | 78.4 | 70.2 | 69.7 | 413 | 269.5 | Headline **s** baseline (throughput path) | [mot17_default_config.md](../reference/mot17_default_config.md) |
| 2026-06-21 | frozen_v2 (see source) | `mamba_whole_graph_m` | SDP | 79.5 | — | — | — | 335 | ~241 | Headline **m** capacity: higher recall/MOTA class, fewer IDs | [mot17_default_config.md](../reference/mot17_default_config.md) |
| 2026-07-09 | `2bc556f2` | `mamba_whole_graph` **A** both | SDP | **78.4** | 78.0 | **70.1** | **70.0** | **425** | ~305 | Dual-stab **keep both** (s primary); A best | [dual_stability_ablation_results_2026-07-09.md](tracker-decision/audit/dual_stability_ablation_results_2026-07-09.md) §3 |
| 2026-07-09 | `2bc556f2` | `mamba_whole_graph` **B** cost-only | SDP | 77.4 | 78.1 | 69.6 | 68.9 | 490 | — | Bid helps when cost on (vs A) | same §3 |
| 2026-07-09 | `2bc556f2` | `mamba_whole_graph` **C** bid-only | SDP | 77.5 | 78.3 | 69.6 | 68.8 | 430 | — | Cost helps when bid on (vs A) | same §3 |
| 2026-07-09 | `2bc556f2` | `mamba_whole_graph` **D** both-off | SDP | 77.6 | 77.8 | 69.9 | 69.7 | 482 | — | No height-stability: loses IDF1/IDs vs A | same §3 |
| 2026-07-09 | `1b8b23cb` | `mamba_whole_graph_m` **A** both | SDP | **80.3** | 81.9 | 74.3 | 72.8 | **359** | ~352 | m capacity A best IDF1; keep-both still OK | same §8.3 |
| 2026-07-09 | `1b8b23cb` | `mamba_whole_graph_m` **B** cost-only | SDP | 78.3 | 81.9 | 72.5 | 70.3 | 401 | — | Bid strongly needed on m | same §8.3 |
| 2026-07-09 | `1b8b23cb` | `mamba_whole_graph_m` **C** bid-only | SDP | 80.1 | 81.7 | **74.5** | **73.2** | 360 | — | **Cost additivity weak/near-noise** on m (A−C≈0.2 IDF1) | same §8.3 |
| 2026-07-09 | `1b8b23cb` | `mamba_whole_graph_m` **D** both-off | SDP | 78.6 | 82.0 | 73.0 | 70.5 | 400 | — | Both-off loses vs A/C on IDF1/AssA | same §8.3 |

**Host for 2026-07-09 rows:** `DESKTOP-0FLA6SQ`, RTX 5070 Ti Laptop, double-buffer.  
**FPS** for P7 rows = overall throughput from eval logs (not frozen_v2 protocol).  
**Em-dash MOTA/HOTA** on older m baseline: source table incomplete; use linked doc.

---

## Decision outcomes (not raw metrics)

| Date | Topic | Decision | Source |
|:--|:--|:--|:--|
| 2026-07-09 | Dual height stability | **Keep both** (`stability_cost_w=0.20` + env bid 0.1); no behavior PR | [results](tracker-decision/audit/dual_stability_ablation_results_2026-07-09.md) |
| 2026-07-09 | m cost additivity | **Weak / near-noise** when bid on; does not flip global default | same §8.6 |
| 2026-07-09 | Guardrails P0–P6 | Contract C1–C9 + CI checker + NO-GO process | [status](tracker-decision/status_2026-07-09.md) |
| ongoing | NO-GO knobs (fuse, NSA, OAO spatial, …) | Stay off headline; promotion bar = 7-seq + bipolar | [no_go_guardrails.md](tracker-decision/audit/no_go_guardrails.md) |

---

## How to add a row

```text
1. Run with pinned SHA, preset, detector, double-buffer flag.
2. Record host/GPU.
3. Fill metrics from OVERALL (and note if per-seq only).
4. One-sentence conclusion (what decision it supports).
5. Link source markdown (or PR); prefer not pasting only into chat.
```

**Do not** add dual-stability behavior flips to this ledger without a new measured
row set and an explicit contract update.
