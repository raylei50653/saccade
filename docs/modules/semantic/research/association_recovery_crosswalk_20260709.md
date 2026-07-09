# Association recovery — experiment ↔ implementation crosswalk

<!-- doc-status: research-synthesis -->
<!-- doc-promotion: not-for-report-citation-yet -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**Purpose:** one对照 sheet so a new AssA / identity experiment can align to **existing
docs, knobs, code paths, NO-GO, and frozen substrates** without re-deriving the
stack. This file is a **map**, not a second baseline and not a long research report.
It is **not** the semantic sole-active experiment (that remains #55 until parked);
status `research-synthesis` means D1 docs-only navigation, not an open workstream.

**Hub for offline geometry discriminability:**
[offline_relink_candidate_analysis.md](offline_relink_candidate_analysis.md) (§0).

**Scripts lookup (path / recipe / wrapper):**
[association_recovery_scripts_index_20260709.md](association_recovery_scripts_index_20260709.md)
— use that file for *which CLI to open*; this crosswalk for *whether / how to experiment*.

**Information source contract (who is truth; what scripts may generate):**
[association_recovery_info_source_contract_20260709.md](association_recovery_info_source_contract_20260709.md)
— Step 0 sealed. **R** registry: [association_tools.yaml](association_tools.yaml)
(`registry_status: populated`, Step 1B). Checker:
`scripts/tools/check_association_tools.py` (Step 2). Maps remain curated **H**
beside **R**; knobs stay subordinate to preset/schema (**C**).

**Governance entry:** [DEVELOPMENT.md](../../../../DEVELOPMENT.md) D1–D2 ·
[module TODO](../TODO.md) (WIP=1 sole active) · [no_go_registry](../../../reference/no_go_registry.md).

**Numbers:** absolute headline metrics → [docs/TODO.md](../../../TODO.md);
citable decision rows → [evidence_ledger](../../../research/evidence_ledger.md).
Do **not** copy metric tables into this file. Not for external report citation until
promotion is upgraded (ledger / report_data) with explicit fact-owner rows.

---

## 0. How to use (30-second protocol)

Before coding or sweeping:

1. **Skim the production stack** (§0.5) so the experiment is placed on the live path, not beside it.
2. **Pick the door** (table §1) — death/birth bridge vs live swap vs post-hoc ID cleanup vs bank substrate.
3. **Read the fact-owner research** + **NO-GO ids** for that door (do not re-open a closed lever without a *new* hypothesis that invalidates the old failure mode).
4. **Touch only the listed knobs / files** — CLI schema defaults ≠ preset production values (§2).
5. **Choose control substrate** (§5): prefer frozen MOT txt when the lever is post-process; live re-run only when the lever is in the tracker critical path.
6. **Respect WIP=1:** semantic sole active is #55 occ-exit until parked; other lines stay doc/probe-only or park first. This crosswalk does **not** count as sole active.
7. **Promotion bar:** default-off until cross-seq non-negative and noise-aware (≲0.2–0.3 pp IDF1 is noise band on many knobs).

```text
idea → production stack (§0.5) → door → research + no_go → knobs/code → control substrate → default-off A/B → (optional) ledger/no_go
```

---

## 0.5 Production association stack

Current production identity recovery is **geometry-first** (headline
`mamba_whole_graph*` · `reid_mode=off`). Narrative only — knobs in §2, code in §3:

```text
detector + GMC substrate
  → auction association
  → bridge relink for death/birth gaps
  → occ-state / same-height / OAO for live swap pressure
  → ReID remains off the sync critical path
  → Cheb-GR is offline/post-hoc cleanup, not live feedback
  → occ-exit / sparse bank remain conditional research lines
  → NO-GO registry prevents reopening settled levers
```

If a new idea does not attach to a stage above (or explicitly replaces one), write
why before sweeping. One-page paper/README prose, if needed later, belongs in a
separate thin `association_recovery_mainline.md` — this file stays the experiment
alignment map.

---

## 1. Door map (what problem are you attacking?)

| Door | Failure mode | Production today | Research fact-owner | Status |
|:--|:--|:--|:--|:--|
| **A — Bridge relink** | track death → later birth (gap) | GPU bidir bridge **default ON** (preset) | [offline_relink_candidate_analysis.md](offline_relink_candidate_analysis.md) §1–§6f | GO; hard-pool geometry weak (~0.65 AUC) |
| **B — Live crossing-swap** | mutual occlude 1–2f → id swap | `occ_state_*` **ON** + OAO ramp; depth probe GO signal | [depth_ordering_crossing_swap.md](depth_ordering_crossing_swap.md); offline §8 | Signal real; **#39** warns some hooks overfit — re-read before reopening |
| **C — Appearance online** | hard look-alikes | **ReID off** on headline | appearance ceiling + #2/#32/#35/#48/#57 | **NO-GO sync critical path** |
| **D — Post-hoc ID cleanup** | output-layer wrong merges | optional flags; not headline | Cheb-GR offline GO + [chebgr_handover_signal_map_20260704.md](chebgr_handover_signal_map_20260704.md) | GO offline; live claims **#56 NO-GO** |
| **E — Bank / sparse emb** | extraction cost / clean refs | Python bank + offline FIFO config | [clean_fifo_bank_substrate_20260704.md](clean_fifo_bank_substrate_20260704.md); sparse bank note | Substrate GO; C++ async **parked** (#57) |
| **F — Occ-exit audit** | ABSORB wrong id after occlude | default-off flags | #55 scope + WP2/WP3 | Probe≠runtime; WP3 net harm; **split_feat_pr** only |

Sibling investigation index (longer): offline hub §0 table.

---

## 2. Production contract vs schema defaults

**Always diff the preset you claim to run.** CLI/dataclass defaults in
`scripts/eval/config/lifecycle.py` often leave bridge **off** (`relink_bridge_enabled: false`);
headline presets turn it **on**.

| Knob family | `mamba_whole_graph` (s) | `mamba_whole_graph_m` (m) | Schema default (no preset) |
|:--|:--|:--|:--|
| `relink_bridge_enabled` | true | true | **false** |
| `relink_bridge_px` | **0.25** | **0.4** (m small-obj) | 0.25 |
| `relink_bridge_margin` | 0.05 | 0.05 | 0.0 |
| `relink_bridge_h_lo` / `h_hi` | 0.75 / 1.33 | **0.6 / 1.7** | 0.0 / 0.0 (off) |
| `relink_bridge_spatial_gate` | 0.0 | 0.0 | 0.0 |
| `relink_bridge_dir_bonus` | **0.8** | **0.0** (do not copy s) | 0.0 |
| `reid_mode` | off | off | off |
| `occ_state_enabled` + `occ_foot_gap` | true / 0.15 | true / 0.15 | see geometry schema |
| `oao_tau` / `oao_ramp_frames` | 0.50 / 25 | 0.50 / 25 | 0 / 0 |
| `interpolate_tracklets` | true (max_gap 35) | true | — |

**Preset files (source of truth for production values):**

- `configs/presets/mamba_whole_graph.yaml`
- `configs/presets/mamba_whole_graph_m.yaml`

**CLI / argparse owners:**

- Bridge + relink + Cheb-GR offline + occ-audit → `scripts/eval/config/lifecycle.py`
- OAO / occ geometry → `scripts/eval/config/geometry.py`
- Human-readable default narrative → [mot17_default_config](../../../reference/mot17_default_config.md)

**Do not A/B bridge knobs on bare schema defaults and call it “baseline.”**

---

## 3. Implementation hot paths (edit / read here)

### 3.1 Live tracker (critical path)

| Concern | Primary files | Notes |
|:--|:--|:--|
| Bridge propose/commit | `src/tracking/tracker_gpu.cu` (`relink_bidir_propose_kernel` and friends) | Greedy, **not** auction |
| Bridge score / gates (GPU + bind) | `src/tracking/relink_gate.cu`, `tracker_gpu_python.cpp` | Keep three-path parity if scoring changes (GPU / C++ semantic / Python) |
| Python foot-bridge mirror | `src/saccade/perception/eval/relink.py` | Offline/CPU path; h_ref = avg EMA heights (§6e lesson) |
| Occ-state / OAO / foot gap | `src/tracking/tracker_gpu.cu` (`set_occ_params`, OAO same-height helpers) | Production `occ_*` + `oao_*` |
| Auction main assoc | `tracker_gpu.cu` auction kernels | Crossing-swap is **here**, not bridge |

### 3.2 Eval / post-process (not critical path)

| Concern | Primary files |
|:--|:--|
| Occ-exit audit (cosine + chebgr log) | `src/saccade/perception/eval/occ_audit.py`, `occ_audit_seq_conditioning.py` |
| Clean FIFO bank | `src/saccade/perception/eval/clean_fifo_bank.py` |
| Cheb-GR offline handover | `src/saccade/perception/eval/cheb_gr_online.py` (+ merge: `cheb_gr_merge.py`) |
| Evaluator wiring | `eval/evaluator.py` / lifecycle hooks (flags in `lifecycle.py`) |

### 3.3 Offline analysis tools (no tracker feedback)

**Full inventory + recipes + wrappers:**  
[association_recovery_scripts_index_20260709.md](association_recovery_scripts_index_20260709.md).

| Tool | Role | Doc |
|:--|:--|:--|
| `scripts/tools/build_relink_candidates.py` | Enumerate (lost→cand) pairs + GT | offline §2 |
| `scripts/tools/analyze_preloss_motion.py` | Area / turn pre-loss | offline §4 |
| `scripts/tools/analyze_turn_baseline.py` | Turn vs interior | offline §4 |
| `scripts/tools/sweep_speed_turn.py` | Speed×turn distribution | offline §5 |
| `scripts/tools/validate_reach_gate.py` | Reach / drift formulations | offline §6b |
| `scripts/tools/optimize_relink_weight.py` | Speed-weight grid + LOSO | offline §6c |
| `scripts/tools/depth_ordering_probe.py` / `_auc.py` / `_gate_sweep.py` | Front/back geometry | depth doc |
| `scripts/eval/diagnostics/cheb_gr_offline_handover_report.py` | Label handover candidates | signal map |
| `scripts/eval/diagnostics/compare_handover_summaries.py` | Cross-run feature drift | TODO / #58 |
| `scripts/eval/diagnostics/synthesize_handover_applicability.py` | stable-veto / condition-sensitive | #58 |
| `scripts/eval/diagnostics/probe_sparse_bank_equivalence.py` | FIFO vs dense bank | sparse bank |
| `scripts/eval/diagnostics/probe_occ_audit_bank_reference.py` | Bank ref vs post-hoc | #55 |
| `scripts/eval/diagnostics/analyze_occ_audit_seq_conditioning.py` | WP2 seq map | WP2 |
| `scripts/eval/diagnostics/run_occ_audit_wp3_promotion.py` | WP3 promotion metrics | WP3 |

### 3.4 Module YAML (optional overlays)

| Config | Use |
|:--|:--|
| `configs/modules/cheb_gr_offline_mnv4.yaml` | Offline handover GO point (h2 / margin 0.05) |
| `configs/modules/cheb_gr_offline_mnv4_fifo20.yaml` | recent-20 bank mode overlay |
| `configs/modules/reid_mnv4_*.yaml` | ReID paths — **not** headline; #57 applies if sync |

---

## 4. Lever checklist — before you invent a new one

### 4.1 Already settled (do not re-sweep without a new mechanism)

| Lever | Verdict | Pointer |
|:--|:--|:--|
| Additive velocity reach `s·G` | dead | offline §6b |
| Per-frame heading / velocity-direction lock on swaps | harmful | offline §8 |
| Hard-pool `bridge_dist` as precision carrier | weak (~0.65); base rate kills | offline §3 |
| `h_ref = lost-only` | regressed; use **avg** EMA | offline §6e |
| Spatial pre-filter alone under tight `bridge_px` | bit-identical / no gain | offline §6f |
| Sync ReID in tracker critical path | #57 NO-GO | semantic README |
| Live Cheb-GR streaming claims | #56 NO-GO (compounding feedback) | registry |
| Birth-time lost-bank appearance relink | NO-GO | semantic README |
| Offline tracklet merge (Cheb-GR path2) | AssA 0.0pp | semantic README |
| occ_cover gap-path live relink | #33 | buffer kills long gap |
| Occ-gated appearance-only relink | #48 | clean subset tiny + geometry saturated |
| Predict-through-occlusion coast | #51 | FP dominated |
| Global occ-exit cosine audit | #55 / WP3 net harm | WP3 |

### 4.2 Production-positive geometry (change carefully)

| Lever | Where | Doc |
|:--|:--|:--|
| Speed-weighted bridge score + margin + scale gate | preset bridge knobs | offline §6c–§6f |
| Occ-state foot-gap latch | `occ_*` in preset | depth + tracker-decision contract |
| OAO duration ramp | `oao_tau` / `oao_ramp_frames` | preset comments; spatial OAO variants failed |

### 4.3 Conditional / research-only

| Lever | Rule |
|:--|:--|
| Depth-ordering *formulations* beyond current `occ_state` | Re-read **#39** (signal real, some hooks seq-overfit) + depth doc |
| Cheb-GR offline accept rules (`best_cost` etc.) | Descriptive frontier; **not** free default gates — signal map |
| Occ-exit seq allowlist | WP3: only 1 enable_candidate (11); needs **feat/** design PR, not silent merge |
| Sparse bank C++ async | Parked; requires async sidecar project (#57) |

---

## 5. Control substrates (reduce eval noise)

| Goal | Recommended control | Notes |
|:--|:--|:--|
| Offline pair / AUC / gate design | relink-OFF interp-OFF dump + `relink_candidates.csv` | offline §1–§2; `mot17.py --preset mamba_whole_graph --detector SDP --no-interpolate-tracklets` (as in offline doc) |
| Post-process only (handover, occ-audit) | Frozen MOT dirs under `results/diag_*` | e.g. `results/diag_m_no_reid_current_20260704` used by WP3 / #58; **local/gitignored** |
| Live tracker knob A/B | Same preset + single knob flip; smoke `MOT17-04-SDP` then 7-seq | D3 if default changes |
| Cross-condition identity claims | At least two genuine regimes (m vs s backbone, or confirm-score regime) | FRCNN suffix is **not** a detector condition (structural) — see semantic TODO #58 notes |

**Repro anchors (local artifacts; may be missing on a clean clone):**

```text
results/diag_m_no_reid_current_20260704          # m no-ReID frozen
results/diag_s_no_reid_current_20260704          # s no-ReID (if present)
results/diag_m_chebgr_offline_mnv4_h2_m005_log_20260704
results/occ_exit_p55_wp3/                        # WP3 treatment + maps
scripts/tools/out/relink_candidates.csv          # offline candidate pool
```

If missing: regenerate from the owning research note’s command block; do not invent metrics from chat.

---

## 6. Research notes index (semantic AssA family)

| Status | File | One-line role |
|:--|:--|:--|
| hub / closed research | [offline_relink_candidate_analysis.md](offline_relink_candidate_analysis.md) | Door A discriminability + kinematics + bridge online path |
| closed | [bidir_relink_data_analysis.md](bidir_relink_data_analysis.md) | Online hard-pool residual ~0.55 (reconciled in offline intro) |
| closed | [bidirectional_relink_roadmap.md](bidirectional_relink_roadmap.md) | Design history / phases |
| closed | [relink_normalization_gate_analysis.md](relink_normalization_gate_analysis.md) | Scale / normalization gates |
| closed | [depth_ordering_crossing_swap.md](depth_ordering_crossing_swap.md) | Door B geometry signal |
| closed | [chebgr_handover_signal_map_20260704.md](chebgr_handover_signal_map_20260704.md) | Door D signal frontier |
| closed | [sparse_key_embedding_bank_20260704.md](sparse_key_embedding_bank_20260704.md) | Sparse ≡ dense; FIFO-20 |
| closed contract | [clean_fifo_bank_substrate_20260704.md](clean_fifo_bank_substrate_20260704.md) | Bank API + hard constraints |
| closed | [online_sparse_reid_handoff_20260704.md](online_sparse_reid_handoff_20260704.md) | Async/sidecar handoff notes |
| active series | [occ_exit_audit_p55_scope_20260709.md](occ_exit_audit_p55_scope_20260709.md) | #55 scope |
| active series | [occ_exit_audit_p55_wp2_seq_conditioning_20260709.md](occ_exit_audit_p55_wp2_seq_conditioning_20260709.md) | Seq labels |
| active series | [occ_exit_audit_p55_wp3_promotion_decision_20260709.md](occ_exit_audit_p55_wp3_promotion_decision_20260709.md) | Promotion = split_feat_pr |
| **this file** | association_recovery_crosswalk | Experiment ↔ code对照 |

Module card GO/NO-GO table: [../README.md](../README.md).

---

## 7. NO-GO ids most relevant to AssA work

| ID | Topic | Open only if… |
|:--|:--|:--|
| #2 / #32 / #35 | Appearance hard-pool ~0.5 | New embedding domain or new pool definition with proof |
| #31 | Bridge scale gate *speed direction* | Different mechanism than rejected formulation |
| #33 | occ_cover live relink | Buffer / long-gap structural change first |
| #39 | Depth hook overfit | New cross-seq consistent hook; do not re-sweep peak ttl/w on s only |
| #43 | Auction stability / Mahalanobis ID | New identity cue, not motion tie-break alone |
| #48 | Occ-gated appearance relink | Contradicts clean-subset saturation evidence |
| #49 / #51 | Velocity damp / predict coast | Non-FP-dominated recovery design |
| #55 | Occ-exit global audit | Seq-conditioned **feat** design with isolatable harm |
| #56 | Live Cheb-GR claims | Offline-only or non-feedback path |
| #57 | Sync ReID critical path | Async drop-if-late with ≥~0.7–1.0 IDF1 bar |
| #58 | Bank mean/dup/quality-select | Respect hard constraints in clean_fifo doc |

Full table: [no_go_registry.md](../../../reference/no_go_registry.md).

---

## 8. Minimal experiment templates

### 8.1 New bridge geometric score (Door A)

```text
1. Offline: build_relink_candidates on fixed substrate → AUC full + hard (bd≤1) + AP/LOSO
2. Reject if hard-pool no better than dist_h / existing blend (offline §6c table)
3. Wire default-off; A/B on mamba_whole_graph SDP 7-seq (not schema defaults)
4. Keep h_ref=avg; do not drop margin/scale without ablation
5. Write research note; index README; no_go or ledger only if citable
```

### 8.2 New live association term (Door B)

```text
1. Define event population (swap / ABSORB) with existing probes
2. Check against depth/occ_state production terms (avoid double-counting)
3. Per-seq sign table required (#39 failure mode)
4. default-off; measure AssA + IDs; watch bridge interaction (state changes can eat bridges)
```

### 8.3 Appearance / graph (Door C–E)

```text
1. Offline-only or frozen MOT first
2. Bank: raw samples only; no mean into Cheb-GR graph
3. Live feedback path forbidden without #56-style bar
4. Sync extract in critical path forbidden without #57 bar
```

### 8.4 Occ-exit (Door F / #55)

```text
1. Read WP3 first — global cosine net negative
2. Frozen substrate protocol (WP3 method)
3. Seq gate design = separate feat/ PR; no silent preset
4. Cheb-GR columns today are log-only unless explicitly promoted
```

---

## 9. Out of scope for this sheet

- Detection training / VGT (detection module)
- GMC implementation details (geometry module) — only as substrate assumption
- Paper prose ([paper_outline](../../../research/paper_outline.md)) — link claims back here / ledger, do not duplicate
- Tracker-decision P0–P8 history (closed) — [status](../../../research/tracker-decision/status_2026-07-09.md) read-only

---

## 10. Maintenance

**Source of truth:** see [information source contract](association_recovery_info_source_contract_20260709.md).
Summary for editors of *this* file:

| What you are changing | Truth class | Do |
|:--|:--|:--|
| §2 knob *values* | **C** preset/schema | Quote from YAML/py; never invent a third table |
| Door / lever / NO-GO *lists* | **R** (planned) + **N**/**V** | Cite no_go / research; after Step 1 prefer registry |
| Metrics | **M** | Link TODO/ledger/note only |
| §0.5 stack prose | **H** | Manual narrative only |
| Path existence of tools | **D** | Prefer scripts_index / future checker |

- When a lever lands GO/NO-GO: add one row to §4 and hub offline §0 if AssA-related; update semantic README GO table if production-facing (**V**, not this map alone).
- When preset knobs change: update §2 **from** preset YAML (**C**), not from memory.
- When sole active changes: fix WIP line in §0 and [TODO.md](../TODO.md).
- This file stays a **pointer map**; long evidence stays in the fact-owner notes.
- **R** populated + checker landed (order: 0 → 1A → 1B → 2). Optional Step 3 full MD render still deferred.
