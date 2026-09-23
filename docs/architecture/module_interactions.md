<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-21 -->
<!-- doc-module: cross -->

# Module interactions — present-day roles in the production full stack

> **What this file is.** The relation semantics, interaction map and one-line
> present-day role of every mechanism that is **on** in the current production
> configuration (`mamba_whole_graph`, ReID off). It answers "does X still earn
> its place *now that everything else is present*", not "why was X added".
>
> **What this file is not.** It owns no benchmark numbers. Every quantitative
> statement points at an existing artifact; the full-stack subtraction evidence
> behind the roles is
> [`reference/benchmarks/module_interactions_20260921.md`](../reference/benchmarks/module_interactions_20260921.md)
> (#418). Knob cards stay in
> [tracker-decision/assoc_knobs.md](../research/tracker-decision/assoc_knobs.md),
> the transcribed formulas in [reference/math_model.md](../reference/math_model.md),
> production values in [reference/mot17_default_config.md](../reference/mot17_default_config.md).
> Issue: [#418](https://github.com/raylei50653/saccade/issues/418). The
> training-regime axis is #423, scene conditioning is #425; both cite this map
> rather than re-deriving it.

---

## 1. Relation vocabulary

Relations are stated **conditional on the current full stack** (`FULL`). A
relation is read from full-stack removal evidence, never from historical
additive deltas (`base + A`), because those were measured on stacks that no
longer exist ([frozen_v2_ablation §3](../reference/benchmarks/frozen_v2_ablation.md)
already showed the deltas are non-additive).

| Relation | Meaning (all conditional on `FULL`) | Read from |
|:--|:--|:--|
| `independent_gain` | Removing A alone materially lowers quality; A's contribution does not depend on any tested partner | `FULL-A` material, pairs additive |
| `complement` | A and B each contribute, and A's contribution is larger when B is present (`I(A,B) > 0`) — one enables or sharpens the other | pair `FULL-A-B` |
| `prerequisite` / `enabler` | B is structurally required for A to act (A has no code path without B); removing B removes A | code path, not measurement |
| `substitute` / `redundant` | Removing A alone or B alone is near-neutral, removing both is not (`I(A,B) < 0`): they cover for each other | pair `FULL-A-B` |
| `orthogonal` | Removing A and B together costs the sum of removing each (`I ≈ 0`) | pair `FULL-A-B` |
| `performance_only` | Output is bit-identical with and without A; A changes only throughput/latency | MOT md5 equality + fps |
| `safety_only` | Output and speed unchanged in the benchmark; A exists for an invariant or fallback that the benchmark does not exercise | code path + documented invariant |
| `conflict` | Removing A alone materially *raises* quality in the current stack | `FULL-A` material positive |
| `dominated` | Removing A alone is neutral and stays neutral whether or not its suspected partner is present | `FULL-A ≈ FULL` and `FULL-A-B ≈ FULL-B` |
| `parked` | Not on in `FULL`; governed by the NO-GO registry or a closed research line | [no_go_registry](../reference/no_go_registry.md) |

**Role labels** (one per component, §4) are derived from the relations under the
pre-declared rules in the harness (`scripts/benchmarks/module_interactions/subtraction.py`
`RULES`): `essential` (independent material loss), `complementary`, `enabler`,
`runtime-only`, `safety-only`, `redundant/substitutable`, `dominated`,
`trade-off` (material loss on one axis, material gain on another), `parked`.
Materiality is the evidence ledger's decision-knob guidance (|Δ| > 0.2 on
IDF1/HOTA/AssA/MOTA or |ΔIDs| > 10, resolved against same-session repeat ranges);
it is a reading threshold, not a significance test.

---

## 2. What is on in `FULL`

Production identity: `configs/presets/mamba_whole_graph.yaml` + the C++ env
defaults below + eval scheduling flags. Anything not listed here is off in
production (see §6 for the parked surface).

| Component | Family | Switch used for subtraction | Where it acts | Formula / card |
|:--|:--|:--|:--|:--|
| GMC (GPU phase-correlation camera motion) | motion | `--no-gmc` | tracker predict input (warp) | [math_model §5](../reference/math_model.md#5-gmccamera-motion-compensation) |
| `kalman_r_scale = 2.8` | motion | `--kalman-r-scale 1.0` | Kalman measurement noise | [math_model §6](../reference/math_model.md#6-kalman-模型) · [assoc_knobs](../research/tracker-decision/assoc_knobs.md) |
| occ_state (same-height front latch) | occlusion | `--no-occ-state-enabled` | association cost term (front-occluder) | [math_model §7.6](../reference/math_model.md#76-front-occluder-cost) |
| OAO penalty (`oao_tau = 0.50`) | occlusion | `--oao-tau 0` | association cost term | [math_model §7.4](../reference/math_model.md#74-oao-penalty) · [revival note](../research/eval/oao_duration_ramp_revival_20260617.md) |
| OAO duration ramp (`oao_ramp_frames = 25`) | occlusion | `--oao-ramp-frames 0` | scales the OAO term by overlap duration | same |
| Multiplicative (log-linear) cost form | association | `--no-multiplicative-cost` | cost composition | [math_model §7.3](../reference/math_model.md#73-multiplicative-cost) |
| Stability cost (`stability_cost_w = 0.20`) | association | `--stability-cost-w 0` | reward term inside the multiplicative cost | [math_model §7.7](../reference/math_model.md#77-stability-reward) · [dual-stability results](../research/tracker-decision/audit/dual_stability_ablation_results_2026-07-09.md) |
| Stability bid (`SACCADE_STABILITY_W`, default 0.1) | association | env `SACCADE_STABILITY_W=0` | auction bid | [math_model §8.2](../reference/math_model.md#82-auction-bid) · same |
| DDA S0 stage (`SACCADE_ENABLE_DDA`, cap 0.12) | association | env `SACCADE_ENABLE_DDA=0` | unambiguous pre-stage before the cascade | [math_model §8.1](../reference/math_model.md#81-stage-definitions) |
| Bridge relink (bidirectional foot-bridge) | recovery | `--no-relink-bridge-enabled` | tracker-core lost↔new identity relink | [math_model §10](../reference/math_model.md#10-bridge-relink-模型) · [semantic README](../modules/semantic/README.md) |
| Bridge reciprocal margin (`0.05`) | recovery | `--relink-bridge-margin 0` | rejects ambiguous bridges | [math_model §10.5](../reference/math_model.md#105-gates-and-commit) |
| Bridge direction bonus (`0.8`, s only) | recovery | `--relink-bridge-dir-bonus 0` | widens px gate for direction-consistent pairs | [math_model §10.4](../reference/math_model.md#104-direction-bonus) |
| Private continuation | recovery | `--no-private-continuation` | association input set (track-gated wider-NMS boxes) | [math_model §9.1](../reference/math_model.md#91-private-continuationinput-set-policy) |
| Tracklet interpolation (`max_gap 35`) | recovery | `--no-interpolate-tracklets` | output layer (offline gap fill) | [math_model §12](../reference/math_model.md#12-output-與-offline-postprocessing) |
| Whole-detect CUDA graph | runtime | derived config `use_whole_graph: false` | detect stage scheduling | [mamba_whole_graph_analysis](../modules/detection/mamba_whole_graph_analysis.md) |
| Head CUDA graph | runtime | derived config `use_cuda_graph: false` (with whole graph off) | detect head | same |
| Tracker CUDA graph | runtime | derived config `use_tracker_graph: false` | tracker update | [tracker_lane_dose_response](../reference/benchmarks/tracker_lane_dose_response_20260907.md) |
| Graphed main NMS (`main_nms_graphed`) | runtime | `--no-main-nms-graphed` | postprocess NMS | issue #56 · [production_pipeline_code_map](../reference/production_pipeline_code_map.md) |
| Double-buffer scheduling | runtime | `--double-buffer` (FULL is serial) | detect(N+1) ‖ tracker(N) | [#419 closure](../reference/benchmarks/resource_scaling_closure_20260920.md) · [frozen_v2 §4](../reference/benchmarks/frozen_v2_ablation.md) |

Not subtracted (no meaningful "off"): match/birth/confirm thresholds,
`sinkhorn_lambda`, the auction itself, detector/postprocess. These are the
substrate, not optional mechanisms.

---

## 3. Interaction map

Edges are read from the #418 full-stack subtraction record
(`FULL` = `mamba_whole_graph` serial, 7-seq SDP; every row two runs, one
distinct output; deltas are variant − FULL and all quoted deltas resolved above
the repeat range). Numbers below are pointers into that record, not a second
owner.

### 3.1 Single removals (Δ(A | FULL∖A))

```text
                       IDF1   HOTA   AssA   IDs      reading
motion      GMC        −4.9   −3.3   −4.6   +358     independent_gain (largest single loss)
            KalmanR    −1.0   −0.6   −1.0   +51      gain conditional on GMC (3.2)
occlusion   OccState   −1.4   −1.1   −2.2   +19      gain conditional on OAO (3.2)
            OAO        −1.6   −1.3   −2.2   +9       gain conditional on OccState (3.2); FP +936
            OAORamp    −0.8   −0.5   −1.1   +37      independent_gain at tau 0.50 (confound named by REF-OAOPlain025: −2.2)
association MultCost   −4.4   −2.2   −4.2   +126     independent_gain; carries the retune confound; also removes StabCost (enabler)
            StabCost   −0.7   −0.4   −0.9   −1       gain conditional on StabBid (3.2)
            StabBid    −1.0   −0.5   −1.0   +71      independent_gain
            DDA        =      =      =      =        output bit-identical → no quality role (runtime row, 3.3)
recovery    Bridge     −2.7   −1.4   −2.0   +94      independent_gain; FP +1292 when off
            BridgeMargin +0.1  =      =     −3       no material role (guard never pays on this benchmark)
            BridgeDirBonus −0.5 −0.3  −0.5  +11      independent_gain (s only; m ships 0)
            PrivCont   −0.3   =      −0.4   −10      marginal (at the materiality edge); FP −789 / FN +516 when off
            Interp     −0.7   −1.0   −0.8   +380     independent_gain; the IDs/FN owner of the output layer
```

### 3.2 Pair removals (I(A,B) = M(FULL) − M(FULL−A) − M(FULL−B) + M(FULL−A−B))

```text
pair                 FULL−A   FULL−B   FULL−A−B   I(IDF1)  I(HOTA)  relation
OccState × OAO       76.9     76.7     76.8       +1.5     +1.2     complement (mutual dependency): removing the second is free
StabCost × StabBid   77.6     77.3     77.3       +0.7     +0.6     complement: cost pays only while the bid is on
GMC × KalmanR        73.4     77.3     73.3       +0.9     +1.2     complement: R scaling pays only while GMC is on
Bridge × Interp      75.6     77.6     75.4       +0.5     +0.5     complement on IDF1/HOTA; on IDs closer to additive (+94 / +380 / +540)
Bridge × PrivCont    75.6     78.0     74.6       −0.7     −0.7     substitute / backup: PrivCont covers 1.0 IDF1 without bridge, 0.3 with it
```

Reading the positive-`I` pairs: each member's marginal value collapses when
its partner is absent, so the pair is a **unit** — neither member is
redundant, but neither should be evaluated or retuned alone. This is the
full-stack counterpart of the additive observation in
[frozen_v2_ablation §3](../reference/benchmarks/frozen_v2_ablation.md) (occ-gate
and OAO were negative until GMC + bridge were present).

### 3.3 Runtime mechanisms (output identity + serial fps)

```text
toggle                 output identical   serial fps vs FULL          relation
DoubleBuffer           yes                +52 % (297 vs 196)           performance_only (headline scheduling)
TrackerGraph           yes                −16 % when off (vs late control) performance_only
NMSGraph               yes                −18 % when off               performance_only (eager fallback stays wired for ONMS-prior frames)
WholeGraph             NO (68.7 IDF1)     −73 % when off              NOT a pure runtime toggle in this preset: the whole-graph forward is
                                                                      also what bypasses the checkpoint's temporal blocks (see 4, 5)
WholeGraph (T pinned)  metric-identical   −39 % when off (−44 % eager) pure scheduling subtraction (`--no-temporal`); md5 differs by 1e-4 score rounding
DDA                    yes                unresolved (192.8 vs 195.5)  no quality role and no resolved runtime role
```

Serial fps drifted downward during the campaign; the record carries a late
same-session `FULL` control and only reads a runtime delta when the fps
ranges do not overlap. Double-buffer throughput ownership stays with
[#419 closure](../reference/benchmarks/resource_scaling_closure_20260920.md)
and [frozen_v2 §4](../reference/benchmarks/frozen_v2_ablation.md).

---

## 4. Present-day role, one line each

| Component | Role | Present-day contribution · dependency · justification | Evidence |
|:--|:--|:--|:--|
| GMC | **essential** | Largest single loss when removed (−4.9 IDF1, IDs ×1.8); prerequisite for the occlusion pair and for R scaling to pay | record §single, §pair GMC×KalmanR; [frozen_v2 §3](../reference/benchmarks/frozen_v2_ablation.md) |
| `kalman_r_scale` 2.8 | **complementary** (to GMC) | −1.0 IDF1 alone; contributes ~0 once GMC is off → a motion-trust setting for the GMC-compensated residual, not a standalone smoother | record §pair; [kalman_h_recalibration](../research/eval/kalman_h_recalibration_20260612.md) (2.2/3.4 both worse) |
| occ_state | **complementary** (with OAO) | −1.4 IDF1 alone; removing it after OAO is free → the two occlusion cost terms act as one unit | record §pair OccState×OAO |
| OAO penalty | **complementary** (with occ_state) | −1.6 IDF1 alone (FP +936 when off); same unit as occ_state | record §pair; [revival note](../research/eval/oao_duration_ramp_revival_20260617.md) |
| OAO duration ramp | **essential** (within OAO) | −0.8 IDF1 at the ramped tau; the plain-OAO operating point (tau 0.25) is −2.2 → the ramp is what makes tau 0.50 safe | record §single + `REF-OAOPlain025` |
| Multiplicative cost form | **enabler** | −4.4 IDF1 when reverted, but the row carries a retune confound and removes the stability cost with it; its justified role is structural (hosts the reward term), not the −4.4 | record §single (confound named) |
| Stability cost 0.20 | **complementary** (to stability bid) | −0.7 IDF1 with the bid on, 0 with the bid off; same direction as the 2026-07-09 4-way | record §pair; [dual-stability results](../research/tracker-decision/audit/dual_stability_ablation_results_2026-07-09.md) |
| Stability bid 0.1 | **essential** | −1.0 IDF1 / +71 IDs alone; the bid, not the cost, carries the dual-stability gain | record §single |
| DDA S0 stage | **redundant/dominated** | Output bit-identical with the stage off (every S0 match is re-made by S1) and serial fps within the control range → **simplification candidate** | record §single, §runtime |
| Bridge relink | **essential** | −2.7 IDF1 / +94 IDs / +1292 FP alone; partially backed up by private continuation (I = −0.7) | record §single, §pair Bridge×PrivCont; [bridge gate evidence](../reference/benchmarks/bridge_gate_cross_dataset_20260808.md) |
| Bridge reciprocal margin 0.05 | **safety-only** | Removing it is +0.1 IDF1 / −3 IDs (below materiality): the guard rejects no useful bridge on MOT17 train; retained only as the documented ambiguity guard, **simplification candidate** if no other benchmark exercises it | record §single |
| Bridge direction bonus 0.8 (s) | **essential** (s) | −0.5 IDF1 alone; m ships 0.0, so the role is preset-specific | record §single; [assoc_knobs](../research/tracker-decision/assoc_knobs.md) |
| Private continuation | **complementary/backup** (to bridge) | −0.3 IDF1 alone (edge of materiality; FP −789 / FN +516 trade); worth 1.0 IDF1 when the bridge is off → a recall-side backup, not an independent gain | record §single, §pair |
| Tracklet interpolation | **essential** (output layer) | −0.7 IDF1 / −1.0 HOTA / +380 IDs alone; its known FP cost is upstream-bound | record §single; [no_go #44](../reference/no_go_registry.md) |
| Double-buffer | **runtime-only** | +52 % serial→DB throughput here, output identical | record §runtime; [#419 closure](../reference/benchmarks/resource_scaling_closure_20260920.md) |
| Tracker CUDA graph | **runtime-only** | −16 % serial fps when off, output identical | record §runtime |
| Graphed main NMS | **runtime-only** | −18 % serial fps when off, output identical; eager path is the ONMS-prior fallback (safety) | record §runtime; issue #56 |
| Whole-detect CUDA graph | **runtime-only + implicit model selector** | −39 % serial fps when off with T pinned (metric-identical output); but in this preset it is also the only thing that forces T=1 through a temporal checkpoint (−73 % fps and IDF1 68.7 without the pin) → a **coupling to fix**, see §5 | record §runtime (`RT-NoWholeGraph` vs `-T1`) |
| Head CUDA graph | **dominated** (by whole graph) | Superseded; only reachable when the whole graph is off | record §runtime |

---

## 5. Simplification / removal candidates and follow-ups

Nothing is removed or refactored in #418; each item below is a separate
follow-up with its own evidence bar.

1. **DDA S0 stage** — no output effect and no resolved fps effect in `FULL`;
   retire the env knob and the stage (`SACCADE_ENABLE_DDA`, `dda_max_cost`)
   after a double-buffer timing check, since tracker-lane work is
   shape-sensitive ([tracker_lane_dose_response](../reference/benchmarks/tracker_lane_dose_response_20260907.md)).
2. **Bridge reciprocal margin** — no material effect on MOT17 train; either
   demonstrate a case it guards on another benchmark (MOT20 / DanceTrack runs
   already exist for the gate study) or drop it from the preset and the knob
   surface.
3. **Whole-graph ↔ temporal coupling** — `use_whole_graph: false` silently
   activates the checkpoint's T=3 streaming path (IDF1 68.7). The preset
   should pin `temporal_T_override=0` / `--no-temporal` explicitly so that
   the scheduling choice and the model forward are separate switches.
4. **Multiplicative cost** — the −4.4 IDF1 reversion row is not a fair
   subtraction (knob retune confound). If anyone wants the additive form
   back, it is a retune study, not a removal.
5. **Occlusion pair, motion pair, stability pair** — treat each as one unit in
   #423/#425: leave-one-out inside a unit measures the unit, not the member.

---

## 6. Parked surface (off in `FULL`)

Listed so that nobody re-reads them as part of the stack. Each is governed by
the NO-GO registry or a closed line; this audit did not run them.

| Mechanism | Why off | Owner |
|:--|:--|:--|
| ReID / appearance bank / semantic relink | geometry-first production decision; revival rules in the registry | [no_go #2 #3 #4](../reference/no_go_registry.md) · [ADR 018](../decisions/018-project-main-line-direction.md) |
| NSA-Kalman (`kalman_adapt_mode`), gating/output decouple | blocked / not actionable | [no_go #8 #50](../reference/no_go_registry.md) |
| Velocity-direction penalty (`vel_dir_weight`) | blocked | [no_go #21](../reference/no_go_registry.md) |
| `fuse_score_weight` | not actionable | [no_go #45](../reference/no_go_registry.md) |
| Freshness bid (`SACCADE_FRESHNESS_W`) | harmful | [no_go #42](../reference/no_go_registry.md) |
| Predict-through-occlusion coast, occ-gated velocity damping | harmful | [no_go #49 #51](../reference/no_go_registry.md) |
| OAO spatial variants (contest/crowd/height/foot gates) | superseded by the duration ramp | [no_go #7](../reference/no_go_registry.md) |
| Lifecycle merge, post-lifecycle merge, multi-birth, birth/stage2 quality gates | no actionable gain / harmful | [no_go #9 #11–#14](../reference/no_go_registry.md) |
| Cheb-GR merge / offline handover / OWDL | closed research lines, sealed or parked | [semantic module](../modules/semantic/README.md) · [claim state registry](../research/contracts/claim_state_registry.md) |
| GMC foreground mask, tile-PCR, residual correction | not actionable | [no_go #20 #34 #40](../reference/no_go_registry.md) |
| `person_geometry_prior`, `detection_quality_scaling`, `geometry_suspect_support`, `id_stability_filter`, `per_seq_adapt`, detection cap | off in preset; registry / legacy | [mot17_default_config §2](../reference/mot17_default_config.md) · [no_go #10](../reference/no_go_registry.md) |
