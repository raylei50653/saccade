# math_model.md Drift Audit (2026-07-09)

Static cross-check of [`docs/reference/math_model.md`](../../../reference/math_model.md) (source audit banner **2026-06-19**) against:

1. Headline presets `mamba_whole_graph` / `mamba_whole_graph_m` (post P1 explicit defaults)
2. `docs/research/tracker-decision/scoring_semantics.md` classifications
3. Python inject (`pipeline.py`) + schema (`scripts/eval/config/*`)
4. Native state / kernels (`tracker_gpu.cu`, `kalman_gpu.cuh`)

**Scope:** documentation drift only. No production code, preset value, or kernel changes in this audit.  
**Does not trigger P4** (no behavior change).

**Method:** code + YAML comparison only; no MOT17 runs.

Status legend:

| Status | Meaning |
|:--|:--|
| `MATCH` | Claim still true for production path |
| `DRIFT` | Claim false or inverted vs current inject/kernel |
| `STALE` | Incomplete / outdated framing; not strictly false for s-only |
| `NO-GO` | Claim correctly treats a rejected path (keep) |
| `UNKNOWN` | Needs deeper code read or runtime probe |

---

## Executive summary

| Bucket | Count (approx) | Headline |
|:--|:--|:--|
| `MATCH` | majority of §7 cost / §8 auction / s bridge scalars | Core multiplicative association math still good |
| `DRIFT` | **critical: occ_state** | math says front-occluder **off**; eval injects **on** |
| `STALE` | private continuation, m-variant, inject file map, line anchors | Gaps vs tracker-decision active contract |
| `NO-GO` | fuse_score, spatial OAO, NSA, ReID-off | Still correct |
| Production risk | **low for s scalars** | **medium for readers who believe occ is off** |

**Recommended next edit to math_model (separate PR, not this audit):**

1. Flip §7.6 / §3.1 occ baseline to **on** with explicit preset values  
2. Add private continuation to §1 baseline contract (input-set policy)  
3. Add `mamba_whole_graph_m` delta table (r_scale, bridge gates, dir_bonus)  
4. Point inject site at `pipeline.py` (keep evaluator as stage orchestrator)  
5. Bump source-audit date after those edits  

Do **not** merge dual stability without a design choice (A/B/C); math already documents both terms correctly.

---

## A. Baseline contract (§1) vs presets

### A1. s headline scalars

| Claim | Current implementation | Status | Evidence | Action |
|:--|:--|:--|:--|:--|
| `match_thresh: 0.50` | s/m YAML 0.50 | MATCH | presets | keep |
| `new_track_thresh: 0.28` | s/m 0.28 | MATCH | presets | keep |
| `kalman_r_scale: 2.8` | s=2.8; **m=3.5** | STALE | m preset | document m delta; keep s claim if scope = s only |
| `oao_tau / ramp: 0.50 / 25` | s/m same | MATCH | presets | keep |
| `multiplicative_cost / λ / stability_cost_w` | true / 10 / 0.20 | MATCH | presets + inject | keep |
| `reid_mode: off` | off | MATCH | presets | keep |
| `relink_bridge_enabled: true` | true | MATCH | presets | keep |
| `gmc: true, downscale 4, fg_mask false` | same | MATCH | presets | keep |
| PRESET-OFF geometry priors list | person/DQS/suspect/id_stability false | MATCH | presets | keep |
| **Private continuation on** | s/m **true** (post 2026-06-30) | **STALE** (omitted from §1) | presets | add to §1 contract |
| **occ_state on** | s/m explicit true (schema was already true) | **DRIFT** vs silence | presets + geometry schema | fix §7.6 (see B4) |
| Scope only `mamba_whole_graph` | m is second headline capacity path | STALE | m preset + pipeline runbook | add m subsection or pointer |

### A2. Bridge scalars (§10 / §3.1)

| Claim | Current | Status | Evidence | Action |
|:--|:--|:--|:--|:--|
| `bridge_px: 0.25` | s=0.25; m=**0.4** | STALE if “all headline” | presets | s MATCH; note m |
| `h_lo/h_hi: 0.75/1.33` | s same; m **0.6/1.7** | STALE | presets | note m |
| `margin: 0.05` | s/m 0.05 | MATCH | presets | keep |
| `dir_bonus: 0.8` | s=0.8; m=**0.0** explicit | STALE | presets (P1) | note m off |
| `spatial_gate: 0.0` | s/m 0 | MATCH | presets | keep |
| unit of `bridge_px` is heights not pixels | kernel uses h-normalized `bdist` | MATCH | `tracker_gpu.cu` bridge block ~2062–2095 | keep |

---

## B. Association cost / gates / weights (§7)

### B1. Candidate gate + enqueue cap

| Claim | Current implementation | Status | Evidence | Action |
|:--|:--|:--|:--|:--|
| `candidate ⇔ IoU>τ ∨ maha²<τ` else `c=1` | `stage1_cost_fused_kernel` same structure | MATCH | cu ~510–517 | keep |
| `c_max = max(DDA, match, stage2)`; enqueue iff `c≤c_max` | `cand_cost_cap = max(dda, match, stage2)` | MATCH | cu ~2919–2920, 622–627 | keep |
| `iou_stage1_gate` / `maha_gate` hardcoded | 0.30 / 9.4877 members | MATCH | cu ~3809, 3816 | keep (document as non-YAML) |

### B2. Multiplicative cost + stability reward

| Claim | Current implementation | Status | Evidence | Action |
|:--|:--|:--|:--|:--|
| `c = clamp(1 − A e^{−Π})` | `1 - fused_iou * expf(-penalty)` + clamp | MATCH | cu ~585–586 | keep |
| `R_stab = (w/λ)/(1+Δh/h_det)` subtracted from Π | same | MATCH | cu ~578–582 | keep |
| Coupled with `sinkhorn_lambda` | λ used in reward and softmin | MATCH | scoring_semantics | keep dual-coupling note |
| Additive path exists when multiplicative off | else branch | MATCH | cu ~587+ | keep as legacy |

### B3. OAO duration ramp

| Claim | Current implementation | Status | Evidence | Action |
|:--|:--|:--|:--|:--|
| `o = o_base * min(1, d/N_ramp)` | duration counter in occlusion kernel | MATCH | cu ~413–421 | keep |
| `P_OAO = τ · o · g_s(s)` | `penalty += oao_tau * d_occ_coeff * oao_score_scale` | MATCH | cu ~541–543 | keep |
| Spatial OAO family off on baseline | contest/score/mode/crowd/h/foot default off | MATCH / NO-GO | presets only tau+ramp | keep NO-GO pointer |
| §3.1 “crowd /0.25 @cu:500” | that 0.25 is **fuse_score crowd damp**, not OAO base | STALE anchor | cu ~528–530 | retarget footnote |

### B4. Front-occluder / occ_state — **critical DRIFT**

| Claim | Current implementation | Status | Evidence | Action |
|:--|:--|:--|:--|:--|
| §7.6: “baseline 沒有設定 `occ_state_enabled`，因此…**關閉**” | **False for Python MOT17 path** | **DRIFT** | see chain below | **correct math_model** |
| §3.1 table: `P_occ_front` baseline **關** | Actually **on** | **DRIFT** | same | flip to 開 + list knobs |

**Chain of truth:**

```text
GeometryConfig.occ_state_enabled default = True
  → preset now explicit true (s/m)
  → pipeline.py set_occ_params(enabled=cfg.geometry.occ_state_enabled, …)
  → tracker_gpu.cu occ_state_enabled_ (native *member* default false is overwritten)
  → stage1_cost_fused_kernel receives occ_front_ttl + occ_cost_weight (0.50)
```

| Sub-claim | Status | Note |
|:--|:--|:--|
| Native C++ member default `false` | MATCH as ABI default | Misleading without inject story |
| Formula for under-foot penalty | MATCH | cu ~559–562; math §7.6 equations OK |
| `occ_cost_weight: 0.50` schema | MATCH | ACTIVE with state |

**tracker-decision alignment:** scoring_semantics / config_surface treat occ_state as ACTIVE — **correct**. math_model is the stale party.

### B5. fuse_score / vel_dir / energy

| Claim | Current | Status | Evidence | Action |
|:--|:--|:--|:--|:--|
| `fuse_score_weight: 0` | 0; NO-GO if raised | MATCH / NO-GO | #45 | keep |
| confirmed relative-drop formula when w>0 | kernel implements | MATCH (latent) | cu ~523–531 | keep as non-baseline |
| `vel_dir_weight` off | 0 | MATCH | presets | keep |
| energy mode off | baseline mode | MATCH | schema | keep |

---

## C. Auction / stages (§8)

| Claim | Current implementation | Status | Evidence | Action |
|:--|:--|:--|:--|:--|
| Not full Sinkhorn IPF; softmin + single-round auction cascade | comments + kernel structure | MATCH | math §8 intro, cu multistage | keep |
| Stage table S0–S2 score bands + cost caps | fused multistage kernel | MATCH | cu ~877+ | keep |
| `SACCADE_DDA_MAX_COST` default 0.12 | env_float 0.12 | MATCH | cu ~2686 | keep |
| `SACCADE_STABILITY_W` default 0.1 **on** | getenv else 0.1f | MATCH | cu ~2996–2998 | keep dual-stability story |
| `SACCADE_FRESHNESS_W` default 0 | documented | MATCH | math §8.2 | keep |
| `G_aspect` thresholds 0.8 / 0.15 | auction value path | MATCH | cu ~917–919 | keep |
| §3.1 says `G_aspect` @ `cu:186` (2.5/1.2) | **cu:186 is quality-scaling Gaussian aspect**, different mechanism | **DRIFT** (anchor) | cu ~204–206 vs ~917 | fix line anchor in §3.1 |

---

## D. Kalman / GMC (§5–6)

| Claim | Current | Status | Evidence | Action |
|:--|:--|:--|:--|:--|
| 8D state + height-scaled Q/R | kalman_gpu.cuh | MATCH | headers | keep |
| `r_scale` multiplies R | `get_R(..., r_scale)` | MATCH | cuh | keep |
| s r_scale 2.8 | s preset | MATCH | YAML | keep |
| m r_scale 3.5 | not in math | STALE | m YAML | add m note |
| NSA off / double-comp risk | nsa false | MATCH / NO-GO | #8 | keep |
| GMC translation phase-corr, ds=4 | preset | MATCH | YAML + gmc | keep |
| FG mask false | false | MATCH / NO-GO | #20 | keep |

---

## E. Lifecycle / birth / output (§9, §12)

| Claim | Current | Status | Evidence | Action |
|:--|:--|:--|:--|:--|
| birth/confirm/buffer 0.28 / 3 / 0.50 / 30 | same | MATCH | presets + schema buffer | keep |
| interpolate 35/5/0 | same | MATCH | §12 + presets | keep |
| private continuation as det input | ACTIVE, not in math §1/§9 | **STALE** | detection_filters / stages | add “input-set policy” subsection |
| `per_seq_adapt: false` | false | MATCH | presets | keep |

---

## F. Inject / packing / source map

| Claim | Current implementation | Status | Evidence | Action |
|:--|:--|:--|:--|:--|
| `evaluator.py` calls `set_params` | **pipeline.py** ~951+ injects tracker params | **DRIFT** (file map) | pipeline.py | update §4.2 note / §13 map |
| evaluator owns stage order | still largely true | MATCH (partial) | evaluator + stages | keep orchestrator role |
| `kalman_r_scale` → `r_scale` arg | set_params remap | MATCH | tracker_gpu.py | already in tracker-decision native_bridge |
| Shared `set_relink_params` packing | bank + bridge + tail | MATCH | pipeline + facade | keep |
| Audit date 2026-06-19 | superseded by this audit | STALE banner | header | bump after math edits |

---

## G. Alignment with tracker-decision semantics

| tracker-decision claim | math_model | Status |
|:--|:--|:--|
| ACTIVE surface small | §1 lists most but misses private + occ | STALE math |
| s/m assoc primary thresh shared | implied s-only | STALE (add m) |
| Dual stability cost vs bid | §3.1 + §8.2 correctly separate | **MATCH** (best dual-stability doc today) |
| Private continuation is input-set not setter | not modeled | STALE |
| occ_state ACTIVE | says OFF | **DRIFT** |
| m dir_bonus 0 / looser bridge | s-only values | STALE |

---

## H. NO-GO claims still valid

| Claim area | Status | Evidence |
|:--|:--|:--|
| Appearance off on baseline | MATCH / intentional | reid_mode off |
| Spatial OAO off | MATCH / NO-GO | only tau+ramp |
| fuse_score 0 | MATCH / NO-GO #45 | preset |
| NSA off | MATCH / NO-GO #8 | schema |
| Semantic relink gate §11 off | MATCH | baseline |

---

## Priority fix list (for a future math_model PR)

| Priority | Item | Why |
|:--|:--|:--|
| **P0** | §7.6 + §3.1 occ_state baseline → **on** + knobs | Wrong decision semantics for readers |
| **P1** | §1 add private_continuation | Input-set changes association population |
| **P1** | §1 / §10 m-variant delta table | Avoid applying s bridge to m blindly |
| **P2** | Inject source map → `pipeline.py` | Wrong file waste debug time |
| **P2** | Fix §3.1 G_aspect line anchor | Quality vs auction confusion |
| **P3** | Bump audit date; link this drift file | Freshness contract |
| Hold | Dual stability merge | Documented correctly; architectural choice A/B/C later |

---

## What not to do from this audit

```text
✗ Change preset numbers
✗ Change setters / packing / kernels
✗ Run 7-seq “because docs drifted”
✗ Rewrite math_model in the same commit as this audit without review
```

P4 (smoke / MOT17-04 / 7-seq) only if a follow-up PR **changes behavior**.

---

## Verification commands used

```bash
# Preset vs schema sampling (uv run)
# Kernel greps: stage1_cost_fused_kernel, compute_track_occlusion_kernel,
#   set_occ_params inject, SACCADE_STABILITY_W, bridge bdist
# Cross-read: scoring_semantics.md, config_surface.md, pipeline.py:975-985
```

---

## Related

- Active contract: [../README.md](../README.md)
- Scoring semantics: [../scoring_semantics.md](../scoring_semantics.md)
- Config surface: [config_surface.md](config_surface.md)
- Native bridge: [native_bridge.md](native_bridge.md)
- Math model (subject): [../../../reference/math_model.md](../../../reference/math_model.md)
- Pipeline path contract (m): [../../pipeline/mot17_mamba_whole_graph_m_sdp_double_buffer.md](../../pipeline/mot17_mamba_whole_graph_m_sdp_double_buffer.md)
