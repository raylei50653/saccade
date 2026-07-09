# Decision-Layer Config Surface

Cross-module inventory of knobs that change **association / identity / lifecycle** decisions under the production path.

**Baseline reference:** `mamba_whole_graph` and `mamba_whole_graph_m` (`reid_mode: off`, bridge relink on, multiplicative cost on).

**How to read status:**

| Tag | Meaning |
|:--|:--|
| `ACTIVE` | On / non-trivial under headline preset(s) |
| `PRESET-OFF` | Schema default often on; headline preset turns off |
| `LATENT` | Wired; off by default; may work if enabled |
| `NO-GO` | Registry or code comment: do not enable as default |
| `ENV` | Controlled by env var, not preset YAML |

Path labels: `NATIVE` (tracker CUDA), `PYTHON` (pre/post tracker), `BOTH`.

---

## Current Active Contract (summary)

See also [../README.md](../README.md) § Current Active Contract.

| Fact | Implication for this surface |
|:--|:--|
| ACTIVE set is small | Prefer editing the ACTIVE tables below; do not treat full dataclass dumps as policy |
| s/m share association primary thresholds | Diffs in `match_thresh` / `new_track_thresh` / OAO / multiplicative between s and m are **bugs** unless re-validated |
| m differs mainly in `kalman_r_scale` + bridge gates (+ explicit `dir_bonus=0`) | Documented in §3 / §5 |
| Private continuation is det-set policy | Listed under §6; never expect a `set_params` field |
| `occ_state_*` ACTIVE | Explicit in headline presets (schema defaults locked in YAML) |

---

## NO-GO / LATENT prohibition table

Conditions under which a non-ACTIVE knob must **not** become default production policy without new paired evidence.

| Knob / family | Tag | Do not enable as default when… | Registry / note |
|:--|:--|:--|:--|
| `fuse_score_weight > 0` | NO-GO | Any aggregate “+IDF1 on one seq” without 7-seq bipolar check | #45 |
| `nsa_kalman` | NO-GO | `kalman_r_scale` already ≥ ~2.8 (double compensation) | #8 |
| OAO spatial family (`contest`, `score_w`, `occ_mode`, `crowd_radius`, `height/foot_gate`) | NO-GO | Goal is separating MOT17-05 harm vs 04 benefit — duration ramp is the GO axis | OAO revival |
| `gmc_fg_mask` | NO-GO | Expecting PCR-dominated BG to become FG-clean | #20 |
| Sync online ReID / `relink_bridge_app_veto` as critical path | NO-GO | Double-buffer throughput budget; tiny AssA gain | #57 |
| `new_track_thresh` ≪ 0.28 | NO-GO as blind lower | Aggregate-only win with cross-seq split (overfit) | preset #38-style |
| Fancy interpolate geometry | NO-GO | Endpoint association is the FP source, not path shape | #44 |
| `person_geometry_prior` / DQS / suspect / id_stability | PRESET-OFF | Re-enabling on Mamba headline without score-dist study | fights recalibration |
| Bank `relink_enabled` | LATENT | Embeddings off (`reid_mode=off`) | needs ReID path |
| Lifecycle / post / cheb-gr merge | LATENT | Offline experiments; some post-merge NO-GO | keep off headline |
| `association_scoring_mode=energy` | LATENT | Extra score/height Π terms without retune of λ/stability | experimental |
| `vel_dir_weight > 0` | LATENT | Unvalidated reverse-vel policy on current cost form | — |
| Env bid knobs without docs | ENV | Changing `SACCADE_STABILITY_W` while also sweeping `stability_cost_w` | dual stability debt |

**Rule of thumb:** LATENT means “wired for ablation,” not “safe default.” NO-GO means “evidence already rejected this as headline policy.”

---

## 1. Association scoring (cost matrix + assignment)

| Field | Schema | s preset | m preset | Role | Path | Status |
|:--|:--|:--|:--|:--|:--|:--|
| `match_thresh` | core `0.75` | **0.50** | **0.50** | Stage cost gate / cand cap input | NATIVE | ACTIVE |
| `high_thresh` | core `0.45` | (default) | (default) | High-score cascade boundary | NATIVE | ACTIVE |
| `mid_thresh` | core `0.10` | (default) | (default) | Mid cascade boundary | NATIVE | ACTIVE |
| `track_thresh` | core `0.05` | (default) | (default) | Low cascade / private floor | NATIVE | ACTIVE |
| `stage2_match_thresh` | geometry `0.5` | (default) | (default) | Stage-2 cost gate / cand cap | NATIVE | ACTIVE |
| `multiplicative_cost` | geometry `false` | **true** | **true** | Cost form `1−A·e^{−Π}` | NATIVE | ACTIVE |
| `sinkhorn_lambda` | geometry `30` | **10** | **10** | Softmin temperature `e^{−λc}` | NATIVE | ACTIVE |
| `stability_cost_w` | geometry `0` | **0.20** | **0.20** | Height-consistency **reward** in Π | NATIVE | ACTIVE |
| `fuse_score_weight` | geometry `0` | **0.0** | **0.0** | Score fused into affinity | NATIVE | PRESET-OFF / NO-GO #45 |
| `vel_dir_weight` | geometry `0` | (default) | (default) | Reverse-velocity penalty | NATIVE | LATENT |
| `association_scoring_mode` | geometry `baseline` | (default) | (default) | `energy` enables extra terms | NATIVE | LATENT |
| `assoc_score_cost_w` | geometry `0` | — | — | Energy score penalty | NATIVE | LATENT |
| `assoc_height_cost_w` | geometry `0` | — | — | Energy log-height penalty | NATIVE | LATENT |
| `iou_stage1_gate` | hardcoded | — | — | IoU OR maha candidate gate | NATIVE | ACTIVE (code) |
| `maha_gate` | hardcoded `9.4877` | — | — | χ²(4) Mahalanobis gate | NATIVE | ACTIVE (code) |
| `SACCADE_STABILITY_W` | env | `0.1` typical | same | Auction **bid** height bias (≠ cost `stability_cost_w`) | NATIVE | ENV ACTIVE |
| `SACCADE_FRESHNESS_W` | env | `0` | `0` | Auction age bid bias | NATIVE | ENV LATENT |
| `SACCADE_ENABLE_DDA` / `SACCADE_DDA_MAX_COST` | env | on / `0.12` | same | Tight confirmed×high stage | NATIVE | ENV ACTIVE |

**Decision note:** `sinkhorn_lambda` is a historical name — assignment uses softmin values + parallel auction, not full Sinkhorn iterations (see `math_model.md` §8).

---

## 2. Occlusion / OAO

| Field | Schema | s / m preset | Role | Path | Status |
|:--|:--|:--|:--|:--|:--|
| `oao_tau` | `0` | **0.50** | Penalty scale on occ coefficient | NATIVE | ACTIVE |
| `oao_ramp_frames` | `0` | **25** | Duration ramp: `min(1, frames/ramp)` | NATIVE | ACTIVE |
| `oao_contest_thresh` | `-1` | (default) | Contention-gated OAO | NATIVE | LATENT / NO-GO spatial family |
| `oao_score_w` | `-1` | (default) | Soft score scale on OAO | NATIVE | LATENT / NO-GO |
| `oao_occ_mode` | `0` | (default) | 0=max IoU, 1=union grid | NATIVE | LATENT / NO-GO |
| `oao_crowd_radius` | `0` | (default) | Local crowd multiplier | NATIVE | LATENT / NO-GO |
| `oao_height_gate` | `0` | (default) | Same-height partner filter | NATIVE | LATENT / NO-GO |
| `oao_foot_gate` | `0` | (default) | Same-foot partner filter | NATIVE | LATENT / NO-GO |
| `occ_state_enabled` | `true` | **true** (explicit) | Front-occluder latch | NATIVE | ACTIVE |
| `occ_iou_thresh` | `0.45` | **0.45** (explicit) | Front latch IoU | NATIVE | ACTIVE |
| `occ_foot_gap` | `0.15` | **0.15** (explicit) | Depth (foot) separation | NATIVE | ACTIVE |
| `occ_ttl` | `4` | **4** (explicit) | Front latch frames | NATIVE | ACTIVE |
| `occ_cost_weight` | `0.50` | **0.50** (explicit) | Under-foot re-acq penalty | NATIVE | ACTIVE |

OAO spatial discriminants (contest/score/union/crowd/height/foot) were tried and failed to separate MOT17-05 harm vs 04 benefit; **duration ramp is the GO axis** (registry OAO revival 2026-06-17).

---

## 3. Kalman + motion (GMC)

| Field | Schema | s preset | m preset | Role | Path | Status |
|:--|:--|:--|:--|:--|:--|:--|
| `kalman_r_scale` | `0.75` | **2.8** | **3.5** | Global measurement R scale | NATIVE | ACTIVE |
| `kalman_adapt_mode` | `0` | (default) | (default) | Per-track R adaptation modes | NATIVE | LATENT |
| `nsa_kalman` | `false` | — | — | Score-adaptive R | NATIVE | NO-GO #8 (double-comp with r_scale) |
| `gmc` | core `true` | **true** | **true** | Enable GMC warp into predict | BOTH | ACTIVE |
| `gmc_downscale` | core `8` | **4** | **4** | Phase-corr resolution | NATIVE | ACTIVE |
| `gmc_fg_mask` | — | **false** | **false** | FG mask during GMC | NATIVE | NO-GO #20 |
| `gmc_mode` | `gpu` | (default) | (default) | GPU vs Python fallback | BOTH | ACTIVE |

Higher `kalman_r_scale` → trust **prediction** more than noisy boxes (m is noisier → 3.5).

---

## 4. Birth / confirm / death

| Field | Schema | s / m | Role | Path | Status |
|:--|:--|:--|:--|:--|:--|
| `new_track_thresh` | core `0.35` | **0.28** | Birth score floor | NATIVE | ACTIVE |
| `confirm_streak` | core `1` | **3** | Hits to confirm | NATIVE | ACTIVE |
| `confirm_score_thresh` | core `0` | **0.50** | Score path to confirm | NATIVE | ACTIVE |
| `adaptive_confirmation` | `false` | (default) | Adaptive confirm | NATIVE | LATENT |
| `track_buffer` | lifecycle `30` | (default) | Lost-track survival frames | NATIVE | ACTIVE |
| `birth_low_score_thresh` | `0` | — | Extra birth gate | NATIVE | LATENT |
| `birth_prox_norm_thresh` | `0` | — | Proximity birth | NATIVE | NO-GO |
| `birth_*_gate` family | false | — | Quality / consecutive birth | PYTHON/NATIVE | NO-GO / LATENT |
| `multi_birth_*` | false | — | Multi-signal birth | PYTHON | LATENT / experimental |

---

## 5. Relink / ID continuity

| Field | Schema | s preset | m preset | Role | Path | Status |
|:--|:--|:--|:--|:--|:--|:--|
| `relink_bridge_enabled` | false | **true** | **true** | Bidirectional foot bridge | NATIVE | ACTIVE |
| `relink_bridge_px` | 0.25 | **0.25** | **0.40** | Bridge distance gate (×h) | NATIVE | ACTIVE |
| `relink_bridge_h_lo` | 0 | **0.75** | **0.60** | Height-ratio lower | NATIVE | ACTIVE |
| `relink_bridge_h_hi` | 0 | **1.33** | **1.70** | Height-ratio upper | NATIVE | ACTIVE |
| `relink_bridge_margin` | 0 | **0.05** | **0.05** | Best-vs-second margin | NATIVE | ACTIVE |
| `relink_bridge_dir_bonus` | 0 | **0.8** | **0.0** (explicit) | Direction consistency bias | NATIVE | ACTIVE (s on / m off by contract) |
| `relink_bridge_spatial_gate` | 0 | **0** | **0** | Center prefilter | NATIVE | LATENT (off) |
| `relink_bridge_at` | 4 | default | default | Fire at hit_streak | NATIVE | ACTIVE |
| `relink_bridge_min_lost` | 2 | default | default | Min lost age | NATIVE | ACTIVE |
| `relink_bridge_ttl` | 120 | default | default | Max lost age for bridge | NATIVE | ACTIVE |
| `relink_bridge_max_speed` | 0 | default | default | Physical speed gate | NATIVE | LATENT (off) |
| `relink_bridge_*_occ_*` / `app_veto` | off | — | — | Occ expand / app veto | NATIVE | LATENT / NO-GO #57 |
| `relink_enabled` (bank ReID) | false | off | off | Birth-time bank relink | NATIVE | LATENT (needs ReID) |
| `lifecycle_merge` / post / cheb-gr | false | off | off | Offline/online ID merge | PYTHON | LATENT / partial NO-GO |
| `interpolate_tracklets` | true | **true** | **true** | Gap fill (eval continuity) | PYTHON | ACTIVE |
| `interpolate_max_gap` | 20 | **35** | **35** | Max fill gap | PYTHON | ACTIVE |

**m-vs-s decision delta:** m relaxes bridge geometry and raises `kalman_r_scale`; primary association thresholds intentionally stay identical (loosening fragments recovered small objects).

---

## 6. Private continuation (association **input** policy)

Not a tracker setter — expands the detection set before association.

| Field | Schema | s / m | Role | Path | Status |
|:--|:--|:--|:--|:--|:--|
| `private_continuation_enabled` | false | **true** | Wider NMS candidates | PYTHON / native post | ACTIVE |
| `private_candidate_nms_iou` | 0.7 | **0.70** | Second NMS IoU | PYTHON | ACTIVE |
| `private_prior_iou_threshold` | 0 | **0.30** | Must overlap active track | PYTHON | ACTIVE |
| `private_min_score` | 0.25 | **0.10** | Score floor | PYTHON | ACTIVE |
| `private_max_candidates` | 0 | **50** | Cap | PYTHON | ACTIVE |
| `private_selection_mode` | global | **global** | Selection policy | PYTHON | ACTIVE |

Scores are clamped **below** `new_track_thresh` so candidates can **continue** tracks but not **birth** ghosts.

---

## 7. Person / quality geometry (mostly PRESET-OFF)

| Field | Schema default | Headline preset | Status |
|:--|:--|:--|:--|
| `person_geometry_prior` | true | **false** | PRESET-OFF |
| `detection_quality_scaling` | true | **false** | PRESET-OFF |
| `geometry_suspect_support` | true | **false** | PRESET-OFF |
| `id_stability_filter` | true | **false** | PRESET-OFF |
| `track_person_only` | true | **false** | PRESET-OFF |
| `per_seq_adapt` | true | **false** | PRESET-OFF |
| crowd mid-scale family | mostly off | off | LATENT |
| `fp_hard_filter_*` | on | (detection defaults) | ACTIVE detection-side |

Headline Mamba path turns hard geometry priors off to avoid fighting recalibrated score distributions.

---

## 8. Surface size & risk summary

| Subdomain | Rough field count (decision-relevant) | Production-active | Risk |
|:--|:--|:--|:--|
| Association + auction | ~20 | ~12 | Softmax temp + cand cap interact |
| OAO / occ | ~12 | ~7 | Spatial OAO knobs look useful but NO-GO |
| Kalman / GMC | ~8 | ~5 | r_scale overlaps NSA |
| Birth / confirm | ~15 | ~5 | Many dead birth experiments |
| Bridge relink | ~20 | ~10 | m/s gate mismatch intentional |
| Private continuation | ~8 | ~6 | False keepalive in motion cam |
| Person/quality priors | ~25 | ~0 on headline | Easy to re-enable by accident |
| Lifecycle merge / ReID relink | ~60+ | ~0 on headline | Large latent surface |

**Main open maintainability issue:** large LATENT / NO-GO surface still in schema and native setters, while ACTIVE decision path is a small subset. Cleanup should start from this ACTIVE list, not from full dataclass dumps.

---

## Related

- Callpoint map: [callpoints.md](callpoints.md)
- Native packing: [native_bridge.md](native_bridge.md)
- Knob cards: [../assoc_knobs.md](../assoc_knobs.md)
- Canonical math: [../../../reference/math_model.md](../../../reference/math_model.md)
