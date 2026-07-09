# Association / Decision Knobs

Knob cards for the production decision layer. Values are **`mamba_whole_graph` (s)** unless noted; **m** deltas called out.

Template per knob:

```text
Decision · Signal · Policy · Callpoint · ↑ effect · ↓ effect · Risk · Path
```

See also: [scoring_semantics.md](scoring_semantics.md), [audit/config_surface.md](audit/config_surface.md).

---

## Quick index (ACTIVE)

| Knob | s | m | Role |
|:--|:--|:--|:--|
| `match_thresh` | 0.50 | 0.50 | cost gate |
| `new_track_thresh` | 0.28 | 0.28 | birth gate |
| `confirm_streak` | 3 | 3 | lifecycle |
| `confirm_score_thresh` | 0.50 | 0.50 | lifecycle |
| `kalman_r_scale` | 2.8 | **3.5** | motion trust |
| `gmc_downscale` | 4 | 4 | motion input |
| `oao_tau` | 0.50 | 0.50 | penalty weight |
| `oao_ramp_frames` | 25 | 25 | penalty norm |
| `multiplicative_cost` | true | true | cost form |
| `sinkhorn_lambda` | 10 | 10 | softmin temp |
| `stability_cost_w` | 0.20 | 0.20 | reward weight |
| `relink_bridge_px` | 0.25 | **0.40** | relink gate |
| `relink_bridge_h_lo/hi` | 0.75/1.33 | **0.6/1.7** | relink gate |
| `relink_bridge_margin` | 0.05 | 0.05 | relink gate |
| `relink_bridge_dir_bonus` | 0.8 | **0** (unset) | relink weight |
| `private_continuation_*` | on | on | det input policy |
| `interpolate_max_gap` | 35 | 35 | post-ID continuity |

---

## Match / assignment

### `match_thresh`

| | |
|:--|:--|
| **Decision** | Accept association if cost ≤ stage threshold |
| **Signal** | Final `c_ij` after gates + penalties |
| **Policy** | Hard gate (not a soft weight) |
| **Callpoint** | `CoreConfig` → preset → `set_params(match_thresh)` → stage thresh + `cand_cost_cap` |
| **↑** | More permissive matches; more sparse candidates; risk of ID switches |
| **↓** | Stricter; more FN fragmentation; fewer candidates |
| **Risk** | Interacts with FPS/candidate races at high values (see `bench_tracker_match_thresh.py`) |
| **Path** | NATIVE |
| **s/m** | 0.50 / 0.50 |

### `high_thresh` / `mid_thresh` / `track_thresh`

| | |
|:--|:--|
| **Decision** | Which cascade stage sees a detection |
| **Signal** | Det score |
| **Policy** | Stage partition gates |
| **Callpoint** | `CoreConfig` defaults (0.45 / 0.10 / 0.05) → `set_params` |
| **↑ high_thresh** | Fewer dets in high stage |
| **Risk** | Moving mid/track changes low-score recovery (ByteTrack second stage) |
| **Path** | NATIVE |
| **Preset** | not overridden (defaults ACTIVE) |

### `stage2_match_thresh`

| | |
|:--|:--|
| **Decision** | Cost gate for low-score stage |
| **Callpoint** | `GeometryConfig` 0.5 → `set_params` → also enters `cand_cost_cap` |
| **Path** | NATIVE |

### `sinkhorn_lambda`

| | |
|:--|:--|
| **Decision** | How peaky softmin values are before auction |
| **Signal** | Transforms cost → value `e^{−λc}` |
| **Policy** | Temperature (not geometric evidence) |
| **Callpoint** | preset 10 → `set_sinkhorn_lambda` |
| **↑** | Harder max; small cost gaps dominate |
| **↓** | Softer; stability rewards matter more |
| **Risk** | Coupled with `stability_cost_w` (reward ÷ λ) |
| **Path** | NATIVE |
| **Note** | Name is historical; not full Sinkhorn IPF |

### `multiplicative_cost`

| | |
|:--|:--|
| **Decision** | How `A` and `Π` combine into `c` |
| **Policy** | Cost algebra switch |
| **Callpoint** | preset true → `set_multiplicative_cost(True)` |
| **On** | `c = clamp(1 − A e^{−Π})` |
| **Off** | Legacy additive clamp chain (no stability reward path) |
| **Path** | NATIVE |

### `stability_cost_w`

| | |
|:--|:--|
| **Decision** | Prefer height-stable match |
| **Signal** | `|h_trk − h_det| / h_det` |
| **Policy** | Bounded **reward** (negative penalty) |
| **Callpoint** | preset 0.20 → `set_stability_cost_w` inside multiplicative branch |
| **↑** | Stronger size-consistent preference |
| **↓** | Rely more on pure IoU/OAO |
| **Risk** | Double-counts with env auction `SACCADE_STABILITY_W` |
| **Path** | NATIVE |

### `fuse_score_weight`

| | |
|:--|:--|
| **Decision** | Blend det score into affinity |
| **Status** | **0.0** preset; **NO-GO #45** if raised |
| **Path** | NATIVE |

---

## Occlusion

### `oao_tau`

| | |
|:--|:--|
| **Decision** | How strongly to discourage matching an occluded track |
| **Signal** | `occ_coeff` (max inter-track IoU, duration-ramped) |
| **Policy** | Penalty weight `Π += τ · occ · …` |
| **Callpoint** | preset 0.50 → `set_oao_params` → cost kernel |
| **↑** | More anti-confusion in crowds; risk FN on true tracks in dense scenes |
| **↓** | More ID switches at persistent overlaps |
| **Path** | NATIVE |

### `oao_ramp_frames`

| | |
|:--|:--|
| **Decision** | When OAO reaches full strength |
| **Signal** | Consecutive overlapped frames |
| **Policy** | Time normalization `min(1, dur/ramp)` |
| **Callpoint** | preset 25 → occlusion kernel |
| **↑** | Longer transient grace (helps brief crossings e.g. MOT17-05) |
| **↓** | Full penalty sooner (helps persistent crowd e.g. MOT17-04) |
| **Path** | NATIVE |
| **Evidence** | `docs/research/eval/oao_duration_ramp_revival_20260617.md` |

### `occ_state_*` / `occ_cost_weight`

| | |
|:--|:--|
| **Decision** | Depth-consistent re-acquisition under occluder |
| **Signal** | Track-track IoU + foot gap; under-foot residual |
| **Policy** | Latch + soft penalty |
| **Callpoint** | schema defaults (enabled) → `set_occ_params` |
| **Risk** | Overlaps conceptually with OAO; different mechanism |
| **Path** | NATIVE |
| **Preset** | silent ACTIVE (not in YAML) |

### OAO spatial family (`contest`, `score_w`, `occ_mode`, `crowd_radius`, `height_gate`, `foot_gate`)

| | |
|:--|:--|
| **Status** | LATENT / **NO-GO** as default policy (cannot separate 05 harm vs 04 benefit) |
| **Path** | NATIVE |

---

## Motion

### `kalman_r_scale`

| | |
|:--|:--|
| **Decision** | Trust measurement vs predict when updating / gating |
| **Signal** | Measurement noise R scaled globally |
| **Policy** | Calibration of Kalman trust |
| **Callpoint** | s **2.8** / m **3.5** → `set_params(r_scale=…)` |
| **↑** | Smoother tracks; trust motion; maha gate wider; can lag true motion |
| **↓** | Follow jittery boxes; more measurement-driven assoc |
| **Risk** | Do not combine with NSA (#8) |
| **Path** | NATIVE |

### `gmc_downscale`

| | |
|:--|:--|
| **Decision** | Quality/cost of camera-motion estimate fed to predict |
| **Signal** | Phase-correlation warp on downscaled gray |
| **Callpoint** | preset 4 (schema default 8) |
| **↑ downscale factor** | Cheaper / coarser GMC |
| **↓** | Finer, costlier |
| **Risk** | Bad warp → systematic association error on moving-cam seqs |
| **Path** | NATIVE GMC + tracker predict |
| **See** | [kalman_gmc_motion.md](kalman_gmc_motion.md) |

---

## Birth / confirm

### `new_track_thresh`

| | |
|:--|:--|
| **Decision** | Unmatched det may birth new ID |
| **Signal** | Det score |
| **Policy** | Hard birth gate |
| **Callpoint** | preset 0.28 → `set_params` + private birth ceiling |
| **↑** | Fewer births / more FN fragments missed |
| **↓** | More births / more FP IDs |
| **Risk** | 0.20 looked good aggregate but failed cross-seq (registry #38 style overfit) |
| **Path** | NATIVE (+ PYTHON private clamp) |

### `confirm_streak` / `confirm_score_thresh`

| | |
|:--|:--|
| **Decision** | Tentative → confirmed (emit-stable ID) |
| **Callpoint** | preset 3 / 0.50 → `set_params` |
| **↑ streak** | Slower confirm; fewer flash IDs |
| **↑ score thresh** | Only high-score paths confirm early |
| **Path** | NATIVE |

### `track_buffer`

| | |
|:--|:--|
| **Decision** | How long lost tracks remain matchable / bridgeable |
| **Callpoint** | lifecycle default 30 → `set_params` |
| **↑** | Longer occlusion recovery; ghost keep-alive risk |
| **Path** | NATIVE |

---

## Relink bridge (summary)

Full semantics: [relink_bridge.md](relink_bridge.md).

| Knob | Role | s | m |
|:--|:--|:--|:--|
| `relink_bridge_enabled` | master switch | true | true |
| `relink_bridge_px` | distance gate (heights) | 0.25 | **0.40** |
| `relink_bridge_h_lo/hi` | scale gate | 0.75–1.33 | **0.6–1.7** |
| `relink_bridge_margin` | ambiguity reject | 0.05 | 0.05 |
| `relink_bridge_dir_bonus` | direction bias | **0.8** | **0.0** (explicit) |
| `relink_bridge_at` | fire at hit_streak | 4 | 4 |
| `relink_bridge_min_lost` / `ttl` | age window | 2 / 120 | same |

**↑ `relink_bridge_px`:** more reconnects, more false merges.  
**↑ h-band width:** recover noisy small boxes (m); risk scale-mismatched people.

---

## Private continuation

| Knob | Role | Value |
|:--|:--|:--|
| `private_continuation_enabled` | master | true |
| `private_candidate_nms_iou` | wider NMS | 0.70 |
| `private_prior_iou_threshold` | must touch active track | 0.30 |
| `private_min_score` | floor | 0.10 |
| `private_max_candidates` | cap | 50 |
| `private_selection_mode` | ranking | global |

| | |
|:--|:--|
| **Decision** | Which extra boxes enter association as **continue-only** evidence |
| **Signal** | Suppressed NMS candidates + prior IoU to active tracks |
| **Policy** | Score clamp `< new_track_thresh` prevents birth |
| **Callpoint** | `DetectionConfig` → `_append_private_continuation_candidates` / native append |
| **↑ prior IoU** | Fewer private cands (stricter track gate) |
| **↓ min_score** | More low-score continues; ghost keepalive risk |
| **Risk** | Moving-cam sequences (e.g. 13) can false-continue |
| **Path** | PYTHON / native postprocess (**not** `set_params`) |

---

## Interpolate (post decision)

| Knob | s/m | Role |
|:--|:--|:--|
| `interpolate_tracklets` | true | fill gaps |
| `interpolate_max_gap` | 35 | max frames |
| `interpolate_min_track_len` | 5 | eligibility |
| `interpolate_min_h` | 0 | min height |

Affects **reported** continuity and FP/FN trade (registry #44: interpolation FP largely from bad endpoints, not path shape).

---

## Explicitly PRESET-OFF (do not re-enable casually)

| Knob | Why off on headline |
|:--|:--|
| `person_geometry_prior` | Mamba score dist; avoid double filtering |
| `detection_quality_scaling` | same |
| `geometry_suspect_support` | same |
| `id_stability_filter` | same |
| `fuse_score_weight` | NO-GO #45 |
| `reid_mode` | off; sync ReID NO-GO #57 |
| `relink_enabled` (bank) | needs appearance |
| `lifecycle_merge` / post / cheb | offline experiments |

---

## Related

- Semantics: [scoring_semantics.md](scoring_semantics.md)
- Surface table: [audit/config_surface.md](audit/config_surface.md)
- Callpoints: [audit/callpoints.md](audit/callpoints.md)
