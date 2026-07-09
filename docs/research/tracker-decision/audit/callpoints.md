# Decision-Layer Callpoints

Maps knobs from **schema → preset → Python inject → native setter → runtime effect**.

Baseline path: `scripts/eval/mot17.py` → `pipeline.py` tracker setup → `GPUByteTracker.update` → `tracker_gpu.cu`.

Format:

```text
knob → schema → preset (s / m) → inject site → native API → effect
```

---

## Injection hub

| Site | File | What it configures |
|:--|:--|:--|
| Tracker thresholds | `src/saccade/perception/eval/pipeline.py` ~951–968 | `set_params(...)` |
| OAO | same ~969–978 | `set_oao_params(...)` |
| Occ-state | same ~979–985 | `set_occ_params(...)` |
| Multiplicative / λ / stability | same ~994–1002 | `set_multiplicative_cost` / `set_sinkhorn_lambda` / `set_stability_cost_w` |
| Association energy | same ~1003–1009 | `set_association_energy_params` |
| Quality scaling | same ~946–950 | `set_quality_params` |
| Bridge + bank relink | same ~578–610 | `set_relink_params(...)` |
| Private continuation | `evaluator.py` / `stages.py` / `detection_filters.py` | expands det set pre-track |
| Interpolate | post-track emit path | host-side gap fill |

Python facade: `src/saccade/perception/tracking/tracker_gpu.py` (wraps extension setters).

Native core: `src/tracking/tracker_gpu.cu`, API `include/tracking/tracker_gpu.hpp`, bindings `src/tracking/tracker_gpu_python.cpp`.

---

## ACTIVE production knobs

### Association gates & cost form

```text
match_thresh
  → CoreConfig.match_thresh (default 0.75)
  → preset 0.50 / 0.50
  → pipeline.set_params(match_thresh=...)
  → GPUByteTracker::set_params → match_thresh_
  → stage thresholds + cand_cost_cap = max(dda, match_thresh, stage2_match_thresh)
  → effect: higher → more permissive matches / more candidates enqueued

high_thresh / mid_thresh / track_thresh
  → CoreConfig defaults 0.45 / 0.10 / 0.05 (not overridden in headline presets)
  → set_params
  → cascade stage score cuts + private score floor coupling
  → effect: which dets enter which association stage

stage2_match_thresh
  → GeometryConfig 0.5
  → set_params
  → stage-2 cost gate + cand_cost_cap
  → effect: low-score recovery stage looser/tighter

multiplicative_cost
  → GeometryConfig false → preset true
  → set_multiplicative_cost(True)
  → multiplicative_cost_ flag in stage1_cost_fused_kernel
  → effect: c = clamp(1 − A·exp(−Π), 0, 1) instead of additive clamp chain

sinkhorn_lambda
  → GeometryConfig 30 → preset 10
  → set_sinkhorn_lambda(10)
  → softmin p ∝ exp(−λ c); also normalizes stability reward (÷λ)
  → effect: lower λ = softer discrimination, rewards matter more

stability_cost_w
  → GeometryConfig 0 → preset 0.20
  → set_stability_cost_w(0.20)
  → penalty −= (w/λ) / (1 + |h_trk−h_det|/h_det)
  → effect: prefer size-stable matches (reward, not gate)

fuse_score_weight
  → GeometryConfig 0 → preset 0.0
  → set_params(fuse_score_weight=0)
  → fused_iou path inactive when w=0
  → effect: NO-GO when raised (registry #45)
```

### OAO / occlusion

```text
oao_tau / oao_ramp_frames
  → GeometryConfig 0 / 0 → preset 0.50 / 25
  → set_oao_params(tau, ..., ramp_frames)
  → compute_track_occlusion_kernel: occ_base *= min(1, dur/ramp)
  → stage1 cost: Π += tau * occ_coeff * score_scale (if contest allows)
  → effect: penalize matching occluded tracks more when overlap is persistent

occ_state_enabled / occ_iou_thresh / occ_foot_gap / occ_ttl / occ_cost_weight
  → GeometryConfig true / 0.45 / 0.15 / 4 / 0.50 (preset does not override)
  → set_occ_params(...)
  → front-ttl latch + under-foot penalty in cost kernel
  → effect: bias re-acquisition against depth-inconsistent under-occluder matches
```

### Kalman / GMC

```text
kalman_r_scale
  → GeometryConfig 0.75 → s:2.8 / m:3.5
  → set_params(r_scale=...)  # name remap
  → kf_gpu::get_R / update / mahalanobis S
  → effect: larger R → trust motion model more; maha gate loosens with R

gmc / gmc_downscale
  → CoreConfig true / 8 → preset true / 4
  → GMC module then warp pointer into tracker update
  → predict_gmc_sinv_fused_kernel applies W_f to state
  → effect: camera translation compensated before association IoU/maha
```

### Birth / confirm / lost

```text
new_track_thresh
  → CoreConfig 0.35 → preset 0.28
  → set_params + private birth ceiling uses same value
  → unmatched det with score ≥ thresh can birth
  → effect: lower → more births / more ID fragments risk

confirm_streak / confirm_score_thresh
  → CoreConfig 1 / 0 → preset 3 / 0.50
  → set_params
  → tentative→confirmed policy
  → effect: delayed confirmed IDs; score shortcut for strong dets

track_buffer
  → LifecycleConfig 30
  → set_params(track_buffer=seq_track_buffer)
  → lost track survival for association + bridge
  → effect: longer buffer → more recovery / more ghost keep-alive risk
```

### Bridge relink

```text
relink_bridge_*
  → LifecycleConfig (enabled false; presets enable + set gates)
  → pipeline set_relink_params(
        enabled=bank ReID path,
        bidirectional / bridge_px / bridge_at / min_lost / ttl /
        h_lo / h_hi / margin / dir_bonus / spatial_gate / ...)
  → native bridge block in tracker_gpu.cu (post-assoc ID rewrite)
  → effect: young confirmed track may adopt lost id if geometry bridge passes

Note: bank ReID fields (sim_thresh, bank_cap, …) share the same setter
but relink_enabled stays false under headline reid_mode=off.
```

### Private continuation (pre-tracker)

```text
private_continuation_*
  → DetectionConfig; preset enables + sets IoU/score/cap
  → evaluator/stages → _append_private_continuation_candidates
     (also native process_private_continuation_append when available)
  → appended boxes with score < new_track_thresh enter tracker det list
  → effect: recover NMS-suppressed overlaps for CONTINUE only, not birth
```

### Interpolate (post-tracker)

```text
interpolate_tracklets / max_gap / min_track_len / min_h
  → LifecycleConfig; preset max_gap 35
  → host postprocess on MOT rows
  → effect: fills short gaps → AssA/IDF1; can add FP if endpoints wrong
```

---

## LATENT / PRESET-OFF (short map)

| Knob family | Inject if enabled | Typical effect if flipped on |
|:--|:--|:--|
| `person_geometry_prior` | detection filter path | drops non-person-shaped boxes |
| `detection_quality_scaling` | `set_quality_params` | rescales det scores |
| `id_stability_filter` | Python emit filter | suppresses unstable IDs |
| `geometry_suspect_support` | det auxiliary path | keep bad boxes as soft support |
| `vel_dir_weight` | `set_params` | reverse-motion penalty in cost |
| `association_scoring_mode=energy` | forces multiplicative + energy weights | extra score/height Π terms |
| `relink_enabled` bank | `set_relink_params(enabled=True)` | needs embeddings |
| lifecycle / cheb-gr merge | separate Python modules | offline/online ID stitch |
| OAO spatial family | `set_oao_params` | NO-GO for default |
| `fuse_score_weight>0` | `set_params` | NO-GO #45 |

---

## Dead / unused risk checklist

Use this when auditing a new knob:

1. Is it in a dataclass? (schema)
2. Does any preset set it? (ACTIVE candidate)
3. Does `pipeline.py` / stages pass it? (inject)
4. Does facade call a real pybind method? (bridge)
5. Does the kernel read the member? (runtime)
6. Does effect differ Python-only vs native eval path?

Known traps:

| Trap | Detail |
|:--|:--|
| Name remap | `kalman_r_scale` → `r_scale` in `set_params` |
| Dual stability | `stability_cost_w` (cost Π) vs `SACCADE_STABILITY_W` (auction bid) |
| Shared setter | `set_relink_params` packs bank ReID + bridge + occ expand + app veto |
| Default-on not preset | `occ_state_enabled=true` is ACTIVE without YAML line |
| m missing dir_bonus | s sets `relink_bridge_dir_bonus: 0.8`; m does not → schema default `0.0` |
| Private not in set_params | decision via det list composition only |

---

## Related

- Surface inventory: [config_surface.md](config_surface.md)
- Bridge packing: [native_bridge.md](native_bridge.md)
- Score math: [../scoring_semantics.md](../scoring_semantics.md)
