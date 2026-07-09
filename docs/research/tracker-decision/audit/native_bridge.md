# Native Bridge: Python → C++/CUDA Decision Fields

Documents how decision-layer knobs cross from Python orchestration into `GPUByteTracker` native state.

---

## Bridge layers

```text
CLI / preset YAML
  → scripts/eval/config/*.py dataclasses
  → saccade.perception.eval.config.EvalConfig (flat attrs)
  → pipeline.py / stages.py inject
  → tracker_gpu.py facade (optional kwargs / try-except for ABI)
  → pybind (tracker_gpu_python.cpp)
  → tracker_gpu.cu Impl members
  → kernels (stage1_cost_fused_kernel, occlusion, bridge, Kalman)
```

There is **no generic config blob** packed as a struct for the whole geometry surface. Fields cross via **typed setters**. Names are **exact** at the pybind layer; some **rename** at the Python inject layer.

---

## Setter inventory (decision-relevant)

| Setter | Facade | Native | Decision fields |
|:--|:--|:--|:--|
| `set_params` | yes | yes | track/high/mid/match thresh, buffer, confirm*, new_track, kalman_adapt_mode, **r_scale**, vel_dir, fuse_score, stage2, birth_* |
| `set_oao_params` | yes | yes | tau, contest, score_w, occ_mode, crowd_radius, height_gate, foot_gate, **ramp_frames** |
| `set_occ_params` | yes | yes | enabled, iou_thresh, foot_gap, ttl, cost_weight |
| `set_multiplicative_cost` | yes | yes | bool |
| `set_sinkhorn_lambda` | yes | yes | λ |
| `set_stability_cost_w` | yes | yes | w |
| `set_association_energy_params` | yes | yes | enabled, score_w, height_w |
| `set_relink_params` | yes | yes | bank ReID + **bridge** + occ expand + app veto (one mega-setter) |
| `set_quality_params` | yes | yes | detection quality scaling weights |
| `set_reid_params` | yes | yes | appearance fusion (baseline off) |

Env knobs (`SACCADE_STABILITY_W`, `SACCADE_DDA_*`, `SACCADE_FRESHNESS_W`, GMC PCR) are read **inside native/Python GMC** without going through preset dataclasses.

---

## Name remaps (collision / confusion risks)

| Config / CLI name | Argument at inject | Native member (concept) | Risk |
|:--|:--|:--|:--|
| `kalman_r_scale` | `r_scale=` | measurement R multiplier | Easy to search wrong name in CUDA |
| `relink_bridge_px` | `bridge_px=` | bridge distance gate | prefix dropped |
| `relink_bridge_h_lo/hi` | `bridge_h_lo/hi=` | height ratio gate | same |
| `relink_enabled` | `enabled=` (first arg of set_relink_params) | bank ReID master | **Not** the bridge master; bridge uses separate bidirectional/enabled logic inside setter |
| `oao_ramp_frames` | last arg of `set_oao_params` | `oao_ramp_frames_` | float type in API |
| `stability_cost_w` | setter | cost Π reward | Collides **conceptually** with `SACCADE_STABILITY_W` bid bias |
| `match_thresh` | same | stage thresh + cand cap | One knob, two effects |

### Shared `set_relink_params` packing

Single call packs:

1. Bank ReID: enabled, bank_cap, sim_thresh, cheb_lambda, spatial_gate, max_age  
2. Bridge: bidirectional/bridge_px/at/min_lost/ttl/speed/person_h/fps/margin/spatial/anchor/h_lo/h_hi/dir_bonus  
3. Occ expand / app veto: optional tail args with `try/except TypeError` for older ABI  

**Collision risk:** changing positional order breaks silently if a binding is older (facade falls back to shorter arity). Always add new fields at the **end** and keep the TypeError fallback.

**Semantic risk:** `relink_bridge_enabled` is **not** the first `enabled` flag. Pipeline sets bank `enabled=False` while still passing bridge fields when bridge is on — verify `pipeline.py` whenever either flag changes.

---

## Python-only decision paths (do not look for native setters)

| Mechanism | Where | Why not native tracker |
|:--|:--|:--|
| Private continuation | det postprocess append | Changes **input** det set |
| Person geometry prior | det filter | Pre-tracker box drop |
| ID stability filter | emit filter | Post-tracker host |
| Lifecycle / cheb-gr merge | Python modules | Offline/online stitch |
| Interpolate tracklets | host MOT post | Eval continuity |
| Double-buffer / graph scheduling | pipeline | Non-decision |

These can still change IDs strongly but are **not** `GPUByteTracker` members.

---

## Behavioral equivalence checklist

Before changing a bridge field:

1. **Same setter on Python eval and any C++ runner** (`cpp_runner.py` notes OAO is tracker-side).  
2. **Facade TypeError fallback** still covers old extensions if required.  
3. **Default when unset:** schema default vs preset override vs silent schema-default ACTIVE (`occ_state_*`).  
4. **m vs s:** only documented deltas should differ (bridge gates, `kalman_r_scale`, weights); do not accidentally desync inject.  
5. **Tests:**  
   - `tests/unit/test_math_model_doc_consistency.py` (preset scalars)  
   - `tests/unit/tracking/test_gpu_bidir_bridge.py` (bridge + multiplicative setters)  
6. **Bit-identity:** toggles that claim default-off must leave cost path unchanged when off (`multiplicative_cost`, bridge, OAO tau=0).

---

## Kernel consumers (where native state lands)

| Kernel / path | Reads |
|:--|:--|
| `stage1_cost_fused_kernel` | IoU/maha gates, fuse, OAO, occ_front, multiplicative, stability, λ, energy, cand cap |
| `compute_conditional_cost_kernel` | same + ReID (off baseline) |
| `compute_track_occlusion_kernel` | OAO modes, ramp, occ front latch inputs |
| Kalman update / S_inv | `r_scale`, adapt mode |
| Predict + GMC fused | warp `W_f` from GMC module |
| Bridge relink block | bridge_* members, foot history |
| Auction / multistage | match/stage thresh, λ, env bid weights |

---

## Required tests before changing bridge fields

| Change type | Minimum verification |
|:--|:--|
| New setter arg | unit test + TypeError-safe facade; document in this file |
| Remap / rename | grep all inject sites (`pipeline`, `stages`, `cpp_runner`) |
| Default flip (e.g. occ_state) | 7-seq or documented single-seq A/B; update preset explicitly |
| Cost formula | math_model consistency test + candidate-count sanity on crowded seq |
| Bridge gate | `test_gpu_bidir_bridge` + false-merge probe notes |

---

## Related

- Callpoints: [callpoints.md](callpoints.md)
- Surface: [config_surface.md](config_surface.md)
- Impl checklist: [../../../reference/math_model_implementation.md](../../../reference/math_model_implementation.md)
