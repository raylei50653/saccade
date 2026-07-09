# Kalman + GMC Motion (Decision Path)

Motion model assumptions that change **matching, gating, and relink geometry** — not a general Kalman tutorial.

Production: GMC on (`gmc_downscale=4`), `kalman_r_scale` = **2.8** (s) / **3.5** (m).

---

## Decision impact summary

| Motion piece | Feeds decision |
|:--|:--|
| Kalman **predict** (+ GMC warp) | Predicted box `B(x)` for IoU affinity & gates |
| Kalman **S / Mahalanobis** | Candidate gate when IoU weak |
| Kalman **update** with R | Next-frame state / smoothness of track box |
| GMC warp quality | Systematic bias of all predicts under camera motion |
| Velocity in state | Latent vel-dir penalty; bridge foot regression uses **foot history**, not necessarily Kalman v |

---

## Kalman state meaning

State (SORT-style, 8D):

```text
x = (cx, cy, a, h, vx, vy, va, vh)
```

| Symbol | Meaning |
|:--|:--|
| `cx, cy` | Box center |
| `a` | Aspect `w/h` |
| `h` | Height |
| `v*` | Constant-velocity derivatives |

Measurement `z = (cx, cy, a, h)` from detection box.  
Box reconstruction `B(x)` used in association IoU.

Source: `include/tracking/kalman_gpu.cuh`, math_model §6.

---

## Measurement noise and `kalman_r_scale`

`get_R(h, …, r_scale)` builds diagonal R with position terms `∝ (h/20)²` scaled by:

```text
multiplier = r_scale * adapt_r_mult * (1 + 2 * light_factor)
```

| | |
|:--|:--|
| **Decision** | How much to trust this frame's box vs the motion model |
| **↑ r_scale** | Larger R → Kalman gain lower → trust **predict** more; smoother tracks; **Mahalanobis gate wider** (S grows with R) |
| **↓ r_scale** | Follow measurements; more box jitter into association |
| **s / m** | 2.8 / **3.5** — m boxes noisier; higher R closes IDF1 gap without loosening association gates |

### Related (off)

| Knob | Status |
|:--|:--|
| `kalman_adapt_mode` | LATENT (0 = off) |
| `nsa_kalman` | **NO-GO #8** — signal real but double-compensates with high `r_scale` |
| `light_factor` | path exists; not a headline decision knob |

**Invariant:** Mahalanobis² scales **inversely** with R. Tuning r_scale without revisiting maha gate changes the IoU-fail recovery population.

---

## GMC role

GMC estimates camera translation (phase correlation on gray frames) → warp `W_f` applied as **control input** during predict so track boxes move with the camera before IoU is computed.

```text
detect ──boxes──┐
                ├→ track (predict+GMC → assoc → update)
GMC ──W_f───────┘
```

| | |
|:--|:--|
| **Decision support** | Without GMC, moving-camera sequences produce systematic IoU miss → FN / ID switch |
| **Not** | Per-object optical flow; FG segmentation (`gmc_fg_mask` **NO-GO #20**) |
| **Mode** | GPU cuFFT path preferred; Python graphed fallback if extension missing |

### `gmc_downscale`

| | |
|:--|:--|
| Schema default | 8 |
| Preset | **4** (higher resolution correlation) |
| **↑ factor** | Faster, coarser; under-compensation risk |
| **↓ factor** | Costlier, finer |

PCR (peak/RMS) can shrink untrusted warps (`SACCADE_GMC_PCR_THRESH`). Low PCR → near-zero translation → under-compensation.

---

## How motion interacts with geometry matching

```text
predict(x) + apply W_f
  → B(x) IoU with dets          [primary affinity]
  → innov / S for Mahalanobis   [secondary gate]
  → update with z and R(r_scale)
```

| Failure of motion | Association symptom |
|:--|:--|
| GMC under-compensates | All tracks lag camera motion; mass IoU drop; ID churn |
| GMC over-compensates | Tracks jump; false high IoU to wrong person |
| r_scale too high | Slow to lock onto true lateral motion; coast through wrong |
| r_scale too low | Jittery boxes; unstable IoU; more false candidates |
| Bad height in state | Wrong R scale (∝ h) and wrong IoU box |

Bridge relink uses **foot point history**, so GMC-affected association history still shapes which tracks die and which feet get stored — motion quality still matters for relink even when bridge is Kalman-free.

---

## Failure cases (motion-side)

### Camera motion under-compensation

- **Symptom:** Moving-cam sequences (MOT17-10/13 family) lose tracks or switch IDs together.  
- **Cause:** Coarse GMC, low PCR shrink, or GMC disabled.  
- **Knobs:** `gmc`, `gmc_downscale`, PCR env; not association thresh first.  
- **Status:** GMC on is baseline; FG mask NO-GO.

### Box jitter amplification

- **Symptom:** Confirmed tracks shake; false stage-2 matches.  
- **Cause:** Low r_scale + noisy dets.  
- **Knobs:** ↑ `kalman_r_scale` (m already 3.5).  
- **Risk:** Over-smooth misses true turns.

### Double compensation (NSA + r_scale)

- **Symptom:** Aggregate regression when enabling NSA on current baseline.  
- **Cause:** Both enlarge effective R.  
- **Status:** NO-GO #8.

### GMC residual vs association

- Residual studies (`docs/research/eval/gmc_residual_correction_*`, kalman recal notes) show residual structure; **decision layer** should not invent unbounded residual terms without cand-cap / cost-range review (`math_model_implementation.md`).

---

## What this file does **not** cover

- Full phase-correlation derivation  
- CUDA graph capture of GMC  
- Double-buffer scheduling of GMC vs detect  

See pipeline docs for those.

---

## Related

- Scoring: [scoring_semantics.md](scoring_semantics.md)
- Knobs: [assoc_knobs.md](assoc_knobs.md)
- Math §5–6: [../../reference/math_model.md](../../reference/math_model.md)
- GMC research: [../../modules/geometry/research/fp_fn_recovery_and_gmc.md](../../modules/geometry/research/fp_fn_recovery_and_gmc.md)
