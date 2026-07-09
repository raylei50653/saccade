# Association Scoring Semantics

Explains **what the association score means** on the production path (`reid_mode: off` → `stage1_cost_fused_kernel`).

Canonical equations also live in [`docs/reference/math_model.md`](../../reference/math_model.md) §7–8. If this file and math_model disagree with `tracker_gpu.cu`, **code wins**.

---

## Decision being made

For each active track `i` and detection `j` in a cascade stage:

> Should track `i` continue with det `j`, or stay unmatched (coast / lost / wait for stage-2 / die)?

Separately, after association:

> Should an unmatched high-score det **birth** a new ID? Should a young track **steal** a lost ID via bridge relink?

This document covers the **cost / assignment** layer. Birth thresholds and bridge gates are identity policy layered on top (see [assoc_knobs.md](assoc_knobs.md), [relink_bridge.md](relink_bridge.md)).

---

## Pipeline of one cost entry

```text
1. Candidate gate     IoU > τ_iou  OR  Mahalanobis² < τ_maha
                      else c_ij = 1 and return (hard reject)

2. Affinity A_ij      A = fused_iou ≈ IoU  (fuse_score_weight=0 on baseline)

3. Penalties Π_ij     OAO + occ_front + (latent vel / energy) − stability reward

4. Cost form          multiplicative (baseline):
                      c = clamp(1 − A · exp(−Π), 0, 1)
                      legacy additive: c = clamp(1−A + Σ penalties)

5. Enqueue gate       only if c ≤ cand_cost_cap
                      cap = max(DDA_max, match_thresh, stage2_match_thresh)

6. Softmin + auction  p ∝ exp(−λ c) · G_aspect · (optional bid biases)
                      multi-stage ByteTrack-style score cascade
```

---

## What is measured?

| Signal | Geometric meaning | Where |
|:--|:--|:--|
| **IoU** | Overlap of predicted track box `B(x)` vs det box | Primary affinity |
| **Mahalanobis²** | Innovation of det measurement vs Kalman `(x,S)` | Gate only when IoU weak |
| **Inter-track IoU** | How much track `i` overlaps other active tracks | OAO `occ_coeff` |
| **Overlap duration** | Consecutive frames with occ_base > 0 | OAO ramp |
| **Foot depth** | Track foot vs det bottom / partner foot | occ_state front latch & under penalty |
| **Height consistency** | `|h_trk − h_det| / h_det` | stability reward (and optional energy height) |
| **Det score** | Detector confidence | cascade stage membership; fuse path off |

**Not measured on baseline:** appearance / ReID cosine (kernel exists in `compute_conditional_cost_kernel` but headline path does not feed embeddings).

---

## What is normalized / calibrated?

| Mechanism | Normalization | Knob |
|:--|:--|:--|
| Process / measurement noise | `σ ∝ h` (height-scaled) | hardcoded weights + `kalman_r_scale` |
| Softmin temperature | `p ∝ e^{−λc}` | `sinkhorn_lambda` (preset **10**) |
| Stability reward vs λ | reward term divided by `λ` so bid boost is λ-independent | `stability_cost_w` |
| Bridge distances | foot residual / `h_ref` | `relink_bridge_px` (post-assoc) |
| OAO duration | `min(1, frames / ramp)` | `oao_ramp_frames` |
| Cost range | clamp to `[0, 1]` always | invariant |

---

## What is a policy weight (not pure evidence)?

| Term | Formula role | Preset | Policy intent |
|:--|:--|:--|:--|
| `oao_tau` | `Π += τ · occ · (score scale)` | 0.50 | Prefer not to steal boxes from heavily overlapped tracks when overlap is persistent |
| `oao_ramp_frames` | scales occ before Π | 25 | Full penalty only after sustained overlap (crowd), not brief crossings |
| `occ_cost_weight` | under-foot depth penalty | 0.50 | Prefer re-acq consistent with occluder geometry |
| `stability_cost_w` | **negative** Π (reward) | 0.20 | Prefer size-stable matches in auction |
| `SACCADE_STABILITY_W` | auction bid bias (separate!) | ~0.1 | Extra height consistency at bid time |
| `SACCADE_DDA_MAX_COST` | tight stage for confirmed×high | 0.12 | High precision first stage |

---

## What is only a gate?

| Gate | Condition | If fail |
|:--|:--|:--|
| IoU / Mahalanobis candidate | neither passes | `c=1`, not enqueued |
| Cost enqueue cap | `c > cand_cost_cap` | stays in dense matrix as high cost but not in sparse top-k list |
| Stage score cascade | det score outside stage band | considered in other stage or not at all |
| Match threshold | assignment rejects above thresh | track unmatched this stage |
| Birth `new_track_thresh` | score too low | no new ID |
| Confirm streak/score | not enough hits / score | stays tentative |
| Bridge gates | geometry bridge fails | ID fragment remains |

Gates do **not** enter `Π` as soft penalties (except where a soft version also exists, e.g. OAO is soft once candidate exists).

---

## What can change identity decisions?

| Layer | Identity effect |
|:--|:--|
| Wrong association (cost) | ID switch or fragmentation this frame |
| Birth threshold | new ID vs missed birth |
| Confirm policy | whether ID is emitted / stable |
| Lost buffer | whether old ID still available to match |
| **Bridge relink** | rewrites young track ID → lost ID (post-assoc) |
| Private continuation | extra det may **continue** correct ID that NMS would drop |
| Interpolate | fills gap under same ID (eval), can glue wrong endpoints |

---

## Multiplicative vs additive (why it matters)

Baseline:

\[
c_{ij} = \mathrm{clamp}\bigl(1 - A_{ij}\, e^{-\Pi_{ij}},\, 0,\, 1\bigr)
\]

- Positive `Π` (OAO, occ, reverse vel) **multiplies down** affinity.
- Negative `Π` (stability reward) **boosts** affinity without needing separate clamp logic.
- Terms do not fight each other via sequential `min(1, c+…)` saturation as easily as additive form.

Stability term in code:

```text
penalty -= (stability_cost_w / max(λ,1)) / (1 + |h_trk − h_det| / h_det)
```

So raising `stability_cost_w` or lowering `sinkhorn_lambda` both strengthen size-consistent preference — they are **coupled**.

---

## Dual stability (architecture debt — document only)

Two different decision stages both encode a **height / stability preference**. They are **not** the same knob.

| Knob | Stage | Role |
|:--|:--|:--|
| `stability_cost_w` | association **cost** shaping (multiplicative `Π`) | Reward size-consistent pairs inside `c_ij` before softmin |
| `SACCADE_STABILITY_W` | **auction / bid-side** preference (env) | Bias bid value after cost→value transform |

```text
status: semantically overlapping but not mechanically equivalent
risk:   double-counting stability or producing hard-to-explain ID preference
preset: stability_cost_w = 0.20 (YAML); SACCADE_STABILITY_W typically ~0.1 (env default)
```

**Do not merge casually.** Architecture options:  
[audit/dual_stability_cleanup.md](audit/dual_stability_cleanup.md).  
**P7 measurement protocol (4-way, no default flip):**  
[audit/dual_stability_ablation_protocol.md](audit/dual_stability_ablation_protocol.md).

```text
A. Keep both layers, rename so stage ownership is obvious
B. Converge to a single stability policy
C. Keep one production; demote the other to experimental / NO-GO
```

Until then: any ablation of “stability” must state **which stage** was changed.

---

## Redundant / overlapping terms

| Pair | Overlap | Baseline status |
|:--|:--|:--|
| OAO vs occ_state | both occlusion geometry | **both on**; OAO is soft match suppress; occ_state is front latch + depth penalty |
| `stability_cost_w` vs `SACCADE_STABILITY_W` | height consistency | **both on**, different stages — see dual stability above |
| `stability_cost_w` vs `SACCADE_STABILITY_W` | same height signal, different stage | **both on** — cost reward + bid bias |
| `fuse_score_weight` vs cascade thresholds | both use det score | fuse off; cascade on |
| Energy score/height vs fuse/stability | alternate score/height encodings | energy off |
| NSA Kalman vs `kalman_r_scale` | both enlarge R | NSA off (NO-GO double-comp) |
| Spatial OAO gates vs duration ramp | both modulate occ | spatial NO-GO; ramp GO |
| Bridge vs bank ReID vs lifecycle merge | all reconnect IDs | only geometry bridge on |

**Refactor candidates (research, not committed):**

1. Unify height-consistency into one calibrated term (cost **or** bid, not both undocumented).
2. Document OAO vs occ_state ownership (who owns “occlusion penalty”).
3. Hide or archive NO-GO OAO spatial knobs from “Tier 1” CLI to shrink surface.
4. Keep multiplicative form; treat additive as legacy.

---

## Cascade stages (ByteTrack-style)

Approximate production cascade (see math_model §8 for full stage table):

| Stage | Score band idea | Cost thresh |
|:--|:--|:--|
| DDA (optional env) | confirmed × high-score | very tight (`~0.12`) |
| High / mid / low stages | `high_thresh` / `mid` / `track` cuts | `match_thresh` / `stage2_match_thresh` |

Candidate sparse list is shared and capped by `cand_cost_cap` so crowd Mahalanobis-only ghosts cannot evict the true IoU match from `K_MAX_CANDIDATES` slots.

---

## Contrast: appearance path (not baseline)

When embeddings are present, `compute_conditional_cost_kernel` may replace or blend affinity with cosine similarity under ReID gates. Headline presets set `reid_mode: off`, so:

- No det embeddings into tracker
- No strong appearance force-match
- Identity recovery after long gap is **geometry bridge only**

See [relink_bridge.md](relink_bridge.md) for geometry-only reconnect semantics vs ReID.

---

## Open questions for decision-policy cleanup

1. Should `occ_state` defaults be explicit in the preset YAML (currently silent ACTIVE)?
2. Should m preset set `relink_bridge_dir_bonus` explicitly (currently inherits 0)?
3. Is dual stability (cost + bid) still necessary after multiplicative retune?
4. Which LATENT knobs can move to an experimental module config without breaking pybind ABI?

---

## Related

- Knob cards: [assoc_knobs.md](assoc_knobs.md)
- Config surface: [audit/config_surface.md](audit/config_surface.md)
- Math: [../../reference/math_model.md](../../reference/math_model.md)
- Implementation checklist: [../../reference/math_model_implementation.md](../../reference/math_model_implementation.md)
