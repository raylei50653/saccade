# Geometry Relink Bridge

Documents **geometry-only** ID reconnection after a track is lost and a new tentative/young track stabilizes.

Production: `relink_bridge_enabled: true`, `reid_mode: off` (no appearance bank).

Formula inventory also in `report_data/relink_gates_and_formulas.md`; this file focuses on **decision semantics**.

---

## What decision is made?

After primary association and Kalman update:

> May a **young live track** adopt the ID of a **still-buffered lost track** by bridging their foot trajectories in height-normalized geometry?

If yes → ID fragment is healed (AssA/IDF1 win).  
If wrongly yes → two people merge (catastrophic ID error).  
If no → fragment remains (higher IDs count).

This is **not** the same as:

| Mechanism | Baseline | Role |
|:--|:--|:--|
| Same-frame association | on | Continue ID with a det |
| Bank ReID `relink_enabled` | **off** | Appearance revive at birth |
| Lifecycle / Cheb-GR merge | **off** | Post-hoc stitch |
| Sync ReID in tracker | **off** (NO-GO #57) | Appearance veto on bridge |

---

## What is a relink bridge?

Kalman-free, appearance-free reconnect:

1. Young candidate reaches `hit_streak == bridge_at` (default 4) with enough foot history.
2. System regresses velocity from last/first **4 foot points** on lost and candidate.
3. Bidirectional midpoint extrapolation produces a residual distance in **units of reference height**.
4. If residual passes gates (distance, height ratio, margin, optional extras), candidate **inherits lost track's ID**.

Trigger sketch:

```text
fire if:
  bridge enabled
  hit_streak[cand] == bridge_at
  foot history length ≥ 4
  track not already revived
  lost track age ∈ [min_lost, ttl]
```

---

## Gates: relaxed vs strict

### Strict on production (reject hard)

| Gate | Rule | s | m |
|:--|:--|:--|:--|
| **Distance** | `bdist ≤ bridge_px` (height-normalized, speed-weighted blend of forward/back residual vs static dist) | 0.25 | **0.40** |
| **Height ratio** | `h_lo ≤ ema_h_lost/ema_h_cand ≤ h_hi` | 0.75–1.33 | **0.6–1.7** |
| **Margin** | second_best − best ≥ margin | 0.05 | 0.05 |
| **Age window** | min_lost ≤ age ≤ ttl | 2 / 120 | same |
| **Trigger** | streak / foot / not revived | at=4 | same |

### Soft / bias (not hard reject when 0)

| Term | Role | s | m |
|:--|:--|:--|:--|
| `relink_bridge_dir_bonus` | Prefer direction-consistent bridges | **0.8** | **0** (unset → schema 0) |
| Spatial prefilter | `cdist ≤ spatial_gate` | **off** (0) | off |
| Physical speed m/s | max speed | **off** (0) | off |
| Occ expand / cover | occlusion-aware expand | off | off |
| `relink_bridge_app_veto` | appearance veto | off / NO-GO sync ReID | off |

**m relaxation rationale:** m detector recovers noisier small boxes; s-tuned h-band and px reject valid small-object bridges. Wider gates recover AssA/IDs; primary association thresholds stay tight.

---

## Signals used

| Signal | Use |
|:--|:--|
| Foot positions (4-pt OLS velocity) | Extrapolate lost forward / cand backward |
| `h_ref = avg(ema_h_lost, ema_h_cand)` | Normalize distances |
| EMA height | Scale gate + normalization |
| Speed of lost (heights/frame) | Blend dynamic vs static residual |
| Optional dir consistency | Bonus / bias toward aligned motion |

**Not used on baseline:** ReID embedding, Kalman predict of lost state for the bridge score (bridge is intentionally Kalman-free).

---

## Geometry vs ReID relink

| | Geometry bridge | Bank / semantic ReID |
|:--|:--|:--|
| Identity evidence | Trajectory continuity + scale | Appearance similarity |
| Works when | Short/medium gap, similar size, motion plausible | Reappear far away with clean crop |
| Fails when | Two people swap near same place | Occlusion / polluted embedding |
| Baseline | **ON** | **OFF** |
| Cost | Cheap (in tracker core) | Sync ReID kills double-buffer budget (#57) |

Design stance: **geometry-first reconnect**; appearance only as future async sidecar, not hard dependency.

---

## Known false-relink / missed-relink modes

### False relink (wrong merge)

| Symptom | Likely cause | Knobs |
|:--|:--|:--|
| Two nearby people collapse to one ID after a gap | px too loose; margin too small | ↓ `bridge_px`, ↑ `margin` |
| Scale-mismatched adults/children merge | h-band too wide | tighten `h_lo/h_hi` |
| Merge after long absence | ttl too long + loose px | ↓ ttl or require spatial gate |

### Missed relink (fragment remains)

| Symptom | Likely cause | Knobs |
|:--|:--|:--|
| Same person new ID after occlusion | px/h too strict (esp. small boxes on m) | m already relaxed; check streak/at |
| Lost purged before cand confirms | `track_buffer` < gap or bridge ttl | ↑ buffer / ttl |
| Cand never fires | streak never hits exactly `bridge_at` or foot_len < 4 | confirm policy interaction |
| Moving-cam residual inflated | bad GMC → wrong geometry | GMC quality (see motion doc) |

---

## Interaction with other policies

| Policy | Interaction |
|:--|:--|
| `track_buffer` | Lost must still exist |
| `confirm_streak` / `bridge_at` | Bridge fires on young stabilization |
| Primary `match_thresh` | If same-frame match succeeds, bridge unneeded |
| Private continuation | Can keep track alive so bridge never needed — or keep wrong track alive |
| Interpolate | Post-hoc gap fill; different from live ID rewrite |
| OAO / occ | Affects whether track dies/fragments in the first place |

---

## Callpoints

```text
LifecycleConfig.relink_bridge_*
  → preset (s tighter / m looser)
  → pipeline.py set_relink_params(...)
  → tracker_gpu.py facade
  → tracker_gpu.cu bridge block
```

Details: [audit/callpoints.md](audit/callpoints.md), [audit/native_bridge.md](audit/native_bridge.md).

---

## Related

- Prior formula dump: `report_data/relink_gates_and_formulas.md`
- Math §10: [../../reference/math_model.md](../../reference/math_model.md)
- Failure modes: [failure_modes.md](failure_modes.md)
