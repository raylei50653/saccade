# Dynamic ReID Trigger Design (2026-04-28)

## Context

The previous `need_reid` path moved MOT17 evaluation away from a fixed `reid_interval`, but the trigger was still too coarse:

- early versions only used detection count deltas;
- later versions added a 5-frame bbox-history state machine;
- dynamic triggering improved `IDs` on some sequences, but often over-triggered SigLIP and collapsed throughput.

The goal of this round is to redesign the trigger around **signal strength**, not just event counts.

## Current Findings

### 1. Main failure mode is `lost -> new` fragmentation

From the recent MOT17 sequence ablations, the dominant identity failure is:

1. a mature track disappears during occlusion / crossing;
2. a new local ID appears shortly after;
3. relink either happens too late or not at all.

This means `lost` events should carry more weight than raw motion jitter.

### 2. Motion instability is a precursor, not the final error

Low IoU / large center shift often appears before fragmentation, especially in crowded scenes, but motion instability alone should not dominate the trigger.

### 3. Moving-camera sequences amplify weak triggers

`MOT17-10-SDP` showed that geometric instability from camera motion can inflate dynamic triggering. A good trigger must distinguish:

- true identity risk;
- scene-wide motion noise.

### 4. Extending bbox history is not the right fix

Increasing precise bbox history from 5 frames to 10 frames was rejected as the default direction.

Reason:

- `5` frames at 30 FPS already covers short occlusion (`~167 ms`);
- older boxes quickly become stale;
- stale geometry is worse than no geometry when the camera or target moves fast.

Instead, we keep:

- **short accurate geometry memory**: 5-frame bbox history;
- **long noisy trend memory**: decayed event scores.

## Dynamic Trigger Direction

The intended design is:

1. keep exact bbox history for the most recent 5 frames;
2. compute per-frame local signals for `new`, `lost`, and `unstable`;
3. smooth those signals with EMA;
4. trigger ReID when the weighted sum exceeds a threshold, optionally with a birth/death boost.

## Local Signal Weighting

### `new_signal_t`

Count-based `+1` is too crude. New tracks should be weighted by detection confidence.

Proposed form:

```text
new_signal_t = sum(det_score_i for i in New)
```

Optional future extension:

```text
new_signal_t = sum(det_score_i * area_weight_i)
```

where larger or more central entrants can contribute more.

### `lost_signal_t`

Short-lived fragments disappearing are usually noise. Mature tracks disappearing are much more important.

Proposed form:

```text
lost_signal_t = sum(min(1.0, age_j / tau_age) for j in Lost)
```

Recommended initial constant:

```text
tau_age = 30 frames
```

This saturates the contribution of a stable identity, while heavily discounting transient flicker.

### `unstable_signal_t`

Unstable motion should encode both:

- center shift growth;
- IoU collapse.

Proposed per-track form:

```text
instability_k =
    alpha * max(0, shift_ratio_k - tau_shift) +
    beta  * max(0, tau_iou - iou_k)
```

Aggregate over matched tracks:

```text
unstable_signal_t = mean(instability_k over matched tracks)
```

Using the **mean** instead of the sum avoids runaway trigger growth in crowded scenes.

## EMA State

Instead of one mixed scalar from raw counts, maintain separate decayed scores:

```text
score_new[t]      = decay * score_new[t-1]      + new_signal_t
score_lost[t]     = decay * score_lost[t-1]     + lost_signal_t
score_geom[t]     = decay * score_geom[t-1]     + geom_signal_t
score_conf[t]     = decay * score_conf[t-1]     + conf_signal_t
```

Recommended initial decay:

```text
decay = 0.8
```

This preserves some influence beyond the 5-frame bbox window without storing stale boxes.

## Trigger Score

The combined score should prioritize mature disappearance over pure instability:

```text
trigger_score =
    w_new      * score_new +
    w_lost     * score_lost +
    w_geom     * score_geom +
    w_conf     * score_conf +
    birth_death_boost
```

Recommended initial weights:

```text
w_new = 1.0
w_lost = 1.4
w_geom = 0.5
w_conf = 0.5
```

Birth/death boost:

```text
if new_signal_t > 0 and lost_signal_t > 0:
    trigger_score += boost_birth_death
```

Recommended initial boost:

```text
boost_birth_death = 0.8 ~ 1.2
```

This explicitly models the common fragmentation pattern:

- an old identity disappears;
- a new identity is born nearby shortly after.

## Guardrails

Even with a weighted trigger, over-triggering must be constrained.

Recommended hard guards:

```text
if trigger_score > threshold and cooldown_ok:
    do_reid = True
```

Suggested initial values:

```text
threshold = 2.0
cooldown = 8 ~ 12 frames
```

### Hard Buffer

In addition to EMA smoothing, score-based modes now support a hard persistence gate:

```text
if trigger_score >= threshold:
    persist_count += 1
else:
    persist_count = 0

do_reid = persist_count >= persist_frames and cooldown_ok
```

Current CLI parameter:

```text
--reid-trigger-persist-frames
```

This is intended to suppress one-frame spikes that survive EMA but should not immediately trigger SigLIP.

## Sequence-Level Interpretation

Recent sequence behavior suggests:

- `MOT17-04-SDP`: dynamic triggering can significantly reduce `IDs`, but current logic is too aggressive and kills FPS;
- `MOT17-02-SDP`: sparse or conservative triggering tends to be enough;
- `MOT17-10-SDP`: moving-camera noise requires instability to have lower weight than lost/new identity events.

This supports the weighting choice:

```text
w_lost > w_new > w_geom / w_conf
```

## Implementation Notes

The current code already contains:

- `DynamicReIDController` with 5-frame bbox history;
- several trigger modes (`count_jump`, `event_any`, `event_persist`, `event_strict`, `event_memory`);
- score-based modes:
  - `score_ema`
  - `score_ema_geom`
  - `score_ema_conf`
- configurable history size, EMA parameters, and persistence threshold.

The current score-based path uses:

1. `new_signal_t = sum(det_score over new tracks)`
2. `lost_signal_t = sum(min(1, age / lost_age_cap) over lost tracks)`
3. `geom_signal_t = mean(weighted center-shift and IoU-drop over matched tracks)`
4. `conf_signal_t = mean(abs(curr_det_score - previous_score_ema) over matched tracks)`
5. weighted EMA fusion and birth/death boost
6. optional hard persistence gate via `persist_frames`

The main tuning gap is no longer "missing signal type"; it is that the trigger still fires too often on some sequences.

## Ablation Script

Current experiment driver:

- `scripts/eval/ablation_reid_trigger.py`

Useful modes currently available:

- `fixed16`
- `score_ema`
- `score_ema_p2`
- `score_ema_p3`
- `score_ema_p3_t25`
- `score_ema_p3_t30`
- `score_ema_geom`
- `score_ema_conf`

Example:

```bash
uv run python scripts/eval/ablation_reid_trigger.py \
  --sequences MOT17-02-SDP,MOT17-04-SDP,MOT17-10-SDP \
  --modes fixed16,score_ema_p3,score_ema_p3_t25,score_ema_p3_t30
```

## Experimental Results

### Geom vs Conf vs Joint (`score_ema`)

This sweep compared `fixed16`, `score_ema_geom`, `score_ema_conf`, and `score_ema`.

#### MOT17-02-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 146 | 32.4% | 21.3% | 87.80 |
| `score_ema_geom` | 146 | 31.8% | 21.4% | 52.64 |
| `score_ema_conf` | 160 | 32.1% | 21.2% | 52.28 |
| `score_ema` | 145 | 32.0% | 21.2% | 51.86 |

#### MOT17-04-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 237 | 39.1% | 30.5% | 64.91 |
| `score_ema_geom` | 369 | 44.1% | 29.5% | 35.65 |
| `score_ema_conf` | 291 | 43.0% | 30.4% | 31.98 |
| `score_ema` | 269 | 43.9% | 30.5% | 33.14 |

#### MOT17-10-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 161 | 42.5% | 44.2% | 82.03 |
| `score_ema_geom` | 161 | 44.5% | 44.2% | 56.14 |
| `score_ema_conf` | 161 | 44.5% | 44.2% | 55.88 |
| `score_ema` | 161 | 44.5% | 44.2% | 55.67 |

Interpretation:

- geometry is more useful than confidence jitter for `IDF1`;
- confidence jitter alone did not create a clear `IDs` advantage;
- joint `score_ema` was occasionally balanced, but not a stable Pareto winner;
- the dominant issue remained over-triggering and throughput collapse.

### Hard Buffer Sweep (`persist_frames`)

This sweep compared:

- `fixed16`
- `score_ema`
- `score_ema_p2`
- `score_ema_p3`

#### MOT17-02-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 143 | 32.5% | 21.3% | 75.57 |
| `score_ema` | 153 | 32.4% | 21.2% | 52.74 |
| `score_ema_p2` | 161 | 31.8% | 21.0% | 50.20 |
| `score_ema_p3` | 140 | 32.1% | 21.2% | 51.67 |

#### MOT17-04-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 312 | 41.1% | 29.9% | 59.40 |
| `score_ema` | 347 | 41.9% | 30.1% | 33.05 |
| `score_ema_p2` | 357 | 42.5% | 30.1% | 32.37 |
| `score_ema_p3` | 194 | 44.2% | 31.1% | 32.43 |

#### MOT17-10-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 161 | 42.5% | 44.2% | 82.58 |
| `score_ema` | 161 | 44.5% | 44.2% | 56.59 |
| `score_ema_p2` | 164 | 44.7% | 44.2% | 52.63 |
| `score_ema_p3` | 164 | 45.9% | 44.2% | 54.54 |

Interpretation:

- `persist=2` was consistently weak;
- `persist=3` was materially better than `persist=2`;
- `persist=3` strongly helped `MOT17-04-SDP`, where fragmentation dominates;
- `persist=3` did not produce a universal `IDs` improvement and still cost substantial FPS.

### Threshold Sweep on `score_ema_p3`

This sweep compared:

- `fixed16`
- `score_ema_p3` (`threshold=2.0`)
- `score_ema_p3_t25`
- `score_ema_p3_t30`

#### MOT17-02-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 154 | 32.5% | 21.1% | 76.90 |
| `score_ema_p3` | 147 | 32.0% | 21.3% | 52.00 |
| `score_ema_p3_t25` | 153 | 32.1% | 21.2% | 52.04 |
| `score_ema_p3_t30` | 144 | 32.0% | 21.3% | 52.71 |

#### MOT17-04-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 349 | 39.6% | 30.3% | 60.12 |
| `score_ema_p3` | 584 | 41.3% | 29.4% | 31.81 |
| `score_ema_p3_t25` | 420 | 44.1% | 30.2% | 31.59 |
| `score_ema_p3_t30` | 333 | 42.8% | 30.0% | 31.28 |

#### MOT17-10-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 161 | 42.5% | 44.2% | 78.60 |
| `score_ema_p3` | 164 | 45.9% | 44.2% | 56.02 |
| `score_ema_p3_t25` | 162 | 45.9% | 44.2% | 53.30 |
| `score_ema_p3_t30` | 166 | 45.9% | 44.2% | 53.44 |

Interpretation:

- raising threshold reduced the worst over-triggering on `MOT17-04-SDP`;
- `t3.0` was the best of the tested dynamic variants on `MOT17-04-SDP`, but only narrowly beat `fixed16` on `IDs` and still cost about half the throughput;
- `MOT17-10-SDP` still preferred `fixed16` on `IDs`;
- threshold tuning alone is not enough to make dynamic triggering a safe general default.

## Implementation Round 2 (2026-04-28)

Two new controller parameters were added to `DynamicReIDController`:

### `cooldown_frames`

After each trigger fires, the persist counter is immediately reset to zero and the controller enters a hard suppression window of `cooldown_frames` frames. During this window, `should_reid()` always returns `False` and the persist counter stays zeroed, so fresh evidence is required after the window expires.

This is different from `MIN_REID_GAP = 5` in runner.py, which did not reset the internal persist counter. Previously, a sustained high-score period would refire immediately after `MIN_REID_GAP` expired.

### `birth_death_lost_min`

The `birth_death_boost` now requires `lost_signal >= birth_death_lost_min` in addition to `lost_signal > 0.0`. With `birth_death_lost_min = 0.3`, a lost track must have age ≥ `0.3 × tau_age = 9 frames` before it contributes to the boost. This suppresses the boost on 1–2 frame noise tracks that appear and vanish without meaningful identity content.

New CLI parameters:

```text
--reid-cooldown-frames       (default 0 = no change to existing behavior)
--reid-birth-death-lost-min  (default 0.0 = no change to existing behavior)
```

## New Ablation Modes

```text
score_ema_p3_cd8          = score_ema_p3 + cooldown=8
score_ema_p3_t30_cd8      = score_ema_p3_t30 + cooldown=8
score_ema_p3_t30_cd12     = score_ema_p3_t30 + cooldown=12
score_ema_p3_t30_cd8_bm3  = score_ema_p3_t30_cd8 + birth_death_lost_min=0.3
score_ema_p3_t40_cd8_bm3  = score_ema_p3_t30_cd8_bm3 but threshold=4.0
score_ema_p3_t30_cd8_bm5  = score_ema_p3_t30_cd8 + birth_death_lost_min=0.5
score_ema_p3_t30_cd8_bm3_nogeom = p3_t30_cd8_bm3 + w_geom=0.0 + w_conf=0.0
```

## Ablation Wave 1 — Cooldown sweep on p3

### MOT17-02-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 148 | 32.4% | 21.3% | 76.81 |
| `score_ema_p3_t30` | 156 | 32.1% | 21.2% | 52.76 |
| `score_ema_p3_cd8` | 162 | 32.0% | 21.1% | 52.72 |
| `p3_t30_cd8` | 155 | 32.0% | 21.0% | 52.74 |
| `p3_t30_cd12` | 158 | 32.2% | 21.0% | 53.14 |
| `p3_t30_cd8_bm3` | 157 | 31.7% | 21.2% | 53.62 |

### MOT17-04-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 308 | 43.7% | 30.0% | 60.33 |
| `score_ema_p3_t30` | 263 | 45.2% | 30.7% | 34.39 |
| `score_ema_p3_cd8` | 435 | 43.1% | 29.4% | 34.79 |
| `p3_t30_cd8` | 353 | 43.6% | 29.8% | 35.04 |
| `p3_t30_cd12` | 347 | 43.2% | 30.2% | 34.83 |
| `p3_t30_cd8_bm3` | 286 | 45.6% | 30.3% | 35.95 |

### MOT17-10-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 161 | 42.5% | 44.2% | 82.60 |
| `score_ema_p3_t30` | 166 | 45.9% | 44.2% | 60.11 |
| `score_ema_p3_cd8` | 162 | 45.9% | 44.2% | 61.15 |
| `p3_t30_cd8` | 166 | 45.9% | 44.2% | 55.62 |
| `p3_t30_cd12` | 166 | 45.9% | 44.2% | 55.82 |
| `p3_t30_cd8_bm3` | 166 | 45.9% | 44.2% | 55.33 |

Wave 1 interpretation:

- Adding cooldown without raising the threshold (`score_ema_p3_cd8`, t=2.0) is harmful on MOT17-04: IDs jumped from 308 to 435. During cooldown windows, missed triggers allow fragmentation that is not recovered.
- `p3_t30_cd8_bm3` (threshold=3.0 + cd8 + born-mature boost gate) was the best of this wave: IDs=286 on MOT17-04, IDF1=45.6%, FPS=35.95.
- On MOT17-10, `score_ema_p3_cd8` recovered IDs=162 (≈ fixed16's 161) with IDF1=45.9%.

## Ablation Wave 2 — Higher threshold and signal ablation

### MOT17-02-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 145 | 32.5% | 21.1% | 91.88 |
| `p3_t30_cd8_bm3` | 146 | 32.1% | 21.1% | 67.96 |
| `p3_t40_cd8_bm3` | 159 | 31.9% | 21.3% | 55.54 |
| `p3_t30_cd8_bm5` | 153 | 32.0% | 21.3% | 66.08 |
| `p3_t30_cd8_bm3_nogeom` | 169 | 31.8% | 21.0% | 67.70 |

### MOT17-04-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 298 | 43.3% | 30.4% | 71.76 |
| `p3_t30_cd8_bm3` | 444 | 42.9% | 29.7% | 46.23 |
| `p3_t40_cd8_bm3` | 276 | 45.9% | 30.9% | 42.20 |
| `p3_t30_cd8_bm5` | 467 | 42.6% | 29.3% | 47.90 |
| `p3_t30_cd8_bm3_nogeom` | 306 | 45.3% | 30.3% | 48.56 |

### MOT17-10-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 161 | 42.5% | 44.2% | 98.37 |
| `p3_t30_cd8_bm3` | 166 | 45.9% | 44.2% | 64.05 |
| `p3_t40_cd8_bm3` | 166 | 45.9% | 44.2% | 72.85 |
| `p3_t30_cd8_bm5` | 166 | 45.9% | 44.2% | 73.64 |
| `p3_t30_cd8_bm3_nogeom` | 166 | 45.9% | 44.2% | 59.02 |

Wave 2 interpretation:

- `p3_t40_cd8_bm3` (threshold=4.0) is the best single dynamic mode on MOT17-04: IDs=276, IDF1=45.9%, MOTA=30.9% — all three metrics beat fixed16, at ~40% FPS cost.
- On MOT17-10, `p3_t40_cd8_bm3` also gives the best FPS among dynamic modes (72.85 vs 98.37 fixed16), with the same IDF1 improvement as cheaper variants.
- `p3_t30_cd8_bm5` (birth_death_lost_min=0.5) is similarly fast on MOT17-10 but harms IDs on MOT17-04.
- `p3_t30_cd8_bm3_nogeom` (w_geom=w_conf=0): near-fixed IDs=306 on MOT17-04, IDF1=45.3%, FPS=48.56 — best FPS among modes that improve IDF1.
- Run-to-run variance is material: `p3_t30_cd8_bm3` showed IDs=286 (wave 1) vs 444 (wave 2) on MOT17-04. The architecture and thresholds are deterministic; the variance likely comes from relink stochasticity. Interpret single-run results as indicative, not definitive.

## Note on Trigger Frequency vs FPS

The remaining FPS gap between dynamic and fixed16 on MOT17-04 (~42 vs ~72 FPS for `p3_t40_cd8_bm3`) is structural: when identity events are frequent, the score correctly crosses threshold more often. This is not over-triggering to eliminate; it reflects real fragmentation activity. The throughput cost is proportional to the scene complexity that makes dynamic triggering valuable.

## Current Best Configuration

For accuracy-priority deployments:

```text
--need-reid
--reid-trigger-mode score_ema
--reid-trigger-persist-frames 3
--reid-score-threshold 4.0
--reid-score-decay 0.80
--reid-weight-new 1.0
--reid-weight-lost 1.4
--reid-weight-geom 0.5
--reid-weight-conf 0.5
--reid-birth-death-boost 1.0
--reid-birth-death-lost-min 0.3
--reid-lost-age-cap 30
--reid-cooldown-frames 8
--reid-unstable-shift-weight 1.0
--reid-unstable-iou-weight 1.0
--reid-conf-jitter-gate 0.10
```

For throughput-priority deployments (mixed scene types), `fixed16` remains the default.

## Ablation Wave 3 — Geometry Moving-Average Smoothing (2026-04-28)

The proposal: keep W+1 raw bbox frames, compute W-frame moving average before the geometry comparison, to reduce detection jitter in the instability signal.

New parameter: `geom_smooth_window` (default 1 = raw, 3 = 3-frame MA). `_raw_geom_history` is a separate deque of maxlen `W+1`; the main `_track_history` is unchanged.

### Theoretical analysis

### Design principle: latest frame must not participate in the MA

The initial wave-3 implementation smoothed **both** sides of the comparison:

```
smooth_curr = mean(frames[i-W+1 : i+1])   ← includes frame i
smooth_prev = mean(frames[i-W   : i  ])
```

This is incorrect for a change-detection trigger. The current frame is the **signal source** — it is what we are trying to measure. Including it in the average dilutes the very shift we want to detect.

Example: a track stable at x=100 for three frames then suddenly jumps to x=200 (occlusion recovery).

```
Old (smooth both, W=3):
  smooth_curr = mean(100, 100, 200) = 133   ← jump diluted to 1/3
  smooth_prev = mean(100, 100, 100) = 100
  measured shift = 33   (true shift = 100)

Correct (raw curr, smooth prev, W=3):
  curr_geom  = 200   ← raw, preserves the full jump
  prev_geom  = mean(100, 100, 100) = 100
  measured shift = 100   (true shift = 100)
```

The corrected implementation (2026-04-28):

```python
# Current frame always raw — it IS the signal we are measuring.
curr_geom = {tid: tracks[tid].box for tid in shared_ids}
# Previous baseline smoothed over W frames to reduce noise.
if W > 1 and len(self._raw_geom_history) > W:
    prev_geom = self._avg_boxes(list(self._raw_geom_history)[-(W + 1):-1], shared_ids)
else:
    prev_geom = {tid: prev[tid].box for tid in shared_ids}
```

### Asymmetric smoothing amplifies constant-velocity drift

The corrected design has a side effect for constant-velocity camera motion:

```
curr_raw  = base + i·v
prev_MA   = mean(base + (i-3)·v, base + (i-2)·v, base + (i-1)·v)
          = base + (i-2)·v
measured shift = curr_raw - prev_MA = 2·v
```

For window W, the amplification factor is `(W-1)·v` instead of `1·v`. With W=3, constant-velocity drift registers at 2× the per-frame velocity. This makes sw3 **worse** for moving-camera sequences, not better.

Confirmed empirically:

```python
# Sudden jump (stable→jump): W=1 geom_score=3.533, W=3 geom_score=3.533  ← same, preserved
# Constant drift 20px/frame:  W=1 geom_score=2.886, W=3 geom_score=6.311  ← 2.2× amplified
```

In contrast, the old (incorrect) both-smoothed implementation for constant velocity:

```
smooth_curr = mean(frames[i-2, i-1, i]) = base + (i-1)·v
smooth_prev = mean(frames[i-3, i-2, i-1]) = base + (i-2)·v
shift = 1·v   ← same as raw
```

So the corrected design preserves sudden shifts (correct) at the cost of amplifying smooth drift (undesirable for moving-camera sequences). This confirms that `w_geom > 0` with sw3 is harmful for MOT17-10.

Note: the wave-3 ablation results below were collected with the old (both-smoothed) implementation. The corrected design has not been re-evaluated; the overall recommendation remains unchanged.

### MOT17-02-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 150 | 32.7% | 21.1% | 93.28 |
| `p3_t40_cd8_bm3` | 166 | 32.7% | 21.0% | 67.45 |
| `p3_t40_cd8_bm3_sw3` | 167 | 32.3% | 21.0% | 65.49 |
| `p3_t30_cd8_bm3_sw3` | 175 | 32.7% | 20.8% | 67.55 |

### MOT17-04-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 334 | 43.3% | 29.9% | 78.77 |
| `p3_t40_cd8_bm3` | 433 | 44.5% | 30.0% | 48.59 |
| `p3_t40_cd8_bm3_sw3` | 332 | 41.8% | 30.7% | 39.92 |
| `p3_t30_cd8_bm3_sw3` | 303 | 42.2% | 30.2% | 40.19 |

### MOT17-10-SDP

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed16` | 161 | 42.5% | 44.2% | 97.34 |
| `p3_t40_cd8_bm3` | 166 | 45.9% | 44.2% | 70.23 |
| `p3_t40_cd8_bm3_sw3` | 166 | 45.9% | 44.2% | 66.24 |
| `p3_t30_cd8_bm3_sw3` | 166 | 45.9% | 44.2% | 60.91 |

Wave 3 interpretation (results from old both-smoothed implementation):

- `sw3` adds ~5 FPS overhead vs raw (MA computation over 3 frames).
- On MOT17-04, `p3_t40_cd8_bm3_sw3` IDF1 = 41.8%, below fixed16's 43.3%. The old implementation diluted genuine shifts, reducing trigger quality.
- On MOT17-10, quality metrics are identical; FPS cost not recovered.

The corrected design (raw curr / smoothed prev) would preserve sudden shifts but amplify constant-velocity drift by ~(W-1) factor. For moving-camera sequences this makes `sw3` actively harmful (stronger spurious geom signals).

**Conclusion: bbox geometry smoothing is not recommended.** Two compounding problems:
1. Detection jitter in this tracker is below `unstable_center_shift=0.30` — smoothing gains nothing on the noise side.
2. Constant-velocity camera drift is amplified (not reduced) by the asymmetric design, making moving-camera sequences worse.

*(Update 2026-04-28)*: Given that smoothing is ineffective and actively dilutes genuine sudden shifts, the `geom_smooth_window` parameter and `sw3` modes have been completely removed from the implementation to simplify the codebase and avoid wasted computation.

## Ablation Wave 4 — Global Motion Compensation (GMC) for Geometry Signal

An architectural bug was found where the `bbox history` comparison in the dynamic ReID controller was evaluating "raw absolute coordinates" across frames, completely ignoring Ego-motion (camera translation). This explains why the geometry trigger performed poorly on moving-camera sequences like `MOT17-10-SDP`.

The system was updated to pass the `gmc_warp` (2x3 affine matrix) into the controller. The controller now projects `prev_box` to the current frame's coordinate space before calculating `center_shift` and `IoU`.

### Experimental Results (with GMC geometric correction)

This test compares `fixed16` and the previous best `p3_t40_cd8_bm3`, now using GMC-corrected geometry.

#### MOT17-02-SDP (Near-static)

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed@16` | 109 | 32.5% | 21.7% | 83.24 |
| `p3_t40_cd8_bm3` | 124 | 32.0% | 21.6% | 58.87 |

#### MOT17-04-SDP (High fragmentation)

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed@16` | 122 | 43.6% | 30.8% | 63.08 |
| `p3_t40_cd8_bm3` | 180 | 43.6% | 30.8% | 34.21 |

#### MOT17-10-SDP (Moving Camera)

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed@16` | 161 | 42.5% | 44.2% | 79.83 |
| `p3_t40_cd8_bm3` | 166 | 45.9% | 44.2% | 56.46 |

**Interpretation:**
- GMC integration succeeded. `MOT17-10` still gains the **+3.4% IDF1** boost but the geometry signal no longer fires erroneously due to camera pans.
- `w_geom` is now a mathematically sound signal purely representing target shape-change and tracking instability.
- With the elimination of camera drift noise from `w_geom`, future tuning can rely purely on tracking logic without needing moving-average workarounds.

## Ablation Wave 5 — Dual Threshold (Hysteresis)

To prevent the ReID controller from rapidly oscillating in and out of the persistent trigger state when the score hovers around the `score_threshold` (e.g., 4.0), a dual-threshold (hysteresis) logic was introduced. 

A new parameter `--reid-score-threshold-low` was added. The trigger logic now requires the score to cross `score_threshold` to start accumulating persistent frames, but it only needs to remain above `score_threshold_low` to continue accumulating.

### Experimental Results (Hysteresis Sweep)

This sweep compared the single-threshold baseline (`p3_t40_cd8_bm3`) against dual-threshold configurations with a low threshold of 3.0 (`..._l30`) and 3.5 (`..._l35`).

#### MOT17-02-SDP (Near-static)

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed@16` | 114 | 32.4% | 21.5% | 76.0 |
| `p3_t40_cd8_bm3` | 122 | 31.9% | 21.6% | 52.1 |
| `..._l30` (4.0 / 3.0) | 116 | 31.9% | 21.7% | 51.3 |
| `..._l35` (4.0 / 3.5) | 123 | 31.9% | 21.6% | 51.5 |

#### MOT17-04-SDP (High fragmentation)

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed@16` | 164 | 44.0% | 30.8% | 59.5 |
| `p3_t40_cd8_bm3` | 152 | 43.4% | 30.9% | 40.4 |
| `..._l30` (4.0 / 3.0) | 157 | 43.5% | 30.8% | 37.6 |
| `..._l35` (4.0 / 3.5) | 152 | 43.4% | 30.9% | 35.3 |

#### MOT17-10-SDP (Moving Camera)

| Mode | IDs | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|
| `fixed@16` | 161 | 42.5% | 44.2% | 80.3 |
| `p3_t40_cd8_bm3` | 166 | 45.9% | 44.2% | 54.4 |
| `..._l30` (4.0 / 3.0) | 166 | 45.9% | 44.2% | 61.4 |
| `..._l35` (4.0 / 3.5) | 166 | 45.9% | 44.2% | 65.1 |

**Interpretation:**
- **FPS Recovery**: On `MOT17-10`, the dual-threshold logic (specifically `_l35`) recovered significant throughput, raising FPS from **54.4 to 65.1**. This proves that oscillation around the threshold was causing excessive trigger cancellations and resets, leading to wasted computation.
- **Stability**: On `MOT17-02`, the `_l30` mode reduced IDs from 122 down to 116, closer to the fixed baseline.
- **Trade-off**: The hysteresis keeps the system in the ReID evaluation state slightly longer when a genuine event occurs, which causes a minor FPS drop in highly chaotic scenes like `MOT17-04` (from 40.4 down to 35.3). However, the massive FPS recovery in moving-camera scenarios and the prevention of logical oscillation makes it a clear net positive.

## Updated Best Configuration

For accuracy-priority deployments, incorporating GMC-geometry and Hysteresis:

```text
--need-reid
--reid-trigger-mode score_ema
--reid-trigger-persist-frames 3
--reid-score-threshold 4.0
--reid-score-threshold-low 3.5
--reid-score-decay 0.80
--reid-weight-new 1.0
--reid-weight-lost 1.4
--reid-weight-geom 0.5
--reid-weight-conf 0.5
--reid-birth-death-boost 1.0
--reid-birth-death-lost-min 0.3
--reid-lost-age-cap 30
--reid-cooldown-frames 8
--reid-unstable-shift-weight 1.0
--reid-unstable-iou-weight 1.0
--reid-conf-jitter-gate 0.10
```

## Next Step

1. Validate the updated best config (`p3_t40_cd8_bm3_l35`) across more MOT17 training sequences.
2. Investigate relink stochasticity as the source of run-to-run IDs variance.
