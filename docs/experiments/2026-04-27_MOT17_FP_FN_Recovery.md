# MOT17 FP/FN Recovery Experiments

Date: 2026-04-27

## Summary

This round started from a pure-FN-reduction probe and then recovered MOTA by preventing low-confidence detections from creating noisy new identities.

Final best SDP-wide setting in this round:

This setting is now the default for `scripts/eval/mot17.py`; lifecycle and post-lifecycle merge remain opt-in and disabled by default.

```bash
--reid-mode semantic \
--conf-threshold 0.05 \
--track-thresh 0.05 \
--mid-thresh 0.10 \
--new-track-thresh 0.45 \
--confirm-streak 1 \
--confirm-score-thresh 0.0 \
--id-stability-filter \
--id-stability-min-hits 2 \
--id-stability-min-iou 0.05 \
--id-stability-max-center-shift 2.0 \
--id-stability-max-gap 1 \
--id-stability-min-score-ema 0.15 \
--person-geometry-prior \
--person-min-height-ratio 0.018 \
--person-min-aspect 1.0 \
--person-max-aspect 5.5 \
--person-min-area-ratio 0.00006 \
--geometry-suspect-support \
--semantic-threshold 0.95 \
--semantic-spatial-gate 0.20 \
--semantic-min-iou 0.20 \
--semantic-mahalanobis-threshold 0
```

Best result:

- IDF1: `41.2%`
- Rcll: `50.3%`
- Prcn: `76.7%`
- FP: `17,155`
- FN: `55,763`
- IDs: `1,117`
- MOTA: `34.1%`
- Eval throughput: `95.94 FPS`

Compared with the earlier stable baseline `tentative_sdp_off`, this improves MOTA from `31.9%` to `34.1%`, improves recall from `48.5%` to `50.3%`, and reduces FN from `57,872` to `55,763`. FP is slightly lower (`17,959 -> 17,155`), while IDs remain higher (`656 -> 1,117`).

## Implemented Controls

### New-Track Threshold

Added a separate tracker parameter:

```bash
--new-track-thresh
```

This decouples low-score association from new ID creation:

- `track_thresh` and `mid_thresh` can stay low so weak detections can continue existing tracks.
- `new_track_thresh` controls which unmatched detections may create new tentative tracks.
- Default behavior remains backward-compatible: if unset, `new_track_thresh = mid_thresh`.

Implemented in:

- `include/tracking/tracker_gpu.hpp`
- `src/tracking/tracker_gpu.cu`
- `src/tracking/tracker_gpu_python.cpp`
- `perception/tracking/tracker_gpu.py`
- `perception/eval/runner.py`
- `scripts/eval/mot17.py`

### ID Stability Filter

Added an eval-layer output gate:

```bash
--id-stability-filter
--id-stability-min-hits
--id-stability-min-iou
--id-stability-max-center-shift
--id-stability-max-gap
--id-stability-score-ema
--id-stability-min-score-ema
```

The filter tracks each raw ID's recent bbox continuity and score EMA. Unstable IDs are not written to MOT output, but the tracker state still updates internally.

### Person Geometry Prior

Added a bbox-shape prior before tracker update:

```bash
--person-geometry-prior
--person-min-height-ratio
--person-min-aspect
--person-max-aspect
--person-min-area-ratio
--person-max-area-ratio
```

This uses only bbox geometry, not image content:

- height ratio
- height/width aspect ratio
- area ratio

It targets obvious non-person false positives and malformed tile artifacts.

### Geometry Suspect Support

Added:

```bash
--geometry-suspect-support
--geometry-suspect-score
```

Instead of dropping detections that fail the person geometry prior, they can be kept as internal auxiliary observations:

- suspect boxes are score-clamped into the low-score band `track_thresh < score < mid_thresh`;
- with ByteTrack's existing stages, these can only support confirmed tracks;
- they cannot open new IDs;
- if the current output is matched to a suspect box, it is not written to MOT results.

This preserves some continuity support from partial/occluded/non-upright people without directly adding MOT false positives.

## Pure Recall Probe

The first probe intentionally lowered every threshold:

```bash
--conf-threshold 0.05 \
--track-thresh 0.05 \
--mid-thresh 0.05 \
--confirm-streak 1 \
--confirm-score-thresh 0.0 \
--reid-mode off
```

| Config | IDF1 | Rcll | Prcn | FP | FN | IDs | MOTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tentative_sdp_off` baseline | 45.8 | 48.5 | 75.2 | 17,959 | 57,872 | 656 | 31.9 |
| pure recall `0.05` | 31.3 | 61.7 | 53.4 | 60,355 | 43,051 | 8,350 | 0.5 |
| pure recall `0.01` | 16.6 | 68.7 | 27.9 | 199,032 | 35,186 | 12,821 | -120.0 |

Conclusion:

- The detector has enough low-score signal to raise recall substantially.
- MOTA collapses because low-score detections create too many new IDs and FP outputs.

## Recovery Experiments

### ID Stability and ReID

| Config | IDF1 | Rcll | Prcn | FP | FN | IDs | MOTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure recall `0.05` | 31.3 | 61.7 | 53.4 | 60,355 | 43,051 | 8,350 | 0.5 |
| ID stability, ReID off | 31.7 | 52.8 | 66.2 | 30,299 | 52,999 | 4,601 | 21.7 |
| ID stability + semantic ReID | 34.5 | 53.1 | 66.7 | 29,754 | 52,720 | 3,890 | 23.1 |

Semantic ReID helped IDF1 and IDs, but FP remained the dominant MOTA limiter.

### New-Track Threshold Sweep

All rows use:

```bash
--conf-threshold 0.05
--track-thresh 0.05
--mid-thresh 0.05
--confirm-streak 1
--confirm-score-thresh 0.0
--reid-mode semantic
--id-stability-filter
--id-stability-min-score-ema 0.15
```

| `new-track-thresh` | IDF1 | Rcll | Prcn | FP | FN | IDs | MOTA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 36.1 | 53.5 | 68.1 | 28,178 | 52,234 | 3,322 | 25.4 |
| 0.20 | 38.4 | 53.4 | 68.7 | 27,371 | 52,323 | 2,662 | 26.7 |
| 0.25 | 38.2 | 52.8 | 69.3 | 26,210 | 53,034 | 2,804 | 26.9 |
| 0.30 | 38.5 | 52.5 | 70.4 | 24,832 | 53,352 | 2,549 | 28.1 |
| 0.35 | 38.1 | 51.9 | 71.4 | 23,344 | 54,021 | 2,127 | 29.2 |
| 0.40 | 39.0 | 51.2 | 72.6 | 21,640 | 54,829 | 1,826 | 30.3 |
| 0.45 | 40.9 | 50.3 | 73.4 | 20,487 | 55,774 | 1,470 | 30.8 |

Conclusion:

- Separating low-score association from new-track creation was the largest structural improvement.
- `0.45` was the best before adding geometry prior.
- Recall remained above the earlier stable baseline while FP approached baseline levels.

### Person Geometry Prior

Using `new-track-thresh=0.45`:

| Config | IDF1 | Rcll | Prcn | FP | FN | IDs | MOTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| no geometry prior | 40.9 | 50.3 | 73.4 | 20,487 | 55,774 | 1,470 | 30.8 |
| geometry mild | 41.7 | 50.8 | 74.9 | 19,131 | 55,265 | 1,199 | 32.7 |
| geometry strict | 41.1 | 50.5 | 76.3 | 17,659 | 55,569 | 1,256 | 33.7 |
| geometry suspect support | 41.2 | 50.3 | 76.7 | 17,155 | 55,763 | 1,117 | 34.1 |

Geometry settings:

Mild:

```bash
--person-min-height-ratio 0.012
--person-min-aspect 0.75
--person-max-aspect 7.0
--person-min-area-ratio 0.00003
```

Strict:

```bash
--person-min-height-ratio 0.018
--person-min-aspect 1.0
--person-max-aspect 5.5
--person-min-area-ratio 0.00006
```

Suspect support used strict geometry plus:

```bash
--mid-thresh 0.10
--geometry-suspect-support
```

Conclusion:

- Geometry prior reduced FP enough to beat the original stable baseline.
- Suspect support was slightly better than hard-dropping suspect boxes: FP and IDs decreased while MOTA improved from `33.7` to `34.1`.

## Current Interpretation

The best working architecture is:

```text
Detector: keep low-score person detections
Association: allow low-score detections to continue confirmed tracks
New ID creation: require high enough score via new-track-thresh
Output: require short ID stability
Appearance: semantic ReID as conservative post-association relink
Geometry: suspicious shapes become auxiliary support, not direct output
```

This preserves some recall benefit from low-confidence detections while keeping FP under control.

Remaining issue:

- IDs are still higher than the earlier stable baseline (`1,117` vs `656`).
- The next useful direction is likely an ID lifecycle penalty or stronger relink/merge policy for short-lived IDs, not further lowering detection thresholds.

## ID Lifecycle and Short-Lived Merge Analysis

After the best geometry-suspect configuration, we tested whether ID fragmentation could be repaired by merging recently-lost IDs into newly-created IDs.

Implemented controls:

```bash
--lifecycle-merge
--lifecycle-ttl
--lifecycle-min-gap
--lifecycle-spatial-gate
--lifecycle-min-iou
--lifecycle-sim-threshold
--lifecycle-require-embedding
--lifecycle-ema
```

This online lifecycle merger runs after semantic relink and before global output-ID mapping. It keeps recently-lost local IDs, then aliases a new local ID back to an older output ID when temporal, spatial, IoU, and optional embedding gates pass.

Also implemented an offline output-tracklet merge:

```bash
--post-lifecycle-merge
--post-lifecycle-ttl
--post-lifecycle-min-gap
--post-lifecycle-velocity-samples
--post-lifecycle-spatial-weight
--post-lifecycle-motion-weight
--post-lifecycle-time-weight
--post-lifecycle-direction-weight
--post-lifecycle-max-cost
```

The offline path builds output tracklets after each sequence, computes a spatio-temporal cost matrix, and solves a one-to-one assignment with Hungarian matching. The cost includes:

- time gap
- raw spatial distance
- forward extrapolation from the lost tracklet into the new tracklet start
- backward extrapolation from the new tracklet into the lost tracklet end
- motion-compensated IoU
- direction cosine penalty

### Lifecycle Results

All rows below start from the best geometry-suspect-support setup unless otherwise noted.

| Config | IDF1 | Rcll | Prcn | FP | FN | IDs | MOTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry suspect support | 41.2 | 50.3 | 76.7 | 17,155 | 55,763 | 1,117 | 34.1 |
| online lifecycle, spatial/optional embedding | 39.1 | 50.3 | 76.7 | 17,173 | 55,762 | 1,025 | 34.1 |
| online lifecycle, require embedding | 40.0 | 50.0 | 76.4 | 17,334 | 56,170 | 1,261 | 33.4 |
| offline Hungarian, loose | 37.2 | 50.2 | 76.6 | 17,223 | 55,909 | 1,185 | 33.8 |
| offline Hungarian, strict | 40.7 | 50.3 | 76.7 | 17,188 | 55,796 | 1,300 | 33.9 |

Conclusion:

- Online lifecycle merge can reduce reported ID switches, but current gates also create wrong merges; IDF1 drops from `41.2` to `39.1`.
- Requiring embeddings was too conservative and still not accurate enough; it reduced recall and MOTA.
- Offline Hungarian merge was not safer by itself. The loose setting over-merged, especially on crowded sequences, and strict matching failed to improve the best setup.
- The implemented lifecycle tools should remain disabled by default.

### Fragmentation Diagnosis

Comparing the earlier baseline and current best:

- output local-ID counts are similar (`924` baseline vs `913` current best);
- MOT ID switches increased (`656 -> 1,117`);
- short-lived IDs increased only modestly;
- the larger issue is fragmentation of longer tracks after gaps, not a flood of one-frame IDs.

This means a simple short-lived-ID suppressor or greedy relinker is not enough. The failure mode is mainly incorrect identity recovery after occlusion/gaps.

### Next Direction

The next lifecycle attempt should focus on cleaner appearance evidence before doing more aggressive merging:

- store tracklet-level ReID history rather than using only the latest embedding;
- keep Top-K high-confidence, geometry-clean embeddings per tracklet;
- compute an EMA or robust mean embedding for merge candidates;
- reject merge candidates whose clean embedding set has low internal consistency;
- use the offline spatio-temporal cost only as a candidate generator, then require strong appearance agreement.

Until that exists, the recommended best configuration remains `geometry_suspect_support_newtrack045_reid_sdp` without lifecycle merge.
