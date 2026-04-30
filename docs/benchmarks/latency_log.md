# 2026-04-30 ReID Trigger and Latency Investigation

## Scope

- Profiled `scripts/eval/mot17.py` on single MOT17 sequences with `--profile-stages`.
- Added trusted stage jitter reporting and native ReID breakdown support.
- Evaluated ReID trigger policy changes against latency and tracking quality.

## Kept Code Changes

- `TrackAppearanceBank` now stores normalized appearance samples on CPU and uploads batched references back to GPU when needed.
- Stage profiling now reports exclusive top-level per-frame jitter for:
  `fetch`, `ingest_preprocess`, `detect`, `postprocess`, `reid_bank_sync`,
  `reid_crop`, `reid_extract`, `lazy_reid`, `track`, `relink_write`,
  `frame_total`.
- Native ReID profiling now breaks `reid_extract` into:
  `native_reid_crop`, `native_reid_pre_normalize`,
  `native_reid_trt_enqueue`, `native_reid_l2_normalize`.

## Main Latency Findings

- `detect` is the largest steady fixed cost, but is not the main jitter source.
- `relink_write` was reduced substantially by moving the appearance bank to CPU.
- Remaining tail latency is dominated by native ReID TensorRT inference.
- Native ReID breakdown on `MOT17-09-SDP` showed:
  - `native_reid_crop`: ~`0.02 ms`
  - `native_reid_pre_normalize`: ~`0.02 ms`
  - `native_reid_trt_enqueue`: ~`6.5-8.0 ms` tail event
  - `native_reid_l2_normalize`: ~`0.04 ms`

## Current Default ReID Trigger Policy

Defaults intentionally kept at:

- `reid_trigger_mode=event_any`
- `reid_score_threshold=2.0`
- `reid_score_threshold_low=2.0`
- `reid_trigger_persist_frames=2`
- `reid_cooldown_frames=4`
- `reid_birth_death_lost_min=0.1`

Reason:

- A more aggressive `score_ema` default improved `MOT17-09-SDP`, but hurt
  `MOT17-04-SDP`.
- A narrower `p2/c2` variant (`persist=2`, `cooldown=2`) looked strong on
  `MOT17-04-SDP` and `MOT17-09-SDP`, but was unstable on `MOT17-02-SDP`.
- `p2/c4` is the safer cross-sequence default for now.

## Sequence Comparison Summary

Comparison baseline:

- `event_any`
- `score_threshold=2.0`
- `score_threshold_low=2.0`
- `persist=1`
- `cooldown=0`
- `birth_death_lost_min=0.0`

Current default (`p2/c4`) versus old baseline:

### MOT17-09-SDP

- `FPS`: `90.17 -> 93.96`
- `mean_ms`: `11.09 -> 10.64`
- `IDF1`: `46.7 -> 45.8`
- `MOTA`: `39.5 -> 41.4`
- `IDs`: `182 -> 132`

### MOT17-04-SDP

- `FPS`: `58.84 -> 58.63`
- `mean_ms`: `17.00 -> 17.06`
- `IDF1`: `45.2 -> 44.3`
- `MOTA`: `30.0 -> 30.8`
- `IDs`: `232 -> 170`

### MOT17-02-SDP

- `FPS`: `77.29 -> 77.41`
- `mean_ms`: `12.94 -> 12.92`
- `IDF1`: `32.2 -> 31.7`
- `MOTA`: `21.1 -> 21.2`
- `IDs`: `152 -> 163`

## Notes

- `p2/c2` was not promoted to default because reruns on `MOT17-02-SDP` showed
  unstable throughput and IDF1.
- Keep using trusted jitter profiling when comparing future trigger-policy
  changes.
