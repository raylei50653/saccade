# Legacy Module-Benchmark Baseline

This file records the 2026-05 module-benchmark baseline and profiling notes.
For current MOT17 headline baseline numbers, use [mot17_default_config.md](../mot17_default_config.md)
and [DATAFLOW.md](../../DATAFLOW.md). Use the native_960 baseline below only when comparing
against the legacy module-benchmark harness:

- Baseline directory: `results/module_benchmark/20260615_141731`
- Entry point: `./scripts/eval/module_benchmark.sh --mode all`
- Detector: `SDP`
- Sequences: `MOT17-04-SDP,MOT17-10-SDP`
- Engine: `models/yolo/yolo26s_960_batch1.engine`
- Tiling: `native_960`
- Max frames: `100`

Legacy module-benchmark summary:

- validate: `65.12 FPS`, `15.36 ms`, `IDF1 9.4%`, `MOTA 4.5%`
- profile main stages: `detect 4.53 ms`, `postprocess 1.40 ms`, `track 1.10 ms`, `reid_extract 1.88 ms`, `relink_write 3.46 ms`
- best current ablation signal: `geometry mid-scale` (`IDF1 9.9%`, `MOTA 4.7%`, `IDs 14`)

Use `results/module_benchmark/20260615_141731/notes.md` and
`results/module_benchmark/20260615_141731/experiment_matrix.md` as the
legacy module-benchmark record. The sections below remain useful as historical profiling
context and optimization notes.

# 2026-05-06 Inter-Frame Relink Pipelining (`--pipeline-relink`)

## Design

- `ThreadPoolExecutor(max_workers=1)`: main thread runs GPU path (fetch→detect→reid→track),
  bg thread runs `relink_write` (CPU Python) for the previous frame.
- GIL release: `torch.cuda.synchronize()` in `time_stage("detect", sync_cuda=True)` releases
  GIL for ~4ms → bg thread runs Python/CPU code during this window.
- Pre-materialization: GPU tensors (`fused_boxes`, `embeddings`, `boxes_gpu`, `gmc_warp`) moved
  to CPU before submit. `motion_candidate_ids` + `motion_snapshots` computed in main thread
  (need main CUDA stream context). Avoids CUDA stream conflicts in background thread.
- Sync point: after postprocess, before `dynamic_reid.should_reid()` (first shared mutable state access).

## Results (7-seq SDP A/B, back-to-back, 2026-05-06)

| Metric | Baseline | `--pipeline-relink` | Δ |
|--------|----------|---------------------|---|
| IDF1 | 47.9% | 47.9% | 0 |
| MOTA | 40.7% | 40.7% | 0 |
| IDs | 651 | 655 | noise |
| **Eval FPS** | **70.05** | **71.80** | **+1.75 (+2.5%)** |
| mean frame | 14.28ms | 13.93ms | **-0.35ms** |

Per-sequence gains: MOT17-04 +4.9% (57.93→60.76), MOT17-02 +6.2%, MOT17-13 +1.4%.

## Why Smaller Than Expected

Profile-stages showed `relink_write` = 5.4ms. Without profiling (no forced GPU syncs per stage),
actual wall-clock relink_write ≈ 2ms (GPU ops overlap asynchronously). Python GIL contention
between bg and main threads adds ~1ms overhead. Net: ~2ms hidden, ~1.5ms overhead → 0.35ms/frame.

Expected gain grows with heavier relink_write (more tracks, longer sequences).
`--profile-stages` is auto-disabled when `--pipeline-relink` is active.

---

# 2026-05-06 Stage Baseline Before ReID Pipelining

## Scope

- Profiled `native_960` path on `MOT17-04-SDP` (1000 frames, `--profile-stages`).
- Establishing per-stage baseline before implementing async ReID pipelining.

## E2E Stage Breakdown — Legacy Baseline (native_960, MOT17-04-SDP, 2026-05-06)

| Stage | Mean | Std | P95 | P99 |
|-------|------|-----|-----|-----|
| `fetch` | 0.55ms | 0.12ms | 0.79ms | 0.99ms |
| `ingest_preprocess` | 1.12ms | 0.41ms | 2.23ms | 2.64ms |
| `detect` | 5.95ms | 1.17ms | 8.23ms | 9.42ms |
| `postprocess` | 2.25ms | 0.88ms | 4.09ms | 4.52ms |
| `reid_bank_sync` | 0.68ms | 0.79ms | 2.12ms | 3.16ms |
| `reid_budget` | 0.86ms | 0.95ms | 2.62ms | 3.30ms |
| `reid_extract` | 2.90ms | 3.00ms | 7.08ms | 7.85ms |
| `track` | 1.42ms | 0.44ms | 2.38ms | 3.29ms |
| `relink_write` | 7.84ms | 6.70ms | 16.99ms | 18.31ms |
| **frame_total** | **24.87ms** | 11.33ms | **39.78ms** | 41.57ms |

Throughput: **40.21 FPS**

ReID TRT breakdown (on ReID frames):
- `native_reid_trt_enqueue`: 5.13ms mean, P95 6.78ms
- crop / pre_normalize / l2_normalize: <0.1ms each

## Async ReID Pipelining Results (2026-05-06)

- `reid_extract` submitted to a side CUDA stream (`reid_side_stream`) after `reid_budget`
- GMC estimation on main stream overlaps with reid on side stream (~1ms overlap)
- Sync point: after GMC / `set_reference_features`, right before `tracker.update_into`
- Tracker and bank/relink still receive fresh embeddings — no accuracy regression

7-seq SDP A/B (2026-05-06):

| Metric | Baseline | `--async-reid` | Δ |
|--------|----------|----------------|---|
| IDF1 | 47.8% | 47.8% | 0 |
| MOTA | 40.6% | 40.5% | -0.1pp (noise) |
| IDs | 661 | 661 | 0 |
| FP | 11,000 | 11,027 | +27 (noise) |
| **Eval FPS** | **54.90** | **56.34** | **+1.44 (+2.6%)** |
| mean frame | 18.21ms | 17.75ms | **-0.46ms** |

Note: `--profile-stages` adds `torch.cuda.synchronize()` at every stage boundary,
serializing all GPU work and eliminating parallelism benefit. Profile measurements
of async-reid show overhead rather than gain; compare without `--profile-stages`.

Implementation: `--async-reid` flag in `scripts/eval/mot17.py`; internal via
`async_reid=True` config in `evaluator.py` (`runner.py` re-exports `run_eval`).

---

# 2026-05-05 E2E Latency Profiling and relink_write Optimizations

## Scope

- Profiled `native_960` path on `MOT17-04-SDP` (200 frames, `--profile-stages`).
- Measured raw TRT inference for `yolo26s_960_batch1.engine` vs `960 2×2 tiled` (batch=4 @ 640).
- Identified `relink_write` as dominant bottleneck via wall-clock sub-stage timing.
- Applied two targeted optimizations without accuracy regression.

## Engine Latency Comparison (raw TRT, no pipeline overhead)

| Engine | Input | Mean | P50 | P99 | FPS |
|--------|-------|------|-----|-----|-----|
| `yolo26s_960_batch1.engine` | 960×960 batch=1 | 4.20ms | 4.21ms | 4.68ms | 238 |
| `yolo26s_batch4.engine` | 640×640 batch=4 (2×2 tiles) | 4.83ms | 4.82ms | 5.28ms | 207 |

`native_960` is 15% faster than 2×2 tiled at the TRT level. Tiled path also carries
additional cross-tile merge overhead (not counted above), further widening the gap.

## E2E Stage Breakdown — Baseline (native_960, MOT17-04-SDP)

Measured with `--profile-stages`, 150 evaluated frames after warmup:

| Stage | Mean | Std | P95 | P99 |
|-------|------|-----|-----|-----|
| `relink_write` | 8.04ms | 7.25ms | 17.46ms | 18.07ms |
| `detect` | 5.34ms | 0.92ms | 7.11ms | 7.63ms |
| `reid_extract` | 2.34ms | 2.35ms | 5.10ms | 6.08ms |
| `postprocess` | 1.54ms | 0.45ms | 2.24ms | 2.98ms |
| `track` | 1.22ms | 0.42ms | 1.98ms | 2.44ms |
| `reid_budget` | 0.63ms | 0.66ms | 1.57ms | 1.91ms |
| `ingest_preprocess` | 0.88ms | 0.12ms | 1.13ms | 1.25ms |
| `reid_bank_sync` | 0.35ms | 0.37ms | 0.81ms | 1.08ms |
| `fetch` | 0.43ms | 0.07ms | 0.57ms | 0.61ms |
| **frame_total** | **21.7ms** | **10.3ms** | **34.1ms** | **35.8ms** |

Throughput: **46 FPS**.

`detect` raw overhead: raw TRT = 4.20ms, pipeline cost = 5.34ms → 1.14ms letterbox/canvas overhead.

## relink_write Root Cause Analysis

Wall-clock sub-stage breakdown inside `relink_write` (8.04ms total):

| Sub-stage | Mean | P95 |
|-----------|------|-----|
| `_prepare_track_candidates` | **9.3ms** | **20.6ms** |
| `_resolve_frame_tracks` | 0.89ms | 2.07ms |
| `_finalize_frame_side_effects` | 0.45ms | 0.60ms |
| `_prepare_host_track_batch` | 0.08ms | 0.10ms |
| `_emit_resolved_tracks` | 0.11ms | 0.15ms |
| `bank_prune` | 0.01ms | 0.02ms |

`_prepare_track_candidates` is the dominant cost. cProfile (tottime) breakdown:

- `_refresh_track` × 892 calls / 100 frames: pairwise cosine `embs @ embs.T` (O(K²) per update)
- `_build_prepared_candidates`: per-candidate GPU scalar extractions (`fused_scores[i]`, `fused_boxes[i].tolist()`, `geometry_suspect_mask[i]`)
- `IdStabilityFilter.accept_many`: Python loop with `_iou` / `_center_shift_ratio` per track (small)

High P95 (20.6ms) driven by ReID trigger frames where ~16 tracks hit `_refresh_track` simultaneously.

## Optimizations Applied (2026-05-05)

### Fix 1: `_refresh_track` — O(K²) → O(1) consistency (`tracker_gpu.py:673`)

**Problem**: `sims = embs @ embs.T` computes full K×K pairwise cosine, done twice (all bank + HQ bank).

**Fix**: Replace with `||mean_unnorm||²` formula, which is mathematically equivalent:
```
mean_pairwise_cosine = (K * ||mean_unnorm||² - 1) / (K - 1)
```
where `mean_unnorm = embs.mean(dim=0)` is already computed for the representative.
O(K²·D) → O(D) (D=768), verified numerically for K=2..5 to <1e-5 error.

Same fix applied to HQ representative consistency block.

### Fix 2: `_build_prepared_candidates` — batch D2H (historical runner-era location)

**Problem**: Phase 3 loop called `float(fused_scores[det_idx])`, `bool(geometry_suspect_mask[det_idx])`,
`fused_boxes[det_idx].tolist()` per candidate — 3 separate GPU→CPU syncs per track.

**Fix**: Extend the Phase 2 batch D2H to include det_score, box coords, and suspect mask
in a single `torch.stack([...], dim=1).cpu().tolist()`. Phase 3 reads from the pre-fetched dict,
eliminating all per-candidate GPU operations.

## Results After Optimization

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| `relink_write` mean | 8.04ms | 5.4ms | **-2.6ms (-32%)** |
| `relink_write` P95 | 17.5ms | 11.7ms | **-5.8ms (-33%)** |
| `frame_total` mean | ~21.7ms | ~21.2ms | ~-0.5ms (noise) |
| FPS | ~46 | ~47 | +1 |

P95 improvement (-5.8ms) is the most meaningful signal: ReID-frame latency spikes are substantially reduced.
Accuracy metrics unchanged (10 tests pass, IDF1/MOTA stable).

## Remaining Opportunities (not yet implemented)

| Area | Estimated Gain | Approach |
|------|---------------|----------|
| `detect` letterbox overhead | ~0.5ms | async canvas fill / precompute scales |
| `_resolve_frame_tracks` (lifecycle merger) | ~0.5ms | batch Python → possible C++ port |
| ReID stream pipelining | ~2ms throughput | overlap ReID extract with next-frame detect |

---

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
