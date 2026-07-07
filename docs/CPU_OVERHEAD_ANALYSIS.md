# CPU Overhead Analysis — mamba_whole_graph_m Preset

> Date: 2026-07-07
> Scope: per-frame hot path, DB (double-buffer) path, m preset
> Method: `--profile-frame-csv` stage timing + code audit

## Profile Summary (MOT17-02-SDP, 250 frames, DB path)

| Stage                    | Mean (ms) | Notes                                     |
|--------------------------|-----------|--------------------------------------------|
| `total_ms`               | 8.148     | End-to-end per-frame latency               |
| `fetch_ms`               | 0.013     | Frame fetch                                |
| `detect_ms`              | 0.003     | TRT enqueue host-side (GPU work async)     |
| `detect_ingest_barrier`  | 0.000     | Eliminated by event mode                   |
| `detect_postproc_barrier`| 0.000     | Eliminated by event mode                   |
| **`post_ms`**            | **1.740** | **Total postprocessing**                   |
| → `post_tensor_prep`     | 0.360     | dtype cast + contiguity (pre-NMS)           |
| → `post_pre_nms`         | 0.900     | tensor prep + `get_state_snapshots`         |
| → `post_finalize`        | 0.841     | NMS output slicing + keypoint match         |
| → `post_tail`            | 0.153     | detection filters (external_fp `.nonzero()`)|
| `gmc_ms`                 | 0.039     | cuFFT graph                                |
| `track_ms`               | 0.274     | tracker graph replay (host-side)           |
| `output_ms`              | 0.002     | emit (`fast_emit_mot_lines`)               |
| `post_any_sync`          | 0.000     | No `.any()` syncs                          |
| `post_item_sync`         | 0.000     | No `.item()` syncs                         |

Throughput: 247 FPS (4.05 ms/frame). Latency 8.11 ms. Overlap benefit: ~2×.

## Key Finding: `get_state_snapshots()` — 634 KB D2H per frame, 1.2% utilized

### Trigger

`_build_active_track_priors` (`detection_filters.py:1137`) is called from
`_run_native_tensor_prep` (`stages.py:818`) for **private continuation priors**
because the m preset has `private_continuation_enabled: true` and
`private_prior_iou_threshold: 0.30 > 0.0`.

### What it does (`tracker_gpu.cu:3494-3529`)

1. **9× `cudaMemcpyAsync`** — copies 9 tracker-state arrays from GPU to host:
   - `d_active_`     (2048 × 1 B  =    2 KB)
   - `d_states_`     (2048 × 32 B =   64 KB)
   - `d_covs_`       (2048 × 256 B = 512 KB) ← 80% of bandwidth, **unused by priors**
   - `d_age_`        (2048 × 4 B  =    8 KB)
   - `d_scores_`     (2048 × 4 B  =    8 KB)
   - `d_classes_`    (2048 × 4 B  =    8 KB)
   - `d_track_ids_`  (2048 × 4 B  =    8 KB)
   - `d_track_uid_`  (2048 × 8 B  =   16 KB)
   - `d_generation_`  (2048 × 4 B  =    8 KB)
   - **Total: 634 KB D2H per frame**
2. **`cudaStreamSynchronize`** — blocks host until all 9 copies complete
   (and any pending GPU work on the stream).
3. **C++ for-loop over 2048 slots** — finds ~24 active tracks.
4. **Python for-loop over ~24 active tracks** — pybind scalar extraction
   (`int(snap.age)`, `float(snap.score)`, `snap.state[0:4]`, `int(snap.class_id)`),
   computes xyxy from Kalman [cx, cy, a, h].
5. **`torch.tensor(prior_boxes, ...)` H2D** — builds GPU tensor from Python list.

### Waste

- 634 KB D2H for ~24 active out of 2048 slots = **1.2% utilization**.
- `d_covs_` (512 KB) is copied but **never read** by `_build_active_track_priors`.
- The `cudaStreamSynchronize` blocks the host, serializing the pipeline.

### Estimated cost

~0.4–0.5 ms/frame (`post_pre_nms` 0.900 ms − `post_tensor_prep` 0.360 ms ≈ 0.540 ms,
portion is Python/pybind overhead).

## GPU-Offloadable Solution: Compaction Kernel (IMPLEMENTED)

### Approach: Sync-free GPU compaction

Replace `get_state_snapshots()` + Python loop with a GPU kernel that:

1. `cudaMemsetAsync` zero-fills the entire output buffer (max_objs × 4 floats =
   32 KB) so inactive slots have `[0,0,0,0]` boxes (IoU=0, below NMS threshold).
2. Kernel reads `d_active_`, `d_states_`, `d_classes_`, `d_age_`, `d_scores_`
   on GPU. For each active slot passing age/score filters:
   - Reads [cx, cy, a, h] from `d_states_[i*8]`
   - Computes [x1, y1, x2, y2] = [cx-w/2, cy-h/2, cx+w/2, cy+h/2] where w=a*h
   - Writes to compacted output buffer via `atomicAdd` counter
3. Returns fixed upper bound (`max_objs_`) as count — **no D2H, no sync**.
4. NMS kernel iterates over all `max_objs` prior slots; inactive ones have
   `[0,0,0,0]` (zero IoU, below threshold) and are effectively ignored.

### Why sync-free works

The `compute_prior_immunity_kernel` (`tracker_gpu.cu:4497`) iterates
`for (int i = 0; i < num_priors; ++i)` checking IoU with each prior box.
A `[0,0,0,0]` box has zero area → IoU = 0 → below `prior_iou_threshold` →
no immunity granted.  Safe for NMS correctness.

### Also eliminated: second `_build_active_track_priors` call

The evaluator (`evaluator.py:1354`) previously called
`_build_active_track_priors` a second time for private motion priors,
with the same filters as the first call (`stages.py:818`).  Replaced with
`_fctx.private_prior_boxes` (reuse of first call's result).

### Results (MOT17-02-SDP, 300 frames, DB path)

| Metric              | Phase 2 (before) | Phase 3 (after) | Delta  |
|---------------------|-------------------|------------------|--------|
| Throughput          | 246.82 FPS        | 273.25 FPS       | +10.7% |
| Mean latency        | 8.11 ms           | 7.30 ms          | -10.0% |
| `total_ms`          | 8.148 ms          | 7.300 ms         | -0.849 |
| `post_ms`           | 1.740 ms          | 1.506 ms         | -0.234 |
| `post_pre_nms_ms`   | 0.900 ms          | 0.712 ms         | -0.188 |
| `post_tensor_prep`  | 0.360 ms          | 0.105 ms         | -0.255 |
| Bit-exact           | —                 | 0 diff           | ✅     |

### Files changed

| File | Change |
|------|--------|
| `src/tracking/tracker_gpu.cu` | `build_track_priors_kernel` + `Impl::build_track_priors_gpu` + `GPUByteTracker::build_track_priors_gpu` + `d_prior_count_`/`h_prior_count_` members |
| `include/tracking/tracker_gpu.hpp` | `build_track_priors_gpu` declaration |
| `src/tracking/tracker_gpu_python.cpp` | pybind binding |
| `src/saccade/perception/tracking/tracker_gpu.py` | `_prior_boxes`/`_prior_classes` pre-alloc + `build_track_priors_gpu` wrapper |
| `src/saccade/perception/eval/detection_filters.py` | `_build_active_track_priors` GPU fast path via `hasattr` |
| `src/saccade/perception/eval/evaluator.py` | Reuse `_fctx.private_prior_boxes` instead of second call |
| `tests/unit/eval/test_evaluator.py` | `MagicMock(spec=...)` for fallback path test |

## Other Findings (secondary)

| Item                              | Location                        | Cost      | GPU-offloadable?                          |
|-----------------------------------|---------------------------------|-----------|-------------------------------------------|
| `post_tensor_prep` dtype casts    | `stages.py:774-776`             | 0.36 ms   | May include detection GPU wait; no-op if already correct dtype |
| `post_finalize`                   | `stages.py:850-967`             | 0.84 ms   | Mostly views + no-op keypoint match; may include NMS graph GPU wait |
| `_apply_external_fp_filter` `.nonzero()` | `detection_filters.py:970` | ~0.08 ms  | Could use bool-mask in-place (like `_fp_hard_reject_mask`) |
| `fast_emit_mot_lines` Python loop | `helpers.py:302-312`            | 0.002 ms  | Inherently CPU (f-string); C++ batch formatter possible |
| `event.synchronize()` (deferred emit) | `helpers.py:177`            | inherent  | Async D2H固有; already overlapped via DB  |

## Not in Per-Frame Hot Path

| Item                  | Location                        | Notes                                              |
|-----------------------|---------------------------------|---------------------------------------------------|
| Interpolation         | `post_merge.py:359-447`         | Post-sequence; pandas + Python loop + sort         |
| Non-DB sync barriers  | `stages.py:1183, 1286`          | ~3-5 ms/frame; already solved by event mode (DB)  |

## Disabled for m Preset (but heavy if enabled)

| Filter                        | Sync overhead if enabled            |
|-------------------------------|-------------------------------------|
| `duplicate_suppression`       | O(n²) IoU + O(n) `.any()` syncs     |
| `multi_birth_enabled`         | O(n_sub) with 4n `.item()` syncs    |
| `enable_onms` (SACCADE_ENABLE_ONMS) | O(n_tracks) per-track pybind syncs |
| `birth_consecutive_gate`      | O(w × n_sub × n_prev) + `.any()`    |
| `adaptive_detection_cap`      | 2 `.item()` syncs                   |
| `stage2_quality_gate`         | 2 `.any()` syncs                    |
