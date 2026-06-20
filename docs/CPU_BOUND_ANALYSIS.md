# CPU-Bound Operations: `mamba_whole_graph` + `--relink-bridge-enabled`

> Command: `uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --relink-bridge-enabled`

---

## Active Configuration Summary

| Feature | Value | Effect |
|---|---|---|
| `use_whole_graph` | `true` | Detect (TRT backbone + Mamba head + postprocess) → single CUDA graph replay |
| `use_cuda_graph` | `true` | NMS graph captured |
| `use_tracker_graph` | `true` | Tracker update graph captured |
| `gmc` | `true` | C++ cuFFT GMC with CUDA graph capture |
| `reid_mode` | `off` | No ReID extraction |
| `external_fp_filter_mode` | `off` (default) | No logistic/softmax3 FP filter |
| `detection_quality_scaling` | `false` | No quality-based score scaling |
| `birth_quality_gate` | `false` (default) | No birth quality gate |
| `birth_consecutive_gate` | `false` (default) | No consecutive birth gate |
| `multi_birth_enabled` | `false` (default) | No multi-signal birth |
| `duplicate_suppression` | `false` (default) | No duplicate suppression |
| `per_frame_detection_cap` | `0` (default) | No detection cap |
| `stage2_quality_gate` | `false` (default) | No stage-2 quality gate |
| `id_stability_filter` | `false` | No ID stability filter |
| `use_semantic_mode` | `false` (derived from reid_mode=off) | No Python/C++ relinker |
| `appearance_bank` | `false` (default) | No primary_appearance_bank |
| `lifecycle_merge` | `false` (default) | Lifecycle merger disabled |
| `post_lifecycle_merge` | `false` (default) | Post-lifecycle merge skipped |
| `cheb_gr_merge_enabled` | `false` (default) | Cheb-GR merge skipped |
| `relink_bridge_enabled` | `true` (CLI) | Bridge params → C++ tracker (GPU, in-graph) |
| `pipeline_relink` | `true` (default) | Background-thread MOT emission |
| `fp_hard_filter_enabled` | `true` (default) | GPU element-wise masking |
| `interpolate_tracklets` | `true` | Post-sequence gap interpolation |

---

## 1. Per-Frame CPU Operations (Hot Loop)

The per-frame loop is in [evaluator.py](../src/saccade/perception/eval/evaluator.py).

### 1.1 Fetch + Ingest + Preprocess

| Location | Operation | Type | Notes |
|---|---|---|---|
| `evaluator.py:2406-2411` | `next(stream_iter)` | GPU | `SACCADE_GPU_DECODE=1` → NVJPG on GPU |
| `evaluator.py:2700-2727` | HWC→CHW + `/255` + `apply_frame_preprocess()` | GPU | All GPU element-wise |

**With GPU decode, this stage is fully GPU-bound.**

### 1.2 Detect

| Location | Operation | Type | Notes |
|---|---|---|---|
| `evaluator.py:2543-2563` | `detect_fn()` → `detect_native_640()` | GPU | One CUDA graph replay: TRT backbone + Mamba head + decode |

**One CUDA graph replay. CPU cost is graph launch overhead (~μs).**

#### Stage barriers (`_run_detect`) — `SACCADE_DETECT_BARRIER`

`_run_detect` issues two unconditional `torch.cuda.synchronize()` full-device
barriers — ingest→detect and detect→postprocess — added as the determinism fix
for the ingest→detect stale-buffer race (commit 3046ae60). They block the host
and serialise the per-frame stages. The env flag `SACCADE_DETECT_BARRIER` selects
the policy:

| Value | Behaviour | FPS (MOT17-02, 200f) | Safety |
|---|---|---|---|
| `full` (default) | Both full barriers | 213.4 (baseline) | Correct for **all** presets |
| `no_postproc` | Drop detect→postprocess barrier only | **217.1 (+1.7%)** | whole_graph only: keeps decode-race guard; the dropped barrier is same-stream-redundant (TRT + postprocess graphs both launch from `current_stream`) |
| `event` | Also drop ingest→detect barrier | 220.3 (+3.2%) | Removes the decode-race guard — **not** validated across all sequences |

`no_postproc` is bit-exact to `full` (CPU decode) and showed zero run-to-run drift
across 30+ GPU-decode runs on MOT17-02. The default stays `full` because the
detect→postprocess barrier is only redundant in whole_graph mode; non-whole-graph
presets may route TRT on a dedicated stream where it is load-bearing. Treat
`event` as experimental pending a full 7-sequence determinism burn-in.

### 1.3 Post-Process (NMS + Filter)

| Location | Operation | Type | Notes |
|---|---|---|---|
| `evaluator.py:2858-3064` | `PerceptionPipeline.process_detections_graph()` | GPU | Filter + NMS in CUDA graph (`_nms_graph.replay()`) |
| `evaluator.py:3463-3477` | `_fp_hard_reject_mask()` | GPU | Element-wise score masking |
| `evaluator.py:3386-3411` | Tail score floor filtering | GPU | Element-wise masking |

**All GPU. CPU is graph launch overhead only.**

### 1.4 GMC (Global Motion Compensation)

| Location | Operation | Type | Notes |
|---|---|---|---|
| `evaluator.py:4252-4296` | C++ `GMC.estimate_into_direct()` | GPU | cuFFT in CUDA graph (`_gmc_cuda_graph[0].replay()`) |
| `evaluator.py:4332-4337` | `pcr_score()` | CPU | Single D2H float read |

**Near-zero CPU after first-frame graph capture.**

### 1.5 Tracker Update

| Location | Operation | Type | Notes |
|---|---|---|---|
| `evaluator.py:4387-4402` | `GraphedTrackerUpdate.replay()` | GPU | CUDA graph replay |
| (in-graph) | Bridge relink | GPU | C++ tracker executes bidirectional relink via `set_relink_params()` |

**One CUDA graph replay. Bridge relink is inside the GPU graph. CPU cost is graph launch only.**

### 1.6 Materialize (GPU → CPU)

| Location | Operation | Cost |
|---|---|---|
| `evaluator.py:4418-4446` | `_materialize_gpu_track_results_pinned()` | **CPU** |
| `helpers.py:68-125` | 5× `copy_(..., non_blocking=True)` + 1× `torch.cuda.synchronize()` | **CPU stall** |

Pinned memory enables async D2H, but `cuda.synchronize()` blocks CPU. Typical cost: ~0.05–0.2 ms for <20 tracks.

### 1.7 Pipeline Relink (Background Thread)

Since `pipeline_relink=True` (default), the MOT emission work runs on a background `ThreadPoolExecutor`, overlapping with the next frame's detect on the main thread.

**Main thread per-frame:**

| Location | Operation | Cost |
|---|---|---|
| `evaluator.py:4562-4588` | Pre-materialize D2H copies (boxes, scores, geom_mask, embeddings, gmc → CPU) | **CPU D2H** |
| `evaluator.py:4590-4603` | `_rw_executor.submit(_bg_relink_write, ...)` | Minor |
| `evaluator.py:3816-3836` | `_bg_future.result()` — wait for prev frame's background thread | **CPU wait** |

**Background thread per-frame:**

| Location | Operation | Cost |
|---|---|---|
| `evaluator.py:2279-2301` | `_prepare_track_candidates()` | **CPU** |
| `helpers.py:667-746` | Collect stability candidates + build `PreparedTrackCandidate` list | Python loops over ~N tracks |
| `evaluator.py:2344-2351` | `_resolve_frame_tracks()` → `lifecycle_merger.resolve_many()` | CPU (no-op when disabled) |
| `evaluator.py:2352-2360` | `_emit_resolved_tracks()` → `_mot_result_line()` per track | **CPU** |
| `helpers.py:492-521` | Python string formatting loop | O(N) `f"...{float}..."` per track |
| `evaluator.py:2371` | `lifecycle_merger.prune()` | CPU (no-op when disabled) |
| `evaluator.py:2372-2382` | `_finalize_frame_side_effects()` | CPU (no-op: relinker=None, bank=None, reid=None) |

**Key observation:** With `use_semantic_mode=False` (from `reid_mode=off`), there is **no relinker** and **no identity resolution** overhead. The background thread does: candidate prep → lifecycle_merger (disabled, returns identity) → MOT line formatting → side effects (no-ops).

### 1.8 Inline Path Alternative (`pipeline_relink=False`)

When pipeline_relink is off, the main thread uses `_fast_emit_mot_lines()`:

| Location | Operation | Cost |
|---|---|---|
| `helpers.py:128-155` | `_fast_emit_mot_lines()` — O(N) string formatting loop | CPU |

This skips all candidate prep, lifecycle merge, and emit_resolved_tracks overhead. **Much lighter than the background thread path.**

---

## 2. Per-Sequence Post-Processing

All run once per sequence after the frame loop ends.

### 2.1 Quality Filtering (no-op)

| Location | Operation | Cost |
|---|---|---|
| `post_merge.py:283-307` | `filter_low_quality_tracklets()` | CPU |

With `min_tracklet_len=1, min_tracklet_score=0.0`, no tracks are filtered. The function still parses MOT lines → groups by ID → computes mean scores → doesn't remove any. **Pure overhead, can be skipped by setting min_tracklet_len=0 or adding an early return.**

### 2.2 Interpolation

| Location | Operation | Cost |
|---|---|---|
| `post_merge.py:359-443` | `interpolate_tracklets()` | **CPU** |

- `pandas.read_csv(io.StringIO(...))` — parse all MOT lines into DataFrame
- `df.groupby("tid")["frame"].transform("count")` — per-tracklet frame count
- `df.sort_values(["tid", "frame"])` — two-column sort
- NumPy vectorized gap detection + broadcasting interpolation per gap
- Full-line sort with key function `line.split(",", 2)` + `int()` casts

### 2.3 MOT Metrics Evaluation

| Location | Operation | Cost |
|---|---|---|
| `metrics.py:95-179` | `run_motmetrics_evaluation()` | **CPU** |

- `glob.glob` for GT/tracker files
- Nested loop matching GT names → tracker output names
- `_calculate_hota()` → full TrackEval pipeline (loads all GT + tracker files, computes MOTA/MOTP/IDF1/HOTA/DetA/AssA)

**Runs once after ALL sequences, not per-sequence.**

---

## 3. Summary: Actual CPU Hotspots

| Rank | Stage | Location | Est. per frame | Mitigation |
|---|---|---|---|---|
| 1 | **Pre-materialize D2H** | `evaluator.py:4562-4588` | 0.01–0.05 ms | Skip when no relinker/semantic mode (use inline path instead) |
| 2 | **Materialize (D2H sync)** | `helpers.py:68-125` | 0.05–0.2 ms | Already pinned memory; `cuda.synchronize()` is fundamental |
| 3 | **Background: MOT line format** | `helpers.py:492-521` | 0.01–0.05 ms | `_fast_emit_mot_lines` avoids `_emit_resolved_tracks` overhead |
| 4 | **Background: candidate prep** | `helpers.py:667-746` | 0.01–0.03 ms | Skip when `id_stability_filter=None` and `bank=None` (no useful work) |
| 5 | **Interpolation** | `post_merge.py:359-443` | Per-sequence | pandas parse + groupby + sort |
| 6 | **MOT metrics (TrackEval)** | `metrics.py:95-179` | Overall (once) | External tool, runs after all sequences |
| 7 | **Quality filter (no-op)** | `post_merge.py:283-307` | Per-sequence | Set `min_tracklet_len=0` to skip |

### Key Finding

With `mamba_whole_graph` + `--relink-bridge-enabled`, **the dominant CPU operations are all in the pipeline-relink background thread path**. Since `use_semantic_mode=False` (no relinker) and `lifecycle_merge=False`, the candidate prep → lifecycle resolve → MOT emit chain produces no useful work beyond simple MOT line formatting.

**The fastest path** would be `pipeline_relink=False`, which triggers `_fast_emit_mot_lines()` — a single O(N) string formatting loop on the main thread — eliminating the background thread overhead entirely without losing any bridge functionality (bridge relink runs inside the C++ tracker's GPU graph).
