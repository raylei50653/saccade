# Saccade Pipeline Dataflow

> This file describes the current MOT17 evaluation dataflow. It is not an
> experiment log; detailed latency runs belong under `reference/benchmarks/`.
>
> Source of truth: `src/saccade/perception/eval/evaluator.py`.
> `src/saccade/perception/eval/runner.py` is only a compatibility re-export of
> `run_eval`; the executable MOT17 CLI enters through `scripts/eval/mot17.py`.

---

## 1. Current Main Path

The two recommended MOT17 baselines are:

```bash
# s-variant (headline throughput, yolo26s backbone)
uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --double-buffer

# m-variant (higher recall/MOTA at IDF1 parity, yolo26m backbone)
uv run scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP --double-buffer
```

Both presets use `reid_mode: off`, so the ReID stages are present in the
evaluator but skipped for the baseline.

At a high level, the **single-frame** path (no double-buffer) is:

```text
frame source
  -> fetch
  -> ingest_preprocess
  -> detect
  -> postprocess
  -> optional ReID stages
  -> gmc
  -> track
  -> materialize
  -> relink_write / MOT output
```

When `--double-buffer` is active, the frame loop interleaves detection and
tracking across frames — see §8 for the full double-buffer dataflow.

The exact non-workbench, non-double-buffer frame loop in source is:

1. `fetch`: `next(stream_iter)` returns the next frame.
2. `ingest_preprocess`: the frame is copied into `AdaptiveFramePool`; optional
   gamma / contrast preprocess is applied unless the preset sets
   `preprocess: none`.
3. `detect`: `detect_fn(...)` runs the configured detector path.
4. `postprocess`: native `PerceptionPipeline` tensor prep / filter / NMS path
   when available, otherwise Python fallback; tiled runs may also run
   repo cross-tile merge.
5. `bg_relink_wait`: if previous frame's relink write was submitted to the
   background executor, the current frame waits here before touching shared
   relinker / bank / dynamic-ReID state.
6. `reid_bank_sync`, `reid_budget`, `reid_crop`, `reid_extract`: only if
   `reid_mode != "off"` or lazy-ReID profiling is explicitly enabled.
7. `gmc`: GPU phase-correlation warp. If async ReID is active, the ReID side
   stream overlaps with GMC and is synchronized immediately before `track`.
8. `track`: `tracker.update_into(...)` consumes detections, optional
   embeddings, and the GMC warp.
9. `materialize`: GPU tracker result buffers are materialized at the MOT output
   boundary, unless deferred emit can postpone readback.
10. `lazy_reid`: optional profiling-only tentative-track embedding probes.
11. `relink_write`: either fast MOT-line emit, or full identity-resolution
    emit pipeline when semantic relink, dynamic ReID, appearance bank, or
    ID-stability filtering is active.

The workbench path is a separate fast facade: it can fold detection, quality
postprocess, tracking, and MOT-line writing into `Workbench.process_frame()` /
`process_detections_quality_aware()`. It is not the headline path described
below.

---

## 2. Profiled Stage Names

`evaluator.py` currently records these top-level stage names when
`--profile-stages` is active:

| Stage | Role | Current baseline behavior |
|:--|:--|:--|
| `fetch` | Read the next frame tensor from the configured sequence source | ON |
| `ingest_preprocess` | Move / normalize frame into reusable GPU buffers when needed | ON |
| `detect` | Detector forward path. For `mamba_whole_graph`, this is TRT backbone + Mamba head + decode in the whole-graph path | ON |
| `postprocess` | Native or Python filter / NMS / merge / geometry tail | ON |
| `reid_bank_sync` | Inject appearance-bank representatives into tracker references | OFF for `reid_mode: off` |
| `reid_budget` | Pick detections for embedding extraction | OFF for `reid_mode: off` |
| `reid_crop` | Crop ROIs for appearance extraction | OFF for `reid_mode: off` |
| `reid_extract` | Run the configured appearance backend | OFF for `reid_mode: off` |
| `lazy_reid` | Optional profiling-only tentative-track embedding checks | OFF unless profiling flag is enabled |
| `gmc` | Estimate camera motion warp with GPU phase correlation | ON |
| `track` | GPUByteTracker update / association / lifecycle update | ON |
| `materialize` | Materialize GPU tracker results at the output boundary | ON unless deferred emit path can avoid immediate readback |
| `bg_relink_wait` | Wait for previous-frame background relink write when pipelined | Conditional |
| `relink_write` | Resolve output IDs, bridge relink side effects, and emit MOT rows | ON |
| `frame_total` | End-to-end per-frame wall time | ON when profiling |

Nested diagnostics include:

- `post_*` and `native_*` breakdowns for postprocess attribution.
- `native_reid_*` breakdowns for native ReID.
- `gmc_gray_downscale`, `gmc_fg_mask`, `gmc_phase_corr`, `gmc_handoff`.
- `post_seg_*` CUDA-event spans for sync-free postprocess partitioning.

---

## 3. Detection And Postprocess

### 3.1 `mamba_whole_graph` (s-variant, headline)

| Field | Value |
|:--|:--|
| `tiling` | `native_640` |
| `preprocess` | `none` |
| `use_whole_graph` | `true` |
| `use_cuda_graph` | `true` |
| `fpn_backbone_engine` | `models/yolo/yolo26s_backbone_640_best.engine` |
| `mamba_ckpt` | `runs/mamba_gt_v14replica_t3_t1/best.ckpt` |
| `track_person_only` | `false` |
| `person_geometry_prior` | `false` |
| `detection_quality_scaling` | `false` |

### 3.2 `mamba_whole_graph_m` (m-variant, higher capacity)

The m-variant uses YOLO26m (FPN 256/512/512 vs s's 128/256/512). Key differences
from the s-variant:

| Field | s-variant | m-variant |
|:--|:--|:--|
| Backbone | yolo26s (128/256/512) | yolo26m (256/512/512) |
| `fpn_backbone_engine` | `yolo26s_backbone_640_best.engine` | `yolo26m_backbone_640_best.engine` |
| `mamba_ckpt` | `mamba_gt_v14replica_t3_t1` | `mamba_gt_yolo26m_v14replica_t3_t1` |
| `mamba_teacher_ckpt` | *(absent — TRT backbone covers it)* | `gated_det_yolo26m_v14replica/epoch_0012.ckpt` |
| `mamba_head_engine` | *(absent — PyTorch head)* | `mamba_head_26m.engine` (TRT head) |
| `kalman_r_scale` | 2.8 | 3.5 (m-tuned) |
| Bridge gate | `h_lo=0.75, h_hi=1.33, px=0.25` | `h_lo=0.6, h_hi=1.7, px=0.4` (relaxed for m) |

The m-variant requires `mamba_teacher_ckpt` because the fine-tuned backbone
comes from the teacher checkpoint (the C++ batched TRT path is hardcoded to s
channels). The Python whole-graph path reads channels from the engine
(256/512/512) and validates against the head. The TRT Mamba head engine
(`mamba_head_26m.engine`) replaces the PyTorch head for lower latency.

The relaxed bridge gate (`h_lo=0.6, h_hi=1.7, px=0.4`) is necessary because
the s-tuned gate `[0.75,1.33]/px0.25` is too strict for m's noisier small-box
heights and would reject valid small-object bridges. The relaxation recovers
+0.6 AssA / +0.3 IDF1 / −17 IDs cleanly.

### 3.3 Whole-graph detection pipeline

In whole-graph mode (`use_whole_graph=true`), the detector forward is captured
as a single CUDA graph that includes:

```
frame_gpu (HWC uint8 CUDA)
  → F.interpolate → 640×640
  → TRT backbone (infer_graph) → p3, p4, p5 (FPN features)
  → TRT Mamba head (infer_graph) or PyTorch _forward_eager → cls_preds, reg_preds
  → _postprocess_mamba_fixed: dist2bbox → top-k (max_det=300) → NMS
  → detections (B, max_det, 6) xyxy
```

The CUDA graph is captured once on the first call and replayed for every
subsequent frame. With `--double-buffer`, the detection outputs are `.clone()`d
after every replay because whole-graph replays mutate the same static buffers.

### 3.4 Postprocess

Postprocess still supports tiled-mode repair paths such as cross-tile merge, but
the current headline presets are native 640 and do not depend on tiled output
reconstruction.

The native postprocess path is preferred when `PerceptionPipeline` bindings are
available. In source, native tensor prep produces contiguous postprocess buffers
for the C++/CUDA facade; Python fallback remains for portability and ablation
coverage. Postprocess may include:

- native/Python filter and NMS;
- optional keypoint alignment;
- optional cross-tile duplicate merge for tiled modes only;
- optional stage-2 quality gate / birth gates, most of which are documented
  NO-GO and disabled in the headline presets;
- FP hard filter from raw defaults unless explicitly overridden.

---

## 4. ReID Branch

The ReID branch exists in the evaluator, but it is not part of the current
headline baseline:

```text
reid_mode: off
appearance_bank: false
semantic_bank_inject: false
```

When ReID is enabled for a targeted ablation, the dataflow is:

```text
postprocessed detections
  -> bank sync
  -> budget selection
  -> ROI crop
  -> embedding extraction
  -> tracker / relink appearance inputs
```

Appearance-bank and semantic relink results should be interpreted as ablations
unless the preset explicitly turns them on.

Source detail: ReID work is scheduled before GMC in `_run_reid_and_gmc()`. With
native ReID and `async_reid=True`, extraction is enqueued on `reid_side_stream`,
GMC runs on the main stream, and the side stream is synchronized immediately
before `track` so the tracker sees fresh embeddings. This overlap only matters
when ReID work is enabled; it is not part of the current headline run.

---

## 5. GMC And Tracking

`mamba_whole_graph` uses GPU GMC:

```text
frame tensor -> GPU phase correlation -> 6-parameter warp -> tracker update
```

Key preset values (s-variant; m-variant uses `kalman_r_scale: 3.5` and relaxed bridge gate):

| Field | Value |
|:--|:--|
| `gmc` | `true` |
| `gmc_downscale` | `4` |
| `gmc_fg_mask` | `false` |
| `match_thresh` | `0.50` |
| `new_track_thresh` | `0.28` |
| `kalman_r_scale` | `2.8` (s) / `3.5` (m) |
| `multiplicative_cost` | `true` |
| `sinkhorn_lambda` | `10` |
| `stability_cost_w` | `0.20` |

The tracker writes into preallocated GPU result buffers, then the evaluator
materializes only the output boundary needed for MOT rows and downstream ID
resolution.

Tracker configuration is applied per sequence in source through
`detector.tracker.set_params(...)`, `set_oao_params(...)`, `set_occ_params(...)`,
and, when enabled, `set_multiplicative_cost(...)` / `set_sinkhorn_lambda(...)`.
The current preset also sets `id_stability_filter=false`, `per_seq_adapt=false`,
and `geometry_suspect_support=false`, overriding raw parser defaults.

---

## 6. Relink And Output

The current baseline keeps appearance relink off and uses tracker-core bridge
relink plus occlusion-aware association:

| Field | Value |
|:--|:--|
| `relink_bridge_enabled` | `true` |
| `relink_bridge_px` | `0.25` |
| `relink_bridge_h_lo` / `relink_bridge_h_hi` | `0.75` / `1.33` |
| `relink_bridge_dir_bonus` | `0.8` |
| `oao_tau` | `0.50` |
| `oao_ramp_frames` | `25` |
| `interpolate_tracklets` | `true` |
| `interpolate_max_gap` | `35` |

`relink_write` is still the output-resolution stage name even when the useful
identity work is mostly tracker-core bridge relink rather than semantic relink.

In the current headline configuration (`reid_mode: off`, no appearance bank,
no semantic relinker, no ID-stability filter), `relink_write` uses the fast
MOT emit path. The full identity-resolution pipeline is only used when one of
those optional emit-stage components is active. `pipeline_relink` can submit
that full emit path to a background executor, but it is disabled while
`--profile-stages` is active and does not change headline accuracy.

---

## 7. Current Headline Baselines

<!-- fact-owner: current-baseline = docs/TODO.md -->
> These numbers mirror the single fact owner in [TODO.md](TODO.md) 「當前 Baseline」; do not re-baseline here.

Frozen runs recorded in [ADR 018](decisions/018-project-main-line-direction.md):

| Preset | IDF1 | MOTA | HOTA | DetA | AssA | IDs | Rcll | Prcn | Eval FPS |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `mamba_whole_graph` (s) | **78.2** | **78.4** | **70.2** | **70.9** | **69.7** | **413** | **81.0** | **97.2** | **269.47** |
| `mamba_whole_graph_m` | **79.5** | — | — | — | — | **335** | — | — | **~241** |

The s-variant is the headline throughput baseline. The m-variant trades ~28 FPS
for +1.3 IDF1 / −78 IDs via higher-capacity backbone (YOLO26m FPN 256/512/512).

Do not compare these FPS numbers directly to older `native_960` profiling tables
unless the measurement protocol is named. Older tables may include profiling
sync, short sequence subsets, different engines, or module-only benchmarks.

---

## 8. Double-Buffer Dataflow

`--double-buffer` enables cross-frame overlap: **detect(N+1)** runs on a side
CUDA stream while **postprocess + GMC + track(N)** runs on the main stream.
This increases throughput (Eval FPS) but does not reduce single-frame latency.

### 8.1 Eligibility

Double-buffer is enabled when ALL of:
1. `--double-buffer` CLI flag → sets `SACCADE_DOUBLE_BUFFER=1`
2. CUDA available
3. No `--profile-stages` or `--workbench`
4. Detector is frame-independent: `_temporal_T == 0` OR `use_whole_graph == true`
   (whole-graph forward bypasses the temporal ring buffer)
5. `SACCADE_DETECT_BARRIER=event` (narrow barrier; auto-set by the flag)

`mamba_whole_graph` and `mamba_whole_graph_m` both satisfy these conditions via
`use_whole_graph: true`.

### 8.2 Frame Loop With Double-Buffer

```text
Frame 1 (prime):
  schedule(1):  fetch → ingest_preprocess → detect(1) on double_buffer_stream
                → clone outputs → record ready_event[0]
  run_frame(1): wait_event(ready_event[0]) → consume detect(1)
                → postprocess → GMC → track(1)
                → copy tracker out → pinned[0] → record emit event
                → defer emit to next frame

  schedule(2):  fetch → ingest_preprocess → detect(2) on double_buffer_stream
                (overlaps with frame 1's postprocess+GMC+track on main stream)
                → clone outputs → record ready_event[1]

Frame 2:
  _flush_db_tracker_out(1): sync emit event → emit MOT lines for frame 1
  run_frame(2): wait_event(ready_event[1]) → consume detect(2)
                → postprocess → GMC → track(2)
                → copy → pinned[1] → defer emit

  schedule(3):  detect(3) on double_buffer_stream
                (overlaps with frame 2's postprocess+GMC+track)

... continue for all frames ...

Last frame: _flush_db_tracker_out(last)
```

### 8.3 Key Implementation Details

| Component | Location | Role |
|:--|:--|:--|
| `double_buffer_pools[0/1]` | `pipeline.py:1036-1042` | Two `AdaptiveFramePool` instances so detect(N+1) does not overwrite frame N's pixels |
| `double_buffer_stream` | `pipeline.py:1043` | Side CUDA stream for detection |
| `double_buffer_events[0/1]` | `pipeline.py:1044-1050` | Parity event pairs: `(input_ready, ready_event)` per frame |
| `_launch_double_buffer_detect()` | `stages.py:990` | Queues detect on the side stream, fences with events, clones outputs |
| `double_buffer_tracker_out_pinned[0/1]` | `pipeline.py:1228-1263` | Parity-slotted pinned CPU buffers for deferred tracker D2H |
| `_flush_db_tracker_out()` | `stages.py:417` | Syncs D2H event, builds CPU track_results, runs emit/relink |
| Frame loop dispatch | `evaluator.py:2516-2557` | Alternates `_schedule(N+1)` then `_run_frame(N)` |

### 8.4 Data Ownership

- **Input pools**: `pool[parity]` is consumed by detect(N) on the side stream.
  The main stream does not touch it until `wait_event(ready_event)`.
- **Detection outputs**: `.clone()`d after the side stream records `ready_event`
  because whole-graph replays mutate shared static buffers. Each frame's
  detection tensors are independent.
- **Tracker output**: After `_run_track()`, GPU buffers are copied (non-blocking)
  to parity-slotted CPU pinned memory, then an event is recorded. The emit
  stage (`_flush_db_tracker_out()`) syncs the event in the *next* frame's
  iteration, so the D2H transfer overlaps with the next frame's GPU work.

### 8.5 Bit-Exactness

Double-buffer is **quality bit-exact** with the serial path: raw MOT result
files have identical MD5 hashes. It changes only the scheduling of independent
work, not any tracking decision.

### 8.6 Performance

| Mode | Eval FPS | Mean Latency | Notes |
|:--|--:|--:|:--|
| No double-buffer | ~144 | 6.34 ms | Lower single-frame latency |
| Double-buffer | ~269 | 7.42 ms | Higher throughput; single-frame latency is *worse* due to side-stream overhead |

Throughput and single-frame latency are different measurements and cannot be
inter-converted.
