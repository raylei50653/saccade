# Saccade Pipeline Dataflow

> This file describes the current MOT17 evaluation dataflow. It is not an
> experiment log; detailed latency runs belong under `reference/benchmarks/`.
>
> Source of truth: `src/saccade/perception/eval/evaluator.py`.
> `src/saccade/perception/eval/runner.py` is only a compatibility re-export of
> `run_eval`; the executable MOT17 CLI enters through `scripts/eval/mot17.py`.

---

## 1. Current Main Path

The current recommended MOT17 baseline is:

```bash
uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP
```

At a high level, the per-frame path is:

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

`mamba_whole_graph` uses `reid_mode: off`, so the ReID stages are present in the
evaluator but skipped for the headline baseline.

The exact non-workbench frame loop in source is:

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
`process_detections_quality_aware()`. It is not the headline
`mamba_whole_graph` path described below.

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

Current `mamba_whole_graph` detection config:

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

Postprocess still supports tiled-mode repair paths such as cross-tile merge, but
the current headline preset is native 640 and does not depend on tiled output
reconstruction.

The native postprocess path is preferred when `PerceptionPipeline` bindings are
available. In source, native tensor prep produces contiguous postprocess buffers
for the C++/CUDA facade; Python fallback remains for portability and ablation
coverage. Postprocess may include:

- native/Python filter and NMS;
- optional keypoint alignment;
- optional cross-tile duplicate merge for tiled modes only;
- optional stage-2 quality gate / birth gates, most of which are documented
  NO-GO and disabled in the headline preset;
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

Key preset values:

| Field | Value |
|:--|:--|
| `gmc` | `true` |
| `gmc_downscale` | `4` |
| `gmc_fg_mask` | `false` |
| `match_thresh` | `0.50` |
| `new_track_thresh` | `0.28` |
| `kalman_r_scale` | `2.8` |
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

## 7. Current Headline Baseline

Frozen run recorded in [ADR 018](decisions/018-project-main-line-direction.md):

| Preset | IDF1 | MOTA | HOTA | DetA | AssA | IDs | Rcll | Prcn | Eval FPS |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `mamba_whole_graph` | **78.2** | **78.4** | **70.2** | **70.9** | **69.7** | **413** | **81.0** | **97.2** | **269.47** |

Do not compare that FPS directly to older `native_960` profiling tables unless
the measurement protocol is named. Older tables may include profiling sync,
short sequence subsets, different engines, or module-only benchmarks.
