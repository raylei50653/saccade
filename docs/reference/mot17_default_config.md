# MOT17 Evaluation Configuration

> Last updated: 2026-06-21.
>
> This file distinguishes the argparse / YAML fallback defaults from the current
> recommended MOT17 baseline. The production evaluation baseline is a preset,
> not the raw CLI fallback.

---

## 1. Recommended Baselines

Two baseline presets are available, each with a different trade-off:

### s-variant (`mamba_whole_graph`, headline throughput)

YOLO26s TensorRT backbone + Mamba v14-replica T3→T1 head + C++/CUDA `GPUByteTracker`:

```bash
uv run scripts/eval/mot17.py \
  --preset mamba_whole_graph \
  --detector SDP \
  --double-buffer \
  --output out/frozen_v2
```

`frozen_v2` run (2026-06-21; MOT17 train / SDP, seven sequences):

| Metric | Value |
|:--|--:|
| IDF1 | **78.2** |
| MOTA | **78.4** |
| HOTA | **70.2** |
| DetA | **70.9** |
| AssA | **69.7** |
| IDs | **413** |
| Recall | **81.0** |
| Precision | **97.2** |
| Eval FPS | **269.47** |
| Mean latency | **7.42 ms** |

### m-variant (`mamba_whole_graph_m`, higher recall/MOTA)

YOLO26m TensorRT backbone + Mamba head + C++/CUDA `GPUByteTracker`. Higher
capacity FPN (256/512/512 vs s's 128/256/512) — stronger small-object recall
and MOTA at IDF1 parity with reduced IDs:

```bash
uv run scripts/eval/mot17.py \
  --preset mamba_whole_graph_m \
  --detector SDP \
  --double-buffer \
  --output out/m_run
```

Key differences from the s-variant:

| Area | s-variant | m-variant |
|:--|:--|:--|
| Backbone | yolo26s (128/256/512) | yolo26m (256/512/512) |
| `fpn_backbone_engine` | `yolo26s_backbone_640_best.engine` | `yolo26m_backbone_640_best.engine` |
| `mamba_ckpt` | `mamba_gt_v14replica_t3_t1` | `mamba_gt_yolo26m_v14replica_t3_t1` |
| `mamba_teacher_ckpt` | *(not needed)* | `gated_det_yolo26m_v14replica/epoch_0012.ckpt` |
| `mamba_head_engine` | *(PyTorch head)* | `mamba_head_26m.engine` (TRT head) |
| `kalman_r_scale` | 2.8 | 3.5 (m-tuned) |
| Bridge gate | `[0.75, 1.33] / px 0.25` | `[0.6, 1.7] / px 0.4` (relaxed) |
| Eval FPS | ~269 | ~241 |
| IDF1 / IDs | 78.2 / 413 | 79.5 / 335 |

The m-variant requires `mamba_teacher_ckpt` because the fine-tuned backbone is
loaded from the teacher checkpoint: the C++ batched TRT path is hardcoded to
yolo26s channels, so the Python whole-graph path reads channels from the m
engine and validates against the head.

Authoritative sources:

- Presets: `configs/presets/mamba_whole_graph.yaml`, `configs/presets/mamba_whole_graph_m.yaml`
- Entry point: `scripts/eval/mot17.py`
- Config parser: `src/saccade/perception/eval/config.py`
- Stage order: `src/saccade/perception/eval/evaluator.py`

The frozen-v2 measurement ran on an NVIDIA GeForce RTX 5070 Ti Laptop GPU
(12 GB), Driver 610.62, CUDA UMD 13.3, with `--double-buffer`. The HOTA family
uses TrackEval; IDF1/MOTA/IDs were recomputed with `calculate_mota.py`. Older
benchmark records under `reference/benchmarks/` may use different protocols
such as short sequence subsets, profiling syncs, different engines, or
module-only benchmarks; do not mix those numbers without naming the protocol.

---

## 2. Current Preset Shape

### `mamba_whole_graph` (s-variant)

`configs/presets/mamba_whole_graph.yaml` currently overrides the raw CLI
fallback in these important ways:

| Area | Current preset |
|:--|:--|
| Detection | `tiling: native_640`, `preprocess: none`, `use_whole_graph: true`, `use_cuda_graph: true` |
| Backbone/head | `fpn_backbone_engine: models/yolo/yolo26s_backbone_640_best.engine`, `mamba_ckpt: runs/mamba_gt_v14replica_t3_t1/best.ckpt` |
| ReID | `reid_mode: off` |
| GMC | `gmc: true`, `gmc_downscale: 4`, `gmc_fg_mask: false` |
| Tracker | `match_thresh: 0.50`, `new_track_thresh: 0.28`, `kalman_r_scale: 2.8`, `fuse_score_weight: 0.0` |
| Relink | `relink_bridge_enabled: true`, `relink_bridge_px: 0.25`, height gate `[0.75, 1.33]`, `relink_bridge_dir_bonus: 0.8` |
| Occlusion | `oao_tau: 0.50`, `oao_ramp_frames: 25`, `occ_state_enabled: true` |
| Output cleanup | `interpolate_tracklets: true`, `interpolate_max_gap: 35`, `interpolate_min_track_len: 5` |
| Disabled vs legacy baseline | `track_person_only: false`, `person_geometry_prior: false`, `detection_quality_scaling: false`, `id_stability_filter: false`, `per_seq_adapt: false`, `geometry_suspect_support: false` |

### `mamba_whole_graph_m` (m-variant)

Differs from the s-variant only where noted above in §1: `fpn_backbone_engine`,
`mamba_ckpt`, `mamba_teacher_ckpt`, `mamba_head_engine`, `kalman_r_scale`,
and the bridge gate values (`bridge_px`, `bridge_h_lo`, `bridge_h_hi`). All other
parameters (GMC, OAO, occlusion, interpolation, disabled features) are identical.

Notes (apply to both presets unless stated otherwise):

- `use_tracker_graph: true` is present in the presets. The evaluator disables the
  captured tracker graph only when semantic relink needs per-detection embeddings
  through `relink_enabled`; bridge relink does not require that fallback.
- `async_reid` and `pipeline_relink` are CLI flags and config fields, but with
  `reid_mode: off` they are not the source of the current headline accuracy.
- Semantic relink, appearance bank, lifecycle merge, Cheb-GR merge, multi-birth,
  and scene adapt are not part of the current recommended baselines.

---

## 3. Raw CLI / Fallback Defaults

There are four default layers in source:

1. argparse defaults from `scripts/eval/config/*.py`;
2. file defaults from `configs/mot17_baseline.yaml` when no `--preset` is supplied;
3. module YAML / preset overrides loaded by `scripts/eval/mot17.py` and applied
   via `parser.set_defaults(...)`;
4. final fallbacks in `src/saccade/perception/eval/config.py` when a key is
   still absent.

The raw parser / baseline-file path still describes a legacy `native_960`
comparison. Treat it as a tracker-core comparison point, not the current
production baseline.

Key raw/fallback defaults:

| Parameter | Raw parser / fallback role |
|:--|:--|
| `--engine` | Raw parser: `models/yolo/yolo26m_960_batch1.engine` |
| `--tiling` | Raw parser: `native_960` |
| `--match-thresh` | Raw parser: `0.75`; `configs/mot17_baseline.yaml`: `0.66` |
| `--new-track-thresh` | Raw parser: `0.35`; `configs/mot17_baseline.yaml`: `0.28` |
| `--gmc` | Raw parser default ON |
| `--gmc-downscale` | Raw parser: `8` |
| `--reid-mode` | Raw parser default `off` |
| `--cross-tile-merge` | Raw parser default ON, mainly relevant to tiled modes |
| `--detection-quality-scaling` | Raw parser default ON; current `mamba_whole_graph` turns it OFF |
| `--id-stability-filter` | Raw parser default ON; current `mamba_whole_graph` turns it OFF |
| `--fp-hard-filter-enabled` | Raw parser default ON |
| `--async-reid` | argparse action is `store_true`, but module/config fallback is `true`; only matters when ReID work exists |
| `--pipeline-relink` | argparse action is `store_true`, module/config fallback is `true`, and `parse_eval_config` disables it when `--profile-stages` is active |

Use `--preset baseline` or `--preset speed` only when you intentionally need the
legacy `native_960` comparison, not for current headline reporting.

---

## 4. Stage Names

`evaluator.py` currently profiles these top-level stage names:

```text
fetch
ingest_preprocess
detect
postprocess
reid_bank_sync
reid_budget
reid_crop
reid_extract
lazy_reid
gmc
track
materialize
bg_relink_wait
relink_write
frame_total
```

Postprocess, native ReID, GMC, and CUDA-event segment breakdowns are nested
diagnostics under those top-level stages.

---

## 5. Update Rules

When changing defaults or headline numbers:

1. Update the appropriate preset file(s) (`configs/presets/mamba_whole_graph.yaml`
   and/or `configs/presets/mamba_whole_graph_m.yaml`) first if the baseline really changes.
2. Re-run or cite a same-run MOT17 result before editing headline metrics.
3. Update [PIPELINE.md](../PIPELINE.md), [DATAFLOW.md](../DATAFLOW.md), [TODO.md](../TODO.md), and this file together.
4. Keep legacy `speed` / `baseline` numbers labelled as legacy comparisons.
