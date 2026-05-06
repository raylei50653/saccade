# Eval Scripts

This directory is now centered on `mot17.py` as the primary evaluation entry
point, with `ablation_mot17.py` as the unified tuning harness for its grouped
parameters.

## Primary Entry Points

- `mot17.py`
  - Main MOT17 evaluation pipeline.
  - Runs detector + tracker + ReID / relink / lifecycle logic.

- `ablation_mot17.py`
  - Unified ablation runner for `mot17.py`.
  - Supports grouped studies by category:
    - `detection`
    - `association`
    - `geometry`
    - `reid`
    - `semantic`
    - `trigger`
    - `lifecycle`
  - Supports multiple categories in one run, for example:
    - `uv run python scripts/eval/ablation_mot17.py --category detection,geometry`

## Support Utilities

- `calculate_mota.py`
  - Evaluate MOT-format result files against GT and print tracking metrics.

- `convert_mot17.py`
  - Dataset / output conversion helper for MOT17-related formats.

## Alternative Workflows

- `mot17_public.py`
  - Evaluate tracking with MOT17 public detections from `det/det.txt`.

- `ultralytics_official_mot17.py`
  - Run Ultralytics official tracking as an external baseline.

- `compare_framework_ultralytics.py`
  - Compare result directories, typically Saccade vs Ultralytics.

## Performance Utility

- `bench_yolo_batch.py`
  - Batch-size throughput / latency benchmark for the detector engine.

## Recommended Usage

1. Run `mot17.py` for the main pipeline.
2. Use `ablation_mot17.py` for parameter studies.
3. Use `calculate_mota.py` when you need standalone metric recomputation.
4. Use the remaining scripts only for alternative baselines, comparisons, or performance checks.

## Module Benchmark Template

When you want to rerun the full module-by-module experiment flow from
`docs/PIPELINE_REFERENCE.md`, use:

```bash
scripts/eval/module_benchmark.sh
```

This wrapper standardizes three recurring steps:

1. `profile`: `mot17.py --profile-stages` for stage latency baseline.
2. `ablation`: `ablation_mot17.py` for grouped parameter sweeps.
3. `validate`: `mot17.py` without profiling for end-to-end comparison.

Useful overrides:

```bash
scripts/eval/module_benchmark.sh --mode profile --sequences MOT17-09-SDP --max-frames 80
scripts/eval/module_benchmark.sh --mode ablation --ablation-categories detection,geometry
scripts/eval/module_benchmark.sh --mode validate --tiling 960p_2x2 --engine models/yolo/yolo26s_batch6.engine
scripts/eval/module_benchmark.sh --mode profile -- --async-reid
```

Defaults are intentionally opinionated:

- detector: `SDP`
- sequences: `MOT17-04-SDP,MOT17-10-SDP`
- engine: `models/yolo/yolo26s_960_batch1.engine`
- tiling: `native_960`
- max frames: `100`

Outputs land under `results/module_benchmark/<timestamp>/`.

Each run also creates:

- `summary.txt`: resolved experiment configuration
- `commands.txt`: exact commands emitted by the wrapper
- `notes.md`: lightweight note template for hypothesis / findings / decision
- `experiment_matrix.md`: fill-in table aligned with `docs/PIPELINE_REFERENCE.md`

## MOT17 Detection / Tiling Notes

`mot17.py` now exposes multiple detector geometry paths:

- `--tiling 960p_2x2`
- `--tiling 960p_3x2`
- `--tiling native_960`

Useful tiled-detection diagnostics / controls:

- `--tile-diagnostics`
- `--tile-seam-score-penalty`
- `--tile-seam-margin-canvas-px`
- `--cross-tile-seam-center-scale`
- `--cross-tile-seam-area-ratio-threshold`
- `--cross-tile-seam-min-overlap-ratio`

Typical two-sequence FN diagnosis command:

```bash
PYTHONPATH=build:. LD_LIBRARY_PATH=build uv run python scripts/eval/mot17.py \
  --detector SDP \
  --sequences MOT17-04-SDP,MOT17-10-SDP \
  --split train \
  --engine models/yolo/yolo26s_batch6.engine \
  --tiling 960p_2x2 \
  --tile-diagnostics
```

Native-960 control:

```bash
PYTHONPATH=build:. LD_LIBRARY_PATH=build uv run python scripts/eval/mot17.py \
  --detector SDP \
  --sequences MOT17-04-SDP,MOT17-10-SDP \
  --split train \
  --engine models/yolo/yolo26s_960_batch1.engine \
  --tiling native_960
```

## Person-Only Top-K Detector Notes

Experimental detector artifacts:

- `models/yolo/yolo26s_person_topk1000.onnx`
- `models/yolo/yolo26s_person_topk1000_batch4.engine`
- `scripts/model/export_yolo_person.py`

Intent:

- Keep only `person` before YOLO end-to-end top-k.
- Raise detector output cap from `300` to `1000` for crowded scenes.
- Avoid non-person classes consuming the detector's fixed top-k budget.

Observed behavior summary (2026-05-06):

- On crowded MOT20 frames, the new engine can recover many additional low-score person detections.
- Under the default MOT17 eval thresholds, aggregate tracking metrics changed little.
- Global threshold lowering (`conf/track=0.02`, `new_track=0.25`) improved recall and IDF1 slightly, but also raised FP / IDs.
- A Python-side crowd-aware threshold switch was tested, but is not recommended as a default path due to extra complexity and unstable tradeoffs.

Current recommendation:

- Keep the new engine as an available experiment.
- Do not replace the default detector engine or default eval thresholds yet.
- If revisiting this direction, prefer a tracker-internal crowded-scene policy over per-frame Python parameter switching.

Example crowded-scene experiment:

```bash
PYTHONPATH=src uv run python scripts/eval/mot17.py \
  --detector SDP \
  --sequences MOT17-04-SDP,MOT17-10-SDP \
  --split train \
  --engine models/yolo/yolo26s_person_topk1000_batch4.engine \
  --tiling 960p_2x2 \
  --reid-mode off \
  --max-frames 100 \
  --crowd-low-score-mode \
  --crowd-low-score-trigger 25 \
  --crowd-conf-threshold 0.02 \
  --crowd-track-thresh 0.02 \
  --crowd-mid-thresh 0.05 \
  --crowd-new-track-thresh 0.25
```
