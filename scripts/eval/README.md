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

- `pipeline_contribution.py`
  - Cumulative cutoff runner for module contribution analysis.
  - Runs profiles such as `tracker_core -> +gmc -> +semantic -> +bank -> full`
    so each `Δprev` is the naked gain of one more downstream module.

## Support Utilities

- `calculate_mota.py`
  - Evaluate MOT-format result files against GT and print tracking metrics.

- `convert_mot17.py`
  - Dataset / output conversion helper for MOT17-related formats.

## Alternative Workflows

- `mot17_public.py`
  - Evaluate tracking with MOT17 public detections from `det/det.txt`.

- `sportsmot.py`
  - Cross-dataset evaluation entry point for `datasets/SportsMOT`.
  - Reuses the main detector + tracker pipeline so changes can be checked for
    generalization beyond MOT17.

- `dancetrack.py`
  - Cross-dataset evaluation entry point for `datasets/DanceTrack/val`.
  - Useful for checking identity stability under large pose variation and
    frequent close interactions.

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
3. Use `pipeline_contribution.py` when you want module-by-module naked uplift.
4. Use `calculate_mota.py` when you need standalone metric recomputation.
5. Use `sportsmot.py` and `dancetrack.py` when a change looks good on MOT17 and
   you want a fast generalization check before doing deeper tuning.
6. Use the remaining scripts only for alternative baselines, comparisons, or performance checks.

## Cross-Dataset Generalization

Two lightweight cross-dataset checks are now available:

- `uv run python scripts/eval/sportsmot.py`
- `uv run python scripts/eval/dancetrack.py`

These should be treated as regression / generalization gates for tracker
changes that were originally tuned on MOT17.

Current workflow recommendation:

1. Start from `configs/mot17_baseline.yaml`.
2. Validate the change on MOT17 first.
3. Re-run `sportsmot.py` and `dancetrack.py`.
4. Only introduce dataset-specific profiles after the shared baseline has been
   checked.

Important findings from the current generalization pass:

- The original MOT17 baseline remains the best shared starting point so far.
- Replacing the baseline wholesale with dataset-specific conservative profiles
  improved throughput, but reduced tracking accuracy on both SportsMOT and
  DanceTrack.
- On SportsMOT, the dominant failure mode is excessive false positives rather
  than catastrophic recall collapse.
- The highest-signal fix so far is to keep the MOT17 baseline and raise
  `new_track_thresh` back to `0.35`.

Observed SportsMOT result from the current best known generalized tweak:

```bash
uv run python scripts/eval/sportsmot.py \
  --config configs/mot17_baseline.yaml \
  --new-track-thresh 0.35
```

Metrics:

- `IDF1 40.7%`
- `MOTA 29.9%`
- `IDs 2233`
- `FP 183207`
- `FN 21806`
- `Rcll 92.6%`
- `Prcn 59.9%`

Interpretation:

- On SportsMOT, track birth policy generalized worse than detection recall.
- Tightening `new_track_thresh` was materially more effective than globally
  raising `conf_threshold` / `track_thresh`.
- If further SportsMOT tuning is needed, continue from the MOT17 baseline plus
  `--new-track-thresh 0.35` before touching broader detector or geometry logic.

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
4. `contribution`: `pipeline_contribution.py` for cumulative cutoff / raw uplift tables.

Useful overrides:

```bash
scripts/eval/module_benchmark.sh --mode profile --sequences MOT17-09-SDP --max-frames 80
scripts/eval/module_benchmark.sh --mode ablation --ablation-categories detection,geometry
scripts/eval/module_benchmark.sh --mode validate --tiling 960p_2x2 --engine models/yolo/yolo26s_batch6.engine
scripts/eval/module_benchmark.sh --mode profile -- --async-reid
scripts/eval/module_benchmark.sh --mode contribution -- --match-thresh 0.78
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

## Pipeline Contribution

When you want to measure raw uplift without executing later modules, use:

```bash
uv run python scripts/eval/pipeline_contribution.py --detector SDP
```

Optional pose sidecar cutoff:

```bash
uv run python scripts/eval/pipeline_contribution.py \
  --detector SDP \
  --pose-engine models/yolo/yolo26s_pose_960_batch1.engine
```

The script writes:

- `contribution_report.md`
- `contribution_report.csv`
- `commands.txt`

and stores per-profile MOT outputs under `runs/<profile>/`.

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
