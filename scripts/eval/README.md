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
