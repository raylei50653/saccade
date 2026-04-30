# Saccade

High-efficiency edge video perception with a TensorRT detector, GPU tracking,
appearance-based ReID, and MOT-style evaluation tooling.

> **🌍 [繁體中文版](README_tw.md)**

## What This Repo Is Now

The repo has two practical centers:

- **Online perception pipeline**
  - TensorRT-based detection and preprocessing
  - GPU tracking, GMC, Kalman, appearance fusion
  - async embedding / storage / orchestration components

- **Offline MOT evaluation and tuning**
  - `scripts/eval/mot17.py` is the main MOT17 evaluation entry point
  - `scripts/eval/ablation_mot17.py` is the unified ablation runner for grouped
    tracker parameters

This README focuses on the current code layout and the workflows that are still
actively used.

## Core Areas

- `src/saccade/perception/`
  - Detector, preprocessing, ReID, relink, tracker coordination, eval runner.

- `src/` and `include/`
  - C++ / CUDA tracking and performance-critical extensions.

- `src/saccade/media/`
  - Stream ingestion and video pipeline integration.

- `src/saccade/pipeline/`
  - Higher-level orchestration across perception, storage, and cognition.

- `src/saccade/storage/`
  - Redis / Chroma-related persistence paths.

- `scripts/eval/`
  - MOT17 evaluation, ablation, conversion, comparison, and metric helpers.

- `tests/`
  - Unit and benchmark coverage.

More architectural background lives in [docs/architecture.md](/home/ray/developer/ai/saccade/docs/architecture.md:1) and [docs/README.md](/home/ray/developer/ai/saccade/docs/README.md:1).

## Environment

- Python: `3.12`
- Package manager: `uv`
- Key runtime dependencies:
  - `torch`
  - `torchvision`
  - `tensorrt-cu12`
  - `nvidia-dali-cuda120`
  - `motmetrics`

Install project dependencies with:

```bash
uv sync
```

If you use the C++ / CUDA extensions, also configure and build the native
targets for your machine.

## Main Workflows

### 1. Run MOT17 Evaluation

```bash
uv run python scripts/eval/mot17.py \
  --detector SDP \
  --output results/MOT17_eval
```

`mot17.py` now groups arguments by capability area:

- detection and preprocessing
- association
- geometry and ID stability
- ReID backbone
- semantic relink
- dynamic ReID trigger policy
- lifecycle merge and cleanup

### 2. Run Ablation Studies

```bash
uv run python scripts/eval/ablation_mot17.py --category detection
uv run python scripts/eval/ablation_mot17.py --category detection,geometry
uv run python scripts/eval/ablation_mot17.py --category all
```

Supported categories:

- `detection`
- `association`
- `geometry`
- `reid`
- `semantic`
- `trigger`
- `lifecycle`

See [scripts/eval/README.md](/home/ray/developer/ai/saccade/scripts/eval/README.md:1) for the current eval script map.

### 3. Recompute Tracking Metrics

```bash
uv run python scripts/eval/calculate_mota.py --results results/MOT17_eval
```

### 4. Compare Against an External Baseline

```bash
uv run python scripts/eval/compare_framework_ultralytics.py \
  --saccade results/MOT17_eval \
  --ultralytics results/MOT17_ultralytics_eval \
  --gt-root datasets/MOT17/train \
  --detector SDP
```

## Alternative Evaluation Paths

- `scripts/eval/mot17_public.py`
  - Runs tracking from MOT17 public detections in `det/det.txt`.

- `scripts/eval/ultralytics_official_mot17.py`
  - Runs Ultralytics tracking as an external baseline.

- `scripts/eval/bench_yolo_batch.py`
  - Measures detector batch throughput / latency.

## Development Notes

- The worktree may contain active experimentation in tracking, GMC, ReID, and
  evaluation scripts.
- The MOT evaluation flow has recently been simplified around:
  - `scripts/eval/mot17.py`
  - `scripts/eval/ablation_mot17.py`
- Older ad-hoc grid search and one-off ablation entry points were removed to
  reduce drift.

## Tests

Run the Python test suite with:

```bash
uv run pytest
```

Some benchmarks and evaluation scripts require:

- CUDA-capable hardware
- TensorRT engines
- local MOT datasets

## Status

If something in this README conflicts with the code, treat the code under
`src/saccade/perception/`, `scripts/eval/`, and `tests/` as the source of truth. This file
has been updated to reflect the current MOT17-centered evaluation workflow, but
the repo is still under active iteration.
