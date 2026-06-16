# Paper-Ready Result Tables

All values below are recomputed from the existing result text files. They are
development-set evidence because the v14 replication lineage used all seven
MOT17-SDP training sequences.

## Main curriculum ablation

Only seeds 20260613 and 20260614 are valid same-seed comparisons.

| Method | Seeds | IDF1 | MOTA | HOTA | AssA | FPS | P99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain GT2 | 2 | 73.10 | 76.36 | 65.40 | 62.76 | 215.00 | 5.41 / 6.26 |
| T3-to-T1 | 2 | 74.45 | 77.57 | 66.55 | 63.92 | 216.47 | 5.19 / 5.25 |
| Paired mean delta | 2 | +1.34 | +1.21 | +1.15 | +1.16 | +1.47 | — |

Suggested caption:

> Development-set ablation of the training-time temporal curriculum. T3-to-T1
> improves the mean association and tracking metrics over two valid paired
> seeds while retaining single-frame inference and post-decode-to-output P99
> below 5.3 ms in both paired runs. These results are not held-out MOT17
> benchmark results.

P99 values come from the current root-level latency profiles for
MOT17-13-SDP, 700 measured frames per run. The timer starts after decoded GPU
frame fetch and ends after tracking output. It excludes decode/fetch waiting
and is not camera-to-result latency.

## Per-seed paired deltas

| Seed | Delta IDF1 | Delta MOTA | Delta HOTA | Delta AssA | Delta IDs | Delta FP | Delta FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260613 | +2.27 | +1.39 | +2.04 | +2.57 | -94 | -1,364 | -101 |
| 20260614 | +0.42 | +1.03 | +0.26 | -0.25 | -88 | -962 | -110 |

The curriculum is positive for IDF1, MOTA, HOTA, identity switches, false
positives, and false negatives in both valid pairs. AssA is seed-sensitive and
slightly negative for seed 20260614.

## Historical and diagnostic runs

| Experiment | IDF1 | MOTA | HOTA | AssA | IDs | FPS | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Legacy v14 | 75.14 | 77.67 | 68.19 | 66.65 | 482 | 216.34 | Historical target |
| Plain replica, seed 20260612 | 73.39 | 77.26 | 65.25 | 61.89 | 605 | 220.26 | Reconstructed baseline |
| T3-to-T1, seed 42 | 75.45 | 77.61 | 67.71 | 65.96 | 496 | 217.25 | Best run, but not seed-paired |
| SSM unfreeze fine-tune | 73.27 | 78.13 | 66.13 | 62.78 | 606 | 212.09 | Better MOTA, no IDF1 gain |

Do not use the seed-42 T3-to-T1 run as a paired ablation against the
seed-20260612 plain replica.

## Supporting bridge ablation

| Method | IDF1 | MOTA | HOTA | AssA | IDs | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bridge off | 73.03 | 76.81 | 66.55 | 63.77 | 558 | 4,092 | 21,387 |
| Bridge on | 75.14 | 77.67 | 68.19 | 66.65 | 482 | 3,514 | 21,082 |
| Delta | +2.12 | +0.85 | +1.64 | +2.88 | -76 | -578 | -305 |

This belongs in a supporting tracking-ablation section. It should not be
merged into the central Mamba contribution without a factorial experiment.

## Figures

- `figures/mamba_t3t1_paired_metrics.png`: aggregate and per-seed paired
  curriculum effects.
- `figures/mamba_t3t1_per_sequence_idf1.png`: sequence-level IDF1 deltas,
  showing both gains and seed sensitivity.

Exact unrounded values and provenance are in `tables/` and
`paper_metrics.json`.
