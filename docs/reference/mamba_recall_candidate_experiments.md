# Mamba recall candidate experiments

_Investigation log, 2026-06-29. Goal: turn the recall-bottleneck discussion into
measurable experiments, identify useful candidates, and define how to combine
them into a final recall-first plan._

Compact decision index: `docs/reference/mamba_usable_signal_inventory.md`.

## 1. Current evidence

### Existing accuracy evidence

From `docs/modules/detection/research/mamba-dual-resolution-original-detail-plan.md`:

| candidate | evidence | readout |
|---|---:|---|
| B1-H high-res shallow detail vs B1-L low-res control | MOT17-02 detector-only @ score 0.25: 4-8 px +0.443pp recall, CI [+0.052, +0.834]; 8-16 px +0.328pp, CI [+0.061, +0.638] | weak positive, mostly confidence lift |
| B2 native high-res YOLO P3 semantic oracle | does not clearly beat B1-H or B0-R in size-binned recall | do not expand semantic depth first |
| B1-H patch 5x5 | all detector-only CIs include 0; tracking regresses | no-go |
| B1-H 64ch encoder | only one low-threshold bin improves; operational score and tracking regress | no-go |
| yolo26m backbone | prior project memory: small-object recall +7pp, MOTA +0.8, FP -14%, AssA -3 before relink retune | strong but heavy fallback |

Interpretation: high-resolution detail has a real but small signal. It appears
to lift some existing candidates into the operational score range, not create a
large number of detections that were absent at low confidence.

### Existing latency evidence

From `docs/reference/mamba_head_recall_bottleneck.md` and
`scripts/eval/bench_reduction_bypass.py`: full-resolution scan / no down-up is
closed for production because it is ~12-14x head-only slower. Uniform sr=2 is
borderline at roughly 3x head-only latency.

### New assigner-capacity evidence

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/tools/mamba_assigner_diagnostics.py \
  --seqs MOT17-02-SDP \
  --batch-size 8 \
  --output report_data/mamba_assigner_diagnostics_mot17_02_full.json
```

Result on full MOT17-02-SDP train sequence (600 frames, 10,993 GT):

| metric | all GT | 4-8 px | 8-16 px | >=16 px |
|---|---:|---:|---:|---:|
| GT count | 10,993 | 2,259 | 4,874 | 3,860 |
| zero-positive before conflict | 0.027% | 0.133% | 0.000% | 0.000% |
| zero-positive after conflict | 1.010% | 4.294% | 0.226% | 0.078% |
| lost all positives to conflict | 0.982% | 4.161% | 0.226% | 0.078% |
| mean positives before conflict | 9.188 | 7.572 | 9.293 | 10.000 |
| mean positives after conflict | 8.185 | 4.276 | 8.690 | 9.836 |
| P3 share of final positives | 48.1% overall | 76.4% | 75.1% | 10.9% |

Interpretation:

- Small GT are almost never missing candidate positives before conflict
  resolution. This argues against "no grid/anchor capacity at all" as the
  primary recall bottleneck.
- The real small-GT capacity issue is narrower: 4-8 px objects lose all positives
  after conflict resolution in ~4.2% of GT cases. This can justify a later P2
  small-only head, but not as the first architecture bet.
- Because 4-8 and 8-16 px positives remain mostly P3-owned after conflict, the
  first structural experiments should target P3 input quality / score lift:
  anti-aliased or phase-preserving reduction before Mamba.

### External method mapping

External scan, 2026-06-29:

| external idea | source | local mapping | current decision |
|---|---|---|---|
| Space-to-depth before convolution | [SPD-Conv](https://arxiv.org/abs/2208.03641) replaces strided conv/pooling with space-to-depth plus non-strided conv for low-resolution images and small objects | `space-to-depth` reduction in `MambaDetectionHead`; exact warm start from legacy stride conv | full30 no-go as standalone despite local 3-epoch weak positive |
| Low-pass before downsampling | [anti-aliased CNN / BlurPool](https://arxiv.org/abs/1904.11486) uses filtering before downsampling to improve shift stability | `blur-conv` reduction | Preliminary no-go; local recall regressed |
| Sliced or adaptive high-resolution inference | [SAHI](https://arxiv.org/abs/2202.06934), [ASAHI](https://www.mdpi.com/2072-4292/15/5/1249), and [ESOD](https://arxiv.org/abs/2407.16424) reduce full high-res cost by slicing/adaptive/sparse processing | track-conditioned high-res second pass or sparse ROI pass, not full-frame high-res Mamba | Tier 1 next prototype |
| Wavelet / frequency-preserving downsampling | [WaveCNet](https://arxiv.org/abs/2107.13335), [Residual Haar Wavelet Downsampling](https://arxiv.org/abs/2603.03788), and [DERNet](https://arxiv.org/abs/2606.23825) decompose/enhance/reconstruct ideas preserve frequency/detail instead of raw stride-only downsampling | `wavelet` reduction in `MambaDetectionHead`; exact warm start from legacy stride conv | preliminary no-go as standalone after matched 3-epoch slice |

Deep-search report mapping from
`/mnt/c/Users/Ray Lei/Downloads/小物件偵測論文分析.md`:

| report bucket | local experiments already run | current interpretation for this project |
|---|---|---|
| Adaptive slicing / GOIS / track-conditioned ROI second pass | latency gates closed full-frame high-res Mamba; no ROI second-pass prototype yet | still useful, but now a residual-localization/absent-box fix after NMS/conflict. It should not be the first move because object records show most misses already have overlapping evidence. |
| Local / atrous / dynamic Mamba scan | `p3_sr2`, `all_sr2`, `p3_space_to_depth_sr2`, and full-scan latency probes | reserve only. P3 sr2 is ~3x head-only and full scan is ~12x, so any scan change must be local/windowed/sparse rather than global. |
| SPD-Conv / anti-aliased / wavelet downsampling | `space-to-depth` full30 gate; `blur-conv` slice; Haar `wavelet` slice | same-grid reduction is weak. `space-to-depth`, `blur-conv`, and `wavelet` are no-go as standalone head replacements. |
| YOLO-P2 / NGLA / NWD / small-object assigner | assigner diagnostics, no P2/NWD implementation yet | conditional. MOT17-02 shows almost no zero positives before conflict, but 4-8 px has ~4.2% lost-all-to-conflict after assignment. P2/NWD should target conflict/crowding, not broad missing-grid recovery. |
| ByteTrack / low-score recovery / temporal aggregation | NMS sweep, no-interp tracking, `stage2_match_thresh`, `new_track_thresh`, `min_tracklet_score` guard sweep | strongest current match. NMS/conflict and low-score temporal policies already produced usable candidates; next step is cross-seq validation and suppressed-candidate/private matching. |

Important caveat: the downloaded report is a strategy source, not primary
evidence. Paper-level AP claims should be verified from primary papers before
they drive implementation, while local MOT17 metrics in this document remain the
decision authority for our codebase.

### NGLA / NBCD assigner probe

The deep-search report suggested NGLA/NWD-style Gaussian assignment for tiny
objects. A minimal local variant was implemented as `NGLAAssigner`: it keeps
Ultralytics TaskAlignedAssigner's inside-GT mask, top-k selection, and target
normalization, but swaps the localization score from CIoU to normalized
Bhattacharyya-distance similarity between Gaussian boxes.

Implementation:

```text
src/saccade/perception/temporal_yolo/ngla_assigner.py
scripts/tools/mamba_assigner_diagnostics.py --assigner {tal,ngla}
scripts/train/temporal_yolo/train_mamba_gt.py --assigner {tal,ngla}
```

Diagnostics command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/tools/mamba_assigner_diagnostics.py \
  --seqs MOT17-02-SDP \
  --batch-size 8 \
  --mamba-ckpt runs/reduction_candidates_full30/baseline_conv/best_recall.ckpt \
  --assigner ngla \
  --output report_data/mamba_assigner_diagnostics_ngla_mot17_02_full.json
```

Full MOT17-02-SDP assigner diagnostics, TAL vs NGLA:

| metric | TAL | NGLA | delta |
|---|---:|---:|---:|
| zero pre rate | 0.027% | 0.000% | -0.027pp |
| zero post rate | 1.010% | 1.137% | +0.127pp |
| lost all to conflict | 0.982% | 1.137% | +0.155pp |
| conflict anchor rate | 0.188% | 0.250% | +0.061pp |
| mean positives before conflict | 9.188 | 9.459 | +0.271 |
| mean positives after conflict | 8.185 | 8.000 | -0.185 |
| 4-8 px zero pre rate | 0.133% | 0.000% | -0.133pp |
| 4-8 px zero post rate | 4.294% | 4.205% | -0.089pp |
| 4-8 px lost all to conflict | 4.161% | 4.205% | +0.044pp |
| 4-8 px mean positives before conflict | 7.572 | 8.878 | +1.306 |
| 4-8 px mean positives after conflict | 4.276 | 4.230 | -0.046 |
| 8-16 px lost all to conflict | 0.226% | 0.431% | +0.205pp |
| >=16 px lost all to conflict | 0.078% | 0.233% | +0.155pp |

Smoke training:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/train/temporal_yolo/train_mamba_gt.py \
  --data-root datasets/MOT17 \
  --yolo-weights models/yolo/yolo26s.pt \
  --teacher-ckpt runs/gated_det_v1/best.ckpt \
  --mamba-ckpt runs/reduction_candidates_full30/baseline_conv/best_recall.ckpt \
  --run-dir runs/smoke_mamba_ngla_assigner \
  --epochs 1 --warmup-epochs 1 --batch-size 2 --clip-len 1 --clip-stride 16 \
  --seqs MOT17-02-SDP --max-batches-per-epoch 1 \
  --best-by train-loss --save-every 1 \
  --assigner ngla
```

Smoke result: loss 1.3744, checkpoint saved. This proves the training path is
wired, but it is not an accuracy result.

Readout:

- NGLA removes the tiny number of zero-positive-before-conflict cases, but this
  was not the real bottleneck.
- It increases conflict anchors and does not improve post-conflict positives.
  For 4-8 px it raises pre-conflict positives by +1.31 per GT but leaves
  post-conflict positives slightly lower.
- Direct NGLA/NBCD swap is a **NO-GO as the next full training arm**. A future
  P2/conflict-specific head could still use Gaussian/NWD ideas, but not as a
  standalone replacement for TAL in the current P3/P4/P5 head.

## 2. New experiment added

Script:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/eval/bench_recall_candidates.py \
  --iters 200 \
  --warmup 50 \
  --variants baseline_sr4 blur_conv_sr4 space_to_depth_sr4 p3_sr2 all_sr2 \
  --json-out report_data/mamba_recall_candidate_latency_confirm.json
```

Smoke including full scan:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/eval/bench_recall_candidates.py \
  --iters 50 \
  --warmup 10 \
  --json-out report_data/mamba_recall_candidate_latency.json
```

Environment:

```text
GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
ckpt metadata: runs/mamba_gt_vgt_mamba_v14/best.ckpt
d_model=128, base spatial_reduction=4, FPN inputs=[128, 256, 512]
synthetic FPN: P3 80x80, P4 40x40, P5 20x20
```

Important limitation: this is head-only eager fp32 synthetic latency. It is not
an accuracy result and not the production CUDA-graph wall time. Use ratios and
scan grids to screen candidates before training.

## 3. New latency results

Confirm run:

| variant | scan grids P3/P4/P5 | tokens | head ms | rel | decision |
|---|---|---:|---:|---:|---|
| baseline_sr4 | 20x20 / 10x10 / 5x5 | 400 / 100 / 25 | 12.541 | 1.00x | baseline |
| blur_conv_sr4 | 20x20 / 10x10 / 5x5 | 400 / 100 / 25 | 12.721 | 1.01x | GO to accuracy test |
| space_to_depth_sr4 | 20x20 / 10x10 / 5x5 | 400 / 100 / 25 | 13.488 | 1.08x | GO to accuracy test |
| p3_sr2 | 40x40 / 10x10 / 5x5 | 1600 / 100 / 25 | 36.977 | 2.95x | reserve, not first |
| all_sr2 | 40x40 / 20x20 / 10x10 | 1600 / 400 / 100 | 44.685 | 3.56x | reserve / borderline |

Smoke-only additional rows:

| variant | scan grids P3/P4/P5 | head ms | rel | decision |
|---|---|---:|---:|---|
| p3_space_to_depth_sr2 | 40x40 / 10x10 / 5x5 | 42.077 | 2.93x | no better than p3_sr2 |
| no_downup_full_scan | 80x80 / 40x40 / 20x20 | 171.288 | 11.91x | closed |

Wavelet follow-up latency, same script with
`--variants baseline_sr4 wavelet_sr4 space_to_depth_sr4 --iters 100 --warmup 25`:

| variant | scan grids P3/P4/P5 | tokens | head ms | rel | decision |
|---|---|---:|---:|---:|---|
| baseline_sr4 | 20x20 / 10x10 / 5x5 | 400 / 100 / 25 | 13.638 | 1.00x | matched baseline |
| wavelet_sr4 | 20x20 / 10x10 / 5x5 | 400 / 100 / 25 | 13.795 | 1.01x | cheap enough for accuracy slice |
| space_to_depth_sr4 | 20x20 / 10x10 / 5x5 | 400 / 100 / 25 | 12.922 | 0.95x | already full30 no-go |

Result: P3 dominates the scan cost. P3-only sr=2 is almost as expensive as the
uniform sr=2 family, so the best first candidates are not higher scan
resolution. They are better information-preserving reduction at the same scan
grid.

## 4. Implementation readiness

Implemented reduction variants in `MambaDetectionHead`:

| variant | module | checkpoint migration |
|---|---|---|
| `conv` | original stride-`spatial_reduction` conv | unchanged |
| `blur-conv` | fixed 3x3 low-pass filter + original stride conv | copies legacy `downsample.{i}.weight/bias` into `.conv` |
| `space-to-depth` | `pixel_unshuffle(r)` + 1x1 projection | converts legacy `r x r` stride conv into exact 1x1 phase-packed projection |
| `wavelet` | repeated 2x2 Haar sub-band packing + 1x1 projection | converts legacy `r x r` stride conv into exact 1x1 Haar-packed projection |

Warm-start verification:

```text
space-to-depth max_abs_diff vs legacy stride conv: 1.55e-6
wavelet max_abs_diff vs legacy stride conv: 1.43e-6 (sr=2), 4.77e-6 (sr=4)
blur-conv / space-to-depth / wavelet load from legacy checkpoint through the reduction migration path
```

Training smoke:

```text
runs/smoke_mamba_reduction_blur_conv/best.ckpt
  reduction_variant=blur-conv, epoch=1, one-batch loss=1.1167

runs/smoke_mamba_reduction_space_to_depth/best.ckpt
  reduction_variant=space-to-depth, epoch=1, one-batch loss=0.7633

runs/smoke_mamba_reduction_wavelet/best.ckpt
  reduction_variant=wavelet, epoch=1, one-batch loss=1.3826
```

These smoke losses are not accuracy evidence. They only prove that the
train/load/save path is wired and that checkpoint metadata round-trips.

## 5. Preliminary accuracy slice

This is the first matched detector-only accuracy slice for the reduction
candidates. It is deliberately smaller than the final experiment: MOT17-02-SDP
only, 3 epochs, train-loss checkpoint selection, no held-out epoch selection.
Use it to screen candidates, not as the final proof.

Training/eval:

```bash
OUT_ROOT=runs/reduction_candidates_slice3 \
REPORT_ROOT=report_data/reduction_candidates_slice3 \
EPOCHS=3 WARMUP_EPOCHS=1 BEST_BY=train-loss SELECT=0 \
SEQS=MOT17-02-SDP EVAL_SEQS=MOT17-02-SDP \
scripts/train/temporal_yolo/run_reduction_candidates.sh train

OUT_ROOT=runs/reduction_candidates_slice3 \
REPORT_ROOT=report_data/reduction_candidates_slice3 \
EVAL_SEQS=MOT17-02-SDP \
scripts/train/temporal_yolo/run_reduction_candidates.sh eval

OUT_ROOT=runs/reduction_candidates_slice3 \
REPORT_ROOT=report_data/reduction_candidates_slice3 \
EVAL_SEQS=MOT17-02-SDP BOOTSTRAP_SAMPLES=20000 \
scripts/train/temporal_yolo/run_reduction_candidates.sh bootstrap
```

Train-loss selected checkpoints:

| arm | best train loss | ckpt |
|---|---:|---|
| baseline_conv | 3.2018 | `runs/reduction_candidates_slice3/baseline_conv/best.ckpt` |
| blur_conv | 3.6621 | `runs/reduction_candidates_slice3/blur_conv/best.ckpt` |
| space_to_depth | 3.2196 | `runs/reduction_candidates_slice3/space_to_depth/best.ckpt` |

Detector-only recall on MOT17-02-SDP:

| arm | score | all | min 4-8 px | min 8-16 px |
|---|---:|---:|---:|---:|
| baseline_conv | 0.001 | 96.170 | 88.800 | 97.784 |
| blur_conv | 0.001 | 95.934 | 88.225 | 97.600 |
| space_to_depth | 0.001 | 96.198 | 89.110 | 97.825 |
| baseline_conv | 0.10 | 94.824 | 87.649 | 96.225 |
| blur_conv | 0.10 | 94.442 | 86.853 | 95.897 |
| space_to_depth | 0.10 | 94.897 | 87.959 | 96.204 |
| baseline_conv | 0.25 | 93.732 | 86.012 | 94.932 |
| blur_conv | 0.25 | 92.895 | 84.418 | 93.989 |
| space_to_depth | 0.25 | 93.796 | 86.233 | 94.932 |

Paired moving-block bootstrap vs matched baseline:

| arm | score | bin | recall delta | 95% CI | P(delta>0) | FN reduction |
|---|---:|---|---:|---:|---:|---:|
| blur_conv | 0.001 | all | -0.237pp | [-0.557,+0.101] | 0.077 | -6.18% |
| blur_conv | 0.10 | all | -0.382pp | [-0.718,-0.055] | 0.011 | -7.38% |
| blur_conv | 0.25 | all | -0.837pp | [-1.316,-0.388] | 0.000 | -13.35% |
| blur_conv | 0.25 | min 4-8 px | -1.594pp | [-2.794,-0.446] | 0.003 | -11.39% |
| blur_conv | 0.25 | min 8-16 px | -0.944pp | [-1.621,-0.364] | 0.000 | -18.62% |
| space_to_depth | 0.001 | all | +0.027pp | [-0.100,+0.164] | 0.624 | +0.71% |
| space_to_depth | 0.001 | min 4-8 px | +0.310pp | [+0.000,+0.688] | 0.958 | +2.77% |
| space_to_depth | 0.10 | min 4-8 px | +0.310pp | [+0.000,+0.675] | 0.967 | +2.51% |
| space_to_depth | 0.25 | all | +0.064pp | [-0.055,+0.207] | 0.810 | +1.02% |
| space_to_depth | 0.25 | min 4-8 px | +0.221pp | [-0.125,+0.640] | 0.859 | +1.58% |

Readout:

- `blur-conv` is a preliminary **NO-GO** for recall. It under-trains relative to
  baseline and shows statistically negative detector recall at score 0.25.
- `space-to-depth` is a **weak positive / continue**. It is near-baseline overall
  and improves 4-8 px recall in the low/mid score bands with high
  `P(delta>0)`, but the operational score 0.25 gain is small and CI crosses 0.
- This pattern matches the earlier B1-H detail result: the available head-side
  signal is mostly confidence/score lift for small candidates, not a large new
  detection source.

## 6. Full validation gate

The 3-epoch slice was followed by a matched 30-epoch gate with recall-selected
checkpoints. `blur-conv` was excluded because the slice was already negative.

Command:

```bash
ARMS="baseline_conv space_to_depth" \
OUT_ROOT=runs/reduction_candidates_full30 \
REPORT_ROOT=report_data/reduction_candidates_full30 \
SEQS=MOT17-02-SDP EVAL_SEQS=MOT17-02-SDP \
EPOCHS=30 WARMUP_EPOCHS=5 BEST_BY=none SELECT=1 SAVE_EVERY=1 \
  scripts/train/temporal_yolo/run_reduction_candidates.sh all
```

Recall-selected checkpoints:

| arm | selected epoch | train best loss | tracking Rcll | IDF1 | MOTA | IDs |
|---|---:|---:|---:|---:|---:|---:|
| baseline_conv | 19 | 1.9934 | 61.00 | 57.3 | 59.3 | 81 |
| space_to_depth | 10 | 1.9909 | 60.90 | 56.0 | 59.0 | 81 |

Detector-only recall on MOT17-02-SDP:

| arm | score | all | min 4-8 px | min 8-16 px |
|---|---:|---:|---:|---:|
| baseline_conv | 0.001 | 96.543 | 91.014 | 97.723 |
| space_to_depth | 0.001 | 96.507 | 90.792 | 97.764 |
| baseline_conv | 0.10 | 95.306 | 89.996 | 96.225 |
| space_to_depth | 0.10 | 95.233 | 89.597 | 96.163 |
| baseline_conv | 0.25 | 94.506 | 88.800 | 95.240 |
| space_to_depth | 0.25 | 94.378 | 88.402 | 95.076 |

Paired moving-block bootstrap, `space_to_depth` vs `baseline_conv`:

| score | bin | recall delta | 95% CI | P(delta>0) | FN reduction |
|---:|---|---:|---:|---:|---:|
| 0.001 | all | -0.036pp | [-0.224,+0.142] | 0.336 | -1.05% |
| 0.001 | min 4-8 px | -0.221pp | [-0.734,+0.254] | 0.159 | -2.46% |
| 0.001 | min 8-16 px | +0.041pp | [-0.250,+0.330] | 0.586 | +1.80% |
| 0.10 | all | -0.073pp | [-0.342,+0.181] | 0.290 | -1.55% |
| 0.10 | min 4-8 px | -0.398pp | [-1.311,+0.346] | 0.155 | -3.98% |
| 0.10 | min 8-16 px | -0.062pp | [-0.485,+0.328] | 0.380 | -1.63% |
| 0.25 | all | -0.127pp | [-0.414,+0.140] | 0.176 | -2.32% |
| 0.25 | min 4-8 px | -0.398pp | [-1.397,+0.431] | 0.176 | -3.56% |
| 0.25 | min 8-16 px | -0.164pp | [-0.618,+0.255] | 0.222 | -3.45% |

Readout:

- `space-to-depth` is a **NO-GO as a standalone reduction replacement**. The
  full gate does not reproduce the 3-epoch weak positive and misses the GO
  criteria at all operational bins.
- The result is not a catastrophic regression: CIs cross 0. But it is not a
  useful candidate because the mean deltas point negative and tracking Rcll is
  slightly lower than the matched baseline.
- The current evidence says "same-grid reduction changes" are unlikely to solve
  the recall bottleneck alone. The next useful experiments should either create
  extra high-resolution evidence only where needed, or directly address small-GT
  conflict/assignment.

### Wavelet / DERNet / WaveCNet follow-up slice

The DERNet / WaveCNet family was mapped to the same concrete local question:
can the head preserve frequency/detail during the P3/P4/P5 reduction before
Mamba without increasing scan tokens? A minimal Haar version was implemented as
`reduction_variant=wavelet`: repeated 2x2 Haar sub-band packing followed by the
same 1x1 projection shape as `space-to-depth`. This keeps the scan grid
unchanged and has an exact legacy stride-conv warm start.

Matched 3-epoch slice:

```bash
ARMS="baseline_conv wavelet" \
OUT_ROOT=runs/reduction_candidates_wavelet_slice3 \
REPORT_ROOT=report_data/reduction_candidates_wavelet_slice3 \
EPOCHS=3 WARMUP_EPOCHS=1 BEST_BY=train-loss SELECT=0 \
SEQS=MOT17-02-SDP EVAL_SEQS=MOT17-02-SDP \
BATCH_SIZE=4 CLIP_LEN=4 CLIP_STRIDE=8 SAVE_EVERY=1 \
  scripts/train/temporal_yolo/run_reduction_candidates.sh all
```

Train-loss selected checkpoints:

| arm | best train loss | ckpt |
|---|---:|---|
| baseline_conv | 3.2018 | `runs/reduction_candidates_wavelet_slice3/baseline_conv/best.ckpt` |
| wavelet | 3.1893 | `runs/reduction_candidates_wavelet_slice3/wavelet/best.ckpt` |

Detector-only recall on MOT17-02-SDP:

| arm | score | all | min 4-8 px | min 8-16 px |
|---|---:|---:|---:|---:|
| baseline_conv | 0.001 | 96.170 | 88.800 | 97.784 |
| wavelet | 0.001 | 96.134 | 88.756 | 97.682 |
| baseline_conv | 0.10 | 94.824 | 87.649 | 96.225 |
| wavelet | 0.10 | 94.751 | 87.605 | 96.081 |
| baseline_conv | 0.25 | 93.732 | 86.012 | 94.932 |
| wavelet | 0.25 | 93.669 | 85.967 | 94.768 |

Paired moving-block bootstrap, `wavelet` vs `baseline_conv`:

| score | bin | recall delta | 95% CI | P(delta>0) | FN reduction |
|---:|---|---:|---:|---:|---:|
| 0.001 | all | -0.036pp | [-0.143,+0.063] | 0.222 | -0.95% |
| 0.001 | min 4-8 px | -0.044pp | [-0.326,+0.210] | 0.301 | -0.40% |
| 0.001 | min 8-16 px | -0.103pp | [-0.269,+0.040] | 0.065 | -4.63% |
| 0.10 | all | -0.073pp | [-0.211,+0.053] | 0.118 | -1.41% |
| 0.10 | min 4-8 px | -0.044pp | [-0.336,+0.239] | 0.309 | -0.36% |
| 0.10 | min 8-16 px | -0.144pp | [-0.338,+0.021] | 0.035 | -3.80% |
| 0.25 | all | -0.064pp | [-0.227,+0.079] | 0.193 | -1.02% |
| 0.25 | min 4-8 px | -0.044pp | [-0.481,+0.358] | 0.376 | -0.32% |
| 0.25 | min 8-16 px | -0.164pp | [-0.400,+0.042] | 0.055 | -3.24% |

Readout:

- `wavelet` is a **NO-GO as a standalone reduction replacement**. It is cheap
  and trains normally, but none of the MOT17-02 score/bin comparisons show a
  positive mean delta.
- This closes the local DERNet/WaveCNet-style "frequency-preserving downsample
  only" hypothesis for the current head. A future frequency module would need
  to be attached to a different mechanism, such as P2/conflict handling or a
  sparse high-res ROI pass, not just swapped into the stride-4 reduction.
- The broader same-grid conclusion is now consistent across SPD-Conv,
  anti-aliased blur-conv, and Haar wavelet reduction.

## 7. Miss taxonomy and NMS gate

The reduction gate suggested that same-grid information preservation was not
the main remaining recall lever. To locate the misses, the baseline full30
checkpoint was rerun with object records:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/eval/detector/mamba_size_binned_recall.py \
  --data-root datasets/MOT17 \
  --sequences MOT17-02-SDP \
  --yolo-weights models/yolo/yolo26s.pt \
  --teacher-ckpt runs/gated_det_v1/best.ckpt \
  --mamba-ckpt runs/reduction_candidates_full30/baseline_conv/best_recall.ckpt \
  --score-thresholds 0.001,0.10,0.25 \
  --save-object-records \
  --output report_data/reduction_candidates_full30/baseline_conv_object_recall.json
```

Miss classification rule:

- `conflict_high_score`: the GT is unmatched, but a prediction at the active
  score threshold overlaps it with IoU >= 0.5. This is mostly one-to-one match
  conflict / duplicate suppression / crowding.
- `score_only_low_score`: no active-threshold match, but a conf-floor prediction
  overlaps with IoU >= 0.5.
- `localization_near`: best conf-floor overlap is IoU 0.3-0.5.
- `absent_or_far`: best conf-floor overlap is below IoU 0.3.

Baseline miss taxonomy:

| score | bin | misses | conflict high-score | low-score only | localization near | absent/far |
|---:|---|---:|---:|---:|---:|---:|
| 0.001 | min 4-8 px | 203 | 140 (69%) | 0 (0%) | 39 (19%) | 24 (12%) |
| 0.001 | min 8-16 px | 111 | 83 (75%) | 0 (0%) | 21 (19%) | 7 (6%) |
| 0.25 | min 4-8 px | 253 | 143 (57%) | 47 (19%) | 39 (15%) | 24 (9%) |
| 0.25 | min 8-16 px | 232 | 108 (47%) | 96 (41%) | 21 (9%) | 7 (3%) |

Readout:

- The dominant miss source is not absence of evidence. Most residual misses
  already have an overlapping candidate, especially at score 0.001.
- At score 0.25, `min_8to16` has a large score-calibration component, while
  `min_4to8` is still mostly conflict/crowding.
- This moves the next useful candidate class toward NMS / duplicate retention /
  association-side use of suppressed candidates, before adding high-res compute.

### Detector-only NMS sweep

The baseline checkpoint was evaluated with wider NMS thresholds. This is not a
tracking decision by itself because extra detections can become FP and IDs.

| NMS IoU | predictions | all recall | min 4-8 px | min 8-16 px | all FN | 4-8 FN | 8-16 FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 10,964 | 94.506 | 88.800 | 95.240 | 604 | 253 | 232 |
| 0.55 | 11,134 | 95.151 | 89.022 | 95.773 | 533 | 248 | 206 |
| 0.60 | 11,327 | 95.561 | 89.464 | 96.102 | 488 | 238 | 190 |
| 0.70 | 11,948 | 96.489 | 90.837 | 97.128 | 386 | 207 | 140 |
| 0.90 | 22,846 | 97.744 | 94.378 | 97.907 | 248 | 127 | 102 |

Paired moving-block bootstrap vs NMS 0.50 at score 0.25:

| NMS IoU | all delta | min 4-8 delta | min 8-16 delta | readout |
|---:|---:|---:|---:|---|
| 0.55 | +0.646pp CI [+0.429,+0.881] | +0.221pp CI [+0.051,+0.417] | +0.533pp CI [+0.282,+0.822] | useful conservative detector gain |
| 0.60 | +1.055pp CI [+0.715,+1.418] | +0.664pp CI [+0.290,+1.123] | +0.862pp CI [+0.453,+1.316] | stronger detector gain, needs FP/ID guard |
| 0.70 | +1.983pp CI [+1.363,+2.618] | +2.036pp CI [+1.008,+3.295] | +1.888pp CI [+1.107,+2.714] | detector upper-bound, likely too many duplicates |
| 0.90 | +3.238pp CI [+2.291,+4.241] | +5.578pp CI [+2.913,+8.487] | +2.667pp CI [+1.724,+3.685] | proof of suppressed evidence, not a deployable raw setting |

### Tracking gate with interpolation controlled

Important confound: `mamba_whole_graph.yaml` enables tracklet interpolation by
default. Interpolation changes FP/FN/IDs, so NMS tracking comparisons must be
read first with `--no-interpolate-tracklets`.

No-interpolation tracking on MOT17-02-SDP:

| NMS IoU | IDF1 | MOTA | IDs | FP | FN | Rcll | Prcn | mean ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 54.7 | 53.9 | 126 | 121 | 8321 | 55.2 | 98.8 | 7.80 |
| 0.55 | 55.6 | 54.1 | 124 | 129 | 8279 | 55.4 | 98.8 | 6.52 |
| 0.575 | 55.5 | 54.0 | 133 | 137 | 8277 | 55.5 | 98.7 | 6.57 |
| 0.60 | 56.6 | 54.1 | 146 | 149 | 8241 | 55.6 | 98.6 | 6.45 |
| 0.625 | 54.9 | 54.1 | 150 | 178 | 8193 | 55.9 | 98.3 | 6.68 |
| 0.65 | 55.2 | 54.2 | 152 | 188 | 8178 | 56.0 | 98.2 | 6.54 |
| 0.70 | 54.4 | 54.2 | 172 | 249 | 8091 | 56.5 | 97.7 | 6.72 |

Default interpolation-on tracking:

| NMS IoU | IDF1 | MOTA | IDs | FP | FN | Rcll | Prcn | mean ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 57.3 | 59.3 | 81 | 239 | 7240 | 61.0 | 97.9 | 6.30 |
| 0.55 | 58.5 | 59.4 | 86 | 268 | 7191 | 61.3 | 97.7 | 6.45 |
| 0.60 | 59.5 | 59.1 | 91 | 349 | 7151 | 61.5 | 97.0 | 6.42 |
| 0.70 | 56.5 | 58.0 | 98 | 525 | 7181 | 61.4 | 95.6 | 6.39 |

Readout:

- Interpolation is a large metric intervention, not a neutral post-process. At
  NMS 0.50 it changes FP 121 -> 239, FN 8321 -> 7240, and IDs 126 -> 81.
- NMS 0.55 is the safest direct candidate: no-interp IDF1 +0.9pp, MOTA +0.2pp,
  IDs -2, FP +8.
- NMS 0.60 is the strongest direct candidate if IDF1/recall are prioritized:
  no-interp IDF1 +1.9pp and FN -80, but IDs +20 and FP +28. It needs a guard
  before promotion.
- NMS 0.625 and above continue to lower FN but start paying too much in FP/IDs
  and lose IDF1. NMS 0.70/0.90 are useful as evidence that suppressed boxes
  contain real GT, not as raw production settings.

### NMS 0.60 guard sweep

NMS 0.60 has the best direct no-interp IDF1, but raw IDs/FP are higher than the
safer NMS 0.55 point. A small guard sweep tested tracker parameters that do not
require architecture changes.

No-interpolation guard sweep:

| setting | IDF1 | MOTA | IDs | FP | FN | Rcll | Prcn | mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NMS 0.50 baseline | 54.7 | 53.9 | 126 | 121 | 8321 | 55.2 | 98.8 | 7.80 |
| NMS 0.55 | 55.6 | 54.1 | 124 | 129 | 8279 | 55.4 | 98.8 | 6.52 |
| NMS 0.55 + new_track 0.40 | 55.7 | 54.3 | 125 | 136 | 8226 | 55.7 | 98.7 | 6.45 |
| NMS 0.60 | 56.6 | 54.1 | 146 | 149 | 8241 | 55.6 | 98.6 | 6.45 |
| NMS 0.60 + new_track 0.32 | 56.6 | 54.2 | 143 | 152 | 8208 | 55.8 | 98.6 | 6.75 |
| NMS 0.60 + new_track 0.35 | 56.7 | 54.3 | 144 | 151 | 8192 | 55.9 | 98.6 | 6.62 |
| NMS 0.60 + new_track 0.40 | 56.9 | 54.4 | 150 | 145 | 8173 | 56.0 | 98.6 | 6.66 |
| NMS 0.60 + stage2 0.60 | 56.0 | 54.1 | 141 | 155 | 8226 | 55.7 | 98.5 | 6.65 |
| NMS 0.60 + stage2 0.70 | 56.0 | 54.2 | 141 | 159 | 8206 | 55.8 | 98.5 | 6.68 |
| NMS 0.60 + min_tracklet_score 0.35 | 56.6 | 54.0 | 142 | 147 | 8251 | 55.6 | 98.6 | 6.71 |
| NMS 0.60 + new_track 0.40 + min_tracklet_score 0.35 | 56.9 | 54.4 | 145 | 141 | 8188 | 55.9 | 98.7 | 6.48 |

Production/default interpolation-on check:

| setting | IDF1 | MOTA | IDs | FP | FN | Rcll | Prcn | mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NMS 0.50 baseline | 57.3 | 59.3 | 81 | 239 | 7240 | 61.0 | 97.9 | 6.30 |
| NMS 0.55 | 58.5 | 59.4 | 86 | 268 | 7191 | 61.3 | 97.7 | 6.45 |
| NMS 0.55 + new_track 0.40 | 58.5 | 59.5 | 86 | 289 | 7154 | 61.5 | 97.5 | 6.44 |
| NMS 0.60 | 59.5 | 59.1 | 91 | 349 | 7151 | 61.5 | 97.0 | 6.42 |
| NMS 0.60 + new_track 0.40 | 59.6 | 59.5 | 87 | 353 | 7085 | 61.9 | 97.0 | 6.66 |
| NMS 0.60 + new_track 0.40 + min_tracklet_score 0.35 | 59.6 | 59.5 | 85 | 340 | 7097 | 61.8 | 97.1 | 6.25 |

Readout:

- `new_track_thresh` is the useful cheap guard. Raising it to 0.40 improves
  no-interp IDF1/MOTA/FN for both NMS 0.55 and NMS 0.60.
- `stage2_match_thresh` is not a useful standalone guard here. It lowers IDs a
  little but gives back IDF1 and raises FP.
- `min_tracklet_score=0.35` is useful only as a light post-filter on top of
  NMS 0.60 + new_track 0.40: it keeps IDF1/MOTA, reduces no-interp FP 145 -> 141
  and IDs 150 -> 145, and reduces production FP 353 -> 340 and IDs 87 -> 85.
- Current best low-risk candidate: **NMS 0.55 + new_track 0.40**. No-interp
  IDF1 +1.0pp vs baseline with IDs nearly unchanged; production MOTA +0.2pp and
  FN -86, but FP +50.
- Current best recall-first candidate: **NMS 0.60 + new_track 0.40 +
  min_tracklet_score 0.35**. No-interp IDF1 +2.2pp, MOTA +0.5pp, FN -133, FP
  +20, IDs +19 vs baseline; production IDF1 +2.3pp, MOTA +0.2pp, FN -143, FP
  +101, IDs +4.

### Private-candidate signal separability audit

The raw NMS sweep proves that suppressed boxes contain recoverable detections,
but it does not prove that a deployed private-candidate path can select the
right boxes without opening an FP birth channel. A new signal-level probe was
added before implementing CenterTrack-lite / PTDS-style continuation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/eval/detector/probe_private_candidate_separability.py \
  --data-root datasets/MOT17 \
  --sequences MOT17-02-SDP \
  --mamba-ckpt runs/reduction_candidates_full30/baseline_conv/best_recall.ckpt \
  --yolo-weights models/yolo/yolo26s.pt \
  --teacher-ckpt runs/gated_det_v1/best.ckpt \
  --baseline-nms-iou 0.50 \
  --candidate-nms-iou <0.60|0.70|0.90> \
  --score-thresholds 0.001,0.10,0.25 \
  --output report_data/private_candidate_separability_mot17_02_nmsXXX_full.json
```

Definitions:

- `private`: kept by wider candidate NMS, not kept by baseline NMS 0.50.
- `potential TP`: private candidate overlaps any GT with IoU >= 0.50.
- `recoverable missed GT`: private candidate overlaps a GT missed by baseline
  NMS 0.50 at the active score threshold.
- AUCs are signal-level rank tests on deployment-available signals. They are
  not tracking accuracy.

Full MOT17-02-SDP result, baseline checkpoint:

| candidate NMS | private boxes | private TP precision | unique recoverable GT @ score 0.25 | recoverable candidate precision | AUC score -> recoverable | P@100 by score |
|---:|---:|---:|---:|---:|---:|---:|
| 0.60 | 4,938 | 61.4% | 193 / 604 (32.0%) | 4.21% | 0.817 | 50.0% |
| 0.70 | 10,932 | 65.8% | 313 / 604 (51.8%) | 3.59% | 0.770 | 59.0% |
| 0.90 | 45,371 | 73.7% | 394 / 604 (65.2%) | 2.48% | 0.609 | 11.0% |

Readout:

- The private pool has real signal: private TP vs FP is separable by detector
  score (AUC 0.63/0.67/0.76 for candidate NMS 0.60/0.70/0.90).
- But "TP" is not the deployment target. Many private TPs are duplicates of GT
  already recalled by baseline. The rarer useful target is **recoverable missed
  GT**, and that becomes hard to rank when the candidate pool is too broad.
- `candidate_nms_iou=0.90` is an upper-bound diagnostic only. It recovers more
  missed GT in theory, but score ranking is weak for the actual useful subset
  (AUC 0.609, P@100 11%).
- `candidate_nms_iou=0.70` is the best first private-matching pool: it can cover
  51.8% of score-0.25 baseline FN and score can rank the useful subset
  reasonably (AUC 0.770, P@100 59%).
- This confirms the user's concern: previous tracking/NMS experiments measured
  end metrics but not enough signal separability. A2 must be retested as
  **private continuation only** with no new births, using a constrained 0.60/0.70
  pool and track/motion gates. Do not treat raw global NMS 0.70/0.90 as the
  deployable candidate.

ReID caveat for PTDS/CenterTrack-style ideas:

- Do **not** make ReID the primary signal. Existing ReID-module evidence already
  marks MOT17 appearance as weak for operational tracking, and the quick
  MOT17-02-only sanity check is consistent with that caution: SigLIP2 looks
  usable on short/medium gaps but rank-1 drops to 29.6% at gap 121+, while
  TransReID rank-1 is 38.4% overall and 8.9% at gap 121+.
- Therefore PTDS should not be copied as a ReID-heavy dense-similarity branch.
  The local mapping is **CenterTrack-lite without appearance as a required
  input**: previous-track/detection heatmap, motion/center offset, private
  candidate gating, and no private births.
- A ReID or dense-similarity branch can only be reintroduced after passing a
  separate same-track-vs-wrong-candidate separability gate on the private pool.

### CenterTrack-lite motion signal audit

A second signal probe tests the CenterTrack-like part directly, without ReID.
It treats baseline NMS detections matched to GT in previous frames as a proxy
active track state, then asks whether private candidates are separable by
motion/previous-box geometry.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/eval/detector/probe_motion_private_candidate_separability.py \
  --data-root datasets/MOT17 \
  --sequences MOT17-02-SDP \
  --mamba-ckpt runs/reduction_candidates_full30/baseline_conv/best_recall.ckpt \
  --yolo-weights models/yolo/yolo26s.pt \
  --teacher-ckpt runs/gated_det_v1/best.ckpt \
  --baseline-nms-iou 0.50 \
  --candidate-nms-iou <0.60|0.70> \
  --score-thresholds 0.001,0.10,0.25 \
  --output report_data/motion_private_candidate_separability_mot17_02_nmsXXX_full.json
```

Full MOT17-02-SDP result at score 0.25:

| candidate NMS | private boxes | baseline FN | active baseline FN | unique active FN covered | nearest-motion covered | AUC score | AUC best last IoU | P@100 score*predIoU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.60 | 4,938 | 604 | 203 | 122 / 203 (60.1%) | 111 / 203 (54.7%) | 0.819 | 0.897 | 43.0% |
| 0.70 | 10,932 | 604 | 203 | 161 / 203 (79.3%) | 147 / 203 (72.4%) | 0.774 | 0.843 | 49.0% |

Readout:

- Motion/previous-box geometry has real separability. For score-0.25 active
  misses, `best_last_iou` reaches AUC 0.897 at candidate NMS 0.60 and 0.843 at
  0.70, both stronger than score alone.
- Candidate NMS 0.70 is the better first prototype despite lower AUC because it
  covers more active missed GT: 161/203 vs 122/203.
- This validates the CenterTrack-lite subset: **private continuation for already
  active tracks**. It does not validate private births, ReID-heavy PTDS, or broad
  NMS 0.90.
- The remaining score-0.25 FN are split into active-but-uncovered and not-active
  cases; those require either lower-score continuation, birth confirmation, P2,
  or high-res ROI, not this motion-only path.

## 8. Candidate ranking

### Tier 1: prototype next

1. **adaptive / crowd-aware NMS**

   Start from two measured settings: conservative `NMS 0.55 + new_track 0.40`
   and recall-first `NMS 0.60 + new_track 0.40 + min_tracklet_score 0.35`. A
   raw global threshold is only the probe; the product candidate should raise
   NMS IoU only in small-object or crowded-overlap contexts, while keeping the
   default stricter threshold elsewhere.

2. **suppressed-candidate track matching**

   Keep NMS-suppressed boxes as private candidates for existing/lost tracks or
   confirmed births, instead of emitting them directly. This targets the large
   `conflict_high_score` bucket while limiting FP and ID switches. It is the
   natural guard for the NMS 0.60/0.70 detector evidence. After the signal
   audit, the first implementation should use a constrained 0.70 private pool,
   motion/previous-box gating, and forbid private candidates from starting new
   IDs.

3. **low-score temporal recovery**

   The `min_8to16` score 0.25 misses include a large `score_only_low_score`
   bucket. This suggests using low-score candidates for track continuation or
   birth confirmation, not simply lowering the global output threshold.

### Tier 2: conditional

4. **P2 small-only conflict head**

   Add P2 output only for small-object cls/reg if NMS/suppressed-candidate gates
   leave a clear 4-8 px conflict bucket. MOT17-02 diagnostics show this is not a
   broad missing-grid problem: 4-8 px zero-positive before conflict is only
   ~0.13%, but lost-all positives after conflict is ~4.2%.

5. **track-conditioned high-res second pass**

   Use tracker/lost-track/near-miss boxes to choose a small set of high-res
   crops. This maps SAHI/ASAHI/ESOD to MOT without full sliding-window cost, but
   it should target the `localization_near` and `absent_or_far` residuals rather
   than the dominant conflict bucket.

### Tier 3: fallback or closed

| option | status | reason |
|---|---|---|
| full-res scan / delete down-up | closed | ~12x head-only slower in fresh smoke, previously ~14x |
| P3 sr=2 / local-window Mamba | reserve | ~3x head-only; use only if NMS/P2/ROI evidence says the signal must be recovered inside Mamba |
| stage2-only guard for NMS 0.60 | no-go as standalone | lowers IDs slightly but loses IDF1 and raises FP on MOT17-02 |
| NGLA/NBCD direct assigner swap | no-go as standalone | removes zero-pre cases but increases conflict anchors and does not improve post-conflict positives |
| space-to-depth sr4 | no-go as standalone | full30 gate: no small-bin gain, tracking Rcll 60.90 vs 61.00 baseline |
| blur-conv sr4 | preliminary no-go | 3-epoch slice shows significant recall regression at score 0.25 |
| wavelet sr4 | no-go as standalone | 3-epoch matched slice has negative mean deltas in all monitored score/bin comparisons |
| patch 5x5 detail | closed | no detector recall gain, tracking worse |
| 64ch dense detail | closed | no operational gain, FP/tracking worse |
| wider d256 head | closed | parameter pile-up without recall movement |
| yolo26m backbone | fallback | strong recall evidence, but full backbone/teacher/relink retrain cost |

## 9. Accuracy experiment gates

Reduction-only gate already run:

```bash
ARMS="baseline_conv space_to_depth" \
OUT_ROOT=runs/reduction_candidates_full30 \
REPORT_ROOT=report_data/reduction_candidates_full30 \
  scripts/train/temporal_yolo/run_reduction_candidates.sh all
```

Preferred reproducible harness:

```bash
# historical full reduction validation:
ARMS="baseline_conv space_to_depth" \
  scripts/train/temporal_yolo/run_reduction_candidates.sh all

# cheap smoke of all reduction train arms without selection
EPOCHS=1 MAX_BATCHES=1 SEQS=MOT17-02-SDP SELECT=0 \
  scripts/train/temporal_yolo/run_reduction_candidates.sh train

# historical wavelet/DERNet/WaveCNet-style slice:
ARMS="baseline_conv wavelet" \
OUT_ROOT=runs/reduction_candidates_wavelet_slice3 \
REPORT_ROOT=report_data/reduction_candidates_wavelet_slice3 \
EPOCHS=3 WARMUP_EPOCHS=1 BEST_BY=train-loss SELECT=0 \
  scripts/train/temporal_yolo/run_reduction_candidates.sh all
```

NMS/conflict gates already run:

```bash
# detector-only NMS probe
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/eval/detector/mamba_size_binned_recall.py \
  --data-root datasets/MOT17 \
  --sequences MOT17-02-SDP \
  --yolo-weights models/yolo/yolo26s.pt \
  --teacher-ckpt runs/gated_det_v1/best.ckpt \
  --mamba-ckpt runs/reduction_candidates_full30/baseline_conv/best_recall.ckpt \
  --nms-iou 0.60 \
  --score-thresholds 0.001,0.10,0.25 \
  --output report_data/reduction_candidates_full30/baseline_conv_nms06_size_recall.json

# tracking probe with interpolation disabled
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/eval/mot17.py \
  --preset mamba_whole_graph \
  --detector SDP \
  --split train \
  --sequences MOT17-02-SDP \
  --mamba-ckpt runs/reduction_candidates_full30/baseline_conv/best_recall.ckpt \
  --mamba-teacher-ckpt runs/gated_det_v1/best.ckpt \
  --nms-iou-threshold 0.60 \
  --no-interpolate-tracklets \
  --output report_data/reduction_candidates_full30/mot17_baseline_nms06_nointerp

# recall-first guarded probe
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/eval/mot17.py \
  --preset mamba_whole_graph \
  --detector SDP \
  --split train \
  --sequences MOT17-02-SDP \
  --mamba-ckpt runs/reduction_candidates_full30/baseline_conv/best_recall.ckpt \
  --mamba-teacher-ckpt runs/gated_det_v1/best.ckpt \
  --nms-iou-threshold 0.60 \
  --new-track-thresh 0.40 \
  --min-tracklet-score 0.35 \
  --no-interpolate-tracklets \
  --output report_data/reduction_candidates_full30/mot17_baseline_nms06_new040_mintrk035_nointerp
```

Private continuation prototype:

```bash
# conservative private continuation, interpolation-on production readout
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/eval/mot17.py \
  --preset mamba_whole_graph \
  --detector SDP \
  --split train \
  --sequences MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP \
  --mamba-ckpt runs/reduction_candidates_full30/baseline_conv/best_recall.ckpt \
  --mamba-teacher-ckpt runs/gated_det_v1/best.ckpt \
  --private-continuation \
  --private-candidate-nms-iou 0.60 \
  --private-min-score 0.10 \
  --private-max-candidates 1 \
  --private-low-stage-only \
  --output report_data/reduction_candidates_full30/mot17_private060_min010_top1_lowstage_interp_7seq
```

Implementation:

- `--private-continuation` builds a wider-NMS candidate pool from the same
  post-filter pre-NMS tensor, then subtracts the baseline NMS keep set.
- Private boxes are appended after birth promotion and are score-clamped below
  `new_track_thresh`, so they can continue tracks but cannot start new IDs.
- `--private-low-stage-only` further clamps below `mid_thresh`, so private boxes
  only enter the low-score association stage.
- `--private-max-candidates 1` is currently the best simple FP/ID guard. Tracker
  prior IoU gates (`--private-prior-iou-threshold`) did not materially change
  MOT17-02, which means current Kalman-prior overlap is not selective enough by
  itself.
- `--private-selection-mode per_track` was added as a diagnostic selector: score
  private candidates against tracker priors by prior IoU, normalized center
  distance, and detector score, then greedily keep at most one private candidate
  per prior and one prior per private box.
- `--private-selection-mode suppressor_aware` was added as the first public
  ownership heuristic: keep a private/prior pair only when the private candidate
  has a baseline public suppressor, that suppressor is compatible with a tracker
  prior, and the suppressor is not compatible with the same prior as the private
  candidate.
- `--private-selection-mode sparse_symmetric` was added as a cheap sparse
  symmetric probe. It does not touch FPN features yet; it samples the pre-NMS
  detection field at a private candidate's center and 8 symmetric offsets, then
  combines center support, paired-offset strength, and paired-offset balance into
  a reranking signal.

MOT17-02-SDP no-interp, baseline NMS 0.50 reference:

| arm | IDF1 | MOTA | IDs | FP | FN | readout |
|---|---:|---:|---:|---:|---:|---|
| baseline NMS 0.50 | 54.7 | 53.9 | 126 | 121 | 8321 | reference |
| private 0.60 min0.10 | 54.5 | 54.1 | 136 | 145 | 8246 | recall gain, IDF1 not improved |
| private 0.60 min0.10 top1 | 54.5 | 54.1 | 133 | 133 | 8266 | cap helps FP/IDs but gives back recall |
| private 0.60 min0.10 top1 low-stage | 54.3 | 54.3 | 123 | 128 | 8238 | best no-interp MOTA/IDs tradeoff, IDF1 down |
| private 0.70 min0.25 top1 low-stage | 54.2 | 54.3 | 121 | 129 | 8240 | wider pool does not recover IDF1 |
| private 0.60 min0.10 per-track center2 top1 low-stage | 54.3 | 54.4 | 120 | 126 | 8230 | slightly better IDs/MOTA than global top1, still IDF1 down |
| private 0.60 min0.10 per-track center2 top3 low-stage | 54.3 | 54.4 | 140 | 135 | 8201 | more recall, but IDs rise too much |
| private 0.60 min0.10 suppressor-aware top1 low-stage | 53.5 | 54.0 | 126 | 123 | 8299 | no-go; weak recall gain but IDF1 regresses |
| private 0.60 min0.10 sparse-symmetric top1 low-stage | 54.3 | 54.3 | 135 | 128 | 8234 | no-go; FN -4 vs global top1 but IDs +12 |
| private 0.60 min0.10 sparse-symmetric center0.5 top1 low-stage | 54.3 | 54.3 | 135 | 128 | 8234 | center gate does not change the outcome |

MOT17-02-SDP interpolation-on:

| arm | IDF1 | MOTA | IDs | FP | FN | readout |
|---|---:|---:|---:|---:|---:|---|
| baseline NMS 0.50 | 57.3 | 59.3 | 81 | 239 | 7240 | reference |
| private 0.60 min0.25 top1 low-stage | 57.3 | 59.8 | 83 | 244 | 7150 | IDF1 flat, MOTA/FN better |
| private 0.60 min0.10 top1 low-stage | 57.3 | 59.8 | 83 | 244 | 7146 | best 02 production readout |
| private 0.60 min0.10 per-track top1 low-stage | 57.2 | 59.5 | 83 | 269 | 7172 | worse than global top1; FP rises |
| private 0.60 min0.10 per-track center2 top3 low-stage | 57.1 | 59.2 | 85 | 369 | 7124 | too much FP/IDs for recall gained |
| private 0.60 min0.10 sparse-symmetric top1 low-stage | 57.3 | 59.3 | 85 | 300 | 7174 | no-go; recall rises but FP/IDs give back the gain |

MOT17-SDP 7-seq interpolation-on, private 0.60/min0.10/top1/low-stage:

| seq | baseline IDF1 | private IDF1 | baseline MOTA | private MOTA | baseline IDs | private IDs | baseline FP | private FP | baseline FN | private FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MOT17-02-SDP | 57.3 | 57.3 | 59.3 | 59.8 | 81 | 83 | 239 | 244 | 7240 | 7146 |
| MOT17-04-SDP | 88.3 | 87.7 | 91.1 | 91.7 | 38 | 37 | 56 | 79 | 4155 | 3832 |
| MOT17-05-SDP | 74.3 | 74.4 | 68.8 | 68.5 | 48 | 49 | 291 | 310 | 1821 | 1822 |
| MOT17-09-SDP | 66.9 | 70.3 | 75.7 | 75.2 | 15 | 14 | 38 | 22 | 1240 | 1282 |
| MOT17-10-SDP | 61.1 | 61.6 | 72.1 | 71.5 | 133 | 133 | 1058 | 1223 | 2397 | 2308 |
| MOT17-11-SDP | 83.3 | 83.0 | 84.4 | 84.3 | 24 | 24 | 116 | 120 | 1336 | 1335 |
| MOT17-13-SDP | 70.5 | 71.2 | 68.2 | 68.2 | 128 | 119 | 1154 | 1182 | 2424 | 2402 |
| OVERALL | 76.4 | 76.5 | 78.6 | 78.8 | 467 | 459 | 2952 | 3180 | 20613 | 20127 |

Readout:

- Private continuation is a small positive production tradeoff on 7-seq:
  IDF1 +0.1, MOTA +0.2, IDs -8, FN -486, FP +228.
- It is not a strong standalone GO because gains are not uniform: 09/13 improve
  IDF1, 04/11 regress, 10 gains IDF1 but loses MOTA from FP.
- Per-track geometry-only selection is not enough. It improves MOT17-02 no-interp
  IDs slightly, but interpolation-on gets more FP than the global top1 selector.
- The first suppressor/public-ownership heuristic is also not enough: it adds too
  few useful boxes and regresses IDF1 on MOT17-02 no-interp.
- The detection-field sparse symmetric probe is not enough either. It behaves
  like a recall-biased selector: FN falls slightly, but IDs/FP rise, so it needs
  tracker cost or true feature-backed scoring before it is worth pursuing.
- The useful signal is real, but the selector likely needs tracker association
  cost or a learned motion-conditioned score, not only candidate-to-prior
  IoU/center or pre-tracker suppressor ownership. The current production
  reference remains global top1 low-stage private continuation until that
  stronger selector beats it.

Next gates:

| gate | experiment | GO criterion | decision use |
|---|---|---|---|
| A1 | cross-seq validation of NMS 0.55 + new_track 0.40 and NMS 0.60 + new_track 0.40 + min_tracklet_score 0.35 | same direction on more MOT17-SDP train sequences; no-interp FP/IDs do not explode; interp-on production run does not give back the gain | choose conservative vs recall-first default |
| A2 | suppressed-candidate track matching | improve the prototype selector beyond global top1 by comparing tracker association cost for public vs private candidates, or by a learned motion-conditioned score; keep no-private-birth invariant | only promote if IDF1 gain is positive on cross-seq, not just MOTA/FN |
| A3 | low-score temporal recovery for 8-16 px | reduces score-only misses without lowering global precision | combine with A1/A2 only for continuation/birth confirmation |
| A4 | CenterTrack-lite motion signal probe | previous-track heatmap or motion-conditioned private candidate score separates same-track continuations from wrong private boxes beyond detector score alone; ReID is not required and private boxes cannot start new IDs | only then add a trainable track-conditioned head branch |
| B1 | P2 small-only conflict head | fixes 4-8 px post-conflict misses and does not regress 8-16 px | use if A1/A2 leave crowded small-person misses |
| B2 | track-conditioned high-res second pass prototype | improves localization/absent buckets without FP explosion | use for residual near-miss/absent cases, not as the first conflict solution |
| B3 | yolo26m or yolo26m-teacher distillation | reproduces prior +recall with acceptable FP/AssA after relink retune | promote if A/B head-side paths stay weak |

After each trained checkpoint or detector candidate:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/eval/detector/mamba_size_binned_recall.py \
  --data-root datasets/MOT17 \
  --mamba-ckpt <candidate>/best.ckpt \
  --score-thresholds 0.001,0.10,0.25 \
  --output report_data/<candidate>_size_recall.json

UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/eval/bootstrap_mamba_size_recall.py \
  --baseline report_data/<baseline>_size_recall.json \
  --candidate report_data/<candidate>_size_recall.json \
  --output report_data/<candidate>_bootstrap.json
```

General GO criteria:

- MOT17-02 score 0.25 small bins positive: 4-8 px and/or 8-16 px CI excludes 0,
  or relative FN reduction is at least comparable to B1-H.
- No clear regression at score 0.001. A pure score-calibration lift is useful,
  but losing low-threshold recall means the detector may be moving boxes rather
  than recovering people.
- Full tracking run does not give back the detector gain through FP or IDs.
- For NMS/association changes, report no-interp tracking first. Interpolation-on
  is a production/default check, not the primary comparison.
- Added compute remains bounded by the intended deployment path: +10% head-only
  for reduction changes, or sparse ROI cost for second-pass changes.

NO-GO criteria:

- No detector-only small-bin positive signal after matched baseline rerun.
- Detector recall improves only by increasing FP enough to hurt MOTA/IDF1.
- Benefit is smaller than B1-H while latency/implementation complexity is worse.

## 10. Combination strategy

1. **Start from NMS/conflict, not architecture.** Object records show many
   missed GT already have overlapping evidence, and the NMS sweep recovers
   significant detector recall without retraining.
2. **Carry two concrete candidates forward.** Use `NMS 0.55 + new_track 0.40`
   when FP/ID budget is strict. Use `NMS 0.60 + new_track 0.40 +
   min_tracklet_score 0.35` when recall/IDF1 are prioritized.
3. **Turn raw NMS into adaptive NMS or private suppressed candidates.** The
   final solution should not globally emit NMS 0.70/0.90 duplicates. Those runs
   prove recoverable evidence exists. The signal audit says 0.70 is a plausible
   private pool; the motion audit says it can cover most active-track FNs. 0.90
   is too broad unless later track/motion gates sharply improve separability.
4. **Use P2 only if the conflict bucket remains.** Current assigner diagnostics
   do not show a broad missing-grid problem; they show a smaller post-conflict
   failure mode for 4-8 px objects.
5. **Use high-res ROI for localization/absent residuals.** It is a second-stage
   residual fix after the cheaper conflict/NMS levers, not the first move.
6. **Do not restart standalone reduction changes.** `blur-conv`,
   `space-to-depth`, and `wavelet` are no-go for recall as standalone head
   replacements.
7. **If head-side and association-side variants stay weak**, promote yolo26m or
   yolo26m-teacher distillation to the main path. That would mean the missing
   detail is upstream of the current head.

## 11. Artifacts

- `scripts/eval/bench_recall_candidates.py`
- `scripts/eval/detector/probe_private_candidate_separability.py`
- `scripts/eval/detector/probe_motion_private_candidate_separability.py`
- `scripts/tools/mamba_assigner_diagnostics.py`
- `scripts/train/temporal_yolo/run_reduction_candidates.sh`
- `src/saccade/perception/temporal_yolo/mamba_head.py`
- `src/saccade/perception/temporal_yolo/ngla_assigner.py`
- `scripts/train/temporal_yolo/train_mamba_gt.py`
- `report_data/mamba_assigner_diagnostics_mot17_02_full.json`
- `report_data/mamba_assigner_diagnostics_ngla_mot17_02_full.json`
- `report_data/ngla_probe_assigner_{tal,ngla}_20b.json`
- `runs/smoke_mamba_ngla_assigner/best.ckpt`
- `runs/reduction_candidates_slice3/{baseline_conv,blur_conv,space_to_depth}/best.ckpt`
- `report_data/reduction_candidates_slice3/*_size_recall.json`
- `report_data/reduction_candidates_slice3/*_vs_baseline_bootstrap.json`
- `runs/reduction_candidates_full30/{baseline_conv,space_to_depth}/best_recall.ckpt`
- `report_data/reduction_candidates_full30/*_size_recall.json`
- `report_data/reduction_candidates_full30/space_to_depth_vs_baseline_bootstrap.json`
- `runs/reduction_candidates_wavelet_slice3/{baseline_conv,wavelet}/best.ckpt`
- `report_data/reduction_candidates_wavelet_slice3/*_size_recall.json`
- `report_data/reduction_candidates_wavelet_slice3/wavelet_vs_baseline_bootstrap.json`
- `report_data/mamba_recall_candidate_wavelet_latency.json`
- `runs/smoke_mamba_reduction_wavelet/best.ckpt`
- `report_data/reduction_candidates_full30/baseline_conv_object_recall.json`
- `report_data/reduction_candidates_full30/baseline_conv_nms*_size_recall.json`
- `report_data/reduction_candidates_full30/baseline_conv_nms*_vs_nms05_bootstrap.json`
- `report_data/private_candidate_separability_mot17_02_nms060_full.json`
- `report_data/private_candidate_separability_mot17_02_nms070_full.json`
- `report_data/private_candidate_separability_mot17_02_full.json`
- `report_data/motion_private_candidate_separability_mot17_02_nms060_full.json`
- `report_data/motion_private_candidate_separability_mot17_02_nms070_full.json`
- `report_data/reduction_candidates_full30/mot17_baseline_nms*_nointerp/`
- `report_data/reduction_candidates_full30/mot17_baseline_nms0*/`
- `report_data/mamba_recall_candidate_latency_confirm.json`
- `report_data/mamba_recall_candidate_latency.json`
- `docs/reference/mamba_head_recall_bottleneck.md`
- `docs/modules/detection/research/mamba-dual-resolution-original-detail-plan.md`
- `/mnt/c/Users/Ray Lei/Downloads/小物件偵測論文分析.md` (external deep-search input)
