# Mamba v14-R Training Protocol

> Canonical training contract, frozen on 2026-06-12.
>
> Protocol revision: `MAMBA-V14R-20260612-R1`.
>
> This document defines what may be called `controlled v14-R` and what is
> required before an experiment may be called `strict-clean v14-R`.

## 1. Current status

Two protocols are distinguished:

| Protocol | Status | Meaning |
|---|---|---|
| Controlled v14-R | Ready | Restart Mamba GT fine-tuning from the existing Stage-1 checkpoint with fixed seed, coverage, LR, and selection rules |
| Strict-clean v14-R | Not ready | Retrain teacher, cache, distillation, and GT stages with one shared held-out split |

Do not report the controlled protocol as an independently held-out experiment.
The current teacher and distillation lineage has already seen every MOT17-SDP
sequence.

## 2. Fixed experiment contract

Unless a new protocol revision is written, use:

| Field | Value |
|---|---|
| Dataset | `datasets/MOT17/train` |
| Detector variant | SDP |
| Selection sequence | `MOT17-02-SDP` |
| Training sequences | `MOT17-04/05/09/10/11/13-SDP` |
| Seed | `20260612` |
| Global input | stretch-resized `640x640` |
| Mamba architecture | Cross-Scan + PixelShuffle + `d_state=16` |
| Clip length / stride | `4 / 4` |
| Batch / accumulation | `4 / 1` |
| LR | `1e-4` |
| Warmup | 5 epochs |
| Gradient clipping | `1.0` |
| Cache gate semantics | ungated, therefore `gt_ratio=0` |
| Candidate interval | every 5 epochs |
| Candidate selection | detector size recall first, then HOTA/IDF1/FP/FN |
| Forbidden selector | training loss alone |

`clip_len=4`, `clip_stride=4` covers 5308/5316 MOT17-SDP frames
(99.85%).

## 3. Current artifact lineage

| Stage | Artifact | SHA-256 | Known limitation |
|---|---|---|---|
| Base detector | `models/yolo/yolo26s.pt` | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` | External pretrained base |
| Gated teacher | `runs/gated_det_v1/best.ckpt` | `d2ace71d47c3358d36867c64795b09fad7e3ae0c794ef59554b17518c387dbcf` | All sequences; stopped at epoch 12; no seed/validation |
| Feature cache | `runs/trt_feat_cache_v2` | No manifest | All sequences; ungated FPN; no recorded teacher hash |
| Stage-1 Mamba | `runs/mamba_distill_cs_n16/best.ckpt` | `c6257b70774d83035b70fed1d53a4992ed3b94740281bb5bb89b6bffb6e5583b` | Distilled from all sequences |

The cache numerically matches the current teacher: sampled P3/P4/P5 features
have cosine similarity above `0.99999`. This is evidence of consistency, not a
replacement for a provenance manifest.

The cache was generated with `gate_input=None`. Gate alphas from
`gated_det_v1` therefore do not affect cached features. The effective teacher
contribution is its MOT17-finetuned YOLO backbone and detect head.

## 4. Controlled v14-R

### 4.1 Train candidates

```bash
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt runs/gated_det_v1/best.ckpt \
    --mamba-ckpt runs/mamba_distill_cs_n16/best.ckpt \
    --cache-dir runs/trt_feat_cache_v2 \
    --run-dir runs/mamba_gt_v14r_holdout02 \
    --holdout-seqs MOT17-02-SDP \
    --img-size 640 --clip-len 4 --clip-stride 4 \
    --epochs 30 --batch-size 4 --accum-steps 1 \
    --lr 1e-4 --warmup-epochs 5 --clip-grad 1.0 \
    --gt-ratio 0 --seed 20260612 \
    --best-by none --save-every 5
```

Expected candidates:

```text
epoch_0005.ckpt
epoch_0010.ckpt
epoch_0015.ckpt
epoch_0020.ckpt
epoch_0025.ckpt
epoch_0030.ckpt
```

No `best.ckpt` is expected because training loss is not the deployment model
selector.

### 4.2 Detector selection

Run every candidate on `MOT17-02-SDP`:

```bash
.venv/bin/python scripts/eval/mamba_size_binned_recall.py \
    --mamba-ckpt runs/mamba_gt_v14r_holdout02/epoch_0020.ckpt \
    --sequences MOT17-02-SDP \
    --score-thresholds 0.001,0.10,0.25 \
    --save-frame-records \
    --output report_data/mamba_size_recall_v14r_holdout02_e20.json
```

Rank candidates using:

1. operational score `0.25` recall for resized 4-8 px and 8-16 px GT;
2. total recall and FN;
3. low-score behavior at `0.10` as diagnostic evidence;
4. reject candidates whose gain comes mainly from uncontrolled FP growth.

### 4.3 Tracking selection

Run the detector shortlist through the production tracking path:

```bash
.venv/bin/python scripts/eval/mot17.py \
    --preset mamba_whole_graph --detector SDP \
    --sequences MOT17-02-SDP \
    --mamba-ckpt runs/mamba_gt_v14r_holdout02/epoch_0020.ckpt \
    --output results/mamba_gt_v14r_holdout02_e20
```

Select the epoch using HOTA and IDF1, with FP/FN, DetA, AssA, recall, and latency
reported as supporting metrics. Record the selected epoch and the exact report
paths before final retraining.

### 4.4 Final all-sequence retrain

Replace `<SELECTED_EPOCH>` with the selected training length:

```bash
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt runs/gated_det_v1/best.ckpt \
    --mamba-ckpt runs/mamba_distill_cs_n16/best.ckpt \
    --cache-dir runs/trt_feat_cache_v2 \
    --run-dir runs/mamba_gt_v14r_final \
    --img-size 640 --clip-len 4 --clip-stride 4 \
    --epochs <SELECTED_EPOCH> --batch-size 4 --accum-steps 1 \
    --lr 1e-4 --warmup-epochs 5 --clip-grad 1.0 \
    --gt-ratio 0 --seed 20260612 \
    --best-by none --save-every 5
```

The final deployment candidate is
`runs/mamba_gt_v14r_final/epoch_<SELECTED_EPOCH>.ckpt`, not a training-loss
`best.ckpt`.

## 5. Resume contract

- `--resume` means exact continuation and requires model, optimizer, and
  scheduler state.
- Changing LR or continuing a legacy checkpoint requires
  `--resume-reset-optimizer`.
- Do not use resume for the controlled comparison unless recovering an
  interrupted run with identical arguments.
- A resumed run must retain the same dataset split, seed, batch size,
  accumulation, clip length, and clip stride.

## 6. Strict-clean v14-R

Strict-clean requires the same six training sequences at every learned stage:

```text
MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,
MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP
```

Required dependency chain:

1. Train a new gated teacher using only the six training sequences.
2. Generate a new cache using that exact teacher and only those sequences.
3. Train a new Cross-Scan + PixelShuffle + N=16 distillation checkpoint from
   that cache.
4. Run Mamba GT fine-tuning with the same split and seed.
5. Use `MOT17-02-SDP` only for candidate selection.

### 6.1 Gated teacher candidates

The gated training entry point now enforces seed, split, scheduler, exact
resume, candidate selection state, RNG restoration, and base-YOLO provenance:

```bash
.venv/bin/python scripts/train/temporal_yolo/train_gated_detector.py \
    --data-root datasets/MOT17 \
    --yolo-weights models/yolo/yolo26s.pt \
    --run-dir runs/gated_det_v14r_frozen_holdout02 \
    --holdout-seqs MOT17-02-SDP \
    --epochs 30 --batch-size 4 \
    --clip-len 2 --clip-stride 2 \
    --lr-gate 1e-3 --lr-yolo 0 \
    --warmup-epochs 5 --clip-grad 10 \
    --gt-ratio 0.5 --seed 20260612 \
    --best-by none --save-every 1 \
    --protocol-revision MAMBA-V14R-20260612-R1
```

`--lr-yolo 0` is required for this stage. It freezes both YOLO parameters and
YOLO BatchNorm running statistics; only the spatial gate is trained. The
earlier `runs/gated_det_v14r_holdout02` run used `--lr-yolo 1e-5`, updated
YOLO BatchNorm statistics, and is not a valid frozen-teacher candidate.

Select gated teacher candidates on `MOT17-02-SDP` using the ungated inference
path, because cache generation uses `gate_input=None`:

```bash
.venv/bin/python scripts/eval/eval_gated_bytetrack.py \
    --ckpt runs/gated_det_v14r_frozen_holdout02/epoch_0001.ckpt \
    --sequences MOT17-02-SDP \
    --no-gate \
    --output results/gated_det_v14r_frozen_holdout02_e01_nogate
```

Use HOTA/IDF1/MOTA/recall/FP/FN to select the teacher epoch. Gate-on evaluation
may be reported as a diagnostic, but it must not replace gate-off selection for
an ungated cache lineage.

The selected six-sequence checkpoint becomes the strict-clean teacher. Do not
retrain it on all seven sequences before cache generation.

### 6.2 Remaining blocker

Strict-clean is still blocked because `train_mamba_head.py` does not yet
enforce the full reproducibility and provenance contract:

- deterministic seed;
- safe optimizer/scheduler resume semantics;
- non-training-loss checkpoint selection;
- artifact provenance manifest.

The gated teacher stage is prepared. Passing `--seqs` manually to the
distillation/cache stage remains necessary but is not sufficient to claim
strict-clean.

The strict-clean cache must be regenerated with the current cache schema. Each
frame stores frozen `P3/P4/P5` plus Detect-head `cls/reg` targets. Distillation
with `--cache-dir` therefore runs only the Mamba student per epoch and rejects
legacy feature-only caches.

Build the strict six-sequence cache with:

```bash
scripts/train/temporal_yolo/build_mamba_teacher_cache.sh \
    runs/mamba_teacher_cache_v14r_holdout02
```

The default uses the original base YOLO directly because gate input is disabled
and YOLO weights/BatchNorm are frozen. Set `TEACHER_CKPT=/path/to/checkpoint`
only when the checkpoint intentionally supplies a different frozen YOLO
lineage. The generated `manifest.json` is mandatory for distillation.

The matching distillation command must use the same empty teacher checkpoint
and exact sequence list:

```bash
.venv/bin/python scripts/train/temporal_yolo/train_mamba_head.py \
    --data-root datasets/MOT17 \
    --yolo-weights models/yolo/yolo26s.pt \
    --teacher-ckpt "" \
    --cache-dir runs/mamba_teacher_cache_v14r_holdout02 \
    --seqs MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP \
    --run-dir runs/mamba_distill_v14r_holdout02 \
    --use-pixel-shuffle --use-cross-scan --d-state 16 \
    --clip-len 1 --clip-stride 1 --seed 20260612 \
    --epochs 20 --batch-size 8 --lr 1e-3
```

## 7. Required provenance

Every new run must record:

```text
protocol_revision
git_commit
git_diff_status
command
seed
training_sequences
selection_sequences
input_resolution
clip_len
clip_stride
batch_size
accum_steps
learning_rate
warmup_epochs
parent_checkpoint_path
parent_checkpoint_sha256
teacher_checkpoint_path
teacher_checkpoint_sha256
cache_path
cache_manifest_sha256
selected_epoch
selection_report_paths
```

The cache builder writes the teacher hash, YOLO hash, sequence list, frame
counts, image size, resize policy, gate mode, dtype, and completion status to
`manifest.json`.

## 8. Reporting terminology

Use these exact labels:

- `legacy v14`: current `runs/mamba_gt_vgt_mamba_v14/best.ckpt`.
- `controlled v14-R`: protocol in Section 4 using the existing lineage.
- `strict-clean v14-R`: complete split-consistent lineage in Section 6.
- `v14-R+B1-H`: detail model retrained from the corresponding v14-R lineage
  with the same seed, split, schedule, and selected epoch.

Do not shorten `controlled v14-R` to `clean v14-R`.

## 9. Related evidence

- [Mamba training history and commands](mamba-head-training.md)
- [gated_det_v1 audit](../../../report_data/gated_det_v1_training_audit.md)
- [legacy v14 audit](../../../report_data/mamba_v14_training_audit.md)
- [Dual-resolution detail plan](research/mamba-dual-resolution-original-detail-plan.md)
