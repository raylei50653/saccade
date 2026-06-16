# Mamba v14 Training Audit

Checkpoint under audit:

```text
runs/mamba_gt_vgt_mamba_v14/best.ckpt
```

## Actual training lineage

The production path was longer than the documented distill -> GT fine-tune flow:

1. `mamba_distill_pixelshuffle_crossscan/best.ckpt`
2. `mamba_gt_pixelshuffle_crossscan/best.ckpt`
   - 30 GT fine-tuning epochs
   - best training loss `3.5667908013`
3. `mamba_gt_vgt_mamba_v14`, epochs 1-30
   - warm-start from the previous GT-fine-tuned checkpoint
   - batch 4, 167 optimizer steps/epoch
   - requested LR `1e-4`, 5 warm-up epochs
   - epoch-30 best loss `2.8703332319`
4. `mamba_gt_vgt_mamba_v14`, epochs 31-60
   - resumed from the same run's `best.ckpt`
   - requested LR `3e-4`, no warm-up
   - current `best.ckpt` is epoch 58, loss `2.8057419440`

## Resume LR did not take effect

The resume path restores the complete optimizer state. The requested `--lr 3e-4`
is only applied if optimizer loading fails. Optimizer loading succeeded.

The actual second-stage LR was:

```text
epoch 30: 1.000e-6
epoch 31: 1.005e-6
epoch 40: 1.500e-6
epoch 50: 2.500e-6
epoch 58: 2.978e-6
epoch 60: 3.000e-6
```

The new cosine scheduler used `eta_min=3e-6`, above the restored `1e-6` LR, so
the LR slowly increased toward `3e-6`. Epochs 31-60 were ultra-low-LR polishing,
not a `3e-4` retraining stage.

Model movement from epoch 30 to 58 was small:

| Module | Relative L2 weight change |
|---|---:|
| input projection | 0.16% |
| Mamba blocks | 0.24% |
| cls head | 0.42% |
| reg head | 0.40% |
| downsample | 0.57% |
| PixelShuffle upsampler | 0.69% |
| whole model | 0.54% |

## Cached features disabled gate feedback

v14 used:

```text
cache_dir=runs/trt_feat_cache_v2
gt_ratio=0.5
```

In cache mode, the training loop loaded P3/P4/P5 tensors directly and skipped
the teacher forward pass. `gate_inputs` were constructed but never applied to
cached features. The effective v14 training configuration was therefore:

```text
ungated cached FPN features
gt_ratio has no effect
```

This contradicts the documented claim that v14 GT fine-tuning used 50% GT gate
feedback. Later B0-R/B1 experiments did not use the feature cache, so their
training also changed the global feature distribution and gate-feedback path.

## Data coverage and reproducibility

Checkpoint optimizer steps show 167 batches/epoch at batch 4, equivalent to
668 clips. With clip length 4 and effective stride 8, the run used:

| Coverage | Used | Available | Ratio |
|---|---:|---:|---:|
| Frames | 2,664 | 5,316 | 50.1% |
| Valid pedestrian GT rows | 47,993 | 95,872 | 50.1% |

The same alternating 4-frame blocks were used every epoch; shuffling only
changed clip order. No Python or Torch random seed was recorded, so data order
and gate sampling in non-cache runs are not reproducible.

Training and evaluation both use MOT17 train sequences. Unused interleaved
frames share scenes and tracks with training frames and are not an independent
validation set.

## Checkpoint selection mismatch

`best.ckpt` is selected only by training loss. There is no detector or tracking
validation loop.

MOT17-02:

| Metric | Epoch 30 | Epoch 58 | Epoch 58 - 30 |
|---|---:|---:|---:|
| IDF1 | 52.5 | 52.1 | -0.4 |
| MOTA | 57.2 | 57.1 | -0.1 |
| HOTA | 43.7 | 43.7 | 0.0 |
| Recall | 59.8 | 59.3 | -0.5 |
| FP | 384 | 319 | -65 |
| FN | 7,464 | 7,559 | +95 |
| IDs | 105 | 92 | -13 |

MOT17-05:

| Metric | Epoch 30 | Epoch 58 | Epoch 58 - 30 |
|---|---:|---:|---:|
| IDF1 | 69.3 | 68.7 | -0.6 |
| MOTA | 65.3 | 65.4 | +0.1 |
| HOTA | 58.6 | 58.2 | -0.4 |
| DetA | 60.1 | 60.0 | -0.1 |
| AssA | 57.5 | 56.8 | -0.7 |
| Recall | 71.4 | 71.8 | +0.4 |
| FP | 351 | 375 | +24 |
| FN | 1,978 | 1,949 | -29 |

Epoch-58 detector-only gains over epoch 30 were small. Most paired bootstrap
confidence intervals included zero. The second stage changed calibration and
precision/recall trade-offs, but did not establish a better tracking checkpoint.

The documentation reports v14 loss `2.87`, which corresponds to epoch 30, while
the current `best.ckpt` is epoch 58 with loss `2.8057`. Metrics documented for
the `best.ckpt` path may therefore refer to a different artifact than the file
currently at that path.

## Conclusion

The current v14 checkpoint is usable but is not a cleanly controlled baseline.
Its training process contains:

- ignored resume LR;
- disabled gate feedback in cache mode;
- fixed 50% frame/GT coverage;
- no recorded seed;
- no independent validation;
- training-loss-only checkpoint selection;
- likely documentation/checkpoint artifact drift.

Before treating dense-detail capacity experiments as an architecture ceiling,
train a controlled v14-R baseline with explicit seed, full or randomized frame
coverage, correct resume/LR semantics, and checkpoint selection on a designated
selection sequence or temporal block. Then retrain the best 32-channel detail
branch from that baseline with the same split and schedule.

## Remediation status

The controlled v14-R re-finetuning path was prepared on 2026-06-12:

- explicit Python/PyTorch/CUDA/DataLoader seed;
- configurable clip stride, with stride 4 covering 5308/5316 MOT17 frames;
- cache mode rejects nonzero GT gate ratio;
- exact resume restores optimizer and scheduler state;
- changed-LR/legacy resume requires an explicit optimizer reset;
- one combined linear-warmup + cosine schedule;
- held-out sequences are removed from the training dataset;
- external recall/HOTA candidate selection replaces training-loss-only model
  selection for the clean recipe.

The Stage-1 clean architecture checkpoint is
`runs/mamba_distill_cs_n16/best.ckpt`.

This path is not a strict held-out experiment because `gated_det_v1`, its
feature cache, and `mamba_distill_cs_n16` were all trained using every MOT17-SDP
sequence. A strict clean v14-R requires retraining the complete teacher/cache/
distillation chain with the selection sequence excluded.
