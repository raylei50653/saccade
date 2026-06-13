# gated_det_v1 training audit

Date: 2026-06-12

Artifact:

```text
runs/gated_det_v1/best.ckpt
sha256 d2ace71d47c3358d36867c64795b09fad7e3ae0c794ef59554b17518c387dbcf
```

## Checkpoint state

- Requested training length: 30 epochs.
- Available checkpoints: epochs 1 through 12 only.
- `best.ckpt` and `latest.ckpt` are identical epoch-12 model states.
- Epoch-12 training loss: 4.660035.
- Loss decreased monotonically over all 12 saved epochs, so training-loss
  convergence and the intended 30-epoch endpoint were not established.
- The checkpoint contains optimizer state but no scheduler state.

## Training behavior

- All seven MOT17-SDP training sequences were used.
- `clip_len=2` and DataLoader stride 2 cover 5314/5316 frames (99.96%).
- No seed was recorded or applied to Python, PyTorch, CUDA, or DataLoader
  shuffle.
- Checkpoint selection used training loss only; there was no independent
  validation sequence.
- Resume restores model and optimizer but not scheduler or `best_loss`.
- The entire YOLO model was trainable at `lr_yolo=1e-5`; gate alphas used
  `lr_gate=1e-3`.
- YOLO parameters differ from the original `yolo26s.pt` by approximately
  7.18% relative L2. This artifact is a MOT17-finetuned YOLO teacher, not only
  a gate wrapper.
- Final gate alphas are small: P3 0.00230, P4 0.01182, P5 0.00231.
- With `clip_len=2`, frame 0 is always ungated and frame 1 is gated with
  probability 0.5. Oracle gate exposure is therefore approximately 25% of all
  frame forwards, or 50% of eligible transitions.

## Cache consistency

`runs/trt_feat_cache_v2` was generated after `gated_det_v1`. Recomputing three
sample frames from the current checkpoint on CPU and comparing them with the
FP16 cache produced cosine similarity above 0.99999 and relative RMS error
between 0.18% and 0.43%. The small difference is consistent with CPU/GPU and
FP16 execution differences.

The cache has no provenance manifest or teacher hash, so this relationship is
numerically supported but not cryptographically recorded.

Cache extraction uses `gate_input=None`. Consequently:

- cached P3/P4/P5 are ungated;
- the learned gate alphas do not affect Mamba distillation or cache-mode GT
  fine-tuning;
- the material contribution from `gated_det_v1` is its MOT17-finetuned YOLO
  backbone and detect head.

## Implication for v14-R

`gated_det_v1`, `trt_feat_cache_v2`, and `mamba_distill_cs_n16` all used every
MOT17-SDP sequence. Excluding MOT17-02 only during Mamba GT fine-tuning is a
controlled re-finetune, not a strict held-out experiment.

A strict clean lineage must use the same sequence split for:

1. gated detector training;
2. feature-cache generation;
3. Mamba distillation;
4. Mamba GT fine-tuning.

The selection sequence must only be used for detector recall and tracking
checkpoint selection.

## Remediation status

The replacement gated-detector training path now provides:

- deterministic Python/PyTorch/CUDA/DataLoader seed;
- explicit clip stride and held-out sequence removal;
- one warmup + cosine schedule for all parameter groups while preserving their
  LR ratio;
- exact resume with optimizer, scheduler, AMP scaler, RNG, and DataLoader
  generator state;
- explicit reset semantics for legacy checkpoints or changed schedules;
- candidate checkpoint mode without training-loss `best.ckpt`;
- base YOLO SHA-256, command, git commit/status, split, and protocol revision
  in checkpoint provenance;
- non-finite loss failure and gradient accumulation.
- `--img-size` is now propagated into `GatedDetConfig`, avoiding a hidden
  640-pixel gate geometry when training another resolution.

The legacy `gated_det_v1` artifact is unchanged. New strict-clean teacher runs
must use the canonical command in
`docs/modules/detection/mamba-v14r-training-protocol.md`.
