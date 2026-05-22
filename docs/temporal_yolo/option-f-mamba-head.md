# Option F: Mamba SSM Detection Head

> **Status**: Prototype（2026-05-22）
> Replaces YOLO Detect head with Mamba selective state-space (S6) blocks.
> CUDA kernel in `src/tracking/mamba_scan.cu`, pure PyTorch module in `src/saccade/perception/temporal_yolo/mamba_head.py`.

---

## Architecture

```
YOLO backbone (layers 0-22)
  ├── P3 (128ch, H/8, W/8)  → downsample strided conv → Mamba scan → upsample
  ├── P4 (256ch, H/16, W/16) → same
  └── P5 (512ch, H/32, W/32) → same
         ↓ concat with original FPN (skip)
  cls_head (per-scale Conv) + reg_head (per-scale Conv)
         ↓
  (B, 80, H_s, W_s) cls + (B, 4, H_s, W_s) reg
```

Key design choices:
- **Spatial reduction** (strided conv) before Mamba to reduce sequence length L 16×
- **Skip connection** concatenates reduced output with original FPN features
- **Pure PyTorch S6 block** with CUDA-accelerated selective scan kernel

---

## CUDA Kernel Benchmark

`selective_scan_fwd` in `src/tracking/mamba_scan.cu`. Each (B, D) pair → one block, N threads per state dimension.

| Config (B, L, D, N) | Time | GFLOPS |
|-----|------|--------|
| (1, 400, 128, 16) P5-like | 0.19 ms | 25.7 |
| (1, 1600, 128, 16) P4-like | 0.75 ms | 26.3 |
| (1, 6400, 128, 16) P3-like | 2.97 ms | 26.5 |
| (1, 6400, 256, 16) double D | 2.98 ms | 52.8 |
| (4, 6400, 128, 16) B=4 | 4.24 ms | 74.1 |
| (8, 6400, 128, 16) B=8 | 5.84 ms | 107.8 |
| (1, 400, 128, 32) N=32 | 0.19 ms | 51.6 |
| (1, 6400, 128, 32) N=32 | 2.98 ms | 52.8 |

Observations:
- Near-perfect linear scaling with L, B
- N ≤ 32 is free (single warp, no reduction overhead)
- D doubling doubles throughput (more blocks, more parallelism)

---

## Full Mamba Head Benchmark

960×960 input. YOLO backbone ~12 ms.

| Config | Mamba (ms) | Total (ms) | FPS | Params |
|--------|-------|-------|-----|--------|
| 4× reduce, 1 block | 3.0 | 15.2 | **66** | 3.0M |
| 4× reduce, 2 blocks | 4.3 | 16.5 | 61 | 3.4M |
| 4× reduce, 4 blocks | 6.5 | 18.9 | 53 | 4.1M |
| 2× reduce, 1 block | 4.9 | 17.1 | 58 | 2.5M |
| 1× (full res), 1 block | 15.7 | 28.1 | 36 | 2.3M |

---

## Comparison: Before vs After CUDA Kernel

960×960, spatial_reduce=4, 1 block:

| Implementation | Mamba (ms) | Total (ms) | FPS |
|--------|-------|-------|-----|
| Pure PyTorch (JIT scan) | 28 | 40 | 25 |
| CUDA kernel | 3 | 15 | **66** |
| Speedup | 9.3× | 2.7× | 2.7× |

---

## Distillation Training

Script: `train/temporal_yolo/train_mamba_head.py`

### Architecture

```
MOT17 frame → frozen YOLO backbone (gate disabled)
                   │
         [hooks capture P3/P4/P5 FPN]
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
teacher Detect    MambaDetectionHead (trainable)
cv2[i], cv3[i]    │
(frozen)          ▼
    │           s_cls[i], s_reg[i]  (per scale)
 t_cls[i], t_reg[i]
    │              │
    └── MSE loss ──┘
```

### Loss

Per FPN scale `i ∈ {0,1,2}`: `MSE(s_cls[i], t_cls[i]) + MSE(s_reg[i], t_reg[i])`

Teacher outputs from `detect_head.cv2[i](feat)` and `detect_head.cv3[i](feat)` directly (per-scale raw feature maps, no anchor decoding).

### Usage

```bash
uv run train/temporal_yolo/train_mamba_head.py \
    --epochs 20 --batch-size 64 --lr 1e-3 \
    --data-root datasets/MOT17 \
    --teacher-ckpt runs/gated_det_v1/best.ckpt
```

### Training Results

Batch=64, LR=1e-3, AdamW + cosine schedule, 640×640, 20 epochs.

| Epoch | Loss | Time |
|-------|------|------|
| 1 | 424.5 | 23s |
| 2 | 37.4 | 23s |
| 5 | 22.2 | 23s |
| 10 | 17.8 | 23s |
| 20 | **14.9** | 23s |

Loss is sum over 3 scales + cls+reg per scale. Per-element MSE at epoch 20 ≈ 10⁻⁷.
Student successfully replicates teacher outputs within numerical noise.

---
