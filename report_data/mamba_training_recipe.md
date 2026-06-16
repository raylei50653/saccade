# Mamba v14 Replica: Paper-Ready Training Recipe

Source of truth:
`docs/modules/detection/mamba-v14-replication-protocol.md`.

## Model

| Component | Setting |
|---|---|
| Backbone | frozen YOLO26s FPN, P3/P4/P5 |
| Base single-frame head parameters | 10,126,636 |
| T3-to-T1 checkpoint parameters | 11,368,540 |
| Model width | 128 |
| SSM state size | 16 |
| Blocks | one Mamba block per FPN scale |
| Spatial reduction | 4 |
| Scan | four-direction cross-scan |
| Upsampling | learned PixelShuffle |
| Prediction | per-scale classification and box-regression heads |
| SSM gradient regime | internal scan dynamics stop-gradient |
| Deployment temporal length | 1 |

The frozen internal SSM parameters are `A_log`, `D`, depthwise `conv1d`,
`x_proj`, and `dt_proj`. Learnable paths include the Mamba gate/readout,
FPN projections, downsampling, PixelShuffle upsampling, and output heads.

## Training chain

### Stage T: MOT-adapted teacher

- initialize from `models/yolo/yolo26s.pt`;
- input 640, clip length 2, batch 4;
- gate LR `1e-3`, YOLO LR `1e-5`;
- GT gate ratio 0.5;
- replica seed `20260612`;
- use epoch 12 to match the historical teacher endpoint.

The teacher's learned weights drift only 1.689% from the raw detector, while
BatchNorm statistics drift 16.98%. Its useful prior is therefore mild
MOT-domain adaptation, not a large detector rewrite.

### Stage 0: teacher cache

Cache P3/P4/P5 and detection targets for all seven SDP sequences. Cache mode is
ungated, so the teacher contributes its adapted YOLO features rather than its
learned spatial gate.

### Stage 1: dense distillation

- 30 epochs, batch 8, LR `1e-3`;
- PixelShuffle, cross-scan, `d_state=16`;
- stop gradients through the selective scan internals.

### Stage 2: mixed GT fine-tuning

- live teacher forward;
- clip length 4, stride 8;
- 30 epochs, batch 4, LR `1e-4`;
- 5 warm-up epochs, gradient clipping 1.0;
- `gt_ratio=0.5`.

This stage provides a transition from dense teacher supervision to direct GT
optimization.

### Stage 3A: plain GT2 baseline

- cached features;
- clip length 4, stride 8;
- 30 epochs, batch 4, LR `1e-4`;
- 5 warm-up epochs;
- `gt_ratio=0`;
- stop-gradient SSM.

### Stage 3B: T3-to-T1 curriculum

Phase A:

- warm-start from the same seed's GT1 checkpoint;
- temporal block enabled;
- `T=3`, stride 6, 15 epochs;
- batch 4, LR `1e-4`, warm-up 3;
- cached features and stop-gradient SSM.

Phase B:

- warm-start from Phase A;
- temporal block bypassed;
- `T=1`, stride 2, 15 epochs;
- otherwise the same schedule.

The whole-graph evaluator bypasses temporal blocks, so deployment cost and
state are single-frame even though the checkpoint retains the temporal block
weights.

## Reconstructed artifacts

| Artifact | Role |
|---|---|
| `runs/gated_det_v14replica/epoch_0012.ckpt` | teacher |
| `runs/mamba_teacher_cache_v14replica` | cached teacher targets |
| `runs/mamba_distill_v14replica/best.ckpt` | distilled initialization |
| `runs/mamba_gt_v14replica_stage1/best.ckpt` | mixed GT stage |
| `runs/mamba_gt_v14replica_final/best.ckpt` | plain GT2 |
| `runs/mamba_gt_v14replica_t3/best.ckpt` | temporal shaping phase |
| `runs/mamba_gt_v14replica_t3_t1/best.ckpt` | single-frame readaptation |

Exact hashes and checkpoint metadata are exported to
`tables/mamba_checkpoint_provenance.csv`.

## Mechanistic interpretation

The evidence supports the following explanation:

1. freezing the backbone preserves external object-detection priors;
2. freezing SSM internals limits trainable capacity on approximately 5,300
   frames;
3. dense teacher supervision creates a stable initialization;
4. mixed and pure GT stages remove teacher bias gradually;
5. temporal training regularizes frame-to-frame box/score consistency;
6. T1 readaptation removes train/inference temporal mismatch.

This remains a hypothesis until validated on a strict held-out lineage.
