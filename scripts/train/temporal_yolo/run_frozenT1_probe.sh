#!/usr/bin/env bash
# status: experiment
# §4.1 attribution discriminator: frozen-SSM + T=1-only continued training from the
# shaped t3_t1 checkpoint. The missing 2×2 cell that separates the AssA-collapse cause.
#
#                 T1-only ×30        T3→T1
#   解凍 sg=N     62.8 (T3T1→ssmft)  65.4 (cw0)
#   凍結 sg=Y     THIS               ~66.0 (t3_t1 itself)
#
# ≈66 (no drift)  -> collapse is caused by SSM UNFREEZE (C1 necessary).
# ~63 (drifts)    -> collapse is generic T=1 continued-training forgetting (C3); SSM
#                    unfreeze exonerated, §4.1 attribution wrong.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
CU13="$ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$CU13:${LD_LIBRARY_PATH:-}"

TEACHER=runs/gated_det_v14replica/epoch_0012.ckpt
CACHE=runs/mamba_teacher_cache_v14replica
START=runs/mamba_gt_v14replica_t3_t1/best.ckpt   # the shaped 75.4/66.0 checkpoint

echo "=== frozen-SSM T=1 continued 30ep (from t3_t1, seed 42) ==="
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$START" \
    --cache-dir "$CACHE" \
    --run-dir runs/mamba_gt_v14replica_frozenT1_30ep \
    --img-size 640 --clip-len 1 --clip-stride 2 \
    --epochs 30 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 3 --clip-grad 1.0 \
    --gt-ratio 0 --seed 42 \
    --scan-stop-grad \
    --best-by train-loss --save-every 5

echo ""
echo "=== TRACKING EVAL (whole-graph) ==="
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP \
    --mamba-ckpt runs/mamba_gt_v14replica_frozenT1_30ep/best.ckpt \
    --output results/mamba_v14replica_frozenT1

echo ""
echo "=== DONE — compare AssA vs 66.0 (t3_t1) / 62.8 (T3T1→ssmft) / 65.4 (cw0) ==="
