#!/usr/bin/env bash
# status: experiment
# Controlled v14 causal probe:
# historical Cross-Scan parent -> fixed N=16 -> frozen-SSM GT fine-tune.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

SEQS=MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP
YOLO=models/yolo/yolo26s.pt
TEACHER=runs/gated_det_v1/best.ckpt
CACHE=runs/trt_feat_cache_v2
PARENT=runs/mamba_gt_pixelshuffle_crossscan/best.ckpt
RUN=runs/mamba_gt_v14_parent_n16_frozen_refit

.venv/bin/python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to run the 30-epoch job on CPU.")
print("CUDA:", torch.cuda.get_device_name(0))
PY

.venv/bin/python scripts/tools/migrate_legacy_mamba_cache_manifest.py \
    --cache-dir "$CACHE" \
    --yolo-weights "$YOLO" \
    --teacher-ckpt "$TEACHER" \
    --img-size 640 \
    --sequences "$SEQS"

resume_args=()
if [[ -s "$RUN/latest.ckpt" ]]; then
    resume_args=(--resume "$RUN/latest.ckpt")
    echo "=== RESUME: $RUN/latest.ckpt ==="
else
    echo "=== START: parent -> fixed N=16 frozen-SSM GT refit ==="
fi

.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --yolo-weights "$YOLO" \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$PARENT" \
    --cache-dir "$CACHE" \
    --run-dir "$RUN" \
    --seqs "$SEQS" \
    --img-size 640 --clip-len 4 --clip-stride 4 \
    --epochs 30 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 5 --clip-grad 1.0 \
    --gt-ratio 0 --seed 42 \
    --scan-stop-grad --no-legacy-n1-scan \
    --best-by train-loss --save-every 5 \
    "${resume_args[@]}"

echo "=== RECALL EVAL ==="
.venv/bin/python scripts/eval/mamba_size_binned_recall.py \
    --mamba-ckpt "$RUN/best.ckpt" \
    --sequences MOT17-02-SDP \
    --score-thresholds 0.001,0.10,0.25 \
    --output results/mamba_v14_parent_n16_frozen_refit_recall_02.json

echo "=== TRACKING EVAL ==="
.venv/bin/python scripts/eval/mot17.py \
    --preset mamba_whole_graph --detector SDP \
    --mamba-ckpt "$RUN/best.ckpt" \
    --output results/mamba_v14_parent_n16_frozen_refit

echo "V14 PARENT N16 FROZEN REFIT DONE"
