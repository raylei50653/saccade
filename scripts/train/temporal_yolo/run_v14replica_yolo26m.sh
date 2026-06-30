#!/usr/bin/env bash
# Rebuild the v14 replica lineage on YOLO26m:
# teacher -> cache -> distill -> GT1 -> plain GT2 control -> T3 -> T1 -> eval.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
SEED="${SEED:-20260612}"
SEQS="${SEQS:-MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-models/yolo/yolo26m.pt}"
RUN_PREFIX="${RUN_PREFIX:-yolo26m_v14replica}"
BUILD_TRT="${BUILD_TRT:-1}"

TEACHER="runs/gated_det_${RUN_PREFIX}/epoch_0012.ckpt"
CACHE="runs/mamba_teacher_cache_${RUN_PREFIX}"
DISTILL="runs/mamba_distill_${RUN_PREFIX}"
GT1="runs/mamba_gt_${RUN_PREFIX}_stage1"
GT2="runs/mamba_gt_${RUN_PREFIX}_final"
T3="runs/mamba_gt_${RUN_PREFIX}_t3"
T3T1="runs/mamba_gt_${RUN_PREFIX}_t3_t1"
BACKBONE_ONNX="models/yolo/yolo26m_backbone_640_best.onnx"
BACKBONE_ENGINE="models/yolo/yolo26m_backbone_640_best.engine"

echo "=== Stage T: YOLO26m gated teacher (historical 30-epoch schedule) ==="
set +e
"$PYTHON" scripts/train/temporal_yolo/train_gated_detector.py \
    --data-root datasets/MOT17 \
    --yolo-weights "$YOLO_WEIGHTS" \
    --run-dir "runs/gated_det_${RUN_PREFIX}" \
    --epochs 30 --batch-size 4 --clip-len 2 \
    --img-size 640 \
    --lr-gate 1e-3 --lr-yolo 1e-5 \
    --gt-ratio 0.5 --seed "$SEED" \
    --warmup-epochs 0 \
    --save-every 1 --best-by train-loss \
    --protocol-revision V14-REPLICA-YOLO26M-20260613
teacher_status=$?
set -e
if [[ ! -f "$TEACHER" ]]; then
    echo "Teacher training stopped before epoch 12 (status=${teacher_status})." >&2
    exit 1
fi
if (( teacher_status != 0 )); then
    echo "Teacher training stopped after epoch 12; continuing with the protocol-selected e12."
fi

echo "=== Stage 0: immutable YOLO26m teacher cache ==="
YOLO_WEIGHTS="$YOLO_WEIGHTS" \
TEACHER_CKPT="$TEACHER" \
SEQS="$SEQS" \
scripts/train/temporal_yolo/build_mamba_teacher_cache.sh "$CACHE"

echo "=== Stage 1: Mamba distillation ==="
"$PYTHON" scripts/train/temporal_yolo/train_mamba_head.py \
    --data-root datasets/MOT17 \
    --yolo-weights "$YOLO_WEIGHTS" \
    --teacher-ckpt "$TEACHER" \
    --cache-dir "$CACHE" \
    --seqs "$SEQS" \
    --run-dir "$DISTILL" \
    --use-pixel-shuffle --use-cross-scan --d-state 16 \
    --scan-stop-grad \
    --epochs 30 --batch-size 8 --lr 1e-3 --seed "$SEED"

echo "=== Stage 2: GT1 live-teacher transition ==="
"$PYTHON" scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --yolo-weights "$YOLO_WEIGHTS" \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$DISTILL/best.ckpt" \
    --run-dir "$GT1" \
    --img-size 640 --clip-len 4 --clip-stride 8 \
    --epochs 30 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 5 --clip-grad 1.0 \
    --gt-ratio 0.5 --seed "$SEED" \
    --scan-stop-grad \
    --best-by train-loss --save-every 5

echo "=== Stage 3 control: plain GT2 ==="
"$PYTHON" scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --yolo-weights "$YOLO_WEIGHTS" \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$GT1/best.ckpt" \
    --cache-dir "$CACHE" \
    --run-dir "$GT2" \
    --img-size 640 --clip-len 4 --clip-stride 8 \
    --epochs 30 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 5 --clip-grad 1.0 \
    --gt-ratio 0 --seed "$SEED" \
    --scan-stop-grad \
    --best-by train-loss --save-every 5

echo "=== Stage 3A: T=3 temporal shaping ==="
"$PYTHON" scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --yolo-weights "$YOLO_WEIGHTS" \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$GT1/best.ckpt" \
    --add-temporal \
    --cache-dir "$CACHE" \
    --run-dir "$T3" \
    --img-size 640 --clip-len 3 --clip-stride 6 \
    --epochs 15 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 3 --clip-grad 1.0 \
    --gt-ratio 0 --seed "$SEED" \
    --scan-stop-grad \
    --best-by train-loss --save-every 5

echo "=== Stage 3B: T=1 spatial re-adaptation ==="
"$PYTHON" scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --yolo-weights "$YOLO_WEIGHTS" \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$T3/best.ckpt" \
    --cache-dir "$CACHE" \
    --run-dir "$T3T1" \
    --img-size 640 --clip-len 1 --clip-stride 2 \
    --epochs 15 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 3 --clip-grad 1.0 \
    --gt-ratio 0 --seed "$SEED" \
    --scan-stop-grad \
    --best-by train-loss --save-every 5

if [[ "$BUILD_TRT" == "1" ]]; then
    echo "=== Export/build matching YOLO26m teacher backbone ==="
    "$PYTHON" scripts/model/export_yolo_backbone_ckpt.py \
        --teacher-ckpt "$TEACHER" \
        --output "$BACKBONE_ONNX" \
        --imgsz 640
    "$PYTHON" scripts/model/build_yolo.py \
        --onnx "$BACKBONE_ONNX" \
        --engine "$BACKBONE_ENGINE" \
        --min-batch 1 --opt-batch 1 --max-batch 1 \
        --img-size 640 --precision fp16
fi

if [[ ! -f "$BACKBONE_ENGINE" ]]; then
    echo "Missing $BACKBONE_ENGINE; set BUILD_TRT=1 on a GPU host." >&2
    exit 1
fi

echo "=== Detector recall eval ==="
"$PYTHON" scripts/eval/mamba_size_binned_recall.py \
    --mamba-ckpt "$T3T1/best.ckpt" \
    --yolo-weights "$YOLO_WEIGHTS" \
    --teacher-ckpt "$TEACHER" \
    --trt-backbone-engine "$BACKBONE_ENGINE" \
    --sequences MOT17-02-SDP \
    --score-thresholds 0.001,0.10,0.25 \
    --output "results/mamba_size_recall_${RUN_PREFIX}_t3t1_02.json"

echo "=== MOT17 tracking eval ==="
"$PYTHON" scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP \
    --mamba-yolo-weights "$YOLO_WEIGHTS" \
    --mamba-teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$T3T1/best.ckpt" \
    --fpn-backbone-engine "$BACKBONE_ENGINE" \
    --output "results/${RUN_PREFIX}_t3t1"

echo "=== YOLO26m v14 replica complete ==="
