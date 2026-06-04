#!/bin/bash
set -e

# Option D 兩階段訓練自動化腳本
# 執行前請確保 Option C (train_joint.py) 已經訓練完成並產生 runs/joint/best.ckpt

DATA_ROOT="datasets/MOT17"
JOINT_CKPT="runs/joint/best.ckpt"
PHASE1_DIR="runs/conditioned"
PHASE2_DIR="runs/conditioned_p2"

echo "========================================================="
echo "  Option D Phase 1: 凍結 Backbone+Decoder，只訓練 Gate"
echo "========================================================="
# Phase 1: 從 Option C 的權重熱啟動，只訓練 Gate (GT oracle boxes)
uv run train/temporal_yolo/train_conditioned.py \
    --data-root "$DATA_ROOT" \
    --resume "$JOINT_CKPT" \
    --phase 1 \
    --epochs 10 \
    --run-dir "$PHASE1_DIR"

echo "========================================================="
echo "  Option D Phase 2: 全部解凍聯合訓練，差分 LR 退火"
echo "========================================================="
# Phase 2: 從 Phase 1 的權重接續訓練，全部解凍 (predicted-box curriculum)
uv run train/temporal_yolo/train_conditioned.py \
    --data-root "$DATA_ROOT" \
    --resume "$PHASE1_DIR/best.ckpt" \
    --phase 2 \
    --epochs 50 \
    --run-dir "$PHASE2_DIR"

echo "========================================================="
echo "  Option D 訓練全部完成！最終權重位於 $PHASE2_DIR/best.ckpt"
echo "========================================================="
