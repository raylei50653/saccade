#!/bin/bash
# status: experiment
# Sweep low match_thresh values and collect overall metrics.
set -e

LOG_FILE="results/sweep_low_mt.log"
mkdir -p results
echo "=== Sweep started at $(date) ===" > "$LOG_FILE"

for gmc_flag in "" "--gmc"; do
  for mt in 0.40 0.45 0.50; do
    gmc_label="no GMC"
    if [ -n "$gmc_flag" ]; then
      gmc_label="with GMC"
    fi
    
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "=== Running match_thresh=$mt ($gmc_label) ===" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    
    # Run eval and filter out the final overall metrics
    uv run scripts/eval/mot17.py \
      --config configs/presets/fpn_reid_baseline.yaml \
      --detector SDP \
      --mamba-ckpt runs/mamba_gt_960_v2/best.ckpt \
      --reid-mode off \
      --fpn-backbone-engine models/yolo/yolo26s_backbone_640_best.engine \
      $gmc_flag \
      --match-thresh $mt 2>&1 | tee -a "$LOG_FILE"
  done
done

echo "=== Sweep completed at $(date) ===" >> "$LOG_FILE"
