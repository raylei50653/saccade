#!/bin/bash
set -e

LOG_FILE="results/sweep_gmc.log"
mkdir -p results
echo "=== GMC Sweep started at $(date) ===" > "$LOG_FILE"

# Config 1: fg_mask=true, downscale=8
echo "==========================================" | tee -a "$LOG_FILE"
echo "=== Running Config 1: fg_mask=true, downscale=8 ===" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
uv run scripts/eval/mot17.py \
  --preset mamba_optimal \
  --detector SDP \
  --gmc-fg-mask \
  --gmc-downscale 8 2>&1 | tee -a "$LOG_FILE"

# Config 2: fg_mask=false, downscale=4
echo "==========================================" | tee -a "$LOG_FILE"
echo "=== Running Config 2: fg_mask=false, downscale=4 ===" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
uv run scripts/eval/mot17.py \
  --preset mamba_optimal \
  --detector SDP \
  --no-gmc-fg-mask \
  --gmc-downscale 4 2>&1 | tee -a "$LOG_FILE"

# Config 3: fg_mask=true, downscale=4
echo "==========================================" | tee -a "$LOG_FILE"
echo "=== Running Config 3: fg_mask=true, downscale=4 ===" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
uv run scripts/eval/mot17.py \
  --preset mamba_optimal \
  --detector SDP \
  --gmc-fg-mask \
  --gmc-downscale 4 2>&1 | tee -a "$LOG_FILE"

echo "=== GMC Sweep completed at $(date) ===" >> "$LOG_FILE"
