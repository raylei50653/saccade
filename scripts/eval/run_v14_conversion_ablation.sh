#!/usr/bin/env bash
# status: archive-candidate
# Zero-training v14 conversion ablation.
# A/B uses identical parent weights; only the selective-scan runtime changes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PARENT=runs/mamba_gt_pixelshuffle_crossscan/best.ckpt
FINAL=runs/mamba_gt_vgt_mamba_v14/best.ckpt
VARIANT_DIR=runs/mamba_runtime_ablation
LEGACY_N1="$VARIANT_DIR/pixelshuffle_crossscan_legacy_n1.ckpt"
FIXED_N16="$VARIANT_DIR/pixelshuffle_crossscan_fixed_n16.ckpt"

.venv/bin/python scripts/tools/set_mamba_checkpoint_runtime.py \
    --input "$PARENT" \
    --output "$LEGACY_N1" \
    --scan-runtime legacy-n1
.venv/bin/python scripts/tools/set_mamba_checkpoint_runtime.py \
    --input "$PARENT" \
    --output "$FIXED_N16" \
    --scan-runtime fixed-n16

run_eval() {
    local tag="$1"
    local checkpoint="$2"
    local recall_output="results/mamba_v14_conversion_${tag}_recall_02.json"
    local tracking_output="results/mamba_v14_conversion_${tag}"

    if [[ -s "$recall_output" ]]; then
        echo "=== RECALL: $tag (skip existing $recall_output) ==="
    else
        echo "=== RECALL: $tag ==="
        .venv/bin/python scripts/eval/mamba_size_binned_recall.py \
            --mamba-ckpt "$checkpoint" \
            --sequences MOT17-02-SDP \
            --score-thresholds 0.001,0.10,0.25 \
            --output "$recall_output"
    fi

    if [[ -d "$tracking_output" ]] \
        && [[ "$(find "$tracking_output" -maxdepth 1 -name 'MOT17-*-SDP.txt' | wc -l)" -eq 7 ]]; then
        echo "=== TRACKING: $tag (skip existing 7-sequence result) ==="
    else
        echo "=== TRACKING: $tag ==="
        .venv/bin/python scripts/eval/mot17.py \
            --preset mamba_whole_graph --detector SDP \
            --mamba-ckpt "$checkpoint" \
            --output "$tracking_output"
    fi
}

# 1 -> 2 measures the direct 77fcc262 runtime conversion with identical weights.
run_eval parent_legacy_n1 "$LEGACY_N1"
run_eval parent_fixed_n16 "$FIXED_N16"

# 2 -> 3 measures the small residual contribution from the later v14 GT run.
run_eval final_v14_n16 "$FINAL"

echo "V14 CONVERSION ABLATION DONE"
