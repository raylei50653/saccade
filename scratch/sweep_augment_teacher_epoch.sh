#!/usr/bin/env bash
# Scan augment teacher epochs on MOT17 held-out.
# Fixed mamba head = runs/mamba_gt_pp22_augment_t3_t1_e30/best.ckpt
# Varies teacher backbone epoch to find recall-best.
set -u
cd /home/ray/developer/ai/saccade || exit 1

TORCHLIB=$(.venv/bin/python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
NVLIBS=$(.venv/bin/python -c "import glob,os,nvidia; base=os.path.dirname(nvidia.__file__); print(':'.join(sorted(set(os.path.dirname(p) for p in glob.glob(base+'/*/lib/*.so*')))))")
export LD_LIBRARY_PATH="$TORCHLIB:$NVLIBS:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build:${PYTHONPATH:-}"

MAMBA_CKPT=runs/mamba_gt_pp22_augment_t3_t1_e30/best.ckpt
SUM=scratch/augment_epoch_sweep.txt
: > "$SUM"
echo "# augment teacher epoch sweep  $(date)" >> "$SUM"
echo "# fixed mamba: $MAMBA_CKPT" >> "$SUM"
echo "" >> "$SUM"

run_epoch() {
    local ep="$1"
    local tag
    tag=$(printf "epoch_%04d" "$ep")
    local teacher="runs/gated_det_pp22_augment/${tag}.ckpt"
    local outdir="out/pp22_augment_ep${ep}_pyt"
    local log="out/_pp22_augment_ep${ep}_pyt.log"

    echo "[$(date +%H:%M:%S)] ep=$ep  teacher=$teacher" | tee -a "$SUM"
    .venv/bin/python scripts/eval/mot17.py \
        --preset mamba_pyt_backbone \
        --detector SDP \
        --mamba-teacher-ckpt "$teacher" \
        --mamba-ckpt "$MAMBA_CKPT" \
        --output "$outdir" \
        > "$log" 2>&1
    grep -E "IDF1:|MOTA:|Rcll:|HOTA:|DetA:|AssA:" "$log" | sed "s/^/  ep${ep} /" | tee -a "$SUM"
    echo "" >> "$SUM"
}

for ep in 5 10 15 20 25 27 30; do
    run_epoch "$ep"
done

echo "[$(date +%H:%M:%S)] DONE" | tee -a "$SUM"
cat "$SUM"
