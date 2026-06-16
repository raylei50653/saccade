#!/usr/bin/env bash
# Route 2 (§5): joint T-clip + T=1 loss in a SINGLE phase — keep cross-frame pressure
# (T=3 all-frames) AND deploy-T=1 quality (--t1-weight aux) simultaneously every step,
# vs T3→T1's sequential A→B staging. Tests whether simultaneous beats sequential and
# whether it resolves the §4.1 antagonism (get DetA without losing AssA).
#
# Two regimes from the same GT1 start as T3→T1, 30ep (= T3→T1's 15+15), seed 42:
#   frozen   : methodological — does 1-phase joint match T3→T1 (75.4/66.0/69.7, frozen)?
#   unfrozen : antagonism test — does joint+unfreeze get DetA>69.7 AND keep AssA~66
#              (vs cw0 75.7/65.4/70.7 which lost AssA, vs T3T1→ssmft 62.8 collapse)?
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
CU13="$ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$CU13:${LD_LIBRARY_PATH:-}"
mkdir -p logs

TEACHER=runs/gated_det_v14replica/epoch_0012.ckpt
CACHE=runs/mamba_teacher_cache_v14replica
GT1=runs/mamba_gt_v14replica_stage1/best.ckpt   # same start as T3→T1
T1W="${1:-1.0}"

for regime in frozen unfrozen; do
  if [ "$regime" = "frozen" ]; then SG=(--scan-stop-grad); else SG=(--no-scan-stop-grad); fi
  tag="${regime}_t1w${T1W/./p}"
  echo "######## joint ${tag} ($(date '+%H:%M:%S')) ########"

  .venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
      --data-root datasets/MOT17 \
      --teacher-ckpt "$TEACHER" \
      --mamba-ckpt "$GT1" \
      --add-temporal --t1-weight "$T1W" "${SG[@]}" \
      --cache-dir "$CACHE" \
      --run-dir "runs/mamba_gt_v14replica_t1joint_${tag}" \
      --img-size 640 --clip-len 3 --clip-stride 6 \
      --epochs 30 --batch-size 4 --lr 1e-4 \
      --warmup-epochs 3 --clip-grad 1.0 \
      --gt-ratio 0 --seed 42 \
      --best-by train-loss --save-every 10 2>&1 | tee "logs/t1joint_${tag}.log"

  echo "--- eval ${tag} ---"
  .venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP \
      --mamba-ckpt "runs/mamba_gt_v14replica_t1joint_${tag}/best.ckpt" \
      --output "results/mamba_v14replica_t1joint_${tag}" 2>&1 | tee -a "logs/t1joint_${tag}.log"
done

echo ""
echo "######## SUMMARY (vs T3→T1 75.4/66.0/69.7 · cw0 75.7/65.4/70.7) ########"
for regime in frozen unfrozen; do
  tag="${regime}_t1w${T1W/./p}"
  echo "--- joint ${tag} ---"
  grep -hiE "IDF1:|MOTA:|HOTA:|DetA:|AssA:" "logs/t1joint_${tag}.log" 2>/dev/null | tail -5
done
echo "=== T1JOINT DONE ($(date '+%H:%M:%S')) ==="
