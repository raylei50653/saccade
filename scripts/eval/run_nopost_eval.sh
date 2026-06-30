#!/usr/bin/env bash
# Implicit vs explicit (matched seeds 42/43/44) re-evaluated with the post-processing
# crutches OFF: interpolation (--no-interpolate-tracklets) + bidir relink bridge
# (--no-relink-bridge-enabled). Tests whether §6's +1.8/+3.0 survives without them,
# and adds detector-only size-binned recall (which is interp/relink-free by design).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CU13="$ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$CU13:${LD_LIBRARY_PATH:-}"
mkdir -p logs

TEACHER=runs/gated_det_v14replica/epoch_0012.ckpt
LABELS=(impl_s42 impl_s43 impl_s44 expl_s42 expl_s43 expl_s44)
CKPTS=(
  runs/mamba_gt_v14replica_implicit_s42/best.ckpt
  runs/mamba_gt_v14replica_implicit_s43/best.ckpt
  runs/mamba_gt_v14replica_implicit_s44/best.ckpt
  runs/mamba_gt_v14replica_t3_t1/best.ckpt
  runs/mamba_gt_v14replica_t3_t1_shared_s43/best.ckpt
  runs/mamba_gt_v14replica_t3_t1_shared_s44/best.ckpt
)

for i in "${!LABELS[@]}"; do
  lbl="${LABELS[$i]}"; ckpt="${CKPTS[$i]}"
  echo "######## ${lbl}  ($(date '+%H:%M:%S')) ########"

  echo "--- tracking (interp OFF + relink OFF) ---"
  .venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP \
      --mamba-ckpt "$ckpt" \
      --no-interpolate-tracklets --no-relink-bridge-enabled \
      --output "results/nopost_${lbl}" 2>&1 | tee "logs/nopost_track_${lbl}.log"

  echo "--- size-binned recall (detector-only) ---"
  .venv/bin/python scripts/eval/mamba_size_binned_recall.py \
      --mamba-ckpt "$ckpt" --teacher-ckpt "$TEACHER" \
      --sequences MOT17-02-SDP \
      --score-thresholds 0.001,0.10,0.25 \
      --output "results/recall_nopost_${lbl}.json" 2>&1 | tee "logs/nopost_recall_${lbl}.log"
done

echo ""
echo "######################## SUMMARY ########################"
.venv/bin/python - <<'PY'
import json, re, pathlib
labels = ["impl_s42","impl_s43","impl_s44","expl_s42","expl_s43","expl_s44"]

print("\n== TRACKING (interp OFF + relink OFF) ==")
print(f"{'label':10s} {'IDF1':>6s} {'MOTA':>6s} {'HOTA':>6s} {'DetA':>6s} {'AssA':>6s} {'Rcll':>6s}")
for lbl in labels:
    t = pathlib.Path(f"logs/nopost_track_{lbl}.log").read_text(errors="ignore")
    def g(metric):
        m = re.findall(rf"{metric}:\s*([\d.]+)%", t)
        return m[-1] if m else "--"
    print(f"{lbl:10s} {g('IDF1'):>6s} {g('MOTA'):>6s} {g('HOTA'):>6s} {g('DetA'):>6s} {g('AssA'):>6s} {g('Rcll'):>6s}")

print("\n== SIZE-BINNED RECALL @0.001 (MOT17-02-SDP, detector-only) ==")
bins = ["all","h_ge128","h_64to128","h_32to64","h_lt32","min_ge16","min_8to16","min_4to8","min_lt4"]
hdr = f"{'label':10s}" + "".join(f"{b:>10s}" for b in bins)
print(hdr)
for lbl in labels:
    p = pathlib.Path(f"results/recall_nopost_{lbl}.json")
    if not p.exists():
        print(f"{lbl:10s} (missing)"); continue
    d = json.loads(p.read_text())
    bb = d["sequences"]["MOT17-02-SDP"]["thresholds"]["0.001"]["bins"]
    row = f"{lbl:10s}"
    for b in bins:
        v = bb.get(b, {}).get("recall")
        row += f"{(f'{v:.3f}' if v is not None else '--'):>10s}"
    print(row)
PY
echo "=== NOPOST EVAL DONE ($(date '+%H:%M:%S')) ==="
