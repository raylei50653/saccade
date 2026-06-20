#!/usr/bin/env bash
# Determinism regression gate: run the same short eval N times and hash output.
# Reports distinct non-empty hashes + failure count. Sleeps between runs to
# avoid rapid-fire teardown contention.
set -u
N="${1:-8}"
FRAMES="${2:-100}"
shift 2 2>/dev/null || true
EXTRA=("$@")
SEQ=MOT17-02-SDP
hashes=()
fails=0
for i in $(seq 1 "$N"); do
  out="results/_det_$i"
  rm -rf "$out"
  .venv/bin/python scripts/eval/mot17.py --preset baseline --detector SDP \
    --sequences "$SEQ" --max-frames "$FRAMES" --output "$out" --mlflow-uri "" \
    "${EXTRA[@]}" >/dev/null 2>&1
  f="$out/$SEQ.txt"
  if [ -s "$f" ]; then
    hashes+=("$(md5sum "$f" | cut -d' ' -f1)")
  else
    fails=$((fails+1))
  fi
  rm -rf "$out"
  sleep 1
done
if (( ${#hashes[@]} == 0 )); then
  echo "runs=$N frames=$FRAMES extra=[${EXTRA[*]}]  ok=0 fails=$fails  DISTINCT=NA"
  exit 1
fi

distinct=$(printf '%s\n' "${hashes[@]}" | sort -u | wc -l)
echo "runs=$N frames=$FRAMES extra=[${EXTRA[*]}]  ok=${#hashes[@]} fails=$fails  DISTINCT=$distinct"
printf '%s\n' "${hashes[@]}" | sort | uniq -c | sort -rn

if (( fails > 0 )); then
  exit 1
fi
