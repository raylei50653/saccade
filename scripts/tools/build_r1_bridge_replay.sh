#!/usr/bin/env bash
# Build the research-only device replay helper used by R1 host R0.
# The .so is gitignored (*.so); rebuild after a clean checkout when running
# verify_r1_temporal_reduction_replay.py on a GPU host.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/src/saccade/perception/eval/_cuda/r1_bridge_replay.cu"
OUT="$ROOT/src/saccade/perception/eval/_cuda/libr1_bridge_replay.so"
if [[ ! -f "$SRC" ]]; then
  echo "missing source: $SRC" >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; device R1 replay requires a CUDA toolkit" >&2
  exit 1
fi
nvcc -shared -Xcompiler -fPIC,-O3 -O3 -o "$OUT" "$SRC"
echo "built $OUT"
