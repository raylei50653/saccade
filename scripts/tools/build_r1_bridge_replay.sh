#!/usr/bin/env bash
# Build the research-only device replay helper used by R1 host R0.
# The .so is gitignored (*.so); rebuild after a clean checkout when running
# authority verify with --require-device.
#
# Architecture:
#   - Default: -arch=native (matches production CMAKE_CUDA_ARCHITECTURES=native
#     on a GPU host used for owner packets).
#   - CI / no-GPU compile: set SACCADE_R1_CUDA_ARCH to an explicit gencode, e.g.
#       SACCADE_R1_CUDA_ARCH="-gencode=arch=compute_75,code=sm_75"
#   - The chosen flags are written next to the .so for replay provenance.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC_DIR="$ROOT/src/saccade/perception/eval/_cuda"
SRC="$SRC_DIR/r1_bridge_replay.cu"
OUT="$SRC_DIR/libr1_bridge_replay.so"
META="$SRC_DIR/libr1_bridge_replay.build.json"

if [[ ! -f "$SRC" ]]; then
  echo "missing source: $SRC" >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; device R1 replay requires a CUDA toolkit" >&2
  exit 1
fi

# Prefer an explicit architecture; fall back to native for local GPU hosts.
ARCH_FLAG="${SACCADE_R1_CUDA_ARCH:--arch=native}"
# shellcheck disable=SC2206
NVCC_FLAGS=(-shared -Xcompiler -fPIC,-O3 -O3 $ARCH_FLAG)

echo "nvcc ${NVCC_FLAGS[*]} -o $OUT $SRC"
nvcc "${NVCC_FLAGS[@]}" -o "$OUT" "$SRC"

# Exported symbols must stay stable for ctypes.
if command -v nm >/dev/null 2>&1; then
  if ! nm -D "$OUT" | grep -q "r1_bridge_anchor4_batch"; then
    echo "built $OUT but missing symbol r1_bridge_anchor4_batch" >&2
    exit 1
  fi
  if ! nm -D "$OUT" | grep -q "r1_bridge_vel4_batch"; then
    echo "built $OUT but missing symbol r1_bridge_vel4_batch" >&2
    exit 1
  fi
fi

NVCC_VERSION="$(nvcc --version | tr '\n' ' ' | sed 's/  */ /g')"
SOURCE_SHA="$(sha256sum "$SRC" | awk '{print $1}')"
BINARY_SHA="$(sha256sum "$OUT" | awk '{print $1}')"
BUILT_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

python3 - "$META" "$NVCC_VERSION" "$SOURCE_SHA" "$BINARY_SHA" "$BUILT_AT" "$ARCH_FLAG" <<'PY'
import json, sys
meta_path, nvcc_version, source_sha, binary_sha, built_at, arch_flag = sys.argv[1:7]
# Reconstruct the flag list for provenance (must match the shell invocation).
compile_flags = ["-shared", "-Xcompiler", "-fPIC,-O3", "-O3", *arch_flag.split()]
payload = {
    "nvcc_version": nvcc_version,
    "compile_flags": compile_flags,
    "cuda_architectures": arch_flag,
    "source": "src/saccade/perception/eval/_cuda/r1_bridge_replay.cu",
    "source_sha256": source_sha,
    "binary_sha256": binary_sha,
    "binary": "src/saccade/perception/eval/_cuda/libr1_bridge_replay.so",
    "built_at_utc": built_at,
}
with open(meta_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(f"wrote {meta_path}")
PY

echo "built $OUT"
echo "  arch: $ARCH_FLAG"
echo "  source_sha256: $SOURCE_SHA"
echo "  binary_sha256: $BINARY_SHA"
