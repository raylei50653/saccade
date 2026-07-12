# R1 device bridge replay (research-only)

`r1_bridge_replay.cu` is a line-for-line copy of Consumer-A `bridge_vel4` /
`bridge_linres4` / `bridge_anchor4` from `src/tracking/tracker_gpu.cu`, exposed
as a tiny batch device API for host R0 replay.

## Build

The `.so` is gitignored. Rebuild after a clean checkout:

```bash
# Local GPU host (matches production CMAKE_CUDA_ARCHITECTURES=native)
bash scripts/tools/build_r1_bridge_replay.sh

# CI / no-GPU compile (explicit architecture required)
SACCADE_R1_CUDA_ARCH="-gencode=arch=compute_75,code=sm_75" \
  bash scripts/tools/build_r1_bridge_replay.sh
```

The script also writes `libr1_bridge_replay.build.json` (nvcc version, flags,
source/binary SHA256) consumed by R0 verifier provenance.

## Authority vs unit tests

| Mode | Backend | How |
|---|---|---|
| Unit tests / diagnostics | host binary32+FMA fallback OK | default |
| Seven-sequence owner packet | **device required** (fail-closed) | `verify_r1_temporal_reduction_replay.py --require-device` or `SACCADE_RESEARCH_R1_REQUIRE_DEVICE_REPLAY=1` |

Silent host fallback must never be the authority packet backend.
