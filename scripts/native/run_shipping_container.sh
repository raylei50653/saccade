#!/usr/bin/env bash
# Run the installed shipping tree in a clean Ubuntu 24.04 container (#465 PR-12).
# status: diagnostic
#
# The container is the pinned ubuntu:24.04 image (no Python, no compiler) or,
# for MODE=strace, that image plus strace only. Mounted read-only: the tree at
# /opt/saccade, the third-party runtime libraries (check_shipping_tree.py deps;
# not bundled, Phase C decides) at /opt/saccade-deps, found through
# LD_LIBRARY_PATH, and the MOT17 train split at /data/MOT17/train. No network.
# The GPU comes from the NVIDIA container toolkit (--gpus all with the compute,
# utility and video driver capabilities: nvJPEG's hardware decoder needs
# libnvcuvid). saccade_track writes OUT/native, OUT/trace, OUT/track_report.json
# (and, for MODE=strace, OUT/strace/s.<tid> from strace -ff); OUT/container.txt
# records the image and the container's OS, glibc and the absent tools.
#
# Usage: run_shipping_container.sh pristine|strace TREE DEPS OUT [SEQUENCE...]
#        (default: the seven MOT17 SDP train sequences, in the oracle's order)
set -euo pipefail

IMAGE=ubuntu@sha256:786a8b558f7be160c6c8c4a54f9a57274f3b4fb1491cf65146521ae77ff1dc54
STRACE_IMAGE=saccade-g2-strace:ubuntu24.04
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ $# -lt 4 ]; then
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | head -n -1
    exit 2
fi
MODE=$1 TREE=$(realpath "$2") DEPS=$(realpath "$3") OUT=$4
shift 4
SEQS=("$@")
if [ ${#SEQS[@]} -eq 0 ]; then
    SEQS=(MOT17-02-SDP MOT17-04-SDP MOT17-05-SDP MOT17-09-SDP MOT17-10-SDP MOT17-11-SDP MOT17-13-SDP)
fi
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
if [ -n "$(ls -A "$OUT")" ]; then
    echo "$OUT is not empty" >&2
    exit 2
fi

case "$MODE" in
    pristine) RUN_IMAGE=$IMAGE ;;
    strace)
        docker build --network host -q -t "$STRACE_IMAGE" - >/dev/null <<EOF
FROM $IMAGE
RUN apt-get update && apt-get install -y --no-install-recommends strace && rm -rf /var/lib/apt/lists/*
EOF
        RUN_IMAGE=$STRACE_IMAGE
        mkdir -p "$OUT/strace"
        ;;
    *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac

M=/opt/saccade/share/saccade
ARGS=(--config $M/configs/shipping/mamba_whole_graph.resolved.json
      --lineage $M/models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json
      --attestation $M/configs/shipping/mamba_head_realization.attestation.json
      --model-root $M --out /out/native --report /out/track_report.json --trace /out/trace)
for s in "${SEQS[@]}"; do ARGS+=("/data/MOT17/train/$s"); done
CMD=(/opt/saccade/bin/saccade_track "${ARGS[@]}")
if [ "$MODE" = strace ]; then
    CMD=(strace -ff -qq -s 4096 -e trace=execve,execveat,open,openat -o /out/strace/s "${CMD[@]}")
fi
DOCKER=(docker run --rm --network none --user "$(id -u):$(id -g)"
        --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
        -v "$TREE:/opt/saccade:ro" -v "$DEPS:/opt/saccade-deps:ro"
        -v "$REPO/datasets/MOT17/train:/data/MOT17/train:ro" -v "$OUT:/out"
        -e LD_LIBRARY_PATH=/opt/saccade-deps)

{
    echo "image: $RUN_IMAGE ($(docker image inspect "$RUN_IMAGE" --format '{{.Id}}'))"
    "${DOCKER[@]}" "$RUN_IMAGE" sh -c '
        . /etc/os-release; echo "os: $PRETTY_NAME"
        echo "glibc: $(ldd --version | head -1)"
        echo "libstdc++: $(ls /usr/lib/x86_64-linux-gnu/libstdc++.so.6.*)"
        for t in python3 python cc gcc c++ g++ clang nvcc ptxas; do
            command -v $t >/dev/null && echo "present: $t" || echo "absent: $t"
        done'
} > "$OUT/container.txt"

set +e
"${DOCKER[@]}" "$RUN_IMAGE" "${CMD[@]}" > "$OUT/saccade_track.log" 2>&1
rc=$?
set -e
echo "exit=$rc" >> "$OUT/saccade_track.log"
exit $rc
