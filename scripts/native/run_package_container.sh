#!/usr/bin/env bash
# Install the shipping package in a clean Ubuntu 24.04 container (#465 Phase C PR-C3).
# status: diagnostic
#
# Runs DIST/<name>.install.sh DIST/<name>.tar.gz /install/<base of TARGET> in
# the pinned ubuntu:24.04 image (no Python, no compiler; /bin/sh is dash) or,
# for install-strace, that image plus strace only. DIST (the three release
# files) is mounted read-only at /dist; the parent of TARGET (which must not
# exist) read-write at /install, so the installed tree lands at TARGET on the
# host. No network, no GPU. OUT gets install.log (with exit=<rc>),
# container.txt (image, OS, glibc, coreutils, tar, the absent tools) and
# after.txt (the parent's entries after the run, as the container sees them).
# install-strace records strace -ff -yy of every file-system call (%file plus
# fchmod, fchown, ftruncate, fallocate) into
# OUT/strace/i.<pid> (check_shipping_package.py install-trace reads it).
#
# Harness knobs for the negative controls (the installer has none):
#   INSTALL_TMPFS=<size>     mount a tmpfs of that size at /install instead of
#                            the parent of TARGET (a full disk); after.txt is
#                            the only record of what was left.
#   INSTALL_SIGNAL="<SIG> <seconds>"  run the installer under coreutils
#                            timeout, which sends SIG to its process group.
#
# verify (PR-C4, §20): the user's check before installing, in the same image
# plus minisign only: `minisign -Vm <name>.sha256 -p <key>` then `sha256sum -c
# <name>.sha256` in /dist (read-only); PUBKEY is mounted read-only. OUT gets
# verify.log (with exit=<rc>) and container.txt.
#
# Usage: run_package_container.sh install|install-strace DIST TARGET OUT
#        run_package_container.sh verify DIST PUBKEY OUT
set -euo pipefail

IMAGE=ubuntu@sha256:786a8b558f7be160c6c8c4a54f9a57274f3b4fb1491cf65146521ae77ff1dc54
STRACE_IMAGE=saccade-g2-strace:ubuntu24.04
MINISIGN_IMAGE=saccade-minisign:ubuntu24.04

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | head -n -1; exit 2; }
case "${1:-}" in
    install|install-strace|verify) [ $# -eq 4 ] || usage ;;
    *) usage ;;
esac
MODE=$1
DIST=$(realpath "$2")
TARGET=$3
OUT=$4
tars=("$DIST"/*.tar.gz)
if [ ${#tars[@]} -ne 1 ] || [ ! -f "${tars[0]}" ]; then
    echo "$DIST must hold exactly one .tar.gz" >&2
    exit 2
fi
NAME=$(basename "${tars[0]}" .tar.gz)
if [ "$MODE" = verify ]; then
    PUBKEY=$(realpath "$3")
    OUT=$4
    mkdir -p "$OUT"
    OUT=$(realpath "$OUT")
    if [ -n "$(ls -A "$OUT")" ]; then
        echo "$OUT is not empty" >&2
        exit 2
    fi
    docker build --network host -q -t "$MINISIGN_IMAGE" - >/dev/null <<EOF
FROM $IMAGE
RUN apt-get update && apt-get install -y --no-install-recommends minisign && rm -rf /var/lib/apt/lists/*
EOF
    DOCKER=(docker run --rm --network none --user "$(id -u):$(id -g)"
            -v "$DIST:/dist:ro" -v "$PUBKEY:/key/minisign.pub:ro" -w /dist)
    {
        echo "image: $MINISIGN_IMAGE ($(docker image inspect "$MINISIGN_IMAGE" --format '{{.Id}}'))"
        "${DOCKER[@]}" "$MINISIGN_IMAGE" sh -c '
            . /etc/os-release; echo "os: $PRETTY_NAME"
            echo "sh: $(readlink -f /bin/sh)"
            echo "minisign: $(minisign -v 2>&1 | head -1)"
            echo "coreutils: $(sha256sum --version | head -1)"'
    } > "$OUT/container.txt"
    set +e
    "${DOCKER[@]}" "$MINISIGN_IMAGE" sh -c \
        'minisign -Vm "$1.sha256" -p /key/minisign.pub && sha256sum -c "$1.sha256"' sh "$NAME" \
        > "$OUT/verify.log" 2>&1
    rc=$?
    set -e
    echo "exit=$rc" >> "$OUT/verify.log"
    exit $rc
fi
PARENT=$(realpath "$(dirname "$TARGET")")
BASE=$(basename "$TARGET")
if [ -z "${INSTALL_TMPFS:-}" ] && [ ! -d "$PARENT" ]; then
    echo "$PARENT is not a directory" >&2
    exit 2
fi
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
if [ -n "$(ls -A "$OUT")" ]; then
    echo "$OUT is not empty" >&2
    exit 2
fi

RUN_IMAGE=$IMAGE
CMD=(sh "/dist/$NAME.install.sh" "/dist/$NAME.tar.gz" "/install/$BASE")
if [ -n "${INSTALL_SIGNAL:-}" ]; then
    read -r sig secs <<<"$INSTALL_SIGNAL"
    CMD=(timeout -s "$sig" "$secs" "${CMD[@]}")
fi
if [ "$MODE" = install-strace ]; then
    docker build --network host -q -t "$STRACE_IMAGE" - >/dev/null <<EOF
FROM $IMAGE
RUN apt-get update && apt-get install -y --no-install-recommends strace && rm -rf /var/lib/apt/lists/*
EOF
    RUN_IMAGE=$STRACE_IMAGE
    mkdir -p "$OUT/strace"
    CMD=(strace -ff -qq -yy -s 4096 -e trace=%file,fchmod,fchown,ftruncate,fallocate -e signal=none -o /out/strace/i "${CMD[@]}")
fi
DOCKER=(docker run --rm --network none --user "$(id -u):$(id -g)"
        -v "$DIST:/dist:ro" -v "$OUT:/out")
if [ -n "${INSTALL_TMPFS:-}" ]; then
    DOCKER+=(--tmpfs "/install:rw,size=$INSTALL_TMPFS,uid=$(id -u),gid=$(id -g),mode=0755")
else
    DOCKER+=(-v "$PARENT:/install")
fi

{
    echo "image: $RUN_IMAGE ($(docker image inspect "$RUN_IMAGE" --format '{{.Id}}'))"
    "${DOCKER[@]}" "$RUN_IMAGE" sh -c '
        . /etc/os-release; echo "os: $PRETTY_NAME"
        echo "glibc: $(ldd --version | head -1)"
        echo "sh: $(readlink -f /bin/sh)"
        echo "coreutils: $(mv --version | head -1)"
        echo "tar: $(tar --version | head -1)"
        for t in python3 python cc gcc c++ g++ clang nvcc perl; do
            command -v $t >/dev/null && echo "present: $t" || echo "absent: $t"
        done'
} > "$OUT/container.txt"

set +e
"${DOCKER[@]}" "$RUN_IMAGE" sh -c '"$@"; rc=$?; ls -la /install > /out/after.txt; exit $rc' sh "${CMD[@]}" \
    > "$OUT/install.log" 2>&1
rc=$?
set -e
echo "exit=$rc" >> "$OUT/install.log"
exit $rc
