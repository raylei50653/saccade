#!/bin/sh
# saccade_track launcher (#465 Phase C PR-C1); installed as <prefix>/bin/saccade_track.
#
# Runs the entrypoint <prefix>/libexec/saccade_track (the PR-12 executable,
# bytes unchanged) through the system's dynamic loader with the bundled
# third-party libraries <prefix>/lib/vendor as its library path and the
# provenance auditor <prefix>/lib/saccade_loader_audit.so (docs/reference/
# native_runtime_resolved_config.md §17). --library-path replaces
# LD_LIBRARY_PATH for this process; LD_PRELOAD / LD_AUDIT / LD_LIBRARY_PATH
# from the caller are dropped. Builtins only: the sh process execs the loader
# and nothing else.
set -eu
case $0 in
    */*) bin_dir=${0%/*} ;;
    *) bin_dir=. ;;
esac
prefix=$(cd -P "$bin_dir/.." && pwd -P)
case $prefix in
    *:*) echo "saccade_track: the install prefix must not contain ':' ($prefix)" >&2; exit 2 ;;
esac
unset LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH
exec /lib64/ld-linux-x86-64.so.2 \
    --library-path "$prefix/lib/vendor" \
    --audit "$prefix/lib/saccade_loader_audit.so" \
    --argv0 "$0" \
    "$prefix/libexec/saccade_track" "$@"
