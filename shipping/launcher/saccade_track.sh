#!/bin/sh
# saccade_track launcher (#465 Phase C PR-C1); installed as <prefix>/bin/saccade_track.
#
# Runs the entrypoint <prefix>/libexec/saccade_track (the PR-12 executable,
# bytes unchanged) through the system's dynamic loader with the bundled
# third-party libraries <prefix>/lib/vendor as its library path and the
# provenance auditor <prefix>/lib/saccade_loader_audit.so (docs/reference/
# native_runtime_resolved_config.md §17). --library-path replaces
# LD_LIBRARY_PATH for this process; LD_PRELOAD / LD_AUDIT / LD_LIBRARY_PATH
# from the caller are dropped.
#
# The loader ignores an audit library it cannot load (missing, truncated), so
# the launcher first runs the same loader with the same auditor as a probe
# (SACCADE_AUDIT_PROBE=1: the auditor prints a ready line and exits before
# /bin/sh runs) and exits 127 unless the auditor initialized (§17.8, A2).
# Builtins only: the probe subshell execs the loader once, then the sh process
# execs the loader and nothing else.
set -eu
case $0 in
    */*) bin_dir=${0%/*} ;;
    *) bin_dir=. ;;
esac
prefix=$(cd -P "$bin_dir/.." && pwd -P)
case $prefix in
    *:*) echo "saccade_track: the install prefix must not contain ':' ($prefix)" >&2; exit 2 ;;
esac
unset LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH SACCADE_AUDIT_PROBE
ready=$(SACCADE_AUDIT_PROBE=1 /lib64/ld-linux-x86-64.so.2 \
    --library-path "$prefix/lib/vendor" \
    --audit "$prefix/lib/saccade_loader_audit.so" \
    /bin/sh -c : 2>/dev/null) || ready=
if [ "$ready" != saccade-loader-audit-ready ]; then
    echo "saccade_track: the loader provenance auditor did not initialize ($prefix/lib/saccade_loader_audit.so)" >&2
    exit 127
fi
exec /lib64/ld-linux-x86-64.so.2 \
    --library-path "$prefix/lib/vendor" \
    --audit "$prefix/lib/saccade_loader_audit.so" \
    --argv0 "$0" \
    "$prefix/libexec/saccade_track" "$@"
