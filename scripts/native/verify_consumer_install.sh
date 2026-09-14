#!/usr/bin/env bash
# status: stable
# Fresh-consumer install test for the native extension (ADR 025).
#
# Reproduces, end to end and outside the checkout, the supported third-party
# install path documented in docs/reference/runbooks/native_extension_install.md:
#
#   1. a new venv that has never seen this repository
#   2. `pip install 'saccade[native-build]'` from the checkout, non-editable
#   3. `cmake -S <checkout> -B <build> -DPYTHON_EXECUTABLE=<venv>/bin/python`
#      and `cmake --build <build> --target saccade_tracking_ext`
#   4. registration of <build> with the venv (saccade_build.pth), and the
#      per-process alternative (SACCADE_BUILD_PATH)
#   5. a smoke that imports saccade from site-packages -- with the checkout
#      absent from sys.path and the working directory -- and constructs
#      GPUByteTracker on the native extension
#
# The smoke fails closed on every checkout-relative assumption the runbook
# promises not to have: saccade.paths.source_checkout_root() must be None,
# saccade.paths.build_dir() must be None without SACCADE_BUILD_PATH, and the
# extension must load from the build directory the venv was pointed at.
#
# Usage:
#   scripts/native/verify_consumer_install.sh [--work DIR] [--jobs N] [--reuse]
#
#   --work DIR  where the venv, build directory and report go
#               (default: a fresh mktemp directory; printed at the end)
#   --jobs N    parallel build jobs (default: nproc)
#   --reuse     keep an existing venv/build under --work instead of starting
#               from empty (re-runs the install and build incrementally)
#
# Host requirements are the runbook's: Linux x86_64, Python 3.12, an NVIDIA
# driver, cmake >= 3.18, a GNU C++ host compiler inside nvcc's supported
# range, system OpenCV and GStreamer development files, pkg-config, and
# network access for the first configure (FetchContent).

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work=""
jobs="$(nproc)"
reuse=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --work) work="$2"; shift 2 ;;
        --jobs) jobs="$2"; shift 2 ;;
        --reuse) reuse=1; shift ;;
        -h|--help) sed -n '2,36p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$work" ]]; then
    work="$(mktemp -d "${TMPDIR:-/tmp}/saccade-consumer.XXXXXX")"
fi
mkdir -p "$work"
work="$(cd "$work" && pwd)"
venv="$work/venv"
build="$work/native"
report="$work/report.json"
log="$work/verify.log"

case "$work" in
    "$repo_root"|"$repo_root"/*)
        echo "--work must lie outside the checkout ($repo_root): $work" >&2
        exit 2 ;;
esac

if [[ $reuse -eq 0 ]]; then
    rm -rf "$venv" "$build" "$report"
fi

step() { printf '\n── %s\n' "$*" | tee -a "$log"; }

: > "$log"
step "consumer work directory: $work"

# ── 1. a venv that has never seen the checkout ──────────────────────────────
if [[ ! -x "$venv/bin/python" ]]; then
    step "create venv (python 3.12)"
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.12 "$venv" 2>&1 | tee -a "$log"
    else
        python3.12 -m venv "$venv" 2>&1 | tee -a "$log"
    fi
fi
python="$venv/bin/python"

pip_install() {
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$python" "$@"
    else
        "$python" -m pip install "$@"
    fi
}

# ── 2. non-editable install of the package plus the build toolchain ─────────
step "pip install 'saccade[native-build]' from $repo_root (non-editable)"
pip_install "saccade[native-build] @ file://$repo_root" 2>&1 | tee -a "$log"

site="$("$python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
[[ -d "$site/saccade" ]] || { echo "saccade did not install into $site" >&2; exit 1; }
[[ -x "$site/nvidia/cu13/bin/nvcc" ]] || { echo "native-build extra did not provide $site/nvidia/cu13/bin/nvcc" >&2; exit 1; }

# ── 3. build the tracker extension for that venv ────────────────────────────
step "cmake configure -> $build"
cmake -S "$repo_root" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_NATIVE_TESTS=OFF \
    "-DPYTHON_EXECUTABLE=$python" 2>&1 | tee -a "$log"

step "cmake build saccade_tracking_ext (-j$jobs)"
cmake --build "$build" --target saccade_tracking_ext --parallel "$jobs" 2>&1 | tee -a "$log"

ext="$(ls "$build"/saccade_tracking_ext*.so | head -n1)"
[[ -f "$ext" ]] || { echo "no saccade_tracking_ext*.so under $build" >&2; exit 1; }

# ── 4 + 5. discovery, both ways, then the smoke from outside the checkout ───
smoke="$work/smoke.py"
cat > "$smoke" <<'PY'
import json
import os
import sys
from pathlib import Path

expected_build = Path(os.environ["SACCADE_VERIFY_BUILD"]).resolve()
expected_site = Path(os.environ["SACCADE_VERIFY_SITE"]).resolve()
checkout = Path(os.environ["SACCADE_VERIFY_CHECKOUT"]).resolve()
mode = os.environ["SACCADE_VERIFY_MODE"]

for entry in sys.path:
    p = Path(entry).resolve() if entry else Path.cwd().resolve()
    assert p != checkout and checkout not in p.parents, f"checkout on sys.path: {entry}"
assert checkout not in Path.cwd().resolve().parents and Path.cwd().resolve() != checkout

import saccade  # noqa: E402
from saccade import paths  # noqa: E402

pkg = Path(saccade.__file__).resolve().parent
assert pkg.parent == expected_site, f"saccade imported from {pkg}, not {expected_site}"
assert paths.source_checkout_root() is None, "installed package must not look like a checkout"
if mode == "pth":
    assert "SACCADE_BUILD_PATH" not in os.environ
    assert paths.build_dir() is None, "no checkout and no env var: build_dir() must be None"
else:
    assert paths.build_dir() == expected_build

tracker = saccade.GPUByteTracker()
assert tracker.is_cuda, "GPUByteTracker is not backed by saccade_tracking_ext"

import saccade_tracking_ext  # noqa: E402
import torch  # noqa: E402

ext_path = Path(saccade_tracking_ext.__file__).resolve()
assert ext_path.parent == expected_build, f"extension loaded from {ext_path}"

tracker.set_frame_size(640, 480)
boxes = torch.tensor(
    [[10.0, 10.0, 60.0, 120.0], [200.0, 50.0, 260.0, 180.0]], device="cuda"
)
scores = torch.tensor([0.9, 0.8], device="cuda")
classes = torch.zeros(2, dtype=torch.int32, device="cuda")
tracks = tracker.update(boxes, scores, classes)
torch.cuda.synchronize()

print(
    json.dumps(
        {
            "mode": mode,
            "saccade": str(pkg),
            "saccade_version": saccade.__version__,
            "extension": str(ext_path),
            "is_cuda": tracker.is_cuda,
            "first_update_tracks": len(tracks),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    )
)
PY

run_smoke() {
    local mode="$1"; shift
    (
        cd "$work"
        env -i PATH="$venv/bin:/usr/bin:/bin" HOME="$work" \
            SACCADE_VERIFY_BUILD="$build" SACCADE_VERIFY_SITE="$site" \
            SACCADE_VERIFY_CHECKOUT="$repo_root" SACCADE_VERIFY_MODE="$mode" \
            "$@" "$python" "$smoke"
    )
}

step "register $build via $site/saccade_build.pth; smoke (mode=pth)"
printf '%s\n' "$build" > "$site/saccade_build.pth"
pth_result="$(run_smoke pth)"
echo "$pth_result" | tee -a "$log"

step "remove the .pth; smoke with SACCADE_BUILD_PATH only (mode=env)"
rm -f "$site/saccade_build.pth"
env_result="$(run_smoke env env SACCADE_BUILD_PATH="$build")"
echo "$env_result" | tee -a "$log"

step "control: neither .pth nor SACCADE_BUILD_PATH -> tracker is not native"
control_result="$(
    cd "$work"
    env -i PATH="$venv/bin:/usr/bin:/bin" HOME="$work" "$python" - <<'PY'
import json
import saccade
t = saccade.GPUByteTracker()
print(json.dumps({"mode": "none", "is_cuda": t.is_cuda}))
PY
)"
echo "$control_result" | tee -a "$log"
[[ "$control_result" == *'"is_cuda": false'* ]] || { echo "control run unexpectedly found the extension" >&2; exit 1; }

# leave the venv in the documented end state
printf '%s\n' "$build" > "$site/saccade_build.pth"

"$python" - "$report" "$repo_root" "$work" "$venv" "$build" "$site" "$ext" "$pth_result" "$env_result" "$control_result" <<'PY'
import json, shutil, subprocess, sys
(report, repo, work, venv, build, site, ext, pth, envr, ctl) = sys.argv[1:]
head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
needed = None
if shutil.which("readelf"):
    needed = [
        l.split("[", 1)[1].rstrip("]")
        for l in subprocess.run(["readelf", "-d", ext], capture_output=True, text=True).stdout.splitlines()
        if "(NEEDED)" in l
    ]
doc = {
    "checkout": repo,
    "checkout_head": head,
    "work": work,
    "venv": venv,
    "site_packages": site,
    "build_dir": build,
    "extension": ext,
    "extension_needed": needed,
    "smoke_pth": json.loads(pth),
    "smoke_env": json.loads(envr),
    "control_without_discovery": json.loads(ctl),
}
with open(report, "w") as fh:
    json.dump(doc, fh, indent=2)
print(json.dumps(doc, indent=2))
PY

step "OK — report: $report (work directory kept: $work)"
