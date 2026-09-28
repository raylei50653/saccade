#!/usr/bin/env python3
"""Exercise the #465 PR-2L runner's L1 worker and L2 child entry on synthetic frames (no MOT17 data).

The PR-2L declaration (§8, §9) allows the runner PR only unit tests and
structural checks that read no MOT17 frame; the formal run executes once. This
check runs the real ``run_l1_worker`` and ``run_arm_child`` -- operator
library and artifact loading, the driver/runtime probe, the JIT-fallback and
PyTorch-head guards, the ``_trt_head`` injection and an unmodified
``mot17.py`` under ``runpy`` (inside the harness's CUDA graph capture) -- on
generated sequences in a temporary data root, then applies the runner's own
validity functions:

* L1: ``l1_problems`` (V2 (a) re-run, (b) shared features unchanged,
  (c) zero JIT-fallback calls, driver/runtime, artifact, L-call checks);
* L2: every arm's sidecar passes ``sidecar_problems`` (V5) and its harness
  ``run_manifest.json`` passes ``argv_problems``; a second ``A_L`` run
  reproduces the txt bytes (V3 rehearsal).

It prints no head-comparison quantity and writes nothing under ``results/``;
its output is not evidence. V1's freeze-tag and lease items and V4's oracle
anchor cannot hold before the runner merge and are covered by unit tests
(``tests/unit/test_native_head_parity_libtorch_runner.py``). Usage (GPU, a
few minutes for torch.compile)::

    .venv/bin/python tools/resctl.py run gpu0 -- .venv/bin/python \\
        scripts/eval/diagnostics/native_head_parity_libtorch_structural_check.py
"""
# status: experiment

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[3]
RUNNER = project_root / "scripts/eval/diagnostics/native_head_parity_libtorch.py"
SYNTHETIC = (("SYN-01-SDP", 1920, 1080), ("SYN-02-SDP", 640, 480))
N_FRAMES = 12
ENV_FIRST: dict[str, Any] = {}  # the first arm's child env overrides


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("pr2l_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_synthetic_dataset(root: Path) -> None:
    """Gray frames with a few dark upright bars drifting right; a matching gt."""
    from PIL import Image, ImageDraw

    for name, w, h in SYNTHETIC:
        seq = root / "train" / name
        (seq / "img1").mkdir(parents=True)
        (seq / "gt").mkdir()
        gt_rows = []
        for f in range(1, N_FRAMES + 1):
            img = Image.new("RGB", (w, h), (128, 128, 128))
            draw = ImageDraw.Draw(img)
            for i in range(3):
                bw, bh = w // 20, h // 4
                x = w // 6 + i * w // 4 + 4 * f
                y = h // 2 - bh // 2
                draw.rectangle([x, y, x + bw, y + bh], fill=(40, 40, 60))
                gt_rows.append(f"{f},{i + 1},{x},{y},{bw},{bh},1,1,1.0")
            img.save(seq / "img1" / f"{f:06d}.jpg", quality=95)
        (seq / "gt" / "gt.txt").write_text("\n".join(gt_rows) + "\n")
        (seq / "seqinfo.ini").write_text(
            "[Sequence]\n"
            f"name={name}\nimDir=img1\nframeRate=30\nseqLength={N_FRAMES}\n"
            f"imWidth={w}\nimHeight={h}\nimExt=.jpg\n"
        )


def _child(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, __file__, *args],
        cwd=project_root,
        capture_output=True,
        text=True,
    )


def run_l1(runner: Any, root: Path) -> list[str]:
    out = root / "l1"
    out.mkdir()
    proc = _child(["--_l1", str(out), "--_data", str(root / "data")])
    if proc.returncode != 0 or not (out / "l1.json").exists():
        print(proc.stdout[-4000:], proc.stderr[-4000:], sep="\n")
        return [f"L1 worker exit {proc.returncode}"]
    l1 = json.loads((out / "l1.json").read_text())
    problems = runner.l1_problems(l1, len(SYNTHETIC) * N_FRAMES)
    if l1["v2"]["frames_checked"] != len(SYNTHETIC) * N_FRAMES:
        problems.append("V2(a) did not re-run every synthetic frame")
    if l1["l_calls"] < 2 * len(SYNTHETIC) * N_FRAMES:
        problems.append("L was not called for every frame and every V2 re-run")
    return [f"L1: {p}" for p in problems]


def run_arm(runner: Any, arm: str, work: Path) -> list[str]:
    """One arm's child in its own process; return V5/argv problems."""
    proc = _child(["--_arm", arm, "--_work", str(work)])
    sidecar_path = work / "out.sidecar.json"
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else None
    problems = []
    if proc.returncode != 0:
        print(proc.stdout[-4000:], proc.stderr[-4000:], sep="\n")
        problems.append(f"child exit {proc.returncode}")
    problems += runner.sidecar_problems(arm, sidecar)
    problems += runner.env_override_problems(
        (sidecar or {}).get("resolved_env_overrides"),
        ENV_FIRST.setdefault("first", (sidecar or {}).get("resolved_env_overrides")),
    )
    manifest = work / "out" / "run_manifest.json"
    if not manifest.exists():
        problems.append("run_manifest.json missing")
    elif sidecar is not None:
        cmdline = json.loads(manifest.read_text()).get("cmdline")
        problems += runner.argv_problems(cmdline, sidecar.get("mot17_argv"))
    for name, _, _ in SYNTHETIC:
        if not (work / "out" / f"{name}.txt").exists():
            problems.append(f"{name}.txt missing")
    return [f"{arm}: {p}" for p in problems]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--_l1", dest="l1", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_data", dest="data", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_arm", dest="arm", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_work", dest="work", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    runner = _load_runner()
    seqs = tuple(n for n, _, _ in SYNTHETIC)

    if args.l1:
        return runner.run_l1_worker(Path(args.l1), seqs, data_root=args.data)
    if args.arm:
        work = Path(args.work)
        return runner.run_arm_child(
            args.arm,
            work / "out",
            work / "out.sidecar.json",
            seqs,
            data_root=str(work / "data"),
        )

    with tempfile.TemporaryDirectory(prefix="pr2l_structural_") as tmp:
        root = Path(tmp)
        write_synthetic_dataset(root / "data")
        assert "MOT17" not in str(root / "data")
        problems = run_l1(runner, root)
        for arm in (*runner.ARMS, "A_L"):  # A_L twice: V3 rehearsal
            work = (
                root
                / f"{arm}_{sum(1 for p in root.iterdir() if p.name.startswith(arm))}"
            )
            work.mkdir()
            (work / "data").symlink_to(root / "data")
            problems += run_arm(runner, arm, work)
        if not problems:
            a, b = root / "A_L_0", root / "A_L_1"
            for name in seqs:
                if (a / "out" / f"{name}.txt").read_bytes() != (
                    b / "out" / f"{name}.txt"
                ).read_bytes():
                    problems.append(
                        f"V3 rehearsal: A_L {name} txt differs between runs"
                    )
        for p in problems:
            print(f"FAIL: {p}")
        if not problems:
            print(
                f"OK: L1 worker and arms {list(runner.ARMS)} (+A_L repeat), "
                f"{len(seqs)} synthetic sequences x {N_FRAMES} frames"
            )
        return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
