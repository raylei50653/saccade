#!/usr/bin/env python3
"""Exercise the #465 compiled-head reverse-engineering runner's R0 children and R1 worker on synthetic frames (no MOT17 data).

The declaration (§7) allows the runner PR only unit tests and structural
checks that read no MOT17 frame; the formal run executes once. This check runs
the real ``run_r0_child`` for every arm (isolated process, compile logging on,
fresh compile cache) and the real ``run_r1_worker`` (all arms in one process)
on generated sequences in a temporary data root, then applies the runner's own
validity functions: ``r0_problems`` for every arm and ``r1_problems`` (graph
construction identity -- R1 executes the isolated R0 arm's graphs call by call,
with cache reuse attributed -- stage identity R1 == isolated R0, no compile
during the frames, repeat identity, no input mutation, finite outputs, wrap
sites, policy, driver/runtime). It prints the per-arm provenance summary
(compiled calls, distinct graphs, own compiles, ``reused_from``). It also
checks that ``decide`` returns a declared scope result on the R1 record.

It prints no head-comparison quantity and writes nothing under ``results/``;
its output is not evidence. V1's freeze-tag/lease items and the V-anchor
(which needs the MOT17 frames) cannot hold here and are covered by unit tests
(``tests/unit/test_compiled_head_reverse_engineering_runner.py``). Usage (GPU,
a few minutes for torch.compile)::

    .venv/bin/python tools/resctl.py run gpu0 -- .venv/bin/python \\
        scripts/eval/diagnostics/compiled_head_reverse_engineering_structural_check.py
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
RUNNER = project_root / "scripts/eval/diagnostics/compiled_head_reverse_engineering.py"
SYNTHETIC = (("SYN-01-SDP", 1920, 1080), ("SYN-02-SDP", 640, 480))
N_FRAMES = 24  # more than the runner's REPEAT_FRAMES, so both paths run


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("cre_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_synthetic_dataset(root: Path) -> None:
    """Gray frames with a few dark upright bars drifting right."""
    from PIL import Image, ImageDraw

    for name, w, h in SYNTHETIC:
        img_dir = root / "train" / name / "img1"
        img_dir.mkdir(parents=True)
        for f in range(1, N_FRAMES + 1):
            img = Image.new("RGB", (w, h), (128, 128, 128))
            draw = ImageDraw.Draw(img)
            for i in range(3):
                bw, bh = w // 20, h // 4
                x = w // 6 + i * w // 4 + 4 * f
                y = h // 2 - bh // 2
                draw.rectangle([x, y, x + bw, y + bh], fill=(40, 40, 60))
            img.save(img_dir / f"{f:06d}.jpg", quality=95)


def _child(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, __file__, *args],
        cwd=project_root,
        capture_output=True,
        text=True,
        env=env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--_r0", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_r1", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_out", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_data", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    runner = _load_runner()
    if args._r0:
        return runner.run_r0_child(args._r0, Path(args._out))
    if args._r1:
        seqs = tuple(name for name, _, _ in SYNTHETIC)
        return runner.run_r1_worker(Path(args._r1), seqs, args._data)

    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="cre_structural_") as tmp:
        root = Path(tmp)
        write_synthetic_dataset(root / "data")
        r0: dict[str, dict[str, Any]] = {}
        for arm in runner.ARM_ORDER:
            out = root / f"r0_{arm}.json"
            proc = _child(
                ["--_r0", arm, "--_out", str(out)],
                runner.child_env(root / "cache" / f"r0_{arm}"),
            )
            rec = json.loads(out.read_text()) if out.exists() else None
            if proc.returncode != 0:
                print(proc.stdout[-3000:], proc.stderr[-3000:], sep="\n")
                problems.append(f"R0 {arm}: child exit {proc.returncode}")
            if rec is not None:
                inventory = runner.op_inventory(proc.stderr)
                if runner.ARM_SPECS[arm]["head"] or runner.ARM_SPECS[arm]["block"]:
                    if not inventory["fx"]:
                        problems.append(
                            f"R0 {arm}: compile log has no Dynamo graph code"
                        )
                    if arm.startswith(("A_", "C")) and not inventory["ops"]:
                        problems.append(f"R0 {arm}: compile log has no AOT operator")
                r0[arm] = rec
            problems += runner.r0_problems(arm, rec)
            print(f"[R0] {arm}: exit {proc.returncode}", flush=True)
        r1_dir = root / "r1"
        r1_dir.mkdir()
        proc = _child(
            ["--_r1", str(r1_dir), "--_data", str(root / "data")],
            runner.child_env(root / "cache" / "r1"),
        )
        r1_path = r1_dir / "r1.json"
        if proc.returncode != 0 or not r1_path.exists():
            print(proc.stdout[-4000:], proc.stderr[-4000:], sep="\n")
            problems.append(f"R1 worker exit {proc.returncode}")
        else:
            r1 = json.loads(r1_path.read_text())
            problems += runner.r1_problems(r1, r0, len(SYNTHETIC) * N_FRAMES)
            prov = r1.get("provenance") or {}
            for arm, a in (prov.get("arms") or {}).items():
                fps = {x["fingerprint"] for x in a["executed"]}
                print(
                    f"[R1 provenance] {arm}: {len(a['executed'])} compiled calls, "
                    f"{len(fps)} distinct graphs, own compiles {len(a['own'])}, "
                    f"reused_from {a['reused_from']}",
                    flush=True,
                )
            if set(r1["anchor"]["rows"]) != {n for n, _, _ in SYNTHETIC} | {"ALL"}:
                problems.append("R1 anchor rows do not cover every sequence + ALL")
            result = runner.decide(not problems, r1)
            if result["scope"] not in runner.SCOPE_RESULTS:
                problems.append(f"decide returned {result['scope']!r}")
            for st in result["stages"].values():
                if st["label"] not in runner.STAGE_LABELS:
                    problems.append(f"decide returned stage {st['label']!r}")
    if problems:
        print("STRUCTURAL CHECK FAILED")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("STRUCTURAL CHECK OK (not evidence; no head-comparison quantity printed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
