#!/usr/bin/env python3
"""Exercise the #465 localization r2 arm worker end to end on synthetic frames (no MOT17 data).

The r2 localization declaration (§9) allows the runner PR only unit tests and
structural checks that read no MOT17 frame; the formal run executes once and
cannot be retried. This check runs the real ``run_arm_worker`` -- patched
detector builder, ``ReplayDetect`` with the S(f) composition and V3 checks, stage probes, the
``_run_emit`` observer and an unmodified ``mot17.py`` -- on generated
sequences in a temporary data root, then checks the worker's coverage:

* one replay call per frame, in frame order (the harness's ``detector_output``
  rows for frame k+1 equal the replay's k-th output, bit for bit);
* every frame V3-checked for hybrid arms;
* ``post_nms``/``tracker_input`` probed and one tracker output emitted on the
  same frames (the harness skips them on frames with no detection above the
  floor); tracker outputs carry ``det_idx``;
* the harness's run_manifest cmdline equals the worker's mot17.py argv;
* with ``--arm all``: a second ``R_T`` run reproduces the txt bytes and every
  evidence array (V2 rehearsal), the §6 report code runs to completion, every
  ``tracker_input`` row maps back to a replay row by full-row match, and every
  frame without a ``tracker_input`` probe has the verified-skip signature.

It prints no head-comparison quantity and writes nothing under ``results/``;
the synthetic images carry no information about the study's data. Its output
is not evidence. The synthetic frames need not truncate the top-300 above
the floor, so S(f) closure on truncated frames is covered by the unit tests
(``tests/unit/test_native_head_failure_localization_r2_runner.py``), not
here. Usage (GPU, a few minutes for torch.compile)::

    .venv/bin/python tools/resctl.py run gpu0 -- .venv/bin/python \\
        scripts/eval/diagnostics/native_head_failure_localization_r2_structural_check.py
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
RUNNER = (
    project_root / "scripts/eval/diagnostics/native_head_failure_localization_r2.py"
)
SYNTHETIC = (("SYN-01-SDP", 1920, 1080), ("SYN-02-SDP", 640, 480))
N_FRAMES = 12


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("hfl_r2_runner", RUNNER)
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


def coverage_problems(
    summary: dict[str, Any], arm: str, hybrid: tuple[str, ...]
) -> list[str]:
    problems = []
    for name, _, _ in SYNTHETIC:
        s = summary["sequences"].get(name)
        if s is None:
            problems.append(f"{name}: no evidence")
            continue
        checks = {
            "replay calls == frames": s["replay_frames"] == N_FRAMES,
            "detector_output == replay frame k+1": s[
                "detector_output_equals_replay_frame_k_plus_1"
            ]
            == N_FRAMES,
            # The harness skips post_nms/tracker_input on frames with no
            # detection above the floor, so only equality between the stages
            # and a non-empty stream on the textured sequence are invariants.
            "post_nms frames == tracker_input frames": s["probe_frames"]["post_nms"]
            == s["probe_frames"]["tracker_input"],
            "one tracker output per tracker_input frame": s["tracker_frames"]
            == s["probe_frames"]["tracker_input"],
            "tracker_input observed (first sequence)": name != SYNTHETIC[0][0]
            or s["probe_frames"]["tracker_input"] > 0,
            "det_idx present": s["det_idx_missing_frames"] == 0,
            "V3 checked every frame (hybrid)": (arm not in hybrid)
            or s["v3_frames_checked"] == N_FRAMES,
        }
        problems += [f"{name}: {k} ({s})" for k, ok in checks.items() if not ok]
    return problems


def determinism_problems(a: Path, b: Path, seqs: tuple[str, ...]) -> list[str]:
    import numpy as np

    problems = []
    for seq in seqs:
        if (a / "out" / f"{seq}.txt").read_bytes() != (
            b / "out" / f"{seq}.txt"
        ).read_bytes():
            problems.append(f"V2 rehearsal: R_T {seq} txt differs between two runs")
        with (
            np.load(a / "out.evidence" / f"{seq}.npz") as za,
            np.load(b / "out.evidence" / f"{seq}.npz") as zb,
        ):
            for k in za.files:
                if not np.array_equal(za[k], zb[k]):
                    problems.append(f"V2 rehearsal: R_T {seq} evidence {k} differs")
    return problems


def run_one(runner: Any, arm: str, work: Path) -> list[str]:
    """Run one arm's worker in its own process; return coverage problems."""
    proc = subprocess.run(
        [sys.executable, __file__, "--arm", arm, "--_child", str(work)],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-4000:], proc.stderr[-4000:], sep="\n")
        return [f"{arm}: worker exit {proc.returncode}"]
    summary = json.loads((work / "out.evidence" / "worker.json").read_text())
    problems = [
        f"{arm}: {p}" for p in coverage_problems(summary, arm, runner.HYBRID_ARMS)
    ]
    manifest = json.loads((work / "out" / "run_manifest.json").read_text())
    if manifest.get("cmdline") != summary["mot17_argv"]:
        problems.append(f"{arm}: run_manifest cmdline != worker argv")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arm",
        default="all",
        help="one arm, or 'all' (every arm once, then the §6 report code)",
    )
    parser.add_argument("--_child", dest="child", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    runner = _load_runner()
    arms = runner.ARMS if args.arm == "all" else (args.arm,)
    if any(a not in runner.ARM_SOURCES for a in arms):
        parser.error(f"--arm must be 'all' or one of {list(runner.ARM_SOURCES)}")
    seqs = tuple(n for n, _, _ in SYNTHETIC)

    if args.child:
        work = Path(args.child)
        return runner.run_arm_worker(
            args.arm,
            work / "out",
            work / "out.evidence",
            seqs,
            data_root=str(work / "data"),
        )

    with tempfile.TemporaryDirectory(prefix="hfl_r2_structural_") as tmp:
        root = Path(tmp)
        write_synthetic_dataset(root / "data")
        assert "MOT17" not in str(root / "data")
        problems: list[str] = []
        by = {}
        for arm in arms:
            work = root / arm
            work.mkdir()
            (work / "data").symlink_to(root / "data")
            problems += run_one(runner, arm, work)
            by[(arm, 1)] = {
                "evidence_dir": str(work / "out.evidence"),
                "output_dir": str(work / "out"),
            }
        if args.arm == "all" and not problems:
            # V2 rehearsal: a second R_T run must reproduce txt bytes and census.
            again = root / "R_T_again"
            again.mkdir()
            (again / "data").symlink_to(root / "data")
            problems += run_one(runner, "R_T", again)
            problems += determinism_problems(root / "R_T", again, seqs)
        if args.arm == "all" and not problems:
            # Exercise the §6 analysis end to end; print only whether it ran.
            try:
                report = runner._report(by, seqs)
                json.dumps(report)
                # The harness passes detector rows to the tracker unchanged, so
                # every tracker_input row must map back by full-row match (a
                # coverage property of the observer, not a head comparison).
                for key in ("delta_flow_R_T", "swap_flow_R_T", "delta_e_flow_R_E"):
                    for seq, flow in report[key].items():
                        if flow["tracker_input_rows"]["unmapped"]:
                            problems.append(f"{key} {seq}: unmapped tracker_input rows")
                        if flow["tracker_input_missing_frames"]:
                            problems.append(
                                f"{key} {seq}: missing tracker_input frames"
                            )
            except Exception as exc:  # noqa: BLE001 — reported as a failure
                problems.append(f"§6 report raised {exc!r}")
        for p in problems:
            print(f"FAIL: {p}")
        if not problems:
            print(
                f"OK: arms {list(arms)}, {len(seqs)} synthetic sequences x {N_FRAMES} frames"
            )
        return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
