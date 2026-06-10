"""Bit-exact golden regression gate for run_eval refactors.

run_eval (src/saccade/perception/eval/evaluator.py) is a ~3900-line function
that produces the graded MOT output. Any refactor of it must leave that output
byte-identical. Eval is deterministic (verified: two runs of the whole-graph
preset produce identical .txt), so we can gate refactors on an exact diff.

Workflow:
    # once, on the PRE-refactor commit — record the golden output
    .venv/bin/python scripts/tools/eval_golden.py capture

    # after each refactor step — must print "PASS" for every sequence
    .venv/bin/python scripts/tools/eval_golden.py check

Covers a couple of short sequences under the production whole-graph preset.
Add presets/sequences to CASES to widen branch coverage as the refactor moves
into other paths.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = ROOT / "tests/golden/eval"

# (label, preset, sequence, max_frames). Each runs the real CLI end-to-end.
# Cover distinct detect/postprocess branches so refactors of either are gated:
#   - mamba_whole_graph: whole-detect CUDA graph (backbone+head+decode fused)
#   - mamba_optimal:      head-only graph + the separate eager postprocess block
CASES: list[tuple[str, str, str, int]] = [
    ("whole_graph_mot02", "mamba_whole_graph", "MOT17-02-SDP", 120),
    ("whole_graph_mot04", "mamba_whole_graph", "MOT17-04-SDP", 120),
    ("optimal_mot02", "mamba_optimal", "MOT17-02-SDP", 120),
]


def _run_case(preset: str, seq: str, max_frames: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts/eval/mot17.py"),
        "--preset",
        preset,
        "--sequences",
        seq,
        "--output",
        str(out_dir),
        "--max-frames",
        str(max_frames),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    txt = out_dir / f"{seq}.txt"
    if res.returncode != 0 or not txt.exists():
        sys.stderr.write(res.stdout[-2000:] + res.stderr[-2000:])
        raise RuntimeError(f"eval failed for {preset}/{seq} (exit {res.returncode})")
    return txt


def capture() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for label, preset, seq, n in CASES:
        txt = _run_case(preset, seq, n, GOLDEN_DIR / "_capture" / label)
        dst = GOLDEN_DIR / f"{label}.txt"
        dst.write_bytes(txt.read_bytes())
        print(f"captured {label}: {sum(1 for _ in dst.open())} lines → {dst}")


def check() -> int:
    failed = 0
    for label, preset, seq, n in CASES:
        golden = GOLDEN_DIR / f"{label}.txt"
        if not golden.exists():
            print(f"{label}: NO GOLDEN (run `capture` first)")
            failed += 1
            continue
        txt = _run_case(preset, seq, n, GOLDEN_DIR / "_check" / label)
        if txt.read_bytes() == golden.read_bytes():
            print(f"{label}: PASS ({sum(1 for _ in golden.open())} lines)")
        else:
            failed += 1
            g = golden.read_text().splitlines()
            c = txt.read_text().splitlines()
            print(f"{label}: ❌ FAIL  golden={len(g)} lines, new={len(c)} lines")
            for i, (a, b) in enumerate(zip(g, c)):
                if a != b:
                    print(
                        f"    first diff @ line {i + 1}:\n      golden: {a}\n      new:    {b}"
                    )
                    break
    print("\n" + ("ALL PASS" if failed == 0 else f"{failed} CASE(S) FAILED"))
    return 1 if failed else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "check"])
    args = ap.parse_args()
    if args.mode == "capture":
        capture()
    else:
        sys.exit(check())


if __name__ == "__main__":
    main()
