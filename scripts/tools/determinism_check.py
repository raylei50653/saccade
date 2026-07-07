"""Determinism gate: run N evals per config and verify byte-identical output.

Usage:
    # capture goldens with legacy (baseline) config
    .venv/bin/python scripts/tools/determinism_check.py capture

    # check determinism for all configs (6 runs each)
    .venv/bin/python scripts/tools/determinism_check.py check
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = ROOT / "tests/golden/determinism"
N_RUNS = 6

# (label, preset, sequence, max_frames)
CASES: list[tuple[str, str, str, int]] = [
    ("mamba_whole_mot04", "mamba_whole_graph", "MOT17-04-SDP", 120),
    ("mamba_opt_mot02", "mamba_optimal", "MOT17-02-SDP", 120),
    ("mamba_speed_mot02", "speed", "MOT17-02-SDP", 120),
]


@dataclasses.dataclass(frozen=True)
class Config:
    name: str
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    cli_args: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        if not self.cli_args:
            return self.name
        return f"{self.name}({','.join(self.cli_args)})"


# Configurations under test.
# detect_post_event and double-buffer should match legacy bit-exact.
# SACCADE_STREAM_MODE is passed via env (no CLI flag for it yet).
# SACCADE_DOUBLE_BUFFER must be set via --double-buffer because
# configure_runtime_env (mot17_args.py:152) unconditionally overwrites the
# env var.
CONFIGS: list[Config] = [
    Config("legacy"),
    Config("detect_post_event", env={"SACCADE_STREAM_MODE": "detect_post_event"}),
    Config(
        "detect_post_event_doublebuf",
        env={"SACCADE_STREAM_MODE": "detect_post_event"},
        cli_args=("--double-buffer",),
    ),
]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _run_case(
    preset: str,
    seq: str,
    max_frames: int,
    out_dir: Path,
    *,
    env: dict[str, str] | None = None,
    cli_args: tuple[str, ...] = (),
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

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
    if cli_args:
        cmd.extend(cli_args)
    res = subprocess.run(cmd, capture_output=True, text=True, env=run_env)
    txt = out_dir / f"{seq}.txt"
    if res.returncode != 0 or not txt.exists():
        tail = res.stdout[-2000:] + res.stderr[-2000:]
        sys.stderr.write(tail)
        raise RuntimeError(f"eval failed for {preset}/{seq} (exit {res.returncode})")
    return txt


def capture() -> None:
    """Capture legacy goldens (run once before refactoring)."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for label, preset, seq, n in CASES:
        txt = _run_case(preset, seq, n, GOLDEN_DIR / "_capture" / label / "legacy")
        dst = GOLDEN_DIR / f"{label}_legacy.txt"
        dst.write_bytes(txt.read_bytes())
        n_lines = sum(1 for _ in dst.open())
        print(f"captured {label}_legacy: {n_lines} lines → {dst}")


def check() -> int:
    failed = 0

    for label, preset, seq, n in CASES:
        for cfg in CONFIGS:
            hashes: list[str] = []
            for run_i in range(1, N_RUNS + 1):
                out_dir = GOLDEN_DIR / "_check" / label / cfg.name / f"run{run_i}"
                try:
                    txt = _run_case(
                        preset,
                        seq,
                        n,
                        out_dir,
                        env=cfg.env or None,
                        cli_args=cfg.cli_args,
                    )
                    hashes.append(_md5(txt))
                except RuntimeError as e:
                    print(f"  {label}/{cfg.label}/run{run_i}: ERROR: {e}")
                    hashes.append("ERROR")
                    break

            unique = set(h for h in hashes if h not in ("ERROR",))
            n_ok = len([h for h in hashes if h not in ("ERROR",)])

            if len(unique) <= 1 and n_ok == N_RUNS:
                print(
                    f"  {label}/{cfg.label}: PASS ({N_RUNS}/{N_RUNS} identical, "
                    f"md5={list(unique)[0][:12] if unique else 'N/A'})"
                )
            else:
                failed += 1
                print(
                    f"  {label}/{cfg.label}: ❌ FAIL — "
                    f"{len(unique)} distinct hashes across {n_ok} runs"
                )
                for h in hashes:
                    flag = "❌" if hashes.count(h) < N_RUNS else "✓"
                    print(f"    {flag} {h}")

    print(f"\n{'ALL PASS' if failed == 0 else f'{failed} CASE(S) FAILED'}")
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
