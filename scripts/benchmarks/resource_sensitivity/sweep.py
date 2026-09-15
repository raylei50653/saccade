"""Reproduce paired serial/double-buffer SM-pressure sweeps with telemetry."""

# status: diagnostic
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha256(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 8, 16, 24, 32])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--pulse-us", type=int, default=5000)
    parser.add_argument("--preset", default="mamba_whole_graph_m")
    parser.add_argument("--sequences", default="MOT17-04-SDP")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=419)
    parser.add_argument("--arch", default="sm_120")
    args = parser.parse_args()
    if (
        args.repeats < 1
        or 0 not in args.levels
        or len(set(args.levels)) != len(args.levels)
        or min(args.levels) < 0
    ):
        parser.error(
            "unique nonnegative levels including zero and positive repeats required"
        )
    if not 100 <= args.pulse_us <= 100000:
        parser.error("pulse-us must be 100..100000")
    if args.max_frames != 0 and args.max_frames <= 50:
        parser.error("max-frames must exceed 50 warmup frames (or be zero for all)")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "0"):
        parser.error("device remapping is unsupported; use physical GPU 0")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        parser.error("CUDA_VISIBLE_DEVICES disables CUDA")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    source_snapshot = args.output / "execution_sources"
    source_snapshot.mkdir()
    for path in HERE.iterdir():
        if path.suffix in {".py", ".cu"}:
            shutil.copy2(path, source_snapshot / path.name)
    identity_command = [
        sys.executable,
        str(HERE / "input_identity.py"),
        "--preset",
        args.preset,
        "--sequences",
        args.sequences,
        "--max-frames",
        str(args.max_frames),
    ]
    subprocess.run(
        [
            *identity_command,
            "--output",
            str(args.output / "input_identity_before.json"),
        ],
        check=True,
        cwd=ROOT,
    )
    library = args.output / "blocker.so"
    build = [
        "nvcc",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-O2",
        f"-arch={args.arch}",
        str(HERE / "blocker.cu"),
        "-o",
        str(library),
    ]
    subprocess.run(build, check=True, cwd=ROOT)
    subprocess.run(
        [
            sys.executable,
            str(HERE / "green_probe.py"),
            "--output",
            str(args.output / "green_probe.json"),
        ],
        check=True,
        cwd=ROOT,
        timeout=60,
    )
    schedule = []
    rng = random.Random(args.seed)
    for rep in range(args.repeats):
        levels = list(args.levels)
        rng.shuffle(levels)
        for k in levels:
            modes = (
                ["serial", "double"]
                if (rep + args.levels.index(k)) % 2 == 0
                else ["double", "serial"]
            )
            schedule.extend({"rep": rep, "blocks": k, "mode": m} for m in modes)
    identity = {
        "schema": "saccade-resource-sweep-v1",
        "kind": "bounded_residency_pressure_proxy",
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "source_sha256": {
            p.name: sha256(p) for p in HERE.iterdir() if p.suffix in {".py", ".cu"}
        },
        "library_sha256": sha256(library),
        "build_command": build,
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
        "gpu": subprocess.check_output(["nvidia-smi", "-q"], text=True),
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("SACCADE_", "CUDA_", "NVIDIA_")) or k == "LD_LIBRARY_PATH"
        },
        "schedule": schedule,
    }
    (args.output / "sweep.json").write_text(json.dumps(identity, indent=2) + "\n")
    for entry in schedule:
        name = f"r{entry['rep']}_k{entry['blocks']}_{entry['mode']}"
        run_dir = args.output / name
        command = [
            sys.executable,
            "-u",
            str(HERE / "run_point.py"),
            "--library",
            str(library),
            "--blocks",
            str(entry["blocks"]),
            "--pulse-us",
            str(args.pulse_us),
            "--output",
            str(run_dir),
            "--",
            "--preset",
            args.preset,
            "--detector",
            "SDP",
            "--sequences",
            args.sequences,
            "--warmup-frames",
            "50",
            "--detect-barrier",
            "event",
        ]
        if args.max_frames:
            command += ["--max-frames", str(args.max_frames)]
        if entry["mode"] == "double":
            command += ["--double-buffer"]
        print(name, flush=True)
        started = time.time()
        with (
            (args.output / f"{name}.telemetry.csv").open("w") as telemetry_file,
            (args.output / f"{name}.log").open("w") as log,
        ):
            telemetry = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--id=0",
                    "--query-gpu=timestamp,uuid,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory",
                    "--format=csv",
                    "-lms",
                    "100",
                ],
                stdout=telemetry_file,
                stderr=subprocess.STDOUT,
            )
            try:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                    timeout=args.timeout,
                )
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code = 124
                log.write(f"\nResource sweep timeout after {args.timeout} seconds\n")
            finally:
                telemetry.terminate()
                telemetry.wait(timeout=10)
        entry.update(
            command=command,
            started_epoch=started,
            finished_epoch=time.time(),
            returncode=code,
        )
        (args.output / "sweep.json").write_text(json.dumps(identity, indent=2) + "\n")
        if code:
            raise RuntimeError(f"{name} failed ({code}); see its retained log")
    subprocess.run(
        [*identity_command, "--output", str(args.output / "input_identity_after.json")],
        check=True,
        cwd=ROOT,
    )
    before = json.loads((args.output / "input_identity_before.json").read_text())[
        "files"
    ]
    after = json.loads((args.output / "input_identity_after.json").read_text())["files"]
    if before != after:
        raise RuntimeError("input files changed during the sweep")
    subprocess.run(
        [sys.executable, str(HERE / "summarize.py"), str(args.output)],
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
