"""Reproduce paired serial/double-buffer sweeps over verified Green Context SM budgets."""

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
PROXY = HERE.parent / "resource_sensitivity"
CUDA_INCLUDE = Path("/opt/cuda/targets/x86_64-linux/include")
CUDA_LIB = Path("/opt/cuda/targets/x86_64-linux/lib")


def sha256(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def parse_level(value):
    if value == "full":
        return "full"
    count = int(value)
    if count <= 0:
        raise argparse.ArgumentTypeError("levels are positive SM counts or 'full'")
    return count


def build_libraries(output, arch):
    audit = output / "partition_audit.so"
    probe = output / "smid_probe.so"
    audit_command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-shared",
        "-fPIC",
        "-Wall",
        "-Wextra",
        f"-I{CUDA_INCLUDE}",
        str(HERE / "partition_audit.cpp"),
        f"-L{CUDA_LIB}",
        f"-Wl,-rpath,{CUDA_LIB}",
        "-lcupti",
        "-o",
        str(audit),
    ]
    probe_command = [
        "nvcc",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-O2",
        f"-arch={arch}",
        str(HERE / "smid_probe.cu"),
        "-o",
        str(probe),
    ]
    subprocess.run(audit_command, check=True, cwd=ROOT)
    subprocess.run(probe_command, check=True, cwd=ROOT)
    return audit, probe, {"audit": audit_command, "probe": probe_command}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--levels",
        nargs="+",
        type=parse_level,
        default=["full", 46, 40, 32, 24, 16, 8],
        help="requested SM counts for Green Contexts, plus 'full' for the "
        "primary full-device baseline routed through the same owner",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--preset", default="mamba_whole_graph_m")
    parser.add_argument("--sequences", default="MOT17-04-SDP")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=419)
    parser.add_argument("--arch", default="sm_120")
    args = parser.parse_args()
    if (
        args.repeats < 1
        or "full" not in args.levels
        or len(set(args.levels)) != len(args.levels)
    ):
        parser.error("unique levels including 'full' and positive repeats required")
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
        if path.suffix in {".py", ".cu", ".cpp"}:
            shutil.copy2(path, source_snapshot / path.name)
    identity_command = [
        sys.executable,
        str(PROXY / "input_identity.py"),
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
    audit, probe, build_commands = build_libraries(args.output, args.arch)
    subprocess.run(
        [
            sys.executable,
            str(PROXY / "green_probe.py"),
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
        for level in levels:
            modes = (
                ["serial", "double"]
                if (rep + args.levels.index(level)) % 2 == 0
                else ["double", "serial"]
            )
            # Every point runs twice with an identical command: an audited twin
            # whose CUPTI trace proves the routing, then the unaudited
            # measurement whose timing is free of CUPTI per-kernel overhead.
            for m in modes:
                schedule.append({"rep": rep, "level": level, "mode": m, "audit": True})
                schedule.append({"rep": rep, "level": level, "mode": m, "audit": False})
    identity = {
        "schema": "saccade-partition-sweep-v1",
        "kind": "green_context_true_partition",
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
            p.name: sha256(p)
            for p in HERE.iterdir()
            if p.suffix in {".py", ".cu", ".cpp"}
        },
        "library_sha256": {"audit": sha256(audit), "probe": sha256(probe)},
        "build_commands": build_commands,
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
        name = point_name(entry)
        run_dir = args.output / name
        command = [
            sys.executable,
            "-u",
            str(HERE / "run_point.py"),
            "--sm-count",
            str(entry["level"]),
            "--probe-library",
            str(probe),
            "--output",
            str(run_dir),
        ]
        if entry["audit"]:
            command[-4:-4] = ["--audit-library", str(audit)]
        command += [
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
                log.write(f"\nPartition sweep timeout after {args.timeout} seconds\n")
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


def point_name(entry):
    audit = "_audit" if entry["audit"] else ""
    return f"r{entry['rep']}_sm{entry['level']}_{entry['mode']}{audit}"


if __name__ == "__main__":
    main()
