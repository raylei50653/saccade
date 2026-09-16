"""Freeze, execute and seal repeated full-pipeline mixed-workload comparisons."""

# status: diagnostic
import argparse
import itertools
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.benchmarks.resource_mixed.report import derive, digest  # noqa: E402
from scripts.benchmarks.resource_partition.sweep import build_libraries  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=350)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--policies", nargs="+", default=["fixed", "headroom", "shared", "dynamic"]
    )
    p.add_argument("--iterations", type=int, nargs="+", default=[2048, 8192])
    p.add_argument("--windows", type=int, nargs="+", default=[1, 4])
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for subdir in ("resource_mixed", "resource_partition", "resource_sensitivity"):
        for source in (ROOT / "scripts/benchmarks" / subdir).iterdir():
            if source.suffix in (".py", ".cpp", ".cu", ".md"):
                rel = source.relative_to(ROOT)
                dest = output / "sources" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                sources[str(rel)] = digest(dest)
    build_libraries(output, "sm_120")
    command = [
        "nvcc",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-O2",
        "-arch=sm_120",
        str(ROOT / "scripts/benchmarks/resource_mixed/elastic.cu"),
        "-lcuda",
        "-o",
        str(output / "elastic.so"),
    ]
    subprocess.run(command, check=True)
    identity_command = [
        sys.executable,
        str(ROOT / "scripts/benchmarks/resource_sensitivity/input_identity.py"),
        "--preset",
        "mamba_whole_graph_m",
        "--sequences",
        "MOT17-04-SDP",
        "--max-frames",
        str(args.frames),
    ]
    subprocess.run(
        [*identity_command, "--output", str(output / "input_before.json")], check=True
    )
    conditions = list(itertools.product(args.policies, args.iterations, args.windows))
    conditions.append(("control", 2048, 1))
    schedule = []
    rng = random.Random(431)
    for rep in range(args.repeats):
        ordered = conditions.copy()
        rng.shuffle(ordered)
        for policy, iterations, window in ordered:
            for audit in [True, False] if rep == 0 else [False]:
                schedule.append(
                    dict(
                        rep=rep,
                        policy=policy,
                        iterations=iterations,
                        window=window,
                        audit=audit,
                        name=f"r{rep}_{policy}_i{iterations}_w{window}"
                        + ("_audit" if audit else ""),
                    )
                )
    manifest = dict(
        schema="saccade-mixed-sweep-v1",
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        created_epoch=time.time(),
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        status=subprocess.check_output(["git", "status", "--short"], text=True),
        source_sha256=sources,
        library_sha256={p.name: digest(p) for p in output.glob("*.so")},
        gpu=subprocess.check_output(["nvidia-smi", "-q"], text=True),
        nvcc=subprocess.check_output(["nvcc", "--version"], text=True),
        native_build_command=command,
        schedule=schedule,
    )

    def save():
        (output / "sweep.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    for row in schedule:
        command = [
            sys.executable,
            "-u",
            str(ROOT / "scripts/benchmarks/resource_mixed/run_point.py"),
            "--policy",
            row["policy"],
            "--window",
            str(row["window"]),
            "--iterations",
            str(row["iterations"]),
            "--frames",
            str(args.frames),
            "--libraries",
            str(output),
            "--output",
            str(output / row["name"]),
        ]
        if row["audit"]:
            command.append("--audit")
        row.update(command=command, started_epoch=time.time())
        print(row["name"], flush=True)
        with (
            (output / (row["name"] + ".log")).open("w") as log,
            (output / (row["name"] + ".telemetry.csv")).open("w") as telemetry,
        ):
            monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,utilization.gpu",
                    "--format=csv",
                    "-lms",
                    "100",
                ],
                stdout=telemetry,
                stderr=subprocess.STDOUT,
            )
            try:
                process = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=300
                )
                row["returncode"] = process.returncode
            except subprocess.TimeoutExpired:
                row["returncode"] = 124
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
        row["finished_epoch"] = time.time()
        if row["returncode"] == 0:
            result = derive(output / row["name"])
            (output / row["name"] / "report.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            row["valid"] = result["valid"]
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "valid",
                            "failed_checks",
                            "service_pass",
                            "response_ms",
                        )
                    }
                ),
                flush=True,
            )
        row["files"] = {
            str(path.relative_to(output)): digest(path)
            for path in (output / row["name"]).rglob("*")
            if path.is_file()
        }
        save()
        if row["returncode"] or not row.get("valid"):
            raise RuntimeError(f"retained invalid run: {row['name']}")
        if any(digest(ROOT / path) != h for path, h in sources.items()):
            raise RuntimeError("execution source changed during sweep")
    subprocess.run(
        [*identity_command, "--output", str(output / "input_after.json")], check=True
    )
    if (
        json.loads((output / "input_before.json").read_text())["files"]
        != json.loads((output / "input_after.json").read_text())["files"]
    ):
        raise RuntimeError("input identity changed")
    manifest["finished_epoch"] = time.time()
    save()


if __name__ == "__main__":
    main()
