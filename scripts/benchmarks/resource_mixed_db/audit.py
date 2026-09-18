"""Independently audit the frozen DB target, periods and borrow protection."""

# status: diagnostic
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(directory):
    manifest = json.loads((directory / "sweep.json").read_text())
    if manifest.get("schema") != "saccade-mixed-db-sweep-v2":
        raise ValueError("wrong sweep schema")
    if "finished_epoch" not in manifest:
        raise ValueError("unfinished experiment")
    totals = dict(
        timing_runs=0,
        audited_runs=0,
        stable_frames=0,
        stable_periods=0,
        deadline_misses=0,
        elastic_units=0,
        borrowed_units=0,
        borrowed_completed_before_cutoff=0,
    )
    maximum_clock_bracket_us = 0.0
    per_run = []
    for entry in manifest["schedule"]:
        root = directory / entry["name"]
        for name, value in entry["files"].items():
            with (directory / name).open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != value:
                    raise ValueError("sealed file changed")
        point = json.loads((root / "point.json").read_text())
        report = json.loads((root / "report.json").read_text())
        args = point["arguments"]
        if (
            args["frames"] != 350
            or args["units"] != 256
            or args["bursts"] != 20
            or args["burst_period_ms"] != 50.0
            or args["deadline_ms"] != 20.0
        ):
            raise ValueError("frozen main contract differs")
        deadline = args["deadline_ms"]
        frames = point["frames"]
        if [row["frame"] for row in frames] != list(range(51, 351)):
            raise ValueError("missing stable work")
        if not all(
            row["start"] <= row["cycle_finished"] <= row["output"] for row in frames
        ):
            raise ValueError("invalid stable timestamp order")
        latency = np.array([(row["output"] - row["start"]) * 1000 for row in frames])
        periods = np.diff([row["output"] for row in frames]) * 1000
        if (
            not np.isfinite(latency).all()
            or not np.isfinite(periods).all()
            or (latency < 0).any()
            or (periods < 0).any()
        ):
            raise ValueError("invalid latency or period")
        misses = int((latency > deadline).sum())
        service = np.percentile(latency, 99) <= deadline and misses / 300 <= 0.01
        if (
            misses != report["miss_count"]
            or float(np.percentile(latency, 99)) != report["latency_ms"]["p99"]
            or float(np.percentile(periods, 99)) != report["period_ms"]["p99"]
            or service != report["service_pass"]
        ):
            raise ValueError("stable metric replay differs")
        transitions = point["transitions"]
        windows = point["windows"]
        in_service = [row for row in windows if row["within_stable_service"]]
        tail = [row for row in windows if not row["within_stable_service"]]
        if (
            [row["scheduled_frame"] for row in transitions] != list(range(51, 351))
            or len(in_service) != 299
            or len(tail) != (0 if args["policy"] == "control" else 1)
        ):
            raise ValueError("transition/window coverage differs")
        bracket = point["native_anchor_after"] - point["origin"]
        if not 0 <= bracket <= 0.001:
            raise ValueError("host clock bracket failed")
        maximum_clock_bracket_us = max(maximum_clock_bracket_us, bracket * 1e6)
        borrowed = borrowed_before = 0
        if args["policy"] != "control":
            raw = np.fromfile(root / "elastic.records", dtype="<u8").reshape(-1, 4)
            if len(raw) != 5120 or not np.array_equal(
                raw[:, 0], np.repeat(np.arange(20), 256)
            ):
                raise ValueError("burst conservation failed")
            cutoff = (
                point["native_origin"]
                + (frames[-1]["output"] - point["native_anchor_after"]) * 1e9
            )
            if point["native_origin"] + 19 * 50_000_000 > cutoff:
                raise ValueError("elastic load released after stable service")
            lanes = point["elastic_pool"]["lanes"]
            expected_lanes = 3 if args["policy"] == "dynamic" else 2
            if len(lanes) != expected_lanes or [lane["borrowed"] for lane in lanes] != [
                False
            ] * 2 + [True] * (expected_lanes - 2):
                raise ValueError("lane construction differs")
            if raw[:, 1].max() >= expected_lanes:
                raise ValueError("record on an unknown lane")
            if args["policy"] == "dynamic":
                borrowed_rows = raw[raw[:, 1] == 2]
                borrowed = len(borrowed_rows)
                borrowed_before = int((borrowed_rows[:, 3] <= cutoff).sum())
                enqueue_lower = (
                    point["origin"]
                    + (borrowed_rows[:, 2].astype(np.float64) - point["native_origin"])
                    / 1e9
                )
                enqueue_upper = enqueue_lower + bracket
                releases = [*in_service, *tail]
                protected = [
                    (transition["admitted"], release["release_requested"])
                    for transition, release in zip(transitions, releases, strict=True)
                ]
                for lower, upper in zip(enqueue_lower, enqueue_upper, strict=True):
                    if any(
                        lower >= start and upper <= stop for start, stop in protected
                    ):
                        raise ValueError("borrow admission during protected DB service")
            if not entry["audit"]:
                totals["elastic_units"] += len(raw)
                totals["borrowed_units"] += borrowed
                totals["borrowed_completed_before_cutoff"] += borrowed_before
        if entry["audit"]:
            totals["audited_runs"] += 1
        else:
            totals["timing_runs"] += 1
            totals["stable_frames"] += len(frames)
            totals["stable_periods"] += len(periods)
            totals["deadline_misses"] += misses
        per_run.append(
            dict(
                name=entry["name"],
                service_pass=bool(service),
                borrowed_units=borrowed,
                borrowed_completed_before_cutoff=borrowed_before,
            )
        )
    return dict(
        verified=True,
        scope=(
            "frozen DB contract, production completion periods and raw host/native "
            "borrow intervals; CUPTI placement/overlap replay is in report.py"
        ),
        totals=totals,
        max_host_clock_bracket_us=maximum_clock_bracket_us,
        per_run=per_run,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = audit(args.directory)
    (args.output or args.directory / "audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result["totals"]))


if __name__ == "__main__":
    main()
