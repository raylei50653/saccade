"""Independently audit the frozen service target, arrivals and borrowing intervals."""

# status: diagnostic
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(directory):
    manifest = json.loads((directory / "sweep.json").read_text())
    if "finished_epoch" not in manifest:
        raise ValueError("unfinished experiment")
    totals = dict(
        timing_runs=0,
        audited_runs=0,
        stable_frames=0,
        deadline_misses=0,
        elastic_units=0,
        borrowed_units=0,
    )
    maximum_clock_bracket_us = 0
    per_run = []
    for entry in manifest["schedule"]:
        root = directory / entry["name"]
        for name, digest in entry["files"].items():
            with (directory / name).open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                    raise ValueError("sealed file changed")
        point = json.loads((root / "point.json").read_text())
        report = json.loads((root / "report.json").read_text())
        args = point["arguments"]
        if (
            args["frames"] != 350
            or args["units"] != 256
            or point["bursts"] != 50
            or args["period_ms"] != 1000 / 60
            or args["deadline_ms"] != 1000 / 60
        ):
            raise ValueError("frozen main contract differs")
        frames = point["frames"]
        if [r["frame"] for r in frames] != list(range(51, 351)):
            raise ValueError("missing stable work")
        response = np.array([(r["finished"] - r["arrival"]) * 1000 for r in frames])
        misses = int((response > 1000 / 60).sum())
        if not np.isfinite(response).all() or (response < 0).any():
            raise ValueError("invalid response")
        if (
            misses != report["miss_count"]
            or float(np.percentile(response, 99)) != report["response_ms"]["p99"]
        ):
            raise ValueError("latency replay differs")
        service = np.percentile(response, 99) <= 1000 / 60 and misses / 300 <= 0.01
        if service != report["service_pass"]:
            raise ValueError("service classification differs")
        bracket = point["native_anchor_after"] - point["origin"]
        if not 0 <= bracket <= 0.001:
            raise ValueError("host clock bracket failed")
        maximum_clock_bracket_us = max(maximum_clock_bracket_us, bracket * 1e6)
        borrowed_count = 0
        requests_with_pending = 0
        if args["policy"] != "control":
            raw = np.fromfile(root / "elastic.records", dtype="<u8").reshape(-1, 4)
            if len(raw) != 12800 or not np.array_equal(
                raw[:, 0], np.repeat(np.arange(50), 256)
            ):
                raise ValueError("burst conservation failed")
            if args["policy"] == "dynamic":
                borrowed = raw[raw[:, 1] == 1]
                borrowed_count = len(borrowed)
                enqueue_lower = (
                    point["origin"]
                    + (borrowed[:, 2].astype(np.float64) - point["native_origin"]) / 1e9
                )
                enqueue_upper = enqueue_lower + bracket
                completion_lower = (
                    point["origin"]
                    + (borrowed[:, 3].astype(np.float64) - point["native_origin"]) / 1e9
                )
                for frame in frames:
                    # Reject definite admissions inside the protected interval;
                    # boundary ambiguity is covered by the GPU non-overlap twin.
                    if (
                        (enqueue_lower >= frame["drained"])
                        & (enqueue_upper <= frame["finished"])
                    ).any():
                        raise ValueError("borrow admission during stable service")
                    requests_with_pending += int(
                        (
                            (enqueue_upper < frame["request"])
                            & (completion_lower > frame["request"])
                        ).any()
                    )
            if not entry["audit"]:
                totals["elastic_units"] += len(raw)
                totals["borrowed_units"] += borrowed_count
        if entry["audit"]:
            totals["audited_runs"] += 1
        else:
            totals["timing_runs"] += 1
            totals["stable_frames"] += len(frames)
            totals["deadline_misses"] += misses
        per_run.append(
            dict(
                name=entry["name"],
                service_pass=bool(service),
                requests_with_observed_pending_borrow=requests_with_pending,
                borrowed_units=borrowed_count,
            )
        )
    return dict(
        verified=True,
        scope="frozen main contract and raw host admission intervals; CUPTI replay is in report.py",
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
