"""Fixed six-case observer qualification. This never invokes production evaluation."""

# status: diagnostic

import argparse
import json
from pathlib import Path
import subprocess
import sys

from analyze import analyze


CASES = (
    "blocking-runtime",
    "nonblocking-runtime",
    "blocking-driver",
    "nonblocking-driver",
    "blocking-joined",
    "python",
)


def qualify(observer: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    analysis_output = output / "analysis"
    analysis_output.mkdir()
    here = Path(__file__).resolve().parent
    helper = observer.resolve().with_name("control_owner.so")
    build_manifest = observer.resolve().with_name("build.json")
    if not helper.is_file() or not build_manifest.is_file():
        raise FileNotFoundError(
            "observer qualification requires sibling control_owner.so and build.json"
        )
    results = {}
    for case in CASES:
        root = output / case
        command = [
            sys.executable,
            str(here / "run.py"),
            "--observer",
            str(observer.resolve()),
            "--output",
            str(root.resolve()),
            "--asset",
            str(build_manifest),
            "--asset",
            str(helper),
            "--",
            str(here / "control.py"),
            case,
        ]
        if case != "python":
            command.extend(("--helper", str(helper)))
        child = subprocess.run(command, capture_output=True, text=True, timeout=60)
        problems = []
        if child.returncode != 0:
            problems.append(f"child_exit:{child.returncode}:{child.stderr}")
        report = analyze(root)
        (analysis_output / f"{case}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        problems.extend(report["problems"])
        control_result = None
        if case == "blocking-runtime":
            try:
                parsed = json.loads((root / "stdout.log").read_text())
            except (OSError, json.JSONDecodeError) as exc:
                problems.append(f"recreate_control_result_unreadable:{exc}")
            else:
                if isinstance(parsed, dict):
                    control_result = parsed
                else:
                    problems.append("recreate_control_result_not_an_object")
        captures = report["captures"]
        lifetimes = report["stream_lifetimes"]
        if case == "python":
            labels = {(c["label"], c["mode"]) for c in captures}
            if labels != {
                ("detector.whole", 0),
                ("nms.main_nocopyback", 1),
                ("gmc.direct", 1),
            }:
                problems.append(f"python_site_or_mode_mismatch:{labels}")
            if any(c["flags"] != 1 for c in captures):
                problems.append("python_capture_not_nonblocking")
        else:
            positive = case.startswith("blocking")
            implicit = [e for e in report["capture_errors"] if e["error"]["rc"] == 906]
            if bool(implicit) != positive:
                problems.append("implicit_positive_negative_mismatch")
            if positive and not all(
                "IsCapturing" in e["error"]["api"] for e in implicit
            ):
                problems.append("wrong_failing_api")
            if positive and not all(
                any(c["tid"] != e["error"]["tid"] for c in e["observed_open_captures"])
                for e in implicit
            ):
                problems.append("missing_cross_thread_capture_overlap")
            if (
                case.endswith("driver")
                and "cudaStreamBeginCapture" in report["api_counts"]
            ):
                problems.append("driver_function_pointer_control_used_runtime_begin")
            expected_flags = 0 if positive and case != "blocking-joined" else 1
            if any(c["flags"] != expected_flags or c["end_rc"] != 0 for c in captures):
                problems.append("flags_or_end_result_mismatch")
            owner_lifetimes = [
                lifetime
                for lifetime in lifetimes
                if lifetime["owner"].get("status") == "resolved"
                and lifetime["owner"]["frame"]["module"]["path"] == str(helper)
            ]
            expected_creation = {
                "blocking-runtime": (
                    "cudaStreamCreate",
                    "attribution_control_cuda_create_default",
                    None,
                ),
                "nonblocking-runtime": (
                    "cudaStreamCreateWithPriority",
                    "attribution_control_cuda_create_priority",
                    0,
                ),
                "blocking-driver": (
                    "cuStreamCreate",
                    "attribution_control_cu_create_flags",
                    None,
                ),
                "nonblocking-driver": (
                    "cuStreamCreateWithPriority",
                    "attribution_control_cu_create_priority",
                    0,
                ),
                "blocking-joined": (
                    "cudaStreamCreateWithFlags",
                    "attribution_control_cuda_create_flags",
                    None,
                ),
            }[case]
            if not owner_lifetimes or any(
                lifetime["logical_creation_api"] != expected_creation[0]
                or lifetime["owner"]["frame"].get("symbol") != expected_creation[1]
                or lifetime["priority"] != expected_creation[2]
                for lifetime in owner_lifetimes
            ):
                problems.append("creation_stack_or_variant_attribution_mismatch")
            if case.endswith("driver") and any(
                lifetime["logical_creation_api"].startswith("cuda")
                for lifetime in owner_lifetimes
            ):
                problems.append("driver_control_used_runtime_creation")
            if case == "blocking-runtime":
                recreate = (
                    control_result.get("recreate")
                    if isinstance(control_result, dict)
                    else None
                )
                recreate_valid = (
                    isinstance(recreate, dict)
                    and type(recreate.get("first")) is int
                    and recreate["first"] > 0
                    and type(recreate.get("second")) is int
                    and recreate["second"] > 0
                    and type(recreate.get("same_handle")) is bool
                    and recreate["same_handle"]
                    == (recreate["first"] == recreate["second"])
                )
                if not recreate_valid:
                    problems.append("recreate_control_result_invalid")
                else:
                    ordered = sorted(
                        owner_lifetimes, key=lambda item: item["created_ns"]
                    )
                    lifetimes_valid = (
                        len(ordered) == 2
                        and ordered[0]["stream"] == recreate["first"]
                        and ordered[1]["stream"] == recreate["second"]
                        and all(item["destroy"] is not None for item in ordered)
                        and ordered[0]["destroy"]["exit_ns"] < ordered[1]["created_ns"]
                    )
                    if not lifetimes_valid:
                        problems.append("destroy_recreate_lifetime_mismatch")
                    elif recreate["same_handle"] and [
                        item["generation"] for item in ordered
                    ] != [1, 2]:
                        problems.append("destroy_recreate_generation_mismatch")
            if case == "blocking-joined":
                if not report["event_edges"] or not any(
                    s["status"] == 1 and s["flags"] == 0
                    for s in report["stream_status_observations"]
                ):
                    problems.append("missing_blocking_participant_or_event_edges")
        results[case] = {
            "passed": not problems,
            "problems": problems,
            "capture_rows": len(captures),
            "capture_error_rows": len(report["capture_errors"]),
            "stream_lifetime_rows": len(lifetimes),
            "ownership_evidence_ok": report["ownership_evidence_ok"],
        }
        print(json.dumps({case: results[case]}), flush=True)
    (output / "qualification.json").write_text(json.dumps(results, indent=2) + "\n")
    if not all(r["passed"] for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.observer, args.output)
