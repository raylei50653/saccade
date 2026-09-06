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
    here = Path(__file__).resolve().parent
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
            "--",
            str(here / "control.py"),
            case,
        ]
        child = subprocess.run(command, capture_output=True, text=True, timeout=60)
        problems = []
        if child.returncode != 0:
            problems.append(f"child_exit:{child.returncode}:{child.stderr}")
        report = analyze(root)
        problems.extend(report["problems"])
        captures = report["captures"]
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
