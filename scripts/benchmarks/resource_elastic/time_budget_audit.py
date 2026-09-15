"""Independent vectorized ledger/timing audit; does not import producer code."""

# status: diagnostic
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    if not __debug__:
        raise RuntimeError("audit requires Python assertions; do not use -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.archive
    result = json.loads((root / "result.json").read_text())
    assert result["status"] == "validated_synthetic_time_budget"
    groups = {}
    trials = 0
    for summary in result["summary"]:
        key = f"{summary['route']}/" + (
            f"W{summary['window']}"
            if summary["window"]
            else f"B{summary['budget_ns'] / 1e6:g}"
        )
        g = groups.setdefault(
            key,
            dict(
                arrivals=0,
                over_budget=0,
                over_estimate=0,
                exhausted=0,
                later_rolling=0,
                later_arrivals=0,
                max_drain_ms=0.0,
                max_overshoot_ms=0.0,
                max_scheduled_ms=0.0,
                max_unretired=0,
                max_unit_ratio=0.0,
                units=0,
                unit_exceed=0,
            ),
        )
        with np.load(root / summary["raw"]) as raw:
            ch = raw["chunk_host"]
            a = raw["arrivals"]
            s = raw["stamps"][:, 512:].reshape(-1, 256, 1024, 3)
            estimate = ch[:, :, 3]
            # j is already admitted and not observed retired when i is submitted.
            i = np.arange(256)[:, None]
            j = np.arange(256)[None, :]
            live = (j <= i)[None, :, :] & (ch[:, None, :, 2] > ch[:, :, None, 0])
            ledger = (live * estimate[:, None, :]).sum(2)
            assert np.array_equal(ledger, ch[:, :, 4])
            if summary["window"]:
                assert live.sum(2).max() <= summary["window"]
            else:
                assert ledger.max() <= summary["budget_ns"]
            drain = (a[:, :, 4] - a[:, :, 1]) / 1e6
            response = (a[:, :, 3] - a[:, :, 1]) / 1e6
            scheduled = (a[:, :, 3] - a[:, :, 0]) / 1e6
            overshoot = (a[:, :, 1] - a[:, :, 0]) / 1e6
            duration = s[:, :, :, 1].max(2) - s[:, :, :, 0].min(2)
            for metric, values in [
                ("stable_response_ms", response),
                ("drain_observed_ms", drain),
                ("stable_scheduled_response_ms", scheduled),
                ("arrival_overshoot_ms", overshoot),
            ]:
                assert summary[metric] == dict(
                    zip(
                        ("p50", "p95", "p99"),
                        np.percentile(values, [50, 95, 99]).tolist(),
                    )
                )
            g["arrivals"] += a.shape[0] * 4
            g["over_budget"] += (
                int(np.sum(drain > summary["budget_ns"] / 1e6))
                if not summary["window"]
                else 0
            )
            g["over_estimate"] += int(np.sum(drain > a[:, :, 7] / 1e6))
            g["exhausted"] += int(np.sum(a[:, :, 6] == 256))
            g["max_drain_ms"] = max(g["max_drain_ms"], float(drain.max()))
            g["max_overshoot_ms"] = max(g["max_overshoot_ms"], float(overshoot.max()))
            g["max_scheduled_ms"] = max(g["max_scheduled_ms"], float(scheduled.max()))
            g["max_unretired"] = max(
                g["max_unretired"], int((a[:, :, 5] - a[:, :, 6]).max())
            )
            g["max_unit_ratio"] = max(
                g["max_unit_ratio"], float((duration / estimate).max())
            )
            g["units"] += duration.size
            g["unit_exceed"] += int(np.sum(duration > estimate))
            for trial in range(len(a)):
                for k in range(1, 4):
                    begin = int(a[trial, k, 1])
                    resume = int(a[trial, k - 1, 8])
                    old = int(a[trial, k - 1, 5])
                    admitted = int(a[trial, k, 5])
                    before = ch[trial, :, 2]
                    rolling = any(
                        any(
                            resume <= int(before[j]) <= int(ch[trial, i, 0]) < begin
                            for j in range(i)
                        )
                        for i in range(old, admitted)
                    )
                    g["later_rolling"] += int(rolling)
                    g["later_arrivals"] += 1
            trials += len(a)
    record = dict(
        trials=trials,
        groups=groups,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Independent ledger/timing audit passed: {trials} trials")


if __name__ == "__main__":
    main()
