"""Independently audit raw C-lane transfers and stable service target counts."""

# status: diagnostic
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(root):
    result = json.loads((root / "result.json").read_text())
    if result["status"] != "validated_synthetic_routing":
        raise ValueError("unvalidated run")
    totals = dict(
        trials=0,
        arrivals=0,
        stable_units=0,
        requests=0,
        requests_with_pending_elastic=0,
        stable_c_jobs=0,
        elastic_to_stable_transfers=0,
        deadline_misses=0,
    )
    minimum_gap = None
    groups = []
    for case in result["summary"]:
        service = []
        for name in case["raw"]:
            with np.load(root / name, allow_pickle=False) as data:
                r, a, s = data["records"], data["arrivals"], data["stamps"]
                stable = r[:, 0] == 0
                c_elastic = (r[:, 0] == 1) & (r[:, 1] == 2)
                totals["trials"] += 1
                totals["arrivals"] += len(a)
                totals["stable_units"] += int(stable.sum())
                completion = np.array(
                    [r[stable & (r[:, 2] == j), 5].max() for j in range(16)]
                )
                if np.any(completion > a[:, 5]):
                    raise ValueError("completion before work")
                latency = (a[:, 5].astype(np.int64) - a[:, 0].astype(np.int64)) / 1e6
                service.extend(latency.tolist())
                totals["deadline_misses"] += int((latency > 4).sum())
                for j, arrival in enumerate(a):
                    request, drain, release = map(int, arrival[[2, 3, 6]])
                    if not request:
                        continue
                    totals["requests"] += 1
                    previous = c_elastic & (r[:, 3] < request)
                    if np.any(previous & (r[:, 5] > request)):
                        totals["requests_with_pending_elastic"] += 1
                    if np.any(previous & (r[:, 5] > drain)):
                        raise ValueError("C not drained")
                    if np.any(c_elastic & (r[:, 3] >= request) & (r[:, 3] < release)):
                        raise ValueError("C admitted while stable owner")
                    future = stable & (r[:, 1] == 2) & (r[:, 2] == j)
                    if not future.any():
                        continue
                    totals["stable_c_jobs"] += 1
                    if int(r[future, 3].min()) < drain:
                        raise ValueError("stable dispatched before drain")
                    if previous.any():
                        gap = int(s[future, :, 0].min()) - int(s[previous, :, 1].max())
                        if gap < 0:
                            raise ValueError("overlapping GPU owners")
                        totals["elastic_to_stable_transfers"] += 1
                        minimum_gap = (
                            gap if minimum_gap is None else min(minimum_gap, gap)
                        )
        p99 = float(np.quantile(service, 0.99))
        misses = int((np.array(service) > 4).sum())
        if not np.isclose(p99, case["metrics"]["stable_ms"]["p99"], rtol=0, atol=1e-12):
            raise ValueError("p99 mismatch")
        groups.append(
            {k: case[k] for k in ("repeat", "policy", "iterations", "window", "swap")}
            | dict(
                stable_p99_ms=p99,
                misses=misses,
                samples=len(service),
                service_target_pass=p99 <= 4 and misses / len(service) <= 0.01,
            )
        )
    return dict(
        archive=str(root),
        totals=totals,
        minimum_gpu_handoff_gap_ns=minimum_gap,
        groups=groups,
        audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Independent transfer/service audit; full integrity/coverage checks belong to report.py",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = audit(args.archive.resolve())
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["totals"]))


if __name__ == "__main__":
    main()
