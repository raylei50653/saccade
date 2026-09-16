"""Reject diagnostic GPU stamp transfers inside the stable-service horizon."""

# status: diagnostic
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.benchmarks.resource_partition.green_owner import (  # noqa: E402
    host_to_cupti,
    parse_trace,
)


def check_copy(point, trace):
    size = point["bursts"] * point["arguments"]["units"] * 256 * 24
    copies = [r for r in trace["copies"] if r["bytes"] == size]
    final, spread = host_to_cupti(
        point["owner"]["calibration"], point["frames"][-1]["finished"]
    )
    if len(copies) != 1 or copies[0]["start"] <= final + spread:
        raise ValueError("diagnostic GPU evidence copy precedes stable completion")
    return (copies[0]["start"] - final) / 1e6


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    manifest = json.loads((args.directory / "sweep.json").read_text())
    rows = []
    for entry in manifest["schedule"]:
        if not entry["audit"] or entry["policy"] == "control":
            continue
        root = args.directory / entry["name"]
        point = json.loads((root / "point.json").read_text())
        gap = check_copy(point, parse_trace(root / "cupti.trace"))
        rows.append(dict(name=entry["name"], copy_after_stable_ms=gap))
    expected = (
        len(manifest["arguments"]["policies"])
        * len(manifest["arguments"]["iterations"])
        * len(manifest["arguments"]["windows"])
    )
    if len(rows) != expected:
        raise ValueError("incomplete audited copy coverage")
    result = dict(
        verified=True,
        rows=rows,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (args.output or args.directory / "copy_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(f"Verified {len(rows)} evidence copies after stable completion")


if __name__ == "__main__":
    main()
