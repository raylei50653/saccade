"""Plot descriptive response/drain/throughput frontiers from a replayed record."""

# status: diagnostic
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    groups = defaultdict(list)
    for row in record["summary"]:
        groups[row["route"], row["window"], row["budget_ns"]].append(row)
    routes = ["disjoint", "shared_budget_priority", "full_priority"]
    colors = {
        1: "#377eb8",
        4: "#4daf4a",
        16: "#984ea3",
        3000000: "#e41a1c",
        6000000: "#ff7f00",
        12000000: "#a65628",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for col, route in enumerate(routes):
        axes[0, col].set_title(route.replace("_", " "))
        for r, w, b in sorted(
            groups, key=lambda k: (k[0], 0 if k[1] else 1, k[1] or k[2])
        ):
            rows = groups[r, w, b]
            if r != route:
                continue
            x = np.array([row["drain_observed_ms"]["p99"] for row in rows])
            for index, (metric, q) in enumerate(
                [("stable_response_ms", "p99"), ("elastic_equivalent_units_s", "p50")]
            ):
                ax = axes[index, col]
                y = np.array([row[metric][q] for row in rows])
                xm, ym = np.median(x), np.median(y)
                label = f"W{w}" if w else f"B{b / 1e6:g}"
                ax.errorbar(
                    xm,
                    ym,
                    xerr=[[xm - min(x)], [max(x) - xm]],
                    yerr=[[ym - min(y)], [max(y) - ym]],
                    fmt="o" if w else "s",
                    color=colors[w or b],
                    capsize=3,
                    label=label,
                )
                ax.grid(alpha=0.2)
                ax.set_xlabel("Observed drain p99 (ms)")
                ax.set_ylabel(
                    "Stable response p99 (ms)"
                    if index == 0
                    else "Elastic equivalent units/s (p50)"
                )
        axes[0, col].legend(fontsize=8, ncol=3)
    fig.suptitle(
        "Time-budget admission: descriptive synthetic frontier\n"
        "Points: median across repetition/role quantiles; whiskers: min–max, not confidence intervals. B in ms."
    )
    fig.savefig(args.output)
    if args.output.suffix.lower() == ".svg":
        args.output.write_text(
            "\n".join(line.rstrip() for line in args.output.read_text().splitlines())
            + "\n"
        )


if __name__ == "__main__":
    main()
