"""Plot DB frame-period tails versus elastic burst completion."""

# status: diagnostic
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.summary.read_text())
    rows = [
        row for row in data["rows"] if not row["audit"] and row["policy"] != "control"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = dict(fixed="#3569b0", shared="#bc6b24", dynamic="#963c8c")
    for axis, iterations in zip(axes, (2048, 8192), strict=True):
        for policy, color in colors.items():
            for window, marker in ((1, "o"), (4, "^")):
                selected = [
                    row
                    for row in rows
                    if row["policy"] == policy
                    and row["window"] == window
                    and row["iterations"] == iterations
                ]
                axis.scatter(
                    [row["burst_completion_ms"]["p95"] for row in selected],
                    [row["period_ms"]["p99"] for row in selected],
                    color=color,
                    marker=marker,
                    label=f"{policy} W{window}",
                    s=48,
                    alpha=0.85,
                )
        axis.set(
            xlabel="Elastic burst completion p95 (ms; lower is better)",
            ylabel="DB completed-frame period p99 (ms)",
            title=f"{iterations} iterations per elastic unit",
        )
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Saturated Saccade double buffer + matched elastic bursts, 32 SMs\n"
        "Each point is one repetition; circles W1, triangles W4"
    )
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
