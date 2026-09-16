"""Plot measured routing service, transition and throughput comparisons."""

# status: diagnostic
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from report import POLICIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    cases = record["result"]["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for index, policy in enumerate(POLICIES):
        points = [c for c in cases if c["policy"] == policy]
        for c in points:
            m = c["metrics"]
            opts = dict(
                color=f"C{index}",
                marker="o" if c["window"] == 1 else "^",
                s=25 if c["iterations"] == 2048 else 65,
                alpha=0.6,
            )
            axes[0].scatter(m["stable_ms"]["p99"], m["elastic_units_s"], **opts)
            if (
                policy in ("borrow", "dynamic")
                and m["transition_ms"]["p99"] is not None
            ):
                axes[1].scatter(m["transition_ms"]["p99"], m["elastic_units_s"], **opts)
            axes[2].scatter(
                100 * m["observed_sm_coverage"], m["stable_ms"]["p99"], **opts
            )
        axes[0].scatter([], [], color=f"C{index}", label=policy)
    axes[0].set_xscale("log")
    axes[2].set_yscale("log")
    axes[0].axvline(4, color="black", linestyle="--", linewidth=1)
    axes[2].axhline(4, color="black", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="Stable scheduled-response p99 (ms, log scale)",
        ylabel="Elastic 2048-iteration units/s",
    )
    axes[1].set(
        xlabel="Request to first C enqueue p99 (ms)",
        ylabel="Elastic 2048-iteration units/s",
    )
    axes[2].set(
        xlabel="Observed block-presence coverage (% of device SMs)",
        ylabel="Stable scheduled-response p99 (ms, log scale)",
    )
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Synthetic routing: repetition/role summaries, not confidence bounds\n"
        "Circle: window 1; triangle: window 4. Small: 2048 iterations; large: 8192. Dashed: 4 ms target.",
        fontsize=11,
    )
    fig.savefig(args.output)
    if args.output.suffix.lower() == ".svg":
        args.output.write_text(
            "\n".join(line.rstrip() for line in args.output.read_text().splitlines())
            + "\n"
        )


if __name__ == "__main__":
    main()
