"""Plot verified-partition serial/double FPS, DB gain and frame p99 versus actual SM count."""

# status: diagnostic
import argparse
import json
from pathlib import Path


def plot(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((root / "summary.json").read_text())
    rows = [r for r in data["rows"] if r["curve_eligible"]]
    pairs = [p for p in data["pairs"] if p["curve_eligible"]]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), layout="constrained")

    def marker(row):
        return "o" if row["partition_kind"] == "green_context" else "s"

    for mode, color in (("serial", "#277da1"), ("double", "#e76f51")):
        for rep in sorted({r["rep"] for r in rows}):
            selected = sorted(
                (r for r in rows if r["mode"] == mode and r["rep"] == rep),
                key=lambda r: r["actual_sm_count"],
            )
            label = mode if rep == 0 else None
            for ax, key in ((axes[0], "fps"), (axes[3], "p99")):
                xs = [r["actual_sm_count"] for r in selected]
                ys = [
                    r["fps"] if key == "fps" else r["latency"]["p99_ms"]
                    for r in selected
                ]
                ax.plot(xs, ys, "-", color=color, alpha=0.5, label=label)
                for x, y, r in zip(xs, ys, selected, strict=True):
                    ax.plot(x, y, marker(r), color=color, alpha=0.8)
    for rep in sorted({p["rep"] for p in pairs}):
        selected = sorted(
            (p for p in pairs if p["rep"] == rep), key=lambda p: p["actual_sm_count"]
        )
        axes[1].plot(
            [p["actual_sm_count"] for p in selected],
            [p["db_gain"] for p in selected],
            "o-",
            alpha=0.7,
            label=f"rep {rep + 1}",
        )
        axes[2].plot(
            [p["actual_sm_count"] for p in selected],
            [p["double_fps"] for p in selected],
            "o-",
            alpha=0.7,
            label=f"rep {rep + 1}",
        )
    for ax, title in zip(
        axes,
        (
            "Serial / double FPS",
            "DB gain (double / serial)",
            "Double-buffer FPS",
            "Frame latency p99 (ms)",
        ),
        strict=True,
    ):
        ax.set_title(title)
        ax.set_xlabel("actual SM count (square = primary full device)")
        ax.grid(alpha=0.2)
    axes[1].axhline(1, color="#555555", linestyle=":", linewidth=1)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(
        "Verified Green Context partitions: curve points are true_partition_validated",
        fontsize=12,
    )
    fig.savefig(root / "resource_partition.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    plot(parser.parse_args().root)
