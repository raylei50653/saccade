"""Plot observed resource-proxy throughput, paired gain and frame p99."""

# status: diagnostic
import argparse
import json
from pathlib import Path


def plot(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((root / "summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for mode, color in (("serial", "#277da1"), ("double", "#e76f51")):
        for rep in sorted({r["rep"] for r in data["rows"]}):
            rows = sorted(
                (r for r in data["rows"] if r["mode"] == mode and r["rep"] == rep),
                key=lambda r: r["blocks"],
            )
            label = mode if rep == 0 else None
            axes[0].plot(
                [r["blocks"] for r in rows],
                [r["fps"] for r in rows],
                "o-",
                color=color,
                alpha=0.65,
                label=label,
            )
            axes[2].plot(
                [r["blocks"] for r in rows],
                [r["latency"]["p99_ms"] for r in rows],
                "o-",
                color=color,
                alpha=0.65,
            )
    for rep in sorted({r["rep"] for r in data["pairs"]}):
        rows = sorted(
            (r for r in data["pairs"] if r["rep"] == rep), key=lambda r: r["blocks"]
        )
        axes[1].plot(
            [r["blocks"] for r in rows],
            [r["db_gain"] for r in rows],
            "o-",
            alpha=0.7,
            label=f"rep {rep + 1}",
        )
    for ax, title in zip(
        axes,
        ("Throughput (FPS)", "Double / serial FPS", "Frame latency p99 (ms)"),
        strict=True,
    ):
        ax.set_title(title)
        ax.set_xlabel("K residency-pressure blocks (proxy)")
        ax.grid(alpha=0.2)
    axes[1].axhline(1, color="#555555", linestyle=":", linewidth=1)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("SM residency pressure: K is not a count of disabled SMs", fontsize=13)
    fig.savefig(root / "resource_sensitivity.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    plot(parser.parse_args().root)
