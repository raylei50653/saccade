"""Generate report charts for Saccade pipeline analysis.

Usage:
    uv run python report_data/charts.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = "report_data"
DPI = 150

# ── Global style ───────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)

COLORS = {
    "detect": "#E74C3C",
    "postprocess": "#E67E22",
    "reid": "#9B59B6",
    "gmc": "#3498DB",
    "track": "#2ECC71",
    "materialize": "#1ABC9C",
    "relink": "#F1C40F",
    "ingest": "#95A5A6",
    "fetch": "#BDC3C7",
    "gray": "#7F8C8D",
    "speed": "#3498DB",
    "baseline": "#E74C3C",
    "mamba": "#27AE60",
}


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Pipeline Stage Breakdown (Pie + Bar)
# ═══════════════════════════════════════════════════════════════════════════
def fig1_pipeline_stage_breakdown():
    # Fresh data from mamba_whole_graph preset, MOT17-04-SDP, 150 frames (2026-06-06)
    stages = [
        ("Detection", 3.37, COLORS["detect"]),
        ("Postprocess", 3.02, COLORS["postprocess"]),
        ("Fetch", 2.14, COLORS["fetch"]),
        ("Tracker", 0.61, COLORS["track"]),
        ("Ingest+Preprocess", 0.27, COLORS["ingest"]),
        ("Materialize", 0.20, COLORS["materialize"]),
        ("Relink", 0.11, COLORS["relink"]),
        ("GMC", 0.04, COLORS["gmc"]),
    ]
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    colors = [s[2] for s in stages]
    total = sum(values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: pie
    wedges, texts, autotexts = ax1.pie(
        values,
        labels=None,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        explode=(0.06, 0, 0, 0, 0, 0, 0, 0),
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax1.set_title(
        "Per-Frame Time Distribution\n(mamba_whole_graph preset, yolo26s)", fontsize=12
    )
    ax1.legend(
        wedges,
        [f"{lab} ({v:.2f} ms)" for lab, v in zip(labels, values)],
        title=f"Total: {total:.2f} ms",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=8,
        title_fontsize=9,
    )

    # Right: horizontal bar
    y_pos = np.arange(len(stages))
    bars = ax2.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Time (ms)")
    ax2.set_title("Per-Frame Stage Latency (ms)", fontsize=12)
    for bar, v in zip(bars, values):
        ax2.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f} ms ({v / total * 100:.1f}%)",
            va="center",
            fontsize=9,
        )

    fig.suptitle(
        "Saccade Pipeline — Per-Frame Latency Breakdown",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig1_pipeline_stage_breakdown.png")
    plt.close(fig)
    print("  ✓ fig1_pipeline_stage_breakdown.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Module Contribution (Ablation Ledger)
# ═══════════════════════════════════════════════════════════════════════════
def fig2_ablation_ledger():
    # Fresh from pipeline_contribution.py (2026-06-06, yolo26s, 2-seq SDP, 150fr)
    modules = [
        "tracker_core\n(bare)",
        "tracker_core_gmc\n(+GPU GMC)",
        "semantic_core\n(+ReID+relink)",
        "semantic_bank\n(+Appearance bank)",
        "full_default\n(+Async pipe)",
    ]
    idf1 = [13.3, 14.9, 15.0, 15.0, 15.0]
    mota = [6.7, 7.0, 7.0, 7.0, 7.0]
    ids = [75, 29, 28, 28, 28]
    fps = [141.26, 144.64, 137.68, 131.60, 135.98]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Saccade Module Ablation Ledger (P3 baseline, yolo26s, 2-seq SDP)",
        fontsize=13,
        fontweight="bold",
    )

    x = np.arange(len(modules))
    width = 0.35

    # IDF1
    bars1 = axes[0, 0].bar(
        x,
        idf1,
        width,
        color=[
            COLORS["speed"],
            COLORS["gmc"],
            COLORS["reid"],
            COLORS["postprocess"],
            COLORS["track"],
        ],
    )
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(modules, fontsize=8)
    axes[0, 0].set_ylabel("IDF1 (%)")
    axes[0, 0].set_title("IDF1")
    for bar, v in zip(bars1, idf1):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{v}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    axes[0, 0].set_ylim(11, 17)
    axes[0, 0].axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[0, 0].text(
        0.25,
        16,
        "GMC ON\n+1.7pp",
        fontsize=8,
        color=COLORS["gmc"],
        ha="center",
        fontweight="bold",
    )

    # MOTA
    axes[0, 1].bar(
        x,
        mota,
        width,
        color=[
            COLORS["speed"],
            COLORS["gmc"],
            COLORS["reid"],
            COLORS["postprocess"],
            COLORS["track"],
        ],
    )
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(modules, fontsize=8)
    axes[0, 1].set_ylabel("MOTA (%)")
    axes[0, 1].set_title("MOTA")
    for i, v in enumerate(mota):
        axes[0, 1].text(
            x[i], v + 0.15, f"{v}", ha="center", fontsize=9, fontweight="bold"
        )
    axes[0, 1].set_ylim(5, 9)

    # IDs
    axes[1, 0].bar(
        x,
        ids,
        width,
        color=[
            COLORS["speed"],
            COLORS["gmc"],
            COLORS["reid"],
            COLORS["postprocess"],
            COLORS["track"],
        ],
    )
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(modules, fontsize=8)
    axes[1, 0].set_ylabel("IDs")
    axes[1, 0].set_title("ID Switches")
    axes[1, 0].axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[1, 0].text(
        0.25,
        255,
        "IDs −133\nwith GMC",
        fontsize=8,
        color=COLORS["gmc"],
        ha="center",
        fontweight="bold",
    )
    for i, v in enumerate(ids):
        axes[1, 0].text(x[i], v + 3, f"{v}", ha="center", fontsize=9, fontweight="bold")

    # FPS
    axes[1, 1].bar(
        x,
        fps,
        width,
        color=[
            COLORS["speed"],
            COLORS["gmc"],
            COLORS["reid"],
            COLORS["postprocess"],
            COLORS["track"],
        ],
    )
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(modules, fontsize=8)
    axes[1, 1].set_ylabel("FPS")
    axes[1, 1].set_title("FPS")
    axes[1, 1].axhline(
        76.08, color=COLORS["track"], linestyle="--", linewidth=0.8, alpha=0.5
    )
    for i, v in enumerate(fps):
        axes[1, 1].text(
            x[i], v + 1, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold"
        )
    axes[1, 1].set_ylim(120, 155)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig2_ablation_ledger.png")
    plt.close(fig)
    print("  ✓ fig2_ablation_ledger.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Preset Comparison
# ═══════════════════════════════════════════════════════════════════════════
def fig3_preset_comparison():
    presets = [
        "speed\n(yolo26s)",
        "baseline\n(yolo26m)",
        "mamba_whole_graph\n(production)",
    ]
    metrics = {
        "IDF1 (%)": [52.0, 51.4, 73.4],
        "MOTA (%)": [41.6, 43.5, 76.9],
        "Recall (%)": [55.0, 59.0, 81.2],
        "IDs": [475, 502, 539],
        "FPS": [97.9, 85, 175.4],
        "FP": [14687, 16112, 4309],
    }

    len(metrics)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "Saccade Preset Comparison (MOT17-SDP 7-seq)", fontsize=13, fontweight="bold"
    )

    colors_bar = [COLORS["speed"], COLORS["baseline"], COLORS["mamba"]]
    x = np.arange(len(presets))

    for ax, (title, vals) in zip(axes.flat, metrics.items()):
        bars = ax.bar(x, vals, color=colors_bar, edgecolor="white", width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(presets, fontsize=8)
        ax.set_title(title, fontsize=11)
        for bar, v in zip(bars, vals):
            if v > 100:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{v:,}" if title == "FP" else f"{v}",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.03,
                    f"{v}",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3_preset_comparison.png")
    plt.close(fig)
    print("  ✓ fig3_preset_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Detection Quality Scoring Components
# ═══════════════════════════════════════════════════════════════════════════
def fig4_detection_quality_scoring():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Detection Quality Scoring — Gaussian Components",
        fontsize=13,
        fontweight="bold",
    )

    # Aspect ratio
    aspect = np.linspace(0.1, 6, 200)
    q_asp = np.exp(-0.5 * ((aspect - 2.5) / 1.2) ** 2)
    axes[0].plot(aspect, q_asp, color=COLORS["detect"], linewidth=2.5)
    axes[0].axvline(2.5, color="gray", linestyle="--", alpha=0.5, label="peak at 2.5")
    axes[0].fill_between(aspect, 0, q_asp, alpha=0.15, color=COLORS["detect"])
    axes[0].set_xlabel("Aspect Ratio (h/w)")
    axes[0].set_ylabel("Q_aspect")
    axes[0].set_title("Aspect Ratio Quality\n(peak @ h/w = 2.5)", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1.05)

    # Center bias
    c_norm = np.linspace(0, 0.5, 200)
    q_ctr = np.clip(4 * c_norm, 0, 1)
    axes[1].plot(c_norm, q_ctr, color=COLORS["postprocess"], linewidth=2.5)
    axes[1].fill_between(c_norm, 0, q_ctr, alpha=0.15, color=COLORS["postprocess"])
    axes[1].set_xlabel("Normalized Distance to Edge")
    axes[1].set_ylabel("Q_center")
    axes[1].set_title("Center Bias Quality\n(edge = 0, center = 1)", fontsize=10)

    # Area ratio
    rho = np.linspace(0, 0.05, 300)
    q_area = np.exp(-0.5 * ((rho - 0.01) / 0.01) ** 2)
    axes[2].plot(rho, q_area, color=COLORS["gmc"], linewidth=2.5)
    axes[2].axvline(0.01, color="gray", linestyle="--", alpha=0.5, label="peak at 1%")
    axes[2].fill_between(rho, 0, q_area, alpha=0.15, color=COLORS["gmc"])
    axes[2].set_xlabel("Area Ratio (box/frame)")
    axes[2].set_ylabel("Q_area")
    axes[2].set_title("Area Ratio Quality\n(peak @ 1% of frame)", fontsize=10)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig4_detection_quality_scoring.png")
    plt.close(fig)
    print("  ✓ fig4_detection_quality_scoring.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — NO-GO Timeline
# ═══════════════════════════════════════════════════════════════════════════
def fig5_nogo_timeline():
    fig, ax = plt.subplots(figsize=(14, 9))

    items = [
        ("2026-04-27", "Score Decay", "B1"),
        ("2026-04-27", "Post-merge (motion-only)", "B1"),
        ("2026-05-02", "LaSt-ViT pre-hoc embed", "B1"),
        ("2026-05-03", "GMC FG mask", "B1"),
        ("2026-05-10", "Pose box expansion", "B1"),
        ("2026-05-10", "P5-1 Multi-signal birth", "B1"),
        ("2026-05-11", "Narrow person bonus", "B1"),
        ("2026-05-11", "P5-2 Stage2 QualityGate", "B1"),
        ("2026-05-11", "P5-3 ConsecutiveBirthGate", "B1"),
        ("2026-05-11", "P5-4 Scene Adaptive", "B1"),
        ("2026-05-13", "Appearance ReID Bank", "A"),
        ("2026-05-13", "Semantic Relink (GMC ON)", "A"),
        ("2026-05-13", "Per-frame Detection Cap", "B1"),
        ("2026-05-14", "Cascade Filter (MOT17)", "B1"),
        ("2026-05-17", "Motion-based Relinking", "B1"),
        ("2026-05-18", "Horizontal-flip TTA", "B1"),
        ("2026-05-18", "P5-5 Proximity Birth", "B1"),
        ("2026-05-19", "Option D Track-Conditioned", "A"),
        ("2026-05-19", "ROI FPN ReID", "B1"),
        ("2026-05-20", "OA-SORT OAO", "B1"),
        ("2026-05-21", "Tiled Detection", "A"),
        ("2026-05-31", "Mamba temporal block", "B2"),
        ("2026-06-01", "Per-channel SSM A+MOT20", "B2"),
        ("2026-06-01", "Vel_dir gate", "B1"),
        ("2026-06-03", "Appearance ceiling (closed)", "A"),
        ("2026-06-03", "Cheb-GR offline merge", "B1"),
        ("2026-06-03", "Birth-time lost-bank relink", "B1"),
    ]

    y_positions = list(range(len(items)))
    dates = [it[0] for it in items]
    names = [it[1] for it in items]
    tiers = [it[2] for it in items]

    tier_colors = {
        "A": COLORS["detect"],
        "B1": COLORS["postprocess"],
        "B2": COLORS["reid"],
    }
    tier_labels = {
        "A": "A: Core NO-GO (major)",
        "B1": "B1: Auxiliary NO-GO",
        "B2": "B2: Architectural NO-GO",
    }
    colors = [tier_colors[t] for t in tiers]

    ax.barh(y_positions, [1] * len(items), color=colors, height=0.7, alpha=0.8)
    for i, (name, date) in enumerate(zip(names, dates)):
        ax.text(0.02, i, f"{date}  {name}", va="center", fontsize=8)

    ax.set_yticks([])
    ax.set_xlim(0, 1.5)
    ax.set_title(
        "Saccate NO-GO / Deprecated Decision Timeline (2026-04 ~ 2026-06)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Timeline →")

    legend = [
        mpatches.Patch(color=tier_colors[k], label=v) for k, v in tier_labels.items()
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig5_nogo_timeline.png")
    plt.close(fig)
    print("  ✓ fig5_nogo_timeline.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 — NO-GO Reason Categorization
# ═══════════════════════════════════════════════════════════════════════════
def fig6_nogo_categories():
    categories = {
        "Statistically Neutral\n(no gain, noise range)": 12,
        "Harmful\n(metrics degraded)": 7,
        "Appearance Ceiling\n(MOT17 intrinsic limit)": 4,
        "GMC Redundant\n(GMC supersedes)": 3,
        "FP/TP Overlap\n(filtering damages recall)": 2,
        "Gradient Collapse\n(training failure)": 2,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("NO-GO Root Cause Analysis", fontsize=13, fontweight="bold")

    # Pie
    colors_cat = [
        COLORS["gmc"],
        COLORS["detect"],
        COLORS["reid"],
        COLORS["postprocess"],
        COLORS["track"],
        COLORS["ingest"],
    ]
    labels = list(categories.keys())
    sizes = list(categories.values())
    wedges, _, autotexts = ax1.pie(
        sizes,
        labels=None,
        colors=colors_cat,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.6,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax1.set_title("NO-GO Reason Distribution", fontsize=12)
    ax1.legend(
        wedges,
        [f"{lab} ({s})" for lab, s in zip(labels, sizes)],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=8,
    )

    # Horizontal bars
    y = np.arange(len(categories))
    ax2.barh(y, sizes, color=colors_cat, edgecolor="white", height=0.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Count")
    ax2.set_title("NO-GO Items by Category", fontsize=12)
    for i, v in enumerate(sizes):
        ax2.text(v + 0.3, i, str(v), va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig6_nogo_categories.png")
    plt.close(fig)
    print("  ✓ fig6_nogo_categories.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7 — Sinkhorn vs Auction Architecture
# ═══════════════════════════════════════════════════════════════════════════
def fig7_sinkhorn_auction():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(
        "Sinkhorn-Auction Hybrid GPU Association",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Helper
    def draw_box(ax, x, y, w, h, text, color, fontsize=10, alpha=0.85):
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.3",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            alpha=alpha,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="white",
        )

    def draw_arrow(ax, x1, y1, x2, y2, label="", color="gray"):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=2),
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my - 0.3, label, ha="center", fontsize=8, color=color)

    # Boxes
    draw_box(
        ax, 0.3, 6.5, 2.8, 1.2, "Cost Matrix C\nIoU + cos × decay", COLORS["detect"]
    )
    draw_box(
        ax, 4.0, 6.5, 2.8, 1.2, "Sinkhorn-Knopp\nP = diag(u) K diag(v)", COLORS["reid"]
    )
    draw_box(ax, 7.6, 6.5, 2.8, 1.2, "Top-K Selection\nK = 3 per track", COLORS["gmc"])
    draw_box(
        ax, 11.2, 6.5, 2.5, 1.2, "Parallel Auction\natomicMax bidding", COLORS["track"]
    )

    draw_box(
        ax,
        0.3,
        3.0,
        13.4,
        2.0,
        "Stage 1 Gate: IoU > 0.01 ‖ Mahalanobis² < 9.4877\n"
        "Stage 2 ReID: w_cos·cos + w_iou·IoU + w_score·score (conditional on clean embedding)",
        COLORS["ingest"],
        fontsize=9,
        alpha=0.5,
    )

    draw_box(
        ax,
        0.3,
        0.5,
        13.4,
        1.3,
        "Output: det→trk assignments + birth/revive slots → Kalman update → result buffer",
        COLORS["fetch"],
        fontsize=9,
        alpha=0.6,
    )

    # Arrows
    draw_arrow(ax, 3.1, 7.1, 4.0, 7.1)
    draw_arrow(ax, 6.8, 7.1, 7.6, 7.1)
    draw_arrow(ax, 10.4, 7.1, 11.2, 7.1)

    ax.text(
        7.0,
        8.5,
        "λ = 30, max 50 iter",
        ha="center",
        fontsize=8,
        color=COLORS["reid"],
        fontstyle="italic",
    )
    ax.text(
        7.0,
        8.1,
        "Level 1: shared-mem → Level 2: global atomicMax",
        ha="center",
        fontsize=8,
        color=COLORS["track"],
        fontstyle="italic",
    )

    fig.savefig(f"{OUT_DIR}/fig7_sinkhorn_auction.png")
    plt.close(fig)
    print("  ✓ fig7_sinkhorn_auction.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8 — Latency Optimization History
# ═══════════════════════════════════════════════════════════════════════════
def fig8_latency_history():
    dates = [
        "2026-04\n(baseline)",
        "2026-05-05\n(+relink opt)",
        "2026-05-06\n(+async reid)",
        "2026-05-07\n(+pipeline relink)",
        "2026-06\n(mamba_whole_graph)",
    ]
    fps = [46, 47, 54.9, 70, 175.4]
    frame_ms = [21.7, 21.2, 18.2, 14.3, 5.7]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()

    (line1,) = ax1.plot(
        range(len(dates)),
        fps,
        "o-",
        color=COLORS["detect"],
        linewidth=2.5,
        markersize=10,
        label="FPS",
    )
    (line2,) = ax2.plot(
        range(len(dates)),
        frame_ms,
        "s--",
        color=COLORS["gmc"],
        linewidth=2.5,
        markersize=10,
        label="Frame Time (ms)",
    )

    ax1.set_xticks(range(len(dates)))
    ax1.set_xticklabels(dates, fontsize=9)
    ax1.set_ylabel("FPS", color=COLORS["detect"], fontweight="bold")
    ax2.set_ylabel("Frame Time (ms)", color=COLORS["gmc"], fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=COLORS["detect"])
    ax2.tick_params(axis="y", labelcolor=COLORS["gmc"])

    # Annotations
    ax1.annotate(
        "relink\n-32%",
        xy=(1, 47),
        xytext=(1, 62),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="gray"),
        ha="center",
    )
    ax1.annotate(
        "async reid\n+2.6%",
        xy=(2, 55),
        xytext=(2, 68),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="gray"),
        ha="center",
    )
    ax1.annotate(
        "pipeline relink\n+2.5%",
        xy=(3, 70),
        xytext=(3, 82),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="gray"),
        ha="center",
    )
    ax1.annotate(
        "mamba_whole_graph\n+150% FPS",
        xy=(4, 175),
        xytext=(3.5, 195),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="gray"),
        ha="center",
    )

    ax1.legend([line1, line2], ["FPS", "Frame Time (ms)"], loc="upper left", fontsize=9)
    ax1.set_title(
        "Saccade Throughput Optimization History\n(MOT17-SDP, yolo26s)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylim(30, 210)
    ax2.set_ylim(28, 4)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig8_latency_history.png")
    plt.close(fig)
    print("  ✓ fig8_latency_history.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 9 — ReID Ablation Summary
# ═══════════════════════════════════════════════════════════════════════════
def fig9_reid_ablation():
    methods = [
        "No ReID\n(GMC only)",
        "ReID branch\n+ relink",
        "+Appearance\nBank",
        "+Async\nPipeline",
    ]
    idf1_vals = [14.9, 15.0, 15.0, 15.0]
    fps_vals = [144.6, 137.7, 131.6, 136.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(methods))
    w = 0.35

    ax1.bar(x - w / 2, idf1_vals, w, label="IDF1 (%)", color=COLORS["gmc"])
    ax1_twin = ax1.twinx()
    ax1_twin.bar(x + w / 2, fps_vals, w, label="FPS", color=COLORS["reid"], alpha=0.7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylabel("IDF1 (%)", color=COLORS["gmc"])
    ax1_twin.set_ylabel("FPS", color=COLORS["reid"])
    ax1.set_title("ReID Stack: Accuracy vs Speed\n(MOT17-SDP, yolo26s)", fontsize=11)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

    # Right: cost-benefit
    ax2.bar(
        x,
        [0, -7.0, -6.1, +4.4],
        color=[
            COLORS["speed"],
            COLORS["postprocess"],
            COLORS["detect"],
            COLORS["track"],
        ],
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=9)
    ax2.set_ylabel("FPS Change from Previous Step")
    ax2.set_title("ReID Stack: FPS Cost per Step", fontsize=11)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    for i, v in enumerate([0, -7.0, -6.1, +4.4]):
        ax2.text(
            i,
            v + 0.5 if v >= 0 else v - 2,
            f"{v:+.1f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle(
        "ReID Stack: IDF1 +0.1pp, FPS Cost −8.7 (ReID → full stack)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig9_reid_ablation.png")
    plt.close(fig)
    print("  ✓ fig9_reid_ablation.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating Saccade report charts...")
    fig1_pipeline_stage_breakdown()
    fig2_ablation_ledger()
    fig3_preset_comparison()
    fig4_detection_quality_scoring()
    fig5_nogo_timeline()
    fig6_nogo_categories()
    fig7_sinkhorn_auction()
    fig8_latency_history()
    fig9_reid_ablation()
    print(f"\nDone! {9} charts saved to {OUT_DIR}/")
