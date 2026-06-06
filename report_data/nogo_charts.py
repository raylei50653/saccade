"""NO-GO mathematical modeling and visualization.

Models applied:
  1. Pareto Frontier — Cost (FPS loss) vs Benefit (IDF1 delta)
  2. Statistical Significance Waterfall — Each module's delta vs noise floor
  3. GMC Dominance / Saturation Curve
  4. Appearance Asymptote / Ceiling Model
  5. ROI Bubble Chart — Effort vs Impact vs Sample Size
  6. Sequential Evidence Accumulation

Usage:
    uv run python report_data/nogo_charts.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = "report_data"
DPI = 150

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)

C = {
    "detect": "#E74C3C",
    "noise": "#E67E22",
    "gmc": "#3498DB",
    "gmc_dark": "#1F618D",
    "good": "#2ECC71",
    "bad": "#E74C3C",
    "medium": "#F39C12",
    "gray": "#95A5A6",
    "border": "#BDC3C7",
    "reid": "#9B59B6",
    "track": "#1ABC9C",
    "mamba": "#27AE60",
    "speed": "#3498DB",
}

NOISE_FLOOR_IDF1 = 0.3  # pp — run-to-run variance
NOISE_FLOOR_MOTA = 0.3
NOISE_FLOOR_IDS = 10

# ── Data: Quantified NO-GO experiments ─────────────────────────────────────
# Each: (name, delta_idf1_pp, delta_fps, delta_ids, effort_days, category)
NOGO_DATA = [
    # Category A — Core NO-GO (major direction)
    ("Option D\nTrack-Cond YOLO", -20.3, -20, +50, 14, "Architecture"),
    ("Appearance Bank\n(on GMC ON)", 0.0, -17.3, -1, 5, "ReID"),
    ("Semantic Relink\n(on GMC ON)", 0.0, -2.2, 0, 3, "ReID"),
    ("Tiled Detection\n(960p 2x2)", -4.0, -20, +36, 7, "Detection"),
    ("Appearance Ceiling\n(5 models)", 0.0, 0, 0, 10, "ReID"),
    # Category B1 — Auxiliary NO-GO
    ("P5-5 Proximity\nBirth Gate", -5.6, 0, +30, 2, "Lifecycle"),
    ("Motion-based\nRelinking", 0.0, -5, 0, 5, "Motion"),
    ("OA-SORT OAO", 0.0, -5, 0, 3, "Motion"),
    ("Narrow Person\nScore Bonus", -0.3, -10, +26, 2, "Detection"),
    ("P5-2 Stage2\nQuality Gate", 0.0, 0, 0, 2, "Detection"),
    ("P5-3 Consecutive\nBirth Gate", 0.0, 0, 0, 2, "Lifecycle"),
    ("P5-4 Scene\nAdaptive", -0.6, -10, 0, 3, "Lifecycle"),
    ("P5-1 Multi-signal\nBirth", 0.0, -20, 0, 5, "Lifecycle"),
    ("LaSt-ViT\nPre-hoc Embed", 0.09, 0, 0, 5, "ReID"),
    ("ROI FPN ReID", 0.0, 0, +5, 5, "ReID"),
    ("Cheb-GR\nOffline Merge", 0.0, 0, 0, 5, "ReID"),
    ("Birth-time\nLost-bank Relink", 0.0, 0, 0, 7, "ReID"),
    ("Cascade Filter\n(MOT17)", 0.0, -5, 0, 5, "Detection"),
    ("Pose Box\nExpansion", -1.0, -60, +5, 3, "Detection"),
    ("Pose Bio Gate\n(Biometric)", 0.0, -47, 0, 5, "ReID"),
    ("Horizontal-flip\nTTA", 0.0, -10, -2, 2, "Detection"),
    ("Vel_dir Gate", 0.0, 0, 0, 2, "Geometry"),
    # Category B2 — Architectural
    ("Mamba Temporal\nBlock (v15/17)", -13.8, +5, +50, 10, "Architecture"),
    ("Per-channel SSM\nA + MOT20", -3.2, 0, +20, 5, "Architecture"),
    ("NSA-Kalman\n(Noise Adaptive)", 0.0, 0, 0, 3, "Geometry"),
    ("PostMerge\n(offline)", -1.0, -5, +10, 5, "Lifecycle"),
    ("Detection Cap\n(per-frame)", -1.5, 0, -5, 2, "Detection"),
]

# ── GO items for contrast ──────────────────────────────────────────────────
GO_DATA = [
    ("GPU GMC\n(phase correlation)", +2.8, -6.9, -133, 7, "Geometry"),
    ("Detection Quality\nScaling (A6)", +1.9, 0, -194, 5, "Detection"),
    ("FP Hard Filter\n(area=40000)", +0.4, 0, -20, 2, "Detection"),
    ("P3 fuse_score\nweight=0.4", +1.7, 0, -1, 2, "Association"),
    ("P3 match_thresh\nre-tune 0.72", +2.0, 0, -50, 2, "Association"),
    ("P0 Tracker\nInterpolation", +0.3, 0, -34, 3, "Lifecycle"),
    ("GR ReID\nBudget 0.2", 0.0, +24, 0, 3, "ReID"),
    ("Async ReID\nPipelining", 0.0, +2.6, 0, 3, "Infra"),
    ("Pipeline Relink\n(async)", 0.0, +2.5, 0, 2, "Infra"),
    ("GMC peak_find\nParallel GPU", 0.0, +5, 0, 3, "Geometry"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Figure A — Pareto Frontier: FPS Cost vs IDF1 Benefit
# ═══════════════════════════════════════════════════════════════════════════
def fig_a_pareto_frontier():
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, d_idf1, d_fps, _, _, cat in NOGO_DATA:
        color = C["bad"] if d_idf1 <= 0 else C["medium"]
        ax.scatter(
            -d_fps,
            d_idf1,
            color=color,
            s=80,
            edgecolors="white",
            linewidth=1,
            zorder=3,
            alpha=0.8,
        )
        ax.annotate(
            name,
            (-d_fps, d_idf1),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            alpha=0.8,
        )

    for name, d_idf1, d_fps, _, _, cat in GO_DATA:
        ax.scatter(
            -d_fps,
            d_idf1,
            color=C["good"],
            s=120,
            edgecolors="white",
            linewidth=1.5,
            zorder=4,
            alpha=0.9,
            marker="^",
        )
        ax.annotate(
            name,
            (-d_fps, d_idf1),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            alpha=0.9,
            color=C["good"],
            fontweight="bold",
        )

    # Region annotations
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.axhline(
        NOISE_FLOOR_IDF1, color=C["noise"], linestyle="--", linewidth=1, alpha=0.6
    )
    ax.axhline(
        -NOISE_FLOOR_IDF1, color=C["noise"], linestyle="--", linewidth=1, alpha=0.6
    )
    ax.fill_between(
        [-100, 50], -NOISE_FLOOR_IDF1, NOISE_FLOOR_IDF1, color=C["noise"], alpha=0.08
    )
    ax.text(
        -85,
        0,
        "Noise Floor\n(+/- 0.3pp)",
        fontsize=8,
        color=C["noise"],
        fontstyle="italic",
        alpha=0.7,
    )

    ax.text(
        30,
        2.5,
        "GO Region\n(positive ΔIDF1)",
        fontsize=9,
        color=C["good"],
        ha="center",
        alpha=0.6,
    )
    ax.text(
        -30,
        -2.0,
        "NO-GO Region\n(harmful)",
        fontsize=9,
        color=C["bad"],
        ha="center",
        alpha=0.6,
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color=C["bad"], alpha=0.8, label="NO-GO (harmful/neutral)"),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=C["good"],
            markersize=10,
            label="GO (positive gain)",
        ),
        mpatches.Patch(
            color=C["noise"], alpha=0.3, label=f"Noise floor (+/- {NOISE_FLOOR_IDF1}pp)"
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    ax.set_xlabel("FPS Cost (negative → FPS loss)", fontweight="bold")
    ax.set_ylabel("IDF1 Gain (pp)", fontweight="bold")
    ax.set_title(
        "Pareto Frontier: FPS Cost vs IDF1 Benefit\n(30 NO-GO + 10 GO modules)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(-95, 50)
    ax.set_ylim(-22, 5)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_a_pareto_frontier.png")
    plt.close(fig)
    print("  OK fig_a_pareto_frontier.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure B — Statistical Significance Waterfall
# ═══════════════════════════════════════════════════════════════════════════
def fig_b_waterfall():
    items = [(n, d) for n, d, _, _, _, _ in NOGO_DATA if d != 0] + [
        (n, d) for n, d, _, _, _, _ in GO_DATA if d != 0
    ]
    items.sort(key=lambda x: x[1])

    names = [x[0] for x in items]
    deltas = [x[1] for x in items]
    colors = [C["bad"] if d <= 0 else C["good"] for d in deltas]

    fig, ax = plt.subplots(figsize=(12, 10))
    y = np.arange(len(items))
    ax.barh(y, deltas, color=colors, edgecolor="white", height=0.7)

    # Noise floor bounds
    ax.axvline(
        NOISE_FLOOR_IDF1, color=C["noise"], linestyle=":", linewidth=1.5, alpha=0.7
    )
    ax.axvline(
        -NOISE_FLOOR_IDF1, color=C["noise"], linestyle=":", linewidth=1.5, alpha=0.7
    )
    ax.fill_betweenx(
        y, -NOISE_FLOOR_IDF1, NOISE_FLOOR_IDF1, color=C["noise"], alpha=0.06
    )
    ax.text(
        NOISE_FLOOR_IDF1 + 0.1,
        len(items) - 1.5,
        f"Noise Floor\n+/- {NOISE_FLOOR_IDF1}pp",
        fontsize=8,
        color=C["noise"],
        fontstyle="italic",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("Delta IDF1 (pp)", fontweight="bold")
    ax.set_title(
        "Statistical Significance Waterfall: IDF1 Delta vs Noise Floor\n"
        "Items within +/- 0.3pp cannot be distinguished from random run-to-run variance",
        fontsize=12,
        fontweight="bold",
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlim(-23, 5)
    ax.grid(axis="x", alpha=0.2)

    # Count annotation
    within_noise = sum(1 for d in deltas if -NOISE_FLOOR_IDF1 <= d <= NOISE_FLOOR_IDF1)
    ax.text(
        3.5,
        3,
        f"{within_noise}/{len(deltas)} items\nwithin noise range",
        fontsize=10,
        color=C["noise"],
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_b_waterfall.png")
    plt.close(fig)
    print("  OK fig_b_waterfall.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure C — GMC Dominance / Module Saturation Curve
# ═══════════════════════════════════════════════════════════════════════════
def fig_c_gmc_saturation():
    # Fresh from pipeline_contribution 2026-06-06 (yolo26s, 2-seq SDP, 150fr)
    layers = [
        ("Bare\nTracker", 13.3, ""),
        ("+ GPU\nGMC", 14.9, "+1.7 pp"),
        ("+ ReID\nBranch", 15.0, "+0.1 pp"),
        ("+ App\nBank", 15.0, "+0.0 pp"),
        ("+ Async\nPipe", 15.0, "+0.0 pp"),
    ]

    x = np.arange(len(layers))
    y = [ly[1] for ly in layers]
    labels = [ly[0] for ly in layers]
    gains = [ly[2] for ly in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: saturation curve
    ax1.plot(
        x,
        y,
        "o-",
        color=C["gmc_dark"],
        linewidth=2.5,
        markersize=10,
        markerfacecolor="white",
        markeredgewidth=2,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("IDF1 (%)", fontweight="bold")
    ax1.set_title("Module Saturation Curve\n(IDF1 vs Stacked Modules)", fontsize=11)

    for i, (xi, yi, g) in enumerate(zip(x, y, gains)):
        ax1.annotate(
            g,
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 15),
            fontsize=9,
            ha="center",
            fontweight="bold",
            color=C["gmc_dark"] if g.startswith("+2.8") else C["gray"],
        )

    ax1.set_ylim(11, 17)
    ax1.grid(True, alpha=0.2)

    # Right: marginal contribution bar
    marginal = [0, 1.7, 0.1, 0.0, 0.0]
    bar_colors = [C["gray"], C["gmc"], C["reid"], C["gray"], C["gray"]]
    bars = ax2.bar(x, marginal, color=bar_colors, edgecolor="white", width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Marginal IDF1 Gain (pp)", fontweight="bold")
    ax2.set_title(
        "Marginal Contribution per Module\n(GMC dominates: 94.4% of total gain)",
        fontsize=11,
    )
    ax2.axhline(
        NOISE_FLOOR_IDF1,
        color=C["noise"],
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"Noise floor ({NOISE_FLOOR_IDF1} pp)",
    )
    ax2.legend(fontsize=8)

    for bar, v in zip(bars, marginal):
        if v > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                f"+{v:.1f}pp",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
        else:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                0.05,
                "0",
                ha="center",
                fontsize=9,
                color=C["gray"],
            )

    fig.suptitle(
        "GMC Dominance Model: The Single Module Driving Saccade's IDF1",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_c_gmc_saturation.png")
    plt.close(fig)
    print("  OK fig_c_gmc_saturation.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure D — Appearance Ceiling / Asymptotic Model
# ═══════════════════════════════════════════════════════════════════════════
def fig_d_appearance_ceiling():
    # Model: IDF1 gain as function of embedding quality
    # The MOT17 dataset has an intrinsic ceiling where identities are hard
    # to distinguish in embedding space

    fig, ax = plt.subplots(figsize=(10, 6))

    # Data points from appearance ceiling investigation
    models = [
        "SigLIP2\n(baseline)",
        "LaSt-ViT\n(no training)",
        "OSNet\n(pretrained)",
        "FastReID\n(pretrained)",
        "TransReID\n(pretrained)",
        "Domain\nFinetuned",
    ]
    idf1_gains = [0.0, 0.09, 0.0, 0.0, 0.0, 0.0]
    effort_ranks = [0, 1, 2, 3, 4, 5]

    # The theoretical asymptote
    x_model = np.linspace(0, 5, 100)
    # Logistic saturation curve
    ceiling = 0.09  # observed max
    asymptote = ceiling * (1 / (1 + np.exp(-(x_model - 1.5))) - 0.18)
    asymptote = np.clip(asymptote, 0, 0.12)

    ax.bar(
        effort_ranks,
        idf1_gains,
        color=[C["reid"], C["gray"], C["gray"], C["gray"], C["gray"], C["gray"]],
        edgecolor="white",
        width=0.5,
    )

    ax.axhline(
        0.09,
        color=C["reid"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label="Observed ceiling: +0.09pp",
    )
    ax.axhline(
        1.0,
        color=C["bad"],
        linestyle=":",
        linewidth=1.5,
        alpha=0.6,
        label="GO threshold: +1.0pp",
    )

    ax.set_xticks(effort_ranks)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("IDF1 Gain (pp)", fontweight="bold")
    ax.set_title(
        "Appearance Ceiling Model\n"
        "All 5 ReID models + 4 mechanisms converge to the same asymptote (+0.09pp)\n"
        "MOT17 identities are intrinsically hard in embedding space",
        fontsize=11,
        fontweight="bold",
    )

    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-0.02, 1.2)

    # Key insight annotation
    ax.annotate(
        "intra-inter gap ≈ 0.03\n200+px clear box rank-1 = 57%\nlong gap rank-1 collapses to 13-37%",
        xy=(1, 0.09),
        xytext=(2.5, 0.7),
        arrowprops=dict(arrowstyle="->", color=C["reid"], lw=1.5),
        fontsize=8,
        color=C["reid"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_d_appearance_ceiling.png")
    plt.close(fig)
    print("  OK fig_d_appearance_ceiling.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure E — ROI Bubble Chart
# ═══════════════════════════════════════════════════════════════════════════
def fig_e_roi_bubble():
    all_data = [(n, d, e, c) for n, d, _, _, e, c in NOGO_DATA] + [
        (n, d, e, c) for n, d, _, _, e, c in GO_DATA
    ]

    fig, ax = plt.subplots(figsize=(12, 7))

    for name, d_idf1, effort, cat in all_data:
        color = C["good"] if d_idf1 > 0.3 else (C["medium"] if d_idf1 > 0 else C["bad"])
        size = max(30, effort * 25)
        ax.scatter(
            effort,
            d_idf1,
            s=size,
            color=color,
            edgecolors="white",
            linewidth=1,
            alpha=0.7,
        )
        ax.annotate(
            name,
            (effort, d_idf1),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=6,
            alpha=0.8,
        )

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axhline(
        NOISE_FLOOR_IDF1, color=C["noise"], linestyle="--", linewidth=1, alpha=0.6
    )
    ax.axhline(
        -NOISE_FLOOR_IDF1, color=C["noise"], linestyle="--", linewidth=1, alpha=0.6
    )
    ax.fill_between(
        [0, 16], -NOISE_FLOOR_IDF1, NOISE_FLOOR_IDF1, color=C["noise"], alpha=0.06
    )

    ax.set_xlabel("Implementation Effort (days)", fontweight="bold")
    ax.set_ylabel("IDF1 Gain (pp)", fontweight="bold")
    ax.set_title(
        "ROI Analysis: Effort vs Impact\n"
        "Bubble size = implementation cost; most NO-GO items are zero-gain with high effort",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.2)

    legend_elements = [
        plt.scatter(
            [], [], s=100, color=C["good"], edgecolors="white", label="GO (>0.3pp)"
        ),
        plt.scatter(
            [],
            [],
            s=100,
            color=C["medium"],
            edgecolors="white",
            label="Marginal (0–0.3pp)",
        ),
        plt.scatter(
            [], [], s=100, color=C["bad"], edgecolors="white", label="NO-GO (<=0pp)"
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    # ROI efficiency line
    ax.plot(
        [0, 14],
        [0, 2.8],
        color=C["gmc"],
        linestyle=":",
        alpha=0.5,
        label="GMC ROI slope",
    )
    ax.annotate(
        "GMC ROI\n(+0.24pp/day)",
        xy=(7, 2.8),
        fontsize=8,
        color=C["gmc"],
        fontstyle="italic",
        alpha=0.7,
    )

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_e_roi_bubble.png")
    plt.close(fig)
    print("  OK fig_e_roi_bubble.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure F — Sequential Evidence Accumulation (Cumulative Gain per Day)
# ═══════════════════════════════════════════════════════════════════════════
def fig_f_evidence_accumulation():
    # Model: accumulate experiments over time, track cumulative IDF1 gain
    # and cumulative person-days spent

    all_items = sorted(NOGO_DATA + GO_DATA, key=lambda x: x[4])  # sort by effort

    cum_days = []
    cum_gain = []
    labels_cum = []
    running_days = 0
    running_gain = 0

    for name, d_idf1, _, _, effort, _ in all_items:
        running_days += effort
        running_gain += d_idf1
        cum_days.append(running_days)
        cum_gain.append(running_gain)
        labels_cum.append(name)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        cum_days,
        cum_gain,
        "o-",
        color=C["gmc_dark"],
        linewidth=2,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
    )

    # Highlight GMC step
    for i, (n, d, g) in enumerate(zip(labels_cum, cum_days, cum_gain)):
        if "GMC" in n and "GPU" in n:
            ax.annotate(
                "GMC\n↑+2.8pp",
                (d, g),
                textcoords="offset points",
                xytext=(15, 20),
                fontsize=8,
                color=C["gmc"],
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["gmc"], lw=1.5),
            )

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axhline(running_gain, color=C["gray"], linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Cumulative Experiment Person-Days", fontweight="bold")
    ax.set_ylabel("Cumulative IDF1 Gain (pp)", fontweight="bold")
    ax.set_title(
        "Sequential Evidence Accumulation\n"
        f"Cumulative gain after {running_days} person-days of experiments: {running_gain:+.1f}pp",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.2)

    ax.annotate(
        f"Final: {running_gain:+.1f}pp\nin {running_days} days",
        xy=(cum_days[-1], cum_gain[-1]),
        textcoords="offset points",
        xytext=(-60, -25),
        fontsize=9,
        fontweight="bold",
        color=C["gmc_dark"] if running_gain > 0 else C["bad"],
    )

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_f_evidence_accumulation.png")
    plt.close(fig)
    print("  OK fig_f_evidence_accumulation.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure G — Decision Boundary Model
# ═══════════════════════════════════════════════════════════════════════════
def fig_g_decision_boundary():
    """Model the GO/NO-GO decision rule as a function of effect size and noise."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Decision function
    effect = np.linspace(-5, 5, 200)
    noise_std = 0.15  # estimated run-to-run std

    # Probability of being GO given effect size
    def p_go(e, sigma):
        return 1 / (1 + np.exp(-(e - NOISE_FLOOR_IDF1) / sigma))

    def p_harm(e, sigma):
        return 1 / (1 + np.exp((e + NOISE_FLOOR_IDF1) / sigma))

    p_go_curve = p_go(effect, noise_std)
    p_harm_curve = p_harm(effect, noise_std)

    ax.fill_between(effect, 0, p_go_curve, color=C["good"], alpha=0.15, label="P(GO)")
    ax.fill_between(
        effect, 0, p_harm_curve, color=C["bad"], alpha=0.15, label="P(harmful)"
    )

    ax.plot(effect, p_go_curve, color=C["good"], linewidth=2.5, label="P(GO)")
    ax.plot(effect, p_harm_curve, color=C["bad"], linewidth=2.5, label="P(harmful)")

    # Overlay actual experiments
    for name, d_idf1, _, _, _, _ in NOGO_DATA:
        ax.axvline(d_idf1, alpha=0.3, color=C["bad"], linewidth=0.8, linestyle="--")
    for name, d_idf1, _, _, _, _ in GO_DATA:
        if d_idf1 > 0:
            ax.axvline(
                d_idf1, alpha=0.4, color=C["good"], linewidth=0.8, linestyle="--"
            )

    ax.axvline(
        NOISE_FLOOR_IDF1,
        color=C["noise"],
        linestyle=":",
        linewidth=2,
        alpha=0.8,
        label=f"GO threshold (+{NOISE_FLOOR_IDF1}pp)",
    )
    ax.axvline(
        -NOISE_FLOOR_IDF1,
        color=C["noise"],
        linestyle=":",
        linewidth=2,
        alpha=0.8,
        label=f"Harm threshold (-{NOISE_FLOOR_IDF1}pp)",
    )

    ax.set_xlabel("Observed Effect Size (IDF1 Δ, pp)", fontweight="bold")
    ax.set_ylabel("Decision Probability", fontweight="bold")
    ax.set_title(
        "NO-GO Decision Boundary Model\n"
        "P(GO) and P(harmful) as sigmoid functions of observed effect size",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_g_decision_boundary.png")
    plt.close(fig)
    print("  OK fig_g_decision_boundary.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure H — Category Breakdown Matrix
# ═══════════════════════════════════════════════════════════════════════════
def fig_h_category_matrix():
    categories = [
        "ReID",
        "Detection",
        "Lifecycle",
        "Motion",
        "Geometry",
        "Architecture",
        "Association",
        "Infra",
    ]
    n_cats = len(categories)

    n_no_go = [0] * n_cats
    n_go = [0] * n_cats
    total_effort_nogo = [0] * n_cats
    total_effort_go = [0] * n_cats

    for name, d_idf1, _, _, effort, cat in NOGO_DATA:
        if cat in categories:
            idx = categories.index(cat)
            n_no_go[idx] += 1
            total_effort_nogo[idx] += effort

    for name, d_idf1, _, _, effort, cat in GO_DATA:
        if cat in categories:
            idx = categories.index(cat)
            n_go[idx] += 1
            total_effort_go[idx] += effort

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    x = np.arange(n_cats)
    width = 0.35

    # Count
    bars1 = ax1.bar(
        x - width / 2, n_go, width, label="GO", color=C["good"], edgecolor="white"
    )
    bars2 = ax1.bar(
        x + width / 2, n_no_go, width, label="NO-GO", color=C["bad"], edgecolor="white"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9, rotation=30)
    ax1.set_ylabel("Module Count")
    ax1.set_title("GO vs NO-GO by Module Category", fontsize=11)
    ax1.legend(fontsize=9)

    for bar, v in zip(bars1, n_go):
        if v > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(v),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
    for bar, v in zip(bars2, n_no_go):
        if v > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(v),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    # Effort
    bars3 = ax2.bar(
        x - width / 2,
        total_effort_go,
        width,
        label="GO (days)",
        color=C["good"],
        edgecolor="white",
    )
    bars4 = ax2.bar(
        x + width / 2,
        total_effort_nogo,
        width,
        label="NO-GO (days)",
        color=C["bad"],
        edgecolor="white",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=9, rotation=30)
    ax2.set_ylabel("Total Person-Days")
    ax2.set_title("Effort Spent by Category", fontsize=11)
    ax2.legend(fontsize=9)

    for bar, v in zip(bars3, total_effort_go):
        if v > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(v),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
    for bar, v in zip(bars4, total_effort_nogo):
        if v > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(v),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    fig.suptitle(
        "NO-GO Category Analysis: Where Effort Was Spent vs Results",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_h_category_matrix.png")
    plt.close(fig)
    print("  OK fig_h_category_matrix.png")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating NO-GO mathematical model charts...")
    fig_a_pareto_frontier()
    fig_b_waterfall()
    fig_c_gmc_saturation()
    fig_d_appearance_ceiling()
    fig_e_roi_bubble()
    fig_f_evidence_accumulation()
    fig_g_decision_boundary()
    fig_h_category_matrix()
    print(f"\nDone! 8 charts saved to {OUT_DIR}/")
