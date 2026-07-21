#!/usr/bin/env python3
"""Analyze relink feature stats: Gaussian separation per gap bin."""
# status: diagnostic

import struct
import sys
import math
from pathlib import Path

STAT_BINS = 5
STAT_SOURCES = 2
STAT_OUTCOMES = 8
STAT_FEATURES = 4
STAT_CELL_SZ = 3
STAT_TOTAL = STAT_BINS * STAT_SOURCES * STAT_OUTCOMES * STAT_FEATURES * STAT_CELL_SZ

GAP_LABELS = ["2-5", "6-15", "16-30", "31-45", "46-120"]
SOURCE_LABELS = ["live", "archive"]
OUTCOME_LABELS = [
    "accepted",
    "rej_bridge_px",
    "rej_kalman",
    "rej_direction",
    "rej_speed",
    "rej_margin",
    "rej_backward",
    "rej_other",
]
FEATURE_LABELS = ["bridge", "kalman_d2", "dir_cos", "speed"]


def stat_idx(bin_id: int, src: int, outcome: int, feat: int) -> int:
    return (
        ((bin_id * STAT_SOURCES + src) * STAT_OUTCOMES + outcome) * STAT_FEATURES + feat
    ) * STAT_CELL_SZ


def load_stats(path: Path) -> list[float]:
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"{len(data) // 4}f", data))


def merge_stats(paths: list[Path]) -> list[float]:
    merged = [0.0] * STAT_TOTAL
    for p in paths:
        data = load_stats(p)
        for i in range(STAT_TOTAL):
            merged[i] += data[i]
    return merged


def cell_stats(stats: list[float], bin_id: int, src: int, outcome: int, feat: int):
    """Return (count, mean, std) for a cell."""
    base = stat_idx(bin_id, src, outcome, feat)
    n = stats[base + 0]
    if n < 1:
        return 0, 0, 0
    mean = stats[base + 1] / n
    var = stats[base + 2] / n - mean * mean
    if var < 0:
        var = 0
    return int(n), mean, math.sqrt(var)


def overlap_coeff(mu1, s1, n1, mu2, s2, n2):
    """Bhattacharyya coefficient between two Gaussians (0=perfectly separated, 1=identical)."""
    if s1 < 1e-9 or s2 < 1e-9:
        return 0.0 if abs(mu1 - mu2) > 1e-6 else 1.0
    var1, var2 = s1 * s1, s2 * s2
    term1 = 0.25 * (mu1 - mu2) ** 2 / (var1 + var2)
    term2 = 0.5 * math.log((var1 + var2) / (2 * s1 * s2))
    bc = math.exp(-(term1 + term2))
    return bc


def cohens_d(mu1, s1, mu2, s2):
    """Cohen's d effect size."""
    pooled = math.sqrt((s1 * s1 + s2 * s2) / 2.0)
    if pooled < 1e-9:
        return 0.0
    return abs(mu1 - mu2) / pooled


def optimal_threshold(mu_a, s_a, mu_r, s_r):
    """Optimal threshold at equal error rate (Gaussian assumption)."""
    if s_a < 1e-9 or s_r < 1e-9:
        return (mu_a + mu_r) / 2.0
    a, b = 1.0 / (2 * s_a * s_a), 1.0 / (2 * s_r * s_r)
    disc = (a * mu_a - b * mu_r) ** 2 - (a - b) * (
        a * mu_a * mu_a - b * mu_r * mu_r + math.log(s_r / s_a)
    )
    if disc < 0:
        return (mu_a + mu_r) / 2.0
    sqrt_disc = math.sqrt(disc)
    t1 = (
        (a * mu_a - b * mu_r + sqrt_disc) / (a - b)
        if abs(a - b) > 1e-9
        else (mu_a + mu_r) / 2.0
    )
    t2 = (
        (a * mu_a - b * mu_r - sqrt_disc) / (a - b)
        if abs(a - b) > 1e-9
        else (mu_a + mu_r) / 2.0
    )
    if min(mu_a, mu_r) <= t1 <= max(mu_a, mu_r):
        return t1
    return t2


def analyze(stats: list[float], src_label: str, src_id: int):
    """Print per-feature, per-gap-bin separation analysis."""
    print(f"\n{'=' * 90}")
    print(f"Source: {src_label}")
    print(f"{'=' * 90}")

    for feat_id in range(STAT_FEATURES):
        print(f"\n--- {FEATURE_LABELS[feat_id]} ---")
        print(
            f"{'gap':>8} {'accept_n':>8} {'accept_μ':>10} {'accept_σ':>8} "
            f"{'reject_n':>8} {'reject_μ':>10} {'reject_σ':>8} "
            f"{'Cohen_d':>8} {'overlap':>8} {'opt_thr':>10} {'weight':>8}"
        )

        weights = []
        for bin_id in range(STAT_BINS):
            n_a, mu_a, s_a = cell_stats(stats, bin_id, src_id, 0, feat_id)  # accepted

            # Merge all rejection outcomes for this feature
            n_r, sum_r, sum_sq_r = 0, 0.0, 0.0
            for out_id in range(1, STAT_OUTCOMES):
                nr, mr, sr = cell_stats(stats, bin_id, src_id, out_id, feat_id)
                if nr > 0:
                    n_r += nr
                    sum_r += stats[stat_idx(bin_id, src_id, out_id, feat_id) + 1]
                    sum_sq_r += stats[stat_idx(bin_id, src_id, out_id, feat_id) + 2]

            if n_a < 2 or n_r < 2:
                weights.append(0.0)
                print(
                    f"{GAP_LABELS[bin_id]:>8} {n_a:>8} {mu_a:>10.4f} {s_a:>8.4f} "
                    f"{'n/a':>8} {'n/a':>10} {'n/a':>8} {'n/a':>8} {'n/a':>8} {'n/a':>10}"
                )
                continue

            mu_r = sum_r / n_r
            var_r = sum_sq_r / n_r - mu_r * mu_r
            if var_r < 0:
                var_r = 0
            s_r = math.sqrt(var_r)

            d = cohens_d(mu_a, s_a, mu_r, s_r)
            bc = overlap_coeff(mu_a, s_a, n_a, mu_r, s_r, n_r)
            thr = optimal_threshold(mu_a, s_a, mu_r, s_r)
            w = d * (1.0 - bc)  # simple weight: effect size × (1 - overlap)
            weights.append(w)

            print(
                f"{GAP_LABELS[bin_id]:>8} {n_a:>8} {mu_a:>10.4f} {s_a:>8.4f} "
                f"{n_r:>8} {mu_r:>10.4f} {s_r:>8.4f} "
                f"{d:>8.3f} {bc:>8.3f} {thr:>10.4f} {w:>8.3f}"
            )

        # Normalize weights
        if weights:
            sum_w = sum(weights)
            if sum_w > 0:
                norm_w = [w / sum_w for w in weights]
                print(f"  normalised weights: {[f'{w:.3f}' for w in norm_w]}")

        # Per-outcome breakdown for bridge feature
        if feat_id == 0:  # bridge
            print("\n  Per-outcome breakdown for bridge:")
            print(f"  {'outcome':<18} {'gap':>6}", end="")
            for bin_id in range(STAT_BINS):
                print(f"  {GAP_LABELS[bin_id]:>12}", end="")
            print()
            for out_id in range(STAT_OUTCOMES):
                n_out, _mu_out, _s_out = 0, 0, 0
                for bin_id in range(STAT_BINS):
                    n, mu, _ = cell_stats(stats, bin_id, src_id, out_id, feat_id)
                    n_out += n
                if n_out < 1:
                    continue
                print(f"  {OUTCOME_LABELS[out_id]:<18} {n_out:>6}", end="")
                for bin_id in range(STAT_BINS):
                    n, mu, s = cell_stats(stats, bin_id, src_id, out_id, feat_id)
                    if n > 0:
                        print(f"  {mu:>8.4f}({n:4d})", end="")
                    else:
                        print(f"  {'':>12}", end="")
                print()


def main():
    data_dir = Path("datasets/MOT17/train")
    paths = sorted(data_dir.glob("*_relink_stats.bin"))
    if not paths:
        print("No stats files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(paths)} stats files...")
    merged = merge_stats(paths)

    analyze(merged, "live", 0)
    analyze(merged, "archive", 1)

    # Summary: accepted counts per gap bin
    print(f"\n{'=' * 60}")
    print("Accepted relinks per gap bin")
    print(f"{'gap':>8} {'live':>8} {'archive':>6} {'total':>6}")
    total = 0
    for bin_id in range(STAT_BINS):
        n_live, _, _ = cell_stats(merged, bin_id, 0, 0, 0)
        n_arch, _, _ = cell_stats(merged, bin_id, 1, 0, 0)
        total += n_live + n_arch
        print(f"{GAP_LABELS[bin_id]:>8} {n_live:>8} {n_arch:>6} {n_live + n_arch:>6}")
    print(f"{'total':>8} {'':>8} {'':>6} {total:>6}")


if __name__ == "__main__":
    main()
