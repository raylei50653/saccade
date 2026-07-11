"""Emit or verify the committed Step-0 audit packet (PR #100).

The packet records per-GT-row atom bits and Hamming distances, tail-track
provenance, k=4/5/6/8 occupancies, and the nominal CP diagnostic.  The raw
``pairs.csv`` is intentionally not committed.  Re-emission therefore requires
``--pairs`` and rejects a source whose SHA256 differs from the manifest seal.

``--verify`` rebuilds the packet in a temporary directory and compares every
declared packet file byte-for-byte with the committed packet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
from scipy.stats import beta

PACKET = Path(__file__).resolve().parent
REPO = PACKET.parents[5]
CANONICAL_SOURCE = Path(
    "out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv"
)
sys.path.insert(0, str(REPO / "scripts/tools"))
import audit_relink_safe_reject as ar  # noqa: E402

ATOMS = [
    ("score_m_bridge", True),
    ("bridge_dist", True),
    ("dist_h", True),
    ("log_h_ratio", True),
    ("resid_mean", True),
    ("dir_cos", False),
    ("speed_mismatch", True),
    ("gap", True),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_cp_upper(x: int, n: int, alpha: float = 0.05) -> float:
    """Return the exact one-sided Clopper--Pearson upper bound.

    For nonzero x, the bound is the beta quantile
    ``Beta^{-1}(1-alpha; x+1, n-x)``.
    """

    if not 0 <= x <= n or n <= 0:
        raise ValueError(f"invalid binomial count x={x}, n={n}")
    if x == n:
        return 1.0
    if x == 0:
        return 1.0 - math.exp(math.log(alpha) / n)

    return float(beta.ppf(1.0 - alpha, x + 1, n - x))


def committed_manifest() -> dict[str, object]:
    decoded = json.loads((PACKET / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("committed manifest must be a JSON object")
    return cast(dict[str, object], decoded)


def manifest_files(manifest: dict[str, object]) -> dict[str, str]:
    decoded = manifest.get("files")
    if not isinstance(decoded, dict) or not all(
        isinstance(name, str) and isinstance(digest, str)
        for name, digest in decoded.items()
    ):
        raise ValueError("committed manifest files must map names to SHA256 strings")
    return cast(dict[str, str], decoded)


def verify_source(pairs: Path) -> None:
    if not pairs.is_file():
        raise FileNotFoundError(f"pairs CSV not found: {pairs}")
    expected = str(committed_manifest()["source_pairs_csv_sha256"])
    actual = sha256(pairs)
    if actual != expected:
        raise ValueError(
            "pairs CSV SHA256 does not match the committed manifest: "
            f"expected {expected}, got {actual}"
        )


def emit(pairs: Path, out: Path) -> None:
    """Rebuild packet outputs into *out* from a manifest-verified source."""

    out.mkdir(parents=True, exist_ok=True)
    pool = ar.load_gt_valid_pool(pairs)
    pool["resid_mean"] = 0.5 * (pool["fwd_resid"] + pool["bwd_resid"])
    ar.ensure_prod_proxy_scores(pool)

    y = pool["gt_match"].astype(bool)
    seq = np.asarray(pool["seq"])
    lost = np.asarray(pool["lost_id"], dtype=object)
    track_key = np.array([f"{s}|{lid}" for s, lid in zip(seq, lost)], dtype=object)
    n_rows = y.size

    thresholds: dict[str, dict[str, object]] = {}
    Z = np.zeros((n_rows, len(ATOMS)), dtype=int)
    for j, (name, lower_is_better) in enumerate(ATOMS):
        values = np.asarray(pool[name], float)
        threshold = float(np.nanmedian(values))
        Z[:, j] = (
            (values <= threshold) if lower_is_better else (values >= threshold)
        ).astype(int)
        thresholds[name] = {
            "pool_median_threshold": threshold,
            "safe_side": "<= threshold" if lower_is_better else ">= threshold",
            "p_z1_pool": float(Z[:, j].mean()),
            "p_z1_gt": float(Z[y, j].mean()),
        }

    names = [name for name, _ in ATOMS]
    gt_idx = np.nonzero(y)[0]
    with (out / "gt_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["seq", "lost_id", "track_key"]
            + [f"z_{name}" for name in names]
            + [f"v_{name}" for name in names]
            + ["d_h_k8"]
        )
        for i in gt_idx:
            row_values = [float(np.asarray(pool[name], float)[i]) for name in names]
            writer.writerow(
                [seq[i], lost[i], track_key[i]]
                + Z[i].tolist()
                + [f"{value:.6g}" for value in row_values]
                + [8 - int(Z[i].sum())]
            )

    for k in (4, 5, 6, 8):
        code = (Z[:, :k] * (1 << np.arange(k))).sum(1)
        gt_tracks: dict[int, set[str]] = {}
        fp_rows: dict[int, int] = {}
        for row_index in range(n_rows):
            cell = int(code[row_index])
            if y[row_index]:
                gt_tracks.setdefault(cell, set()).add(str(track_key[row_index]))
            else:
                fp_rows[cell] = fp_rows.get(cell, 0) + 1
        with (out / f"cell_occupancy_k{k}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["cell_code", "bits_atom0_first", "n_gt_tracks", "n_fp_rows"]
            )
            for cell in range(1 << k):
                if cell in gt_tracks or cell in fp_rows:
                    writer.writerow(
                        [
                            cell,
                            format(cell, f"0{k}b")[::-1],
                            len(gt_tracks.get(cell, set())),
                            fp_rows.get(cell, 0),
                        ]
                    )

    hamming = 8 - Z[:, :8].sum(1)
    best: dict[str, int] = {}
    for i in gt_idx:
        key = str(track_key[i])
        best[key] = min(best.get(key, 9), int(hamming[i]))

    per_sequence_tail: dict[str, int] = {}
    tail: dict[str, object] = {}
    for key, distance in best.items():
        if distance < 3:
            continue
        sequence = key.split("|", maxsplit=1)[0]
        per_sequence_tail[sequence] = per_sequence_tail.get(sequence, 0) + 1
        rows = [
            {
                "frame_row_bits_atom0_first": "".join(map(str, Z[i, :8])),
                "d_h": int(hamming[i]),
            }
            for i in gt_idx
            if str(track_key[i]) == key
        ]
        tail[key] = {"min_d_h": distance, "gt_rows": rows}
    (out / "tail_tracks.json").write_text(
        json.dumps(
            {
                "definition": "GT tracks with min over their gt rows of d_H(k=8) >= 3",
                "representation": "descriptive min-d_H layer only (framework §19.4)",
                "n_tail_tracks": len(tail),
                "per_sequence_tail_counts": per_sequence_tail,
                "tracks": tail,
            },
            indent=1,
        )
    )

    n_tracks = len(best)
    distance_histogram: dict[int, int] = {}
    for distance in best.values():
        distance_histogram[distance] = distance_histogram.get(distance, 0) + 1
    (out / "cp_ucb.json").write_text(
        json.dumps(
            {
                "trial_unit": "lost_track(seq,lost_id)",
                "n_tracks": n_tracks,
                "numerator_x": len(tail),
                "numerator_definition": "far-Hamming descriptive tail (k=8, min d_H >= 3)",
                "method": "Clopper-Pearson one-sided 95% upper bound",
                "value_x0": exact_cp_upper(0, n_tracks),
                "value_x": exact_cp_upper(len(tail), n_tracks),
                "status": "nominal; not cluster-adjusted (sequence-level residual clustering declared)",
                "boundary_use": "forbidden for epsilon_morph classification (framework §19.5 UCB validity)",
                "min_d_h_histogram_k8": {
                    str(distance): count
                    for distance, count in sorted(distance_histogram.items())
                },
            },
            indent=1,
        )
    )

    old_manifest = committed_manifest()
    files = {
        name: (out / name) if (out / name).exists() else (PACKET / name)
        for name in manifest_files(old_manifest)
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"packet static files missing: {missing}")
    manifest = {
        "study_id": "gt_support_morphology_step0_20260711",
        "source_pairs_csv": str(CANONICAL_SOURCE),
        "source_pairs_csv_sha256": sha256(pairs),
        "pool_filter": "gt_valid==1 (audit_relink_safe_reject.load_gt_valid_pool)",
        "n_pool_rows": int(n_rows),
        "n_gt_rows": int(y.sum()),
        "n_gt_tracks": n_tracks,
        "atom_order": names,
        "binarization": "pool median (audit-only; best-case occupancy upper bound)",
        "atom_thresholds": thresholds,
        "trial_mapping": "descriptive min-d_H representative; set-valued H_C(u) NOT computed here",
        "files": {name: sha256(path) for name, path in files.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def verify(pairs: Path) -> None:
    """Verify that re-emission reproduces the committed packet exactly."""

    expected_manifest = committed_manifest()
    expected_names = sorted(["manifest.json", *manifest_files(expected_manifest)])
    with tempfile.TemporaryDirectory(prefix="gt-support-morphology-") as tmp:
        rebuilt = Path(tmp) / PACKET.name
        shutil.copytree(PACKET, rebuilt)
        emit(pairs, rebuilt)
        mismatched = [
            name
            for name in expected_names
            if (PACKET / name).read_bytes() != (rebuilt / name).read_bytes()
        ]
    if mismatched:
        raise AssertionError(f"packet is not bit-identical: {mismatched}")
    print("packet verification passed: bit-identical re-emission")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="source pairs.csv")
    parser.add_argument(
        "--out", type=Path, default=PACKET, help="packet output directory"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="rebuild in a temporary directory and compare to the committed packet",
    )
    args = parser.parse_args()
    verify_source(args.pairs)
    if args.verify:
        if args.out != PACKET:
            parser.error("--verify cannot be combined with --out")
        verify(args.pairs)
        return
    emit(args.pairs, args.out)
    print(f"packet emitted: {args.out}")


if __name__ == "__main__":
    main()
