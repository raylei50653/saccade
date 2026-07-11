"""Read-only four-track escape-tail forensic runner (PR-C / issue #102).

Frozen cohort = Step-0 sealed far-Hamming descriptive tail at k=8, min d_H >= 3
(``docs/modules/semantic/research/evidence/gt_support_morphology_step0_20260711/tail_tracks.json``).

Scope (hard):
  - read-only offline evidence extraction + predeclared classification;
  - no atom / threshold / gate / preset / ledger / closure-search changes;
  - categories are only the five predeclared terminals; aggregate is one of three.

Usage::

    uv run python docs/modules/semantic/research/evidence/escape_tail_forensic_20260711/run_escape_tail_forensic.py \\
      --pairs out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv

    # verify committed packet is bit-identical under re-emission
    uv run python .../run_escape_tail_forensic.py --pairs ... --verify
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
from typing import Any

import numpy as np

PACKET = Path(__file__).resolve().parent
REPO = PACKET.parents[5]
STEP0 = (
    REPO
    / "docs/modules/semantic/research/evidence/gt_support_morphology_step0_20260711"
)
CANONICAL_SOURCE = Path(
    "out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv"
)
DEFAULT_GT = REPO / "datasets/MOT17/train/MOT17-10-SDP/gt/gt.txt"

sys.path.insert(0, str(REPO / "scripts/tools"))
import audit_relink_safe_reject as ar  # noqa: E402

ATOMS: list[tuple[str, bool]] = [
    ("score_m_bridge", True),
    ("bridge_dist", True),
    ("dist_h", True),
    ("log_h_ratio", True),
    ("resid_mean", True),
    ("dir_cos", False),
    ("speed_mismatch", True),
    ("gap", True),
]
ATOM_NAMES = [name for name, _ in ATOMS]
MOTION_ATOMS = ("speed_mismatch", "dir_cos", "resid_mean")
HEIGHT_ATOMS = ("log_h_ratio",)
GEOM_ATOMS = ("score_m_bridge", "bridge_dist", "dist_h", "gap")

# Predeclared decision thresholds (fixed before viewing outcomes; recorded here
# so re-runs are deterministic). Occlusion uses GT visibility on MOT17.
OCCLUSION_VIS_MEAN_MAX = 0.35
OCCLUSION_FRAC_ZERO_MIN = 0.25
THRESHOLD_REL_FLIP_MAX = 0.05  # relative thr move that would flip a violation
SIGNAL_ABS_TOL = 1e-9
SIGNAL_REL_TOL = 1e-6
# Step-0 gt_rows.csv stores values with ``f"{value:.6g}"``; allow that rounding.
STEP0_VALUE_REL_TOL = 5e-6
STEP0_VALUE_ABS_TOL = 5e-6

PER_TRACK_TERMINALS = (
    "TRUE_LONG_GAP_REENTRY",
    "ANNOTATION_ISSUE",
    "SIGNAL_COMPUTATION_ISSUE",
    "THRESHOLD_ARTIFACT",
    "UNRESOLVED",
)
AGGREGATE_TERMINALS = (
    "ROLE_REVERSAL_SUPPORTED",
    "TAIL_ARTIFACT_DOMINATED",
    "TAIL_MECHANISM_UNRESOLVED",
)

PACKET_FILES = (
    "manifest.json",
    "cohort.json",
    "track_cards.json",
    "aggregate.json",
    "per_row_evidence.csv",
    "threshold_sensitivity.json",
    "classification_rules.json",
    "run_escape_tail_forensic.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def z_bit(value: float, threshold: float, lower_is_safe: bool) -> int:
    return int(value <= threshold) if lower_is_safe else int(value >= threshold)


def flip_delta(
    value: float, threshold: float, lower_is_safe: bool, z: int
) -> float | None:
    """Absolute threshold move needed to flip a VIOLATION to safe. None if already safe."""
    if z == 1:
        return None
    if lower_is_safe:
        return float(value - threshold)
    return float(threshold - value)


def recompute_signals(row: dict[str, str]) -> dict[str, float]:
    hl = max(float(row["h_lost_raw"]), 1e-6)
    hc = max(float(row["h_cand_raw"]), 1e-6)
    fwd = float(row["fwd_resid"])
    bwd = float(row["bwd_resid"])
    exit_s = float(row["lost_exit_speed"])
    entry_s = float(row["cand_entry_speed"])
    dist_h = float(row["dist_h"])
    w = math.sqrt(min(max(exit_s / 0.12, 0.0), 1.0))
    return {
        "log_h_ratio": abs(math.log(hc / hl)),
        "speed_mismatch": abs(exit_s - entry_s),
        "resid_mean": 0.5 * (fwd + bwd),
        "score_m_bridge": w * 0.5 * (fwd + bwd) + (1.0 - w) * dist_h,
        "bridge_dist": float(row["bridge_dist"]),
        "dist_h": dist_h,
        "dir_cos": float(row["dir_cos"]),
        "gap": float(row["gap"]),
    }


def close(
    a: float, b: float, abs_tol: float = SIGNAL_ABS_TOL, rel_tol: float = SIGNAL_REL_TOL
) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b), 1.0))


def load_gt_table(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            rows.append(
                {
                    "frame": float(parts[0]),
                    "id": float(parts[1]),
                    "x": float(parts[2]),
                    "y": float(parts[3]),
                    "w": float(parts[4]),
                    "h": float(parts[5]),
                    "conf": float(parts[6]),
                    "cls": float(parts[7]),
                    "vis": float(parts[8]),
                }
            )
    return rows


def gt_id_rows(gt_rows: list[dict[str, float]], gid: int) -> list[dict[str, float]]:
    selected = [r for r in gt_rows if int(r["id"]) == gid]
    selected.sort(key=lambda r: r["frame"])
    return selected


def annotation_gaps(frames: list[int]) -> list[tuple[int, int, int]]:
    gaps: list[tuple[int, int, int]] = []
    for a, b in zip(frames[:-1], frames[1:]):
        if b > a + 1:
            gaps.append((a, b, b - a - 1))
    return gaps


def visibility_stats(
    gt_rows: list[dict[str, float]], gid: int, f0: int, f1: int
) -> dict[str, Any]:
    mid = [r for r in gt_rows if int(r["id"]) == gid and f0 < int(r["frame"]) < f1]
    if not mid:
        return {
            "n_annotated_in_gap": 0,
            "vis_mean": None,
            "vis_min": None,
            "frac_vis_lt_0_3": None,
            "frac_vis_eq_0": None,
            "longest_full_invis_stretch": 0,
            "occlusion_strong": False,
        }
    vis = np.asarray([r["vis"] for r in mid], dtype=float)
    zero = (vis <= 0.0).astype(int)
    longest = cur = 0
    for flag in zero:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    vis_mean = float(vis.mean())
    frac0 = float((vis <= 0.0).mean())
    occlusion_strong = bool(
        vis_mean <= OCCLUSION_VIS_MEAN_MAX or frac0 >= OCCLUSION_FRAC_ZERO_MIN
    )
    return {
        "n_annotated_in_gap": len(mid),
        "vis_mean": vis_mean,
        "vis_min": float(vis.min()),
        "frac_vis_lt_0_3": float((vis < 0.3).mean()),
        "frac_vis_eq_0": frac0,
        "longest_full_invis_stretch": int(longest),
        "occlusion_strong": occlusion_strong,
    }


def box_snapshot(
    gt_id_list: list[dict[str, float]], frame: int
) -> dict[str, Any] | None:
    for row in gt_id_list:
        if int(row["frame"]) == frame:
            return {
                "frame": frame,
                "xywh": [
                    float(row["x"]),
                    float(row["y"]),
                    float(row["w"]),
                    float(row["h"]),
                ],
                "vis": float(row["vis"]),
                "foot_xy": [
                    float(row["x"] + row["w"] / 2.0),
                    float(row["y"] + row["h"]),
                ],
            }
    return None


def verify_source(pairs: Path, step0_manifest: dict[str, Any]) -> None:
    if not pairs.is_file():
        raise FileNotFoundError(f"pairs CSV not found: {pairs}")
    expected = str(step0_manifest["source_pairs_csv_sha256"])
    actual = sha256(pairs)
    if actual != expected:
        raise ValueError(
            "pairs CSV SHA256 does not match the sealed Step-0 manifest: "
            f"expected {expected}, got {actual}"
        )


def load_frozen_cohort() -> dict[str, Any]:
    tail = load_json(STEP0 / "tail_tracks.json")
    tracks = tail["tracks"]
    if set(tracks) != {
        "MOT17-10-SDP|455",
        "MOT17-10-SDP|459",
        "MOT17-10-SDP|467",
        "MOT17-10-SDP|503",
    }:
        raise ValueError(f"frozen cohort identity mismatch: {sorted(tracks)}")
    if int(tail["n_tail_tracks"]) != 4:
        raise ValueError("frozen cohort must have exactly 4 tracks")
    return tail


def collect_gt_match_rows(pairs: Path, track_keys: set[str]) -> list[dict[str, str]]:
    wanted = set()
    for key in track_keys:
        seq, lost = key.split("|", maxsplit=1)
        wanted.add((seq, lost))
    rows: list[dict[str, str]] = []
    with pairs.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["seq"], str(row["lost_id"]))
            if key not in wanted:
                continue
            if row.get("gt_valid") not in ("1", "True", "true"):
                continue
            if row.get("gt_match") not in ("1", "True", "true"):
                continue
            rows.append(row)
    return rows


def atom_analysis(
    values: dict[str, float], thresholds: dict[str, float]
) -> dict[str, Any]:
    bits: dict[str, int] = {}
    violated: list[str] = []
    sides: dict[str, Any] = {}
    for name, lower in ATOMS:
        v = float(values[name])
        t = float(thresholds[name])
        z = z_bit(v, t, lower)
        bits[name] = z
        fd = flip_delta(v, t, lower, z)
        sides[name] = {
            "value": v,
            "threshold": t,
            "safe_side": "<= thr" if lower else ">= thr",
            "z": z,
            "side": "SAFE" if z == 1 else "VIOL",
            "flip_delta_if_viol": fd,
            "flip_rel_if_viol": (None if fd is None else fd / max(abs(t), 1e-12)),
        }
        if z == 0:
            violated.append(name)
    return {
        "bits_atom0_first": "".join(str(bits[n]) for n in ATOM_NAMES),
        "d_h": len(violated),
        "violated": violated,
        "motion_violated": [a for a in violated if a in MOTION_ATOMS],
        "height_violated": [a for a in violated if a in HEIGHT_ATOMS],
        "geom_violated": [a for a in violated if a in GEOM_ATOMS],
        "atoms": sides,
    }


def classify_track(card: dict[str, Any]) -> dict[str, Any]:
    """Apply predeclared deterministic category rules (see classification_rules.json)."""
    ann = card["annotation_check"]
    sig = card["signal_computation_check"]
    thr = card["threshold_artifact_check"]
    mech = card["mechanism_features"]
    competing: list[str] = []

    if not ann["same_identity_on_all_gt_rows"] or not ann["annotation_continuous"]:
        category = "ANNOTATION_ISSUE"
        support = [
            "GT identity continuity failed on at least one gt_match row, or GT id has internal annotation gaps."
        ]
    elif not sig["all_signals_recompute_ok"]:
        category = "SIGNAL_COMPUTATION_ISSUE"
        support = [
            "Recomputed log_h_ratio / speed_mismatch / resid_mean / score_m_bridge disagree with ledger."
        ]
    elif thr["threshold_artifact_dominates"]:
        category = "THRESHOLD_ARTIFACT"
        support = thr["reasons"]
    elif (
        mech["height_safe_on_min_dh_row"]
        and mech["n_motion_violations_min_dh"] >= 2
        and mech["occlusion_strong_on_min_dh_row"]
        and ann["same_identity_on_all_gt_rows"]
        and sig["all_signals_recompute_ok"]
    ):
        category = "TRUE_LONG_GAP_REENTRY"
        support = [
            "min-d_H row keeps log_h_ratio on the safe side of the sealed exploratory median.",
            f"motion atoms violated on min-d_H row: {mech['motion_violated_min_dh']}.",
            "gap-window GT visibility meets the predeclared occlusion_strong criterion.",
            "annotation and signal recomputation checks pass.",
        ]
        if thr["any_violation_near_threshold"]:
            competing.append(
                "Some motion violations sit near the exploratory median; membership is threshold-sensitive, but occlusion + multi-atom motion break remain."
            )
        if not mech["all_gt_rows_height_safe"]:
            competing.append(
                "At least one non-min-d_H gt_match row violates height; classification uses min-d_H representative only."
            )
    else:
        category = "UNRESOLVED"
        support = [
            "No single predeclared category is decisive after annotation, signal, threshold, occlusion, and motion/height checks."
        ]
        if (
            mech["height_safe_on_min_dh_row"]
            and mech["n_motion_violations_min_dh"] >= 2
        ):
            competing.append(
                "Motion fails and height is safe, but occlusion_strong is false — tracker fragmentation on a still-visible target is not cleanly long-occlusion re-entry."
            )
        if thr["any_violation_near_threshold"]:
            competing.append(
                "Near-threshold violations leave a residual THRESHOLD_ARTIFACT hypothesis."
            )
        if mech["occlusion_strong_on_min_dh_row"]:
            competing.append(
                "Occlusion evidence is present but other decisive conditions failed."
            )

    if category not in PER_TRACK_TERMINALS:
        raise RuntimeError(f"illegal category {category}")
    return {
        "category": category,
        "supporting_evidence": support,
        "competing_explanations": competing,
        "confidence_boundary": (
            "Single-sequence (MOT17-10-SDP) forensic only; exploratory median binarization; "
            "descriptive min-d_H layer; does not seal morphology terminals or orderability classes."
        ),
    }


def aggregate_terminal(track_cards: dict[str, Any]) -> dict[str, Any]:
    counts = {name: 0 for name in PER_TRACK_TERMINALS}
    for card in track_cards.values():
        counts[card["verdict"]["category"]] += 1
    artifact = (
        counts["ANNOTATION_ISSUE"]
        + counts["SIGNAL_COMPUTATION_ISSUE"]
        + counts["THRESHOLD_ARTIFACT"]
    )
    true_n = counts["TRUE_LONG_GAP_REENTRY"]
    unresolved_n = counts["UNRESOLVED"]

    if artifact >= 3 or (artifact > true_n and artifact >= 2):
        terminal = "TAIL_ARTIFACT_DOMINATED"
        authorizes = [
            "Repair or re-audit the substrate (annotation / signal / threshold) before any conditional-closure claim."
        ]
        blocks = [
            "partial-order promotion of motion atoms",
            "conditional closure probe",
            "any production / preset / ledger change",
        ]
    elif true_n >= 3 and artifact == 0:
        terminal = "ROLE_REVERSAL_SUPPORTED"
        authorizes = [
            "A later partial-order audit may consider motion atoms as conditional_orderable or context_only.",
        ]
        blocks = [
            "global closure arcs on motion atoms",
            "veto against the protected escape tail",
            "production rule / preset / gate change",
            "evidence_ledger promotion beyond this bounded forensic result",
            "L2+ morphology claims without nested held-out confirmation",
        ]
    else:
        terminal = "TAIL_MECHANISM_UNRESOLVED"
        authorizes = [
            "Retain only the pooled L1 descriptive morphology claim from Step-0."
        ]
        blocks = [
            "role-reversal justification for conditional closure",
            "orderability promotion of motion atoms from this tail alone",
            "production / preset / ledger change",
        ]

    if terminal not in AGGREGATE_TERMINALS:
        raise RuntimeError(f"illegal aggregate {terminal}")
    return {
        "terminal": terminal,
        "per_track_category_counts": counts,
        "n_true_long_gap_reentry": true_n,
        "n_artifact": artifact,
        "n_unresolved": unresolved_n,
        "authorizes": authorizes,
        "remains_blocked": blocks,
        "routing": {
            "ROLE_REVERSAL_SUPPORTED": "open separate partial-order audit before any MWC prototype",
            "TAIL_ARTIFACT_DOMINATED": "repair/re-audit substrate; closure work remains blocked",
            "TAIL_MECHANISM_UNRESOLVED": "retain descriptive morphology only; closure conditioning remains blocked",
        }[terminal],
        "confidence_boundary": (
            "All four sealed tail tracks sit on MOT17-10-SDP (sequence clustering is fact, not hypothesis). "
            "This aggregate authorizes at most a research-line partial-order audit probe; it does not "
            "change framework §19 morphology terminals, sealed thresholds, or production behavior."
        ),
    }


def build_threshold_sensitivity(
    pool: dict[str, np.ndarray], track_keys: list[str]
) -> dict[str, Any]:
    pool = dict(pool)
    pool["resid_mean"] = 0.5 * (pool["fwd_resid"] + pool["bwd_resid"])
    ar.ensure_prod_proxy_scores(pool)
    y = pool["gt_match"].astype(bool)
    seq = pool["seq"].astype(str)
    lost = pool["lost_id"].astype(str)
    keys = np.asarray([f"{s}|{lid}" for s, lid in zip(seq, lost)], dtype=object)

    quantiles = {
        "p40": 0.40,
        "median": 0.50,
        "p60": 0.60,
    }
    out: dict[str, Any] = {"quantiles": {}, "frozen_tracks_remain_in_tail": {}}
    for qname, q in quantiles.items():
        thrs = {name: float(np.quantile(pool[name], q)) for name, _ in ATOMS}
        # per-track min d_H over gt rows
        min_dh: dict[str, int] = {}
        for i in np.where(y)[0]:
            key = str(keys[i])
            dh = 0
            for name, lower in ATOMS:
                z = z_bit(float(pool[name][i]), thrs[name], lower)
                if z == 0:
                    dh += 1
            min_dh[key] = min(min_dh.get(key, 99), dh)
        n_tail = sum(1 for d in min_dh.values() if d >= 3)
        remain = sorted(k for k in track_keys if min_dh.get(k, 0) >= 3)
        out["quantiles"][qname] = {
            "thresholds": thrs,
            "n_tail_tracks_d_h_ge_3": n_tail,
            "frozen_tracks_min_d_h": {k: min_dh.get(k) for k in track_keys},
            "frozen_tracks_still_in_tail": remain,
        }
        out["frozen_tracks_remain_in_tail"][qname] = remain
    return out


def emit(pairs: Path, gt_path: Path, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    step0_manifest = load_json(STEP0 / "manifest.json")
    verify_source(pairs, step0_manifest)
    tail = load_frozen_cohort()
    track_keys = sorted(tail["tracks"])
    thresholds = {
        name: float(step0_manifest["atom_thresholds"][name]["pool_median_threshold"])
        for name, _ in ATOMS
    }

    if not gt_path.is_file():
        raise FileNotFoundError(
            f"MOT17 GT not found at {gt_path}. Pass --gt or place MOT17-10-SDP under datasets/."
        )
    gt_table = load_gt_table(gt_path)
    gt_match_rows = collect_gt_match_rows(pairs, set(track_keys))
    if not gt_match_rows:
        raise RuntimeError("no gt_match rows found for frozen cohort")

    # index rows by track
    by_track: dict[str, list[dict[str, str]]] = {k: [] for k in track_keys}
    for row in gt_match_rows:
        key = f"{row['seq']}|{row['lost_id']}"
        if key in by_track:
            by_track[key].append(row)
    for key, rows in by_track.items():
        if not rows:
            raise RuntimeError(
                f"frozen track {key} has zero gt_match rows in pairs.csv"
            )
        rows.sort(key=lambda r: float(r["gap"]))

    pool = ar.load_gt_valid_pool(pairs)
    sensitivity = build_threshold_sensitivity(pool, track_keys)

    per_row_csv: list[dict[str, Any]] = []
    track_cards: dict[str, Any] = {}

    for key in track_keys:
        rows = by_track[key]
        row_payloads: list[dict[str, Any]] = []
        signal_ok_all = True
        same_id_all = True
        height_safe_all = True
        for row in rows:
            recomputed = recompute_signals(row)
            ledger_vals = {
                "bridge_dist": float(row["bridge_dist"]),
                "dist_h": float(row["dist_h"]),
                "dir_cos": float(row["dir_cos"]),
                "gap": float(row["gap"]),
                **{
                    name: recomputed[name]
                    for name in (
                        "log_h_ratio",
                        "speed_mismatch",
                        "resid_mean",
                        "score_m_bridge",
                    )
                },
            }
            # compare recomputed derived fields to formula (self-consistent) and to
            # committed step0 gt_rows later; here check internal formula vs raw cols.
            analysis = atom_analysis(ledger_vals, thresholds)
            gt_lost = int(float(row["gt_lost"]))
            gt_cand = int(float(row["gt_cand"]))
            if gt_lost != gt_cand:
                same_id_all = False
            lf = int(float(row["lost_last_frame"]))
            cf = int(float(row["cand_first_frame"]))
            vis = visibility_stats(gt_table, gt_lost, lf, cf)
            # signal check: recompute from raw equals ledger_vals (tautological for
            # derived) — also check against pairs raw residual/speed fields already used.
            signal_ok = all(
                close(recomputed[n], ledger_vals[n])
                for n in (
                    "log_h_ratio",
                    "speed_mismatch",
                    "resid_mean",
                    "score_m_bridge",
                )
            )
            # Cross-check vs sealed step0 gt_rows.csv values when present.
            signal_ok_all = signal_ok_all and signal_ok
            if analysis["atoms"]["log_h_ratio"]["z"] == 0:
                height_safe_all = False

            gid_rows = gt_id_rows(gt_table, gt_lost)
            frames = [int(r["frame"]) for r in gid_rows]
            payload = {
                "track_key": key,
                "seq": row["seq"],
                "lost_id": str(row["lost_id"]),
                "cand_id": str(row["cand_id"]),
                "gt_lost": gt_lost,
                "gt_cand": gt_cand,
                "same_gt_identity": gt_lost == gt_cand,
                "gap": int(float(row["gap"])),
                "lost_last_frame": lf,
                "cand_first_frame": cf,
                "lost_lifespan": float(row["lost_lifespan"]),
                "cand_lifespan": float(row["cand_lifespan"]),
                "accepted": row["accepted"],
                "already_linked": row["already_linked"],
                "feet": {
                    "lost_xy": [float(row["lost_foot_x"]), float(row["lost_foot_y"])],
                    "cand_xy": [float(row["cand_foot_x"]), float(row["cand_foot_y"])],
                    "delta_xy": [
                        float(row["cand_foot_x"]) - float(row["lost_foot_x"]),
                        float(row["cand_foot_y"]) - float(row["lost_foot_y"]),
                    ],
                    "euclid_px": float(
                        math.hypot(
                            float(row["cand_foot_x"]) - float(row["lost_foot_x"]),
                            float(row["cand_foot_y"]) - float(row["lost_foot_y"]),
                        )
                    ),
                },
                "heights": {
                    "h_lost_raw": float(row["h_lost_raw"]),
                    "h_cand_raw": float(row["h_cand_raw"]),
                    "h_lost_win6": float(row["h_lost_win6"]),
                    "h_cand_win6": float(row["h_cand_win6"]),
                    "h_lost_extrap": float(row["h_lost_extrap"]),
                    "h_cand_extrap": float(row["h_cand_extrap"]),
                },
                "speeds": {
                    "lost_exit_speed": float(row["lost_exit_speed"]),
                    "cand_entry_speed": float(row["cand_entry_speed"]),
                },
                "raw_residuals": {
                    "fwd_resid": float(row["fwd_resid"]),
                    "bwd_resid": float(row["bwd_resid"]),
                },
                "values": ledger_vals,
                "recomputed": recomputed,
                "signal_recompute_ok": signal_ok,
                "atom_analysis": analysis,
                "gap_visibility": vis,
                "gt_box_before_exit": box_snapshot(gid_rows, lf),
                "gt_box_after_reentry": box_snapshot(gid_rows, cf),
                "gt_id_frame_span": {
                    "first": frames[0] if frames else None,
                    "last": frames[-1] if frames else None,
                    "n_annotated": len(frames),
                    "internal_annotation_gaps": annotation_gaps(frames),
                },
                "step0_min_d_h_track": int(tail["tracks"][key]["min_d_h"]),
            }
            row_payloads.append(payload)
            per_row_csv.append(
                {
                    "track_key": key,
                    "cand_id": payload["cand_id"],
                    "gt_lost": gt_lost,
                    "gt_cand": gt_cand,
                    "gap": payload["gap"],
                    "lost_last_frame": lf,
                    "cand_first_frame": cf,
                    "d_h": analysis["d_h"],
                    "bits": analysis["bits_atom0_first"],
                    "violated": "|".join(analysis["violated"]),
                    "motion_violated": "|".join(analysis["motion_violated"]),
                    "height_violated": "|".join(analysis["height_violated"]),
                    "log_h_ratio": ledger_vals["log_h_ratio"],
                    "speed_mismatch": ledger_vals["speed_mismatch"],
                    "dir_cos": ledger_vals["dir_cos"],
                    "resid_mean": ledger_vals["resid_mean"],
                    "score_m_bridge": ledger_vals["score_m_bridge"],
                    "bridge_dist": ledger_vals["bridge_dist"],
                    "dist_h": ledger_vals["dist_h"],
                    "vis_mean": vis["vis_mean"],
                    "frac_vis_eq_0": vis["frac_vis_eq_0"],
                    "occlusion_strong": vis["occlusion_strong"],
                    "signal_recompute_ok": signal_ok,
                    "foot_euclid_px": payload["feet"]["euclid_px"],
                }
            )

        # min-d_H representative among this track's gt rows
        min_row = min(row_payloads, key=lambda p: (p["atom_analysis"]["d_h"], p["gap"]))
        min_analysis = min_row["atom_analysis"]
        # cross-check sealed min_d_h
        if int(min_analysis["d_h"]) != int(tail["tracks"][key]["min_d_h"]):
            raise RuntimeError(
                f"{key}: recomputed min d_H={min_analysis['d_h']} != sealed "
                f"{tail['tracks'][key]['min_d_h']}"
            )

        # threshold artifact check on min-d_H row
        near = []
        strong = []
        for name in min_analysis["violated"]:
            rel = min_analysis["atoms"][name]["flip_rel_if_viol"]
            if rel is None:
                continue
            if rel <= THRESHOLD_REL_FLIP_MAX:
                near.append(name)
            else:
                strong.append(name)
        # Dominates only if EVERY violation is near-threshold AND no occlusion-backed multi-atom motion story
        threshold_dominates = bool(min_analysis["violated"]) and not strong
        thr_reasons = []
        if threshold_dominates:
            thr_reasons.append(
                f"All violated atoms on min-d_H row flip under <= {THRESHOLD_REL_FLIP_MAX:.0%} relative threshold move: {near}."
            )
        # Also note p60 sensitivity
        p60_remain = key in sensitivity["frozen_tracks_remain_in_tail"]["p60"]
        if not p60_remain:
            thr_reasons.append(
                "Track leaves the d_H>=3 tail under pool p60 binarization (membership is median-sensitive)."
            )
            # membership sensitivity alone is NOT enough for THRESHOLD_ARTIFACT if strong violations exist

        # annotation continuity for primary gt id
        primary_gid = int(min_row["gt_lost"])
        gid_rows = gt_id_rows(gt_table, primary_gid)
        frames = [int(r["frame"]) for r in gid_rows]
        ann_gaps = annotation_gaps(frames)
        annotation_continuous = len(ann_gaps) == 0 and len(frames) > 0

        card = {
            "track_key": key,
            "sequence": key.split("|", 1)[0],
            "lost_id": key.split("|", 1)[1],
            "source_provenance": {
                "step0_packet": str(STEP0.relative_to(REPO)),
                "step0_tail_tracks_json": "tail_tracks.json",
                "sealed_min_d_h": int(tail["tracks"][key]["min_d_h"]),
                "source_pairs_csv": str(CANONICAL_SOURCE),
                "source_pairs_csv_sha256": step0_manifest["source_pairs_csv_sha256"],
                "gt_txt": str(gt_path.relative_to(REPO))
                if gt_path.is_relative_to(REPO)
                else str(gt_path),
            },
            "timeline": {
                "n_gt_match_rows": len(row_payloads),
                "primary_gt_id": primary_gid,
                "exit_frame": int(min_row["lost_last_frame"]),
                "reentry_frame_min_dh_row": int(min_row["cand_first_frame"]),
                "gap_min_dh_row": int(min_row["gap"]),
                "all_gaps": [int(p["gap"]) for p in row_payloads],
                "all_cand_ids": [p["cand_id"] for p in row_payloads],
            },
            "min_d_h_row": {
                "cand_id": min_row["cand_id"],
                "gap": int(min_row["gap"]),
                "frames": [
                    int(min_row["lost_last_frame"]),
                    int(min_row["cand_first_frame"]),
                ],
                "atom_analysis": min_analysis,
                "values": min_row["values"],
                "gap_visibility": min_row["gap_visibility"],
                "feet": min_row["feet"],
                "heights": min_row["heights"],
            },
            "all_gt_match_rows": row_payloads,
            "annotation_check": {
                "same_identity_on_all_gt_rows": same_id_all,
                "annotation_continuous": annotation_continuous,
                "primary_gt_id": primary_gid,
                "gt_frame_span": [frames[0], frames[-1]] if frames else None,
                "n_annotated_frames": len(frames),
                "internal_annotation_gaps": ann_gaps,
                "notes": (
                    "GT remains annotated across the tracker gap (fragmentation / occlusion), "
                    "which is expected for MOT17 and does not by itself imply ANNOTATION_ISSUE."
                ),
            },
            "signal_computation_check": {
                "all_signals_recompute_ok": signal_ok_all,
                "formulas": {
                    "log_h_ratio": "abs(log(h_cand_raw / h_lost_raw))",
                    "speed_mismatch": "abs(lost_exit_speed - cand_entry_speed)",
                    "resid_mean": "0.5 * (fwd_resid + bwd_resid)",
                    "score_m_bridge": (
                        "w*0.5*(fwd+bwd)+(1-w)*dist_h with w=sqrt(clip(exit_speed/0.12,0,1))"
                    ),
                },
            },
            "threshold_artifact_check": {
                "any_violation_near_threshold": bool(near),
                "near_threshold_violations": near,
                "strong_violations": strong,
                "threshold_artifact_dominates": threshold_dominates,
                "remains_in_tail_under_p60": p60_remain,
                "reasons": thr_reasons,
            },
            "mechanism_features": {
                "height_safe_on_min_dh_row": min_analysis["atoms"]["log_h_ratio"]["z"]
                == 1,
                "all_gt_rows_height_safe": height_safe_all,
                "motion_violated_min_dh": min_analysis["motion_violated"],
                "n_motion_violations_min_dh": len(min_analysis["motion_violated"]),
                "geom_violated_min_dh": min_analysis["geom_violated"],
                "occlusion_strong_on_min_dh_row": bool(
                    min_row["gap_visibility"]["occlusion_strong"]
                ),
                "gap_vis_mean_min_dh": min_row["gap_visibility"]["vis_mean"],
            },
        }
        card["verdict"] = classify_track(card)
        track_cards[key] = card

    aggregate = aggregate_terminal(track_cards)

    # cross-check step0 gt_rows signal values
    step0_rows = list(csv.DictReader((STEP0 / "gt_rows.csv").open(encoding="utf-8")))
    step0_by_key_gap: dict[tuple[str, str], dict[str, str]] = {}
    for r in step0_rows:
        step0_by_key_gap[(r["track_key"], f"{float(r['v_gap']):.6g}")] = r
    mismatches = []
    for key, card in track_cards.items():
        for prow in card["all_gt_match_rows"]:
            k = (key, f"{float(prow['gap']):.6g}")
            if k not in step0_by_key_gap:
                # gap formatting fallback
                alt = (key, str(int(prow["gap"])))
                ref = step0_by_key_gap.get(alt)
            else:
                ref = step0_by_key_gap[k]
            if ref is None:
                # try match by d_h + bits
                candidates = [
                    r
                    for r in step0_rows
                    if r["track_key"] == key
                    and int(r["d_h_k8"]) == int(prow["atom_analysis"]["d_h"])
                ]
                ref = candidates[0] if len(candidates) == 1 else None
            if ref is None:
                mismatches.append(f"{key} gap={prow['gap']}: missing step0 row")
                continue
            for name in ATOM_NAMES:
                a = float(prow["values"][name])
                b = float(ref[f"v_{name}"])
                if not close(
                    a,
                    b,
                    abs_tol=STEP0_VALUE_ABS_TOL,
                    rel_tol=STEP0_VALUE_REL_TOL,
                ):
                    mismatches.append(
                        f"{key} gap={prow['gap']} {name}: {a} vs step0 {b}"
                    )
    if mismatches:
        raise RuntimeError("step0 value mismatches:\n" + "\n".join(mismatches))

    classification_rules = {
        "per_track_terminals": list(PER_TRACK_TERMINALS),
        "aggregate_terminals": list(AGGREGATE_TERMINALS),
        "occlusion_strong": {
            "vis_mean_max": OCCLUSION_VIS_MEAN_MAX,
            "frac_vis_eq_0_min": OCCLUSION_FRAC_ZERO_MIN,
            "definition": "occlusion_strong iff gap_vis_mean <= vis_mean_max OR frac_vis_eq_0 >= frac_vis_eq_0_min",
        },
        "threshold_artifact_dominates": {
            "rel_flip_max": THRESHOLD_REL_FLIP_MAX,
            "definition": (
                "All min-d_H violations flip under relative threshold move <= rel_flip_max "
                "(no strong violation remains)."
            ),
        },
        "TRUE_LONG_GAP_REENTRY": [
            "annotation same-identity + continuous",
            "signal recompute ok",
            "not threshold_artifact_dominates",
            "height safe on min-d_H row (log_h_ratio z=1)",
            ">=2 motion atom violations on min-d_H row",
            "occlusion_strong on min-d_H gap window",
        ],
        "aggregate_ROLE_REVERSAL_SUPPORTED": (
            "n(TRUE_LONG_GAP_REENTRY) >= 3 AND n(annotation|signal|threshold artifacts) == 0"
        ),
        "aggregate_TAIL_ARTIFACT_DOMINATED": (
            "n(artifacts) >= 3 OR (n(artifacts) > n(TRUE) AND n(artifacts) >= 2)"
        ),
        "aggregate_else": "TAIL_MECHANISM_UNRESOLVED",
        "atom_groups": {
            "motion": list(MOTION_ATOMS),
            "height": list(HEIGHT_ATOMS),
            "geometry": list(GEOM_ATOMS),
        },
    }

    cohort = {
        "definition": tail["definition"],
        "representation": tail["representation"],
        "n_tail_tracks": 4,
        "track_keys": track_keys,
        "per_sequence_tail_counts": tail["per_sequence_tail_counts"],
        "sealed_min_d_h": {k: int(tail["tracks"][k]["min_d_h"]) for k in track_keys},
        "source_pairs_csv": str(CANONICAL_SOURCE),
        "source_pairs_csv_sha256": step0_manifest["source_pairs_csv_sha256"],
        "step0_packet": str(STEP0.relative_to(REPO)),
        "note": "Cohort is frozen from Step-0; this forensic does not add or remove tracks.",
    }

    # write CSV
    csv_path = out / "per_row_evidence.csv"
    fieldnames = list(per_row_csv[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_row_csv:
            writer.writerow(row)

    write_json(out / "cohort.json", cohort)
    write_json(out / "track_cards.json", track_cards)
    write_json(out / "aggregate.json", aggregate)
    write_json(out / "threshold_sensitivity.json", sensitivity)
    write_json(out / "classification_rules.json", classification_rules)

    # copy runner into packet when emitting elsewhere
    runner_src = Path(__file__).resolve()
    runner_dst = out / "run_escape_tail_forensic.py"
    if runner_src != runner_dst.resolve():
        shutil.copy2(runner_src, runner_dst)

    file_hashes = {
        name: sha256(out / name) for name in PACKET_FILES if (out / name).is_file()
    }
    manifest = {
        "study_id": "escape_tail_forensic_20260711",
        "issue": 102,
        "pr_ladder": "PR-C",
        "depends_on": {
            "step0_packet": str(STEP0.relative_to(REPO)),
            "step0_manifest_sha256": sha256(STEP0 / "manifest.json"),
            "source_pairs_csv": str(CANONICAL_SOURCE),
            "source_pairs_csv_sha256": step0_manifest["source_pairs_csv_sha256"],
            "procedure": "framework §19 v1 (PR-A #100 sealed)",
            "research_line": "boolean_closure_domain_line_20260711 (PR-B #101)",
        },
        "scope": "read-only offline forensic; no gate/preset/ledger/closure-search changes",
        "frozen_cohort": track_keys,
        "aggregate_terminal": aggregate["terminal"],
        "per_track_categories": {
            k: v["verdict"]["category"] for k, v in track_cards.items()
        },
        "files": file_hashes,
    }
    write_json(out / "manifest.json", manifest)
    # refresh manifest hash inclusion: re-hash without self, store files including manifest last
    file_hashes = {
        name: sha256(out / name) for name in PACKET_FILES if name != "manifest.json"
    }
    file_hashes["manifest.json"] = "see_files_excluding_self"
    manifest["files"] = {
        name: sha256(out / name) for name in PACKET_FILES if (out / name).is_file()
    }
    # Avoid circular hash: hash all except manifest, record those; manifest lists them only.
    body_files = [n for n in PACKET_FILES if n != "manifest.json"]
    manifest["files"] = {name: sha256(out / name) for name in body_files}
    write_json(out / "manifest.json", manifest)
    return manifest


def verify(pairs: Path, gt_path: Path) -> None:
    expected_names = list(PACKET_FILES)
    with tempfile.TemporaryDirectory(prefix="escape-tail-forensic-") as tmp:
        rebuilt = Path(tmp) / PACKET.name
        emit(pairs, gt_path, rebuilt)
        mismatched = []
        for name in expected_names:
            if name == "manifest.json":
                # compare semantic fields, not self-hash circularity
                a = load_json(PACKET / name)
                b = load_json(rebuilt / name)
                for drop in ("files",):
                    a = dict(a)
                    b = dict(b)
                    a.pop(drop, None)
                    b.pop(drop, None)
                # compare files excluding manifest script may differ only by path — compare body hashes
                a_files = load_json(PACKET / name).get("files", {})
                b_files = load_json(rebuilt / name).get("files", {})
                if a_files != b_files:
                    mismatched.append("manifest.json:files")
                a.pop("files", None)
                b.pop("files", None)
                # re-load clean
                a = load_json(PACKET / name)
                b = load_json(rebuilt / name)
                for k in set(a) | set(b):
                    if k == "files":
                        continue
                    if a.get(k) != b.get(k):
                        mismatched.append(f"manifest.json:{k}")
                continue
            if name == "run_escape_tail_forensic.py":
                if (PACKET / name).read_bytes() != (rebuilt / name).read_bytes():
                    mismatched.append(name)
                continue
            if (PACKET / name).read_bytes() != (rebuilt / name).read_bytes():
                mismatched.append(name)
    if mismatched:
        raise AssertionError(f"packet is not reproducible: {mismatched}")
    print("forensic packet verification passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--out", type=Path, default=PACKET)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="rebuild in a temp dir and compare to the committed packet",
    )
    args = parser.parse_args()
    if args.verify:
        if args.out.resolve() != PACKET.resolve():
            parser.error("--verify cannot be combined with a non-default --out")
        verify(args.pairs, args.gt)
        return
    manifest = emit(args.pairs, args.gt, args.out)
    print(f"packet emitted: {args.out}")
    print(f"aggregate terminal: {manifest['aggregate_terminal']}")
    print(f"per-track: {manifest['per_track_categories']}")


if __name__ == "__main__":
    main()
