"""Read-only four-track escape-tail forensic runner (PR-C / issue #102).

Frozen cohort = Step-0 sealed far-Hamming descriptive tail at k=8, min d_H >= 3
(``.../gt_support_morphology_step0_20260711/tail_tracks.json``).

Scope (hard):
  - read-only offline evidence extraction + category assignment;
  - no atom / threshold / gate / preset / ledger / closure-search changes;
  - per-track categories are only the five #102 terminals; aggregate is one of three.

**Rule provenance (research-owner review, PR #104):**
  Issue #102 predeclared the *terminal vocabulary* and qualitative meanings only.
  Numerical operational cutoffs used by this runner
  (``occlusion_strong``, ``>=2`` motion violations, aggregate ``>=3 TRUE``)
  are **PR-C implementation-time operationalizations**, not sealed in #102.
  They are recorded for reproducibility and remain subject to research-owner
  interpretation before research acceptance.

**Signal-computation check (non-tautological):**
  Derived atoms are recomputed from pairs.csv *raw* columns under declared
  formulas and compared to the *independent sealed* Step-0 ``gt_rows.csv``
  values (plus frame-gap consistency). Builder-emitted residual/speed/dir
  trajectories are **not** pixel-replayed; that residual risk is declared
  untested at the substrate level.

Usage::

    uv run python docs/modules/semantic/research/evidence/escape_tail_forensic_20260711/run_escape_tail_forensic.py \\
      --pairs out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv

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

import cv2
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
DEFAULT_IMG_ROOT = REPO / "datasets/MOT17/train/MOT17-10-SDP/img1"

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
DERIVED_ATOMS = ("log_h_ratio", "speed_mismatch", "resid_mean", "score_m_bridge")

# ---------------------------------------------------------------------------
# PR-C implementation-time operationalization (NOT sealed in issue #102).
# Recorded for deterministic re-runs; research acceptance may override.
# ---------------------------------------------------------------------------
OCCLUSION_VIS_MEAN_MAX = 0.35
OCCLUSION_FRAC_ZERO_MIN = 0.25
MIN_MOTION_VIOLATIONS_FOR_TRUE = 2
AGGREGATE_MIN_TRUE_FOR_ROLE_REVERSAL = 3
THRESHOLD_REL_FLIP_MAX = 0.05
NEARBY_TOP_K = 5
TRUNCATION_MARGIN_PX = 8
SCENE_MID_SAMPLES = 3
CONTACT_SHEET_MAX_WIDTH = 1600

SIGNAL_ABS_TOL = 1e-9
SIGNAL_REL_TOL = 1e-6
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

# Research-owner stamp (PR #104 review, 2026-07-11). Emitted into committed packet.
RESEARCH_ACCEPTANCE: dict[str, Any] = {
    "status": "ACCEPTED_WITH_LIMITS",
    "recorded": "2026-07-11",
    "authority": "PR #104 research-owner review",
    "pr_url": "https://github.com/raylei50653/saccade/pull/104",
    "issue": 102,
    "accepted_aggregate": "ROLE_REVERSAL_SUPPORTED",
    "accepted_per_track": {
        "MOT17-10-SDP|455": "TRUE_LONG_GAP_REENTRY",
        "MOT17-10-SDP|459": "UNRESOLVED",
        "MOT17-10-SDP|467": "TRUE_LONG_GAP_REENTRY",
        "MOT17-10-SDP|503": "TRUE_LONG_GAP_REENTRY",
    },
    "claim_ceiling": "L1 single-sequence forensic (MOT17-10-SDP only)",
    "authorizes": [
        "a separate partial-order audit treating motion atoms as conditional_orderable / context_only candidates",
    ],
    "does_not_authorize": [
        "global motion closure arcs",
        "MWC conclusions",
        "escape-tail veto",
        "production / preset / gate changes",
        "evidence_ledger promotion",
        "multi-sequence generalization from this cohort alone",
        "L2+ morphology claims without nested confirmation",
    ],
}

# Body files hashed in manifest (manifest itself excluded to avoid self-hash).
PACKET_BODY_FILES = (
    "cohort.json",
    "track_cards.json",
    "aggregate.json",
    "per_row_evidence.csv",
    "threshold_sensitivity.json",
    "classification_rules.json",
    "scene_evidence.json",
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
    if z == 1:
        return None
    if lower_is_safe:
        return float(value - threshold)
    return float(threshold - value)


def close(
    a: float,
    b: float,
    abs_tol: float = SIGNAL_ABS_TOL,
    rel_tol: float = SIGNAL_REL_TOL,
) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b), 1.0))


def recompute_derived_from_pairs_raw(row: dict[str, str]) -> dict[str, float]:
    """Derive atom values from pairs.csv raw columns under declared formulas."""
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


def box_at(gt_id_list: list[dict[str, float]], frame: int) -> dict[str, Any] | None:
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
            "occlusion_strong_operational": False,
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
        "occlusion_strong_operational": occlusion_strong,
    }


def truncation_flags(
    xywh: list[float], frame_w: int, frame_h: int, margin: int = TRUNCATION_MARGIN_PX
) -> dict[str, Any]:
    x, y, w, h = xywh
    left = x <= margin
    top = y <= margin
    right = x + w >= frame_w - margin
    bottom = y + h >= frame_h - margin
    return {
        "left": bool(left),
        "top": bool(top),
        "right": bool(right),
        "bottom": bool(bottom),
        "any": bool(left or top or right or bottom),
        "margin_px": margin,
        "frame_wh": [frame_w, frame_h],
    }


def nearby_identities(
    gt_rows: list[dict[str, float]],
    frame: int,
    primary_gid: int,
    top_k: int = NEARBY_TOP_K,
) -> list[dict[str, Any]]:
    primary = None
    others: list[dict[str, float]] = []
    for row in gt_rows:
        if int(row["frame"]) != frame:
            continue
        if int(row["id"]) == primary_gid:
            primary = row
        else:
            others.append(row)
    if primary is None:
        return []
    px = primary["x"] + primary["w"] / 2.0
    py = primary["y"] + primary["h"]
    ranked: list[dict[str, Any]] = []
    for row in others:
        cx = row["x"] + row["w"] / 2.0
        cy = row["y"] + row["h"]
        # axis-aligned IoU
        x1 = max(primary["x"], row["x"])
        y1 = max(primary["y"], row["y"])
        x2 = min(primary["x"] + primary["w"], row["x"] + row["w"])
        y2 = min(primary["y"] + primary["h"], row["y"] + row["h"])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a1 = primary["w"] * primary["h"]
        a2 = row["w"] * row["h"]
        iou = inter / max(a1 + a2 - inter, 1e-9)
        ranked.append(
            {
                "id": int(row["id"]),
                "foot_dist_px": float(math.hypot(cx - px, cy - py)),
                "iou": float(iou),
                "vis": float(row["vis"]),
                "xywh": [
                    float(row["x"]),
                    float(row["y"]),
                    float(row["w"]),
                    float(row["h"]),
                ],
            }
        )
    ranked.sort(key=lambda item: (item["foot_dist_px"], -item["iou"]))
    return ranked[:top_k]


def camera_motion_proxy(
    gt_rows: list[dict[str, float]], f0: int, f1: int
) -> dict[str, Any]:
    """Median per-track foot displacement between f0 and f1 as a camera/scene motion proxy.

    Not a full GMC estimate; auditable and independent of the relink ledger.
    """
    by_id: dict[int, dict[int, tuple[float, float]]] = {}
    for row in gt_rows:
        frm = int(row["frame"])
        if frm not in (f0, f1):
            continue
        gid = int(row["id"])
        by_id.setdefault(gid, {})[frm] = (
            float(row["x"] + row["w"] / 2.0),
            float(row["y"] + row["h"]),
        )
    displacements: list[float] = []
    for pts in by_id.values():
        if f0 in pts and f1 in pts:
            displacements.append(
                float(math.hypot(pts[f1][0] - pts[f0][0], pts[f1][1] - pts[f0][1]))
            )
    if not displacements:
        return {
            "n_tracks_with_both_frames": 0,
            "median_foot_disp_px": None,
            "mean_foot_disp_px": None,
            "p90_foot_disp_px": None,
        }
    arr = np.asarray(displacements, dtype=float)
    return {
        "n_tracks_with_both_frames": int(arr.size),
        "median_foot_disp_px": float(np.median(arr)),
        "mean_foot_disp_px": float(arr.mean()),
        "p90_foot_disp_px": float(np.quantile(arr, 0.90)),
        "definition": (
            "median Euclidean foot displacement of GT ids present at both "
            f"frame {f0} and {f1}; proxy for camera+scene motion, not GMC"
        ),
    }


def frame_path(img_root: Path, frame: int) -> Path:
    return img_root / f"{frame:06d}.jpg"


def render_contact_sheet(
    img_root: Path,
    gt_rows: list[dict[str, float]],
    primary_gid: int,
    frames: list[int],
    labels: list[str],
    out_path: Path,
    frame_w: int,
    frame_h: int,
) -> dict[str, Any]:
    """Draw GT boxes for primary + nearby ids on selected frames; write a contact sheet."""
    panels: list[np.ndarray] = []
    panel_meta: list[dict[str, Any]] = []
    for frame, label in zip(frames, labels):
        path = frame_path(img_root, frame)
        if not path.is_file():
            panel_meta.append(
                {
                    "frame": frame,
                    "label": label,
                    "image_found": False,
                    "path": str(path),
                }
            )
            # grey placeholder
            panel = np.full((270, 480, 3), 40, dtype=np.uint8)
            cv2.putText(
                panel,
                f"missing {frame}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )
            panels.append(panel)
            continue
        img = cv2.imread(str(path))
        if img is None:
            panel_meta.append(
                {
                    "frame": frame,
                    "label": label,
                    "image_found": False,
                    "path": str(path),
                }
            )
            panel = np.full((270, 480, 3), 40, dtype=np.uint8)
            panels.append(panel)
            continue
        # draw all boxes this frame; primary in green, others in orange
        for row in gt_rows:
            if int(row["frame"]) != frame:
                continue
            x, y, w, h = (
                int(row["x"]),
                int(row["y"]),
                int(row["w"]),
                int(row["h"]),
            )
            gid = int(row["id"])
            color = (0, 220, 0) if gid == primary_gid else (0, 140, 255)
            thickness = 3 if gid == primary_gid else 1
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(
                img,
                f"{gid}",
                (x, max(15, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        # banner
        cv2.rectangle(img, (0, 0), (img.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(
            img,
            f"{label}  f={frame}  GT={primary_gid}",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        # resize panel to fixed height
        target_h = 360
        scale = target_h / float(img.shape[0])
        resized = cv2.resize(img, (int(img.shape[1] * scale), target_h))
        panels.append(np.asarray(resized, dtype=np.uint8))
        # Prefer repo-relative path when possible; never embed temp out dirs.
        try:
            img_rel = str(path.relative_to(REPO))
        except ValueError:
            img_rel = path.name
        panel_meta.append(
            {
                "frame": frame,
                "label": label,
                "image_found": True,
                "path": img_rel,
                "sha256": sha256(path),
            }
        )

    if not panels:
        raise RuntimeError("no panels for contact sheet")
    # pad to same height (already), stack horizontally then maybe wrap
    max_h = max(p.shape[0] for p in panels)
    normed = []
    for p in panels:
        if p.shape[0] != max_h:
            p = cv2.resize(p, (int(p.shape[1] * max_h / p.shape[0]), max_h))
        normed.append(p)
    sheet = np.hstack(normed)
    if sheet.shape[1] > CONTACT_SHEET_MAX_WIDTH:
        scale = CONTACT_SHEET_MAX_WIDTH / sheet.shape[1]
        sheet = cv2.resize(
            sheet, (CONTACT_SHEET_MAX_WIDTH, int(sheet.shape[0] * scale))
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # deterministic JPEG quality
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    # Packet-relative path only (stable under --verify temp rebuilds).
    packet_rel = f"scene_sheets/{out_path.name}"
    return {
        "contact_sheet": packet_rel,
        "contact_sheet_sha256": sha256(out_path),
        "panels": panel_meta,
        "frame_wh": [frame_w, frame_h],
    }


def load_step0_by_track_gap() -> dict[tuple[str, int], dict[str, str]]:
    """Independent sealed expected atom values from Step-0 gt_rows.csv."""
    out: dict[tuple[str, int], dict[str, str]] = {}
    with (STEP0 / "gt_rows.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["track_key"], int(float(row["v_gap"])))
            out[key] = row
    return out


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


def signal_computation_check(
    row: dict[str, str],
    recomputed: dict[str, float],
    step0_row: dict[str, str] | None,
) -> dict[str, Any]:
    """Non-tautological signal checks.

    1. Recompute derived atoms from pairs raw columns under declared formulas.
    2. Compare to independent sealed Step-0 gt_rows values (same row identity).
    3. Frame-gap consistency: gap == cand_first_frame - lost_last_frame.
    4. Builder field domain checks (|dir_cos|<=1+eps, non-neg distances/heights).

    Explicitly **untested**: pixel-level replay of builder residual / speed /
    dir_cos emission from raw trajectories.
    """
    failures: list[str] = []
    step0_compare: dict[str, Any] = {
        "available": step0_row is not None,
        "mismatches": [],
    }
    if step0_row is None:
        failures.append("missing sealed Step-0 gt_rows entry for this (track, gap)")
    else:
        for name in ATOM_NAMES:
            a = float(recomputed[name])
            b = float(step0_row[f"v_{name}"])
            if not close(
                a, b, abs_tol=STEP0_VALUE_ABS_TOL, rel_tol=STEP0_VALUE_REL_TOL
            ):
                failures.append(f"step0 mismatch {name}: recompute={a} sealed={b}")
                step0_compare["mismatches"].append(
                    {"atom": name, "recompute": a, "step0": b}
                )

    lf = int(float(row["lost_last_frame"]))
    cf = int(float(row["cand_first_frame"]))
    gap = int(float(row["gap"]))
    if gap != cf - lf:
        failures.append(
            f"gap inconsistency: gap={gap} != cand_first-lost_last={cf - lf}"
        )

    domain: list[str] = []
    dir_cos = float(row["dir_cos"])
    if abs(dir_cos) > 1.0 + 1e-5:
        domain.append(f"|dir_cos|={abs(dir_cos)} > 1")
    for col in (
        "dist_h",
        "bridge_dist",
        "fwd_resid",
        "bwd_resid",
        "h_lost_raw",
        "h_cand_raw",
        "gap",
    ):
        val = float(row[col])
        if val < -1e-9:
            domain.append(f"{col}={val} < 0")
    if domain:
        failures.extend(domain)

    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "step0_compare": step0_compare,
        "frame_gap_check": {
            "lost_last_frame": lf,
            "cand_first_frame": cf,
            "gap": gap,
            "expected_gap_eq_cand_minus_lost": True,
            "ok": gap == cf - lf,
        },
        "domain_check_ok": len(domain) == 0,
        "tested": [
            "derived atoms recomputed from pairs raw columns vs sealed Step-0 gt_rows",
            "gap == cand_first_frame - lost_last_frame",
            "builder field domain: |dir_cos|<=1, non-negative distances/heights/gap",
        ],
        "untested": [
            "pixel/trajectory replay of builder-emitted fwd_resid/bwd_resid/dir_cos/speeds",
            "GMC / camera-compensation correctness inside residual construction",
        ],
        "formulas": {
            "log_h_ratio": "abs(log(h_cand_raw / h_lost_raw))",
            "speed_mismatch": "abs(lost_exit_speed - cand_entry_speed)",
            "resid_mean": "0.5 * (fwd_resid + bwd_resid)",
            "score_m_bridge": (
                "w*0.5*(fwd+bwd)+(1-w)*dist_h with w=sqrt(clip(exit_speed/0.12,0,1))"
            ),
        },
    }


def classify_track(card: dict[str, Any]) -> dict[str, Any]:
    """Apply PR-C operational category rules (not #102-sealed numerical bounds)."""
    ann = card["annotation_check"]
    sig = card["signal_computation_check"]
    thr = card["threshold_artifact_check"]
    mech = card["mechanism_features"]
    scene = card["scene_evidence_summary"]
    competing: list[str] = []

    if not ann["same_identity_on_all_gt_rows"] or not ann["annotation_continuous"]:
        category = "ANNOTATION_ISSUE"
        support = [
            "GT identity continuity failed on at least one gt_match row, or GT id has internal annotation gaps."
        ]
    elif not sig["all_rows_ok"]:
        category = "SIGNAL_COMPUTATION_ISSUE"
        support = [
            "Independent signal checks failed (Step-0 sealed value mismatch, gap/frame inconsistency, or domain violation).",
            f"row failures: {sig.get('failed_rows', [])}",
        ]
    elif thr["threshold_artifact_dominates"]:
        category = "THRESHOLD_ARTIFACT"
        support = thr["reasons"]
    elif (
        mech["height_safe_on_min_dh_row"]
        and mech["n_motion_violations_min_dh"] >= MIN_MOTION_VIOLATIONS_FOR_TRUE
        and mech["occlusion_strong_operational_on_min_dh_row"]
        and ann["same_identity_on_all_gt_rows"]
        and sig["all_rows_ok"]
        and scene["contact_sheet_available"]
        and scene["scene_supports_occlusion_or_crowd"]
    ):
        category = "TRUE_LONG_GAP_REENTRY"
        support = [
            "min-d_H row keeps log_h_ratio on the safe side of the sealed exploratory median.",
            f"motion atoms violated on min-d_H row: {mech['motion_violated_min_dh']} "
            f"(operational cutoff: >={MIN_MOTION_VIOLATIONS_FOR_TRUE}).",
            "gap-window GT visibility meets the operational occlusion_strong cutoff.",
            "scene packet provides contact sheet + nearby-id / truncation / camera-motion proxy.",
            "annotation and independent signal checks pass.",
        ]
        if thr["any_violation_near_threshold"]:
            competing.append(
                "Some motion violations sit near the exploratory median; membership is threshold-sensitive."
            )
        if scene["truncation_any_on_endpoints"]:
            competing.append(
                "Endpoint truncation flags present; partial-box effects remain a residual alternative."
            )
        if scene["nearby_close_on_reentry"]:
            competing.append(
                "Nearby GT identities at re-entry; identity-swap / interference remains residual."
            )
        if not mech["all_gt_rows_height_safe"]:
            competing.append(
                "At least one non-min-d_H gt_match row violates height; classification uses min-d_H representative only."
            )
    else:
        category = "UNRESOLVED"
        support = [
            "No single #102 category is decisive under the operational rule set and available scene evidence."
        ]
        if (
            mech["height_safe_on_min_dh_row"]
            and mech["n_motion_violations_min_dh"] >= MIN_MOTION_VIOLATIONS_FOR_TRUE
            and not mech["occlusion_strong_operational_on_min_dh_row"]
        ):
            competing.append(
                "Motion fails and height is safe, but operational occlusion_strong is false "
                "(visible fragmentation / non-occlusion loss remains open)."
            )
        if thr["any_violation_near_threshold"]:
            competing.append(
                "Near-threshold violations leave a residual THRESHOLD_ARTIFACT hypothesis."
            )
        if not scene["contact_sheet_available"]:
            competing.append("Contact sheet missing; scene audit incomplete.")
        if sig.get("untested_residual_risk"):
            competing.append(
                "Builder residual/speed trajectory emission is not pixel-replayed "
                "(declared untested substrate risk)."
            )

    if category not in PER_TRACK_TERMINALS:
        raise RuntimeError(f"illegal category {category}")
    return {
        "category": category,
        "supporting_evidence": support,
        "competing_explanations": competing,
        "rule_provenance": "PR-C implementation-time operationalization; numerical cutoffs not sealed in #102",
        "confidence_boundary": (
            "Single-sequence (MOT17-10-SDP) forensic only; exploratory median binarization; "
            "descriptive min-d_H layer; operational numerical cutoffs subject to research-owner "
            "interpretation; builder residual/speed pixel-replay untested; does not seal morphology "
            "terminals or orderability classes."
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

    # Operational aggregate mapping (not sealed in #102).
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
    elif true_n >= AGGREGATE_MIN_TRUE_FOR_ROLE_REVERSAL and artifact == 0:
        terminal = "ROLE_REVERSAL_SUPPORTED"
        authorizes = [
            "A later partial-order audit may consider motion atoms as conditional_orderable or context_only "
            "(research-line Phase B / PR-D prep) — contingent on research-owner acceptance of this operational mapping.",
        ]
        blocks = [
            "global closure arcs on motion atoms",
            "veto against the protected escape tail",
            "production rule / preset / gate change",
            "evidence_ledger promotion beyond this bounded forensic result",
            "L2+ morphology claims without nested held-out confirmation",
            "claiming signal bugs are fully ruled out at pixel-trajectory level (builder path untested)",
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
        "rule_provenance": (
            "PR-C implementation-time operationalization of #102 aggregate terminals; "
            f"ROLE_REVERSAL_SUPPORTED uses operational cutoff "
            f"n(TRUE)>={AGGREGATE_MIN_TRUE_FOR_ROLE_REVERSAL} and n(artifact)==0"
        ),
        "research_acceptance": RESEARCH_ACCEPTANCE,
        "routing": {
            "ROLE_REVERSAL_SUPPORTED": "open separate partial-order audit before any MWC prototype",
            "TAIL_ARTIFACT_DOMINATED": "repair/re-audit substrate; closure work remains blocked",
            "TAIL_MECHANISM_UNRESOLVED": "retain descriptive morphology only; closure conditioning remains blocked",
        }[terminal],
        "confidence_boundary": (
            "ACCEPTED_WITH_LIMITS (PR #104): L1 single-sequence (MOT17-10-SDP) forensic only. "
            "Numerical cutoffs remain PR-C operationalizations (not sealed in #102). "
            "Builder residual/speed pixel-replay untested. Authorizes only a separate "
            "partial-order audit; not global motion closure, MWC conclusions, tail veto, "
            "production, or ledger promotion."
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

    quantiles = {"p40": 0.40, "median": 0.50, "p60": 0.60}
    out: dict[str, Any] = {"quantiles": {}, "frozen_tracks_remain_in_tail": {}}
    for qname, q in quantiles.items():
        thrs = {name: float(np.quantile(pool[name], q)) for name, _ in ATOMS}
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
    tail_raw = load_json(STEP0 / "tail_tracks.json")
    if not isinstance(tail_raw, dict):
        raise TypeError("tail_tracks.json must be a JSON object")
    tail: dict[str, Any] = tail_raw
    tracks = tail["tracks"]
    if not isinstance(tracks, dict):
        raise TypeError("tail_tracks.json 'tracks' must be an object")
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
    wanted: set[tuple[str, str]] = set()
    for track_key in track_keys:
        seq, lost = track_key.split("|", maxsplit=1)
        wanted.add((seq, lost))
    rows: list[dict[str, str]] = []
    with pairs.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pair_key = (str(row["seq"]), str(row["lost_id"]))
            if pair_key not in wanted:
                continue
            if row.get("gt_valid") not in ("1", "True", "true"):
                continue
            if row.get("gt_match") not in ("1", "True", "true"):
                continue
            rows.append(row)
    return rows


def emit(pairs: Path, gt_path: Path, img_root: Path, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    sheets_dir = out / "scene_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    step0_manifest = load_json(STEP0 / "manifest.json")
    verify_source(pairs, step0_manifest)
    tail = load_frozen_cohort()
    track_keys = sorted(tail["tracks"])
    thresholds = {
        name: float(step0_manifest["atom_thresholds"][name]["pool_median_threshold"])
        for name, _ in ATOMS
    }
    step0_by_gap = load_step0_by_track_gap()

    if not gt_path.is_file():
        raise FileNotFoundError(
            f"MOT17 GT not found at {gt_path}. Pass --gt or place MOT17-10-SDP under datasets/."
        )
    if not img_root.is_dir():
        raise FileNotFoundError(
            f"MOT17 image root not found at {img_root}. Pass --img-root."
        )

    # frame geometry from first available image
    sample_img = frame_path(img_root, 1)
    if not sample_img.is_file():
        raise FileNotFoundError(f"expected frame image missing: {sample_img}")
    sample = cv2.imread(str(sample_img))
    if sample is None:
        raise RuntimeError(f"failed to read {sample_img}")
    frame_h, frame_w = sample.shape[:2]

    gt_table = load_gt_table(gt_path)
    gt_match_rows = collect_gt_match_rows(pairs, set(track_keys))
    if not gt_match_rows:
        raise RuntimeError("no gt_match rows found for frozen cohort")

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
    scene_bundle: dict[str, Any] = {}

    for key in track_keys:
        rows = by_track[key]
        row_payloads: list[dict[str, Any]] = []
        signal_row_results: list[dict[str, Any]] = []
        same_id_all = True
        height_safe_all = True

        for row in rows:
            recomputed = recompute_derived_from_pairs_raw(row)
            # values used for morphology = recompute from pairs raw (explicit source)
            values = dict(recomputed)
            analysis = atom_analysis(values, thresholds)
            gt_lost = int(float(row["gt_lost"]))
            gt_cand = int(float(row["gt_cand"]))
            if gt_lost != gt_cand:
                same_id_all = False
            lf = int(float(row["lost_last_frame"]))
            cf = int(float(row["cand_first_frame"]))
            vis = visibility_stats(gt_table, gt_lost, lf, cf)

            step0_row = step0_by_gap.get((key, int(float(row["gap"]))))
            sig_row = signal_computation_check(row, recomputed, step0_row)
            signal_row_results.append(
                {
                    "gap": int(float(row["gap"])),
                    "cand_id": str(row["cand_id"]),
                    **sig_row,
                }
            )
            if analysis["atoms"]["log_h_ratio"]["z"] == 0:
                height_safe_all = False

            gid_rows = gt_id_rows(gt_table, gt_lost)
            frames = [int(r["frame"]) for r in gid_rows]
            exit_box = box_at(gid_rows, lf)
            reentry_box = box_at(gid_rows, cf)
            exit_trunc = (
                truncation_flags(exit_box["xywh"], frame_w, frame_h)
                if exit_box
                else None
            )
            reentry_trunc = (
                truncation_flags(reentry_box["xywh"], frame_w, frame_h)
                if reentry_box
                else None
            )
            nearby_exit = nearby_identities(gt_table, lf, gt_lost)
            nearby_reentry = nearby_identities(gt_table, cf, gt_lost)
            cam = camera_motion_proxy(gt_table, lf, cf)

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
                "values_source": "recomputed_from_pairs_raw_columns",
                "values": values,
                "atom_analysis": analysis,
                "signal_computation_check": sig_row,
                "gap_visibility": vis,
                "gt_box_before_exit": exit_box,
                "gt_box_after_reentry": reentry_box,
                "truncation": {"exit": exit_trunc, "reentry": reentry_trunc},
                "nearby_identities": {
                    "exit_frame": nearby_exit,
                    "reentry_frame": nearby_reentry,
                },
                "camera_motion_proxy": cam,
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
                    "log_h_ratio": values["log_h_ratio"],
                    "speed_mismatch": values["speed_mismatch"],
                    "dir_cos": values["dir_cos"],
                    "resid_mean": values["resid_mean"],
                    "score_m_bridge": values["score_m_bridge"],
                    "bridge_dist": values["bridge_dist"],
                    "dist_h": values["dist_h"],
                    "vis_mean": vis["vis_mean"],
                    "frac_vis_eq_0": vis["frac_vis_eq_0"],
                    "occlusion_strong_operational": vis["occlusion_strong_operational"],
                    "signal_ok": sig_row["ok"],
                    "foot_euclid_px": payload["feet"]["euclid_px"],
                    "reentry_truncation_any": bool(
                        reentry_trunc["any"] if reentry_trunc else False
                    ),
                    "nearby_min_foot_dist_reentry": (
                        nearby_reentry[0]["foot_dist_px"] if nearby_reentry else None
                    ),
                    "cam_median_foot_disp_px": cam["median_foot_disp_px"],
                }
            )

        min_row = min(row_payloads, key=lambda p: (p["atom_analysis"]["d_h"], p["gap"]))
        min_analysis = min_row["atom_analysis"]
        if int(min_analysis["d_h"]) != int(tail["tracks"][key]["min_d_h"]):
            raise RuntimeError(
                f"{key}: recomputed min d_H={min_analysis['d_h']} != sealed "
                f"{tail['tracks'][key]['min_d_h']}"
            )

        # threshold artifact check on min-d_H row (operational)
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
        threshold_dominates = bool(min_analysis["violated"]) and not strong
        thr_reasons = []
        if threshold_dominates:
            thr_reasons.append(
                f"All violated atoms on min-d_H row flip under <= {THRESHOLD_REL_FLIP_MAX:.0%} "
                f"relative threshold move: {near}."
            )
        p60_remain = key in sensitivity["frozen_tracks_remain_in_tail"]["p60"]
        if not p60_remain:
            thr_reasons.append(
                "Track leaves the d_H>=3 tail under pool p60 binarization (membership is median-sensitive)."
            )

        primary_gid = int(min_row["gt_lost"])
        gid_rows = gt_id_rows(gt_table, primary_gid)
        frames = [int(r["frame"]) for r in gid_rows]
        ann_gaps = annotation_gaps(frames)
        annotation_continuous = len(ann_gaps) == 0 and len(frames) > 0

        # scene contact sheet for min-d_H timeline
        lf = int(min_row["lost_last_frame"])
        cf = int(min_row["cand_first_frame"])
        mid_frames: list[int] = []
        if cf - lf > 1:
            mids = np.linspace(lf + 1, cf - 1, num=min(SCENE_MID_SAMPLES, cf - lf - 1))
            mid_frames = sorted({int(round(x)) for x in mids})
        sheet_frames = [lf, *mid_frames, cf]
        sheet_labels = (
            ["exit"] + [f"gap{i + 1}" for i in range(len(mid_frames))] + ["reentry"]
        )
        sheet_name = key.replace("|", "_") + "_min_dh.jpg"
        sheet_path = sheets_dir / sheet_name
        sheet_meta = render_contact_sheet(
            img_root,
            gt_table,
            primary_gid,
            sheet_frames,
            sheet_labels,
            sheet_path,
            frame_w,
            frame_h,
        )

        nearby_re = min_row["nearby_identities"]["reentry_frame"]
        nearby_close = bool(
            nearby_re and nearby_re[0]["foot_dist_px"] < 40.0
        )  # diagnostic, not a sealed cutoff
        trunc_any = bool(
            (min_row["truncation"]["exit"] or {}).get("any")
            or (min_row["truncation"]["reentry"] or {}).get("any")
        )
        # Scene supports occlusion/crowd if operational occlusion OR (mid-gap low vis + nearby)
        scene_supports = bool(
            min_row["gap_visibility"]["occlusion_strong_operational"]
            or (
                (min_row["gap_visibility"]["vis_mean"] is not None)
                and min_row["gap_visibility"]["vis_mean"] < 0.5
                and nearby_close
            )
        )

        failed_rows = [
            {"gap": r["gap"], "cand_id": r["cand_id"], "failures": r["failures"]}
            for r in signal_row_results
            if not r["ok"]
        ]
        signal_summary = {
            "all_rows_ok": len(failed_rows) == 0,
            "failed_rows": failed_rows,
            "n_rows_checked": len(signal_row_results),
            "tested": signal_row_results[0]["tested"] if signal_row_results else [],
            "untested": signal_row_results[0]["untested"] if signal_row_results else [],
            "untested_residual_risk": True,
            "formulas": signal_row_results[0]["formulas"] if signal_row_results else {},
            "per_row": signal_row_results,
        }

        scene_summary = {
            "contact_sheet_available": bool(sheet_meta.get("contact_sheet")),
            "contact_sheet": sheet_meta.get("contact_sheet"),
            "contact_sheet_sha256": sheet_meta.get("contact_sheet_sha256"),
            "truncation_any_on_endpoints": trunc_any,
            "nearby_close_on_reentry": nearby_close,
            "scene_supports_occlusion_or_crowd": scene_supports,
            "camera_motion_proxy_min_dh": min_row["camera_motion_proxy"],
            "panels": sheet_meta.get("panels"),
        }
        scene_bundle[key] = {
            **scene_summary,
            "min_dh_frames": sheet_frames,
            "exit_nearby": min_row["nearby_identities"]["exit_frame"],
            "reentry_nearby": nearby_re,
            "truncation": min_row["truncation"],
            "gap_visibility": min_row["gap_visibility"],
        }

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
                "img_root": str(img_root.relative_to(REPO))
                if img_root.is_relative_to(REPO)
                else str(img_root),
            },
            "timeline": {
                "n_gt_match_rows": len(row_payloads),
                "primary_gt_id": primary_gid,
                "exit_frame": lf,
                "reentry_frame_min_dh_row": cf,
                "gap_min_dh_row": int(min_row["gap"]),
                "all_gaps": [int(p["gap"]) for p in row_payloads],
                "all_cand_ids": [p["cand_id"] for p in row_payloads],
            },
            "min_d_h_row": {
                "cand_id": min_row["cand_id"],
                "gap": int(min_row["gap"]),
                "frames": [lf, cf],
                "atom_analysis": min_analysis,
                "values": min_row["values"],
                "gap_visibility": min_row["gap_visibility"],
                "feet": min_row["feet"],
                "heights": min_row["heights"],
                "truncation": min_row["truncation"],
                "nearby_identities": min_row["nearby_identities"],
                "camera_motion_proxy": min_row["camera_motion_proxy"],
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
                    "GT remaining annotated across the tracker gap is expected for MOT17 "
                    "and does not by itself imply ANNOTATION_ISSUE."
                ),
            },
            "signal_computation_check": signal_summary,
            "threshold_artifact_check": {
                "any_violation_near_threshold": bool(near),
                "near_threshold_violations": near,
                "strong_violations": strong,
                "threshold_artifact_dominates": threshold_dominates,
                "remains_in_tail_under_p60": p60_remain,
                "reasons": thr_reasons,
                "rule_provenance": "PR-C operationalization (THRESHOLD_REL_FLIP_MAX)",
            },
            "mechanism_features": {
                "height_safe_on_min_dh_row": min_analysis["atoms"]["log_h_ratio"]["z"]
                == 1,
                "all_gt_rows_height_safe": height_safe_all,
                "motion_violated_min_dh": min_analysis["motion_violated"],
                "n_motion_violations_min_dh": len(min_analysis["motion_violated"]),
                "geom_violated_min_dh": min_analysis["geom_violated"],
                "occlusion_strong_operational_on_min_dh_row": bool(
                    min_row["gap_visibility"]["occlusion_strong_operational"]
                ),
                "gap_vis_mean_min_dh": min_row["gap_visibility"]["vis_mean"],
            },
            "scene_evidence_summary": scene_summary,
        }
        card["verdict"] = classify_track(card)
        track_cards[key] = card

    aggregate = aggregate_terminal(track_cards)

    classification_rules = {
        "terminal_vocabulary_source": "issue #102 (predeclared categories + aggregate names)",
        "numerical_cutoff_provenance": (
            "PR-C implementation-time operationalization — NOT sealed in #102; "
            "subject to research-owner interpretation before research acceptance"
        ),
        "per_track_terminals": list(PER_TRACK_TERMINALS),
        "aggregate_terminals": list(AGGREGATE_TERMINALS),
        "operational_cutoffs": {
            "occlusion_strong": {
                "vis_mean_max": OCCLUSION_VIS_MEAN_MAX,
                "frac_vis_eq_0_min": OCCLUSION_FRAC_ZERO_MIN,
                "definition": (
                    "occlusion_strong_operational iff gap_vis_mean <= vis_mean_max "
                    "OR frac_vis_eq_0 >= frac_vis_eq_0_min"
                ),
            },
            "min_motion_violations_for_true": MIN_MOTION_VIOLATIONS_FOR_TRUE,
            "aggregate_min_true_for_role_reversal": AGGREGATE_MIN_TRUE_FOR_ROLE_REVERSAL,
            "threshold_rel_flip_max": THRESHOLD_REL_FLIP_MAX,
            "truncation_margin_px": TRUNCATION_MARGIN_PX,
        },
        "TRUE_LONG_GAP_REENTRY_operational": [
            "annotation same-identity + continuous",
            "independent signal checks ok (step0 sealed compare + gap/frame + domain)",
            "not threshold_artifact_dominates (operational)",
            "height safe on min-d_H row (log_h_ratio z=1)",
            f">={MIN_MOTION_VIOLATIONS_FOR_TRUE} motion atom violations on min-d_H row",
            "occlusion_strong_operational on min-d_H gap window",
            "contact sheet available",
            "scene_supports_occlusion_or_crowd",
        ],
        "signal_computation_check": {
            "tested": [
                "recompute derived atoms from pairs raw vs sealed Step-0 gt_rows",
                "gap == cand_first_frame - lost_last_frame",
                "domain checks on builder fields",
            ],
            "untested": [
                "pixel/trajectory replay of builder residual/speed/dir_cos emission",
            ],
        },
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
    write_json(out / "scene_evidence.json", scene_bundle)

    runner_src = Path(__file__).resolve()
    runner_dst = out / "run_escape_tail_forensic.py"
    if runner_src != runner_dst.resolve():
        shutil.copy2(runner_src, runner_dst)

    # hash body files + every scene sheet
    file_hashes: dict[str, str] = {
        name: sha256(out / name) for name in PACKET_BODY_FILES if (out / name).is_file()
    }
    for sheet in sorted(sheets_dir.glob("*.jpg")):
        rel = f"scene_sheets/{sheet.name}"
        file_hashes[rel] = sha256(sheet)

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
        "rule_provenance": (
            "terminal vocabulary from #102; numerical cutoffs are PR-C "
            "implementation-time operationalization (not sealed in #102)"
        ),
        "research_acceptance": RESEARCH_ACCEPTANCE,
        "frozen_cohort": track_keys,
        "aggregate_terminal": aggregate["terminal"],
        "per_track_categories": {
            k: v["verdict"]["category"] for k, v in track_cards.items()
        },
        "files": file_hashes,
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def _compare_packet(expected_dir: Path, rebuilt_dir: Path) -> list[str]:
    """Strict packet equality: non-files manifest fields + files map digests + bytes."""
    mismatched: list[str] = []
    exp_manifest = load_json(expected_dir / "manifest.json")
    reb_manifest = load_json(rebuilt_dir / "manifest.json")
    if not isinstance(exp_manifest, dict) or not isinstance(reb_manifest, dict):
        return ["manifest.json:not_an_object"]

    for key in set(exp_manifest) | set(reb_manifest):
        if key == "files":
            continue
        if exp_manifest.get(key) != reb_manifest.get(key):
            mismatched.append(f"manifest.json:{key}")

    exp_files = exp_manifest.get("files", {})
    reb_files = reb_manifest.get("files", {})
    if not isinstance(exp_files, dict) or not isinstance(reb_files, dict):
        mismatched.append("manifest.json:files:not_an_object")
        return mismatched

    exp_names = set(exp_files)
    reb_names = set(reb_files)
    for name in sorted(exp_names - reb_names):
        mismatched.append(f"manifest.json:files:missing_in_rebuilt:{name}")
    for name in sorted(reb_names - exp_names):
        mismatched.append(f"manifest.json:files:extra_in_rebuilt:{name}")

    for name in sorted(exp_names & reb_names):
        exp_digest = str(exp_files[name])
        reb_digest = str(reb_files[name])
        path_e = expected_dir / name
        path_r = rebuilt_dir / name
        if not path_e.is_file():
            mismatched.append(f"missing_on_disk:expected:{name}")
            continue
        if not path_r.is_file():
            mismatched.append(f"missing_on_disk:rebuilt:{name}")
            continue
        actual_e = sha256(path_e)
        actual_r = sha256(path_r)
        if exp_digest != actual_e:
            mismatched.append(f"manifest.json:files:stale_digest:expected:{name}")
        if reb_digest != actual_r:
            mismatched.append(f"manifest.json:files:stale_digest:rebuilt:{name}")
        if exp_digest != reb_digest:
            mismatched.append(f"manifest.json:files:digest_mismatch:{name}")
        if path_e.read_bytes() != path_r.read_bytes():
            mismatched.append(f"bytes_mismatch:{name}")

    return mismatched


def verify(pairs: Path, gt_path: Path, img_root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="escape-tail-forensic-") as tmp:
        rebuilt = Path(tmp) / PACKET.name
        emit(pairs, gt_path, img_root, rebuilt)
        mismatched = _compare_packet(PACKET, rebuilt)
    if mismatched:
        raise AssertionError(f"packet is not reproducible: {mismatched}")
    print("forensic packet verification passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--img-root", type=Path, default=DEFAULT_IMG_ROOT)
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
        verify(args.pairs, args.gt, args.img_root)
        return
    manifest = emit(args.pairs, args.gt, args.img_root, args.out)
    print(f"packet emitted: {args.out}")
    print(f"aggregate terminal: {manifest['aggregate_terminal']}")
    print(f"per-track: {manifest['per_track_categories']}")
    acc = manifest.get("research_acceptance", {})
    status = acc.get("status", acc) if isinstance(acc, dict) else acc
    print(f"research_acceptance: {status}")


if __name__ == "__main__":
    main()
