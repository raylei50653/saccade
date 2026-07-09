#!/usr/bin/env python3
"""Summarize offline relink candidate pairs into a B1 study directory.

Post-process only: reads a CSV from ``build_relink_candidates.py`` and writes
fixed artifacts (distribution background + AUC + threshold table). Does not run
MOT or rebuild candidates.

Mirrors the full-vs-hard reporting style of offline_relink_candidate_analysis.md
and color_relink_features.py, but persists results for m remeasures without
embedding tables in markdown.

Usage:
  uv run python scripts/tools/summarize_relink_pairs.py \\
      --pairs scripts/tools/out/relink_candidates.csv \\
      --study-dir out/signal_study/m_b1_demo \\
      --hard-dist 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

project_root = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(project_root / "src"))

from saccade.perception.eval.signal_tables import (  # noqa: E402
    B1_OUTPUT_FILES,
    CONTEXT_FILENAME,
    DEFAULT_RELINK_GAP_BINS,
    DEFAULT_RELINK_THR_GRID,
    METRICS_AUC_FILENAME,
    METRICS_THR_FILENAME,
    RELINK_PAIR_REQUIRED,
    UniverseId,
    apply_hard_pool_mask,
    auc_full_and_hard_pool,
    offline_threshold_curve,
    validate_columns,
)


def _git_commit_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def _as_bool01(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip()
    if s == "":
        return False
    return bool(int(float(s)))


def _percentile(xs: np.ndarray, q: float) -> float:
    if xs.size == 0:
        return float("nan")
    return float(np.percentile(xs, q))


def load_valid_pairs(
    path: Path,
    *,
    score_field: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"empty or header-less CSV: {path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    missing = [c for c in RELINK_PAIR_REQUIRED if c not in fieldnames]
    # score_field may be bridge_dist (required) or another column
    if score_field not in fieldnames:
        missing.append(score_field)
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")

    validate_columns(UniverseId.U_RELINK_PAIR, fieldnames, extra_ok=True)

    valid: list[dict[str, Any]] = []
    for r in rows:
        if not _as_bool01(r.get("gt_valid", 0)):
            continue
        try:
            score = float(r[score_field])
        except (TypeError, ValueError):
            continue
        valid.append(
            {
                "seq": r.get("seq", ""),
                "gt_match": _as_bool01(r.get("gt_match", 0)),
                "score": score,
                "gap": int(float(r.get("gap", 0))),
                "raw": r,
            }
        )
    return valid, fieldnames


def score_dist_block(scores: np.ndarray) -> dict[str, float]:
    return {
        "median": _percentile(scores, 50),
        "p05": _percentile(scores, 5),
        "p50": _percentile(scores, 50),
        "p95": _percentile(scores, 95),
        "mean": float(scores.mean()) if scores.size else float("nan"),
        "n": int(scores.size),
    }


def gap_bin_counts(gaps: np.ndarray, y: np.ndarray) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for label, lo, hi in DEFAULT_RELINK_GAP_BINS:
        mask = (gaps >= lo) & (gaps <= hi)
        out[label] = {
            "n": int(mask.sum()),
            "n_pos": int(y[mask].sum()) if mask.any() else 0,
        }
    return out


def thr_rows_for_pool(
    scores: list[float],
    y: list[bool],
    thresholds: list[float],
    *,
    pool: str,
) -> list[dict[str, Any]]:
    # Distance: accept if score <= thr
    curve = offline_threshold_curve(scores, y, thresholds, accept_if_ge=False)
    rows: list[dict[str, Any]] = []
    for rec in curve:
        rows.append(
            {
                "pool": pool,
                "threshold": rec["threshold"],
                "tp": rec["tp"],
                "fp": rec["fp"],
                "fn": rec["fn"],
                "precision": rec["precision"],
                "recall": rec["recall"],
                "f1": rec["f1"],
            }
        )
    return rows


def build_context(
    *,
    study_id: str,
    pairs_path: Path,
    n_rows_raw: int,
    score_field: str,
    hard_pool_rule: str,
    hard_dist: float,
    y: np.ndarray,
    scores: np.ndarray,
    gaps: np.ndarray,
    hard_mask: np.ndarray,
    substrate: dict[str, Any],
    e2e: dict[str, Any] | None,
    commit: str,
    preset: str,
    detector: str,
) -> dict[str, Any]:
    y_bool = y.astype(bool)
    pos = scores[y_bool]
    neg = scores[~y_bool]
    hard_y = y_bool[hard_mask]
    n_full = int(y_bool.size)
    n_pos = int(y_bool.sum())
    n_neg = n_full - n_pos
    n_hard = int(hard_mask.sum())
    n_pos_h = int(hard_y.sum())
    n_neg_h = n_hard - n_pos_h

    ctx: dict[str, Any] = {
        "study_id": study_id,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit,
        "preset": preset,
        "detector": detector,
        "substrate": substrate,
        "input": {
            "pairs_csv": str(pairs_path),
            "n_rows_raw": n_rows_raw,
            "n_rows_gt_valid": n_full,
            "score_field": score_field,
        },
        "pool": {
            "full": {
                "n": n_full,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "base_rate": (n_pos / n_full) if n_full else 0.0,
            },
            "hard": {
                "hard_pool_rule": hard_pool_rule,
                "hard_dist": hard_dist,
                "n": n_hard,
                "n_pos": n_pos_h,
                "n_neg": n_neg_h,
                "base_rate": (n_pos_h / n_hard) if n_hard else 0.0,
            },
        },
        "score_dist": {
            "field": score_field,
            "pos": score_dist_block(pos),
            "neg": score_dist_block(neg),
        },
        "gap_bins": gap_bin_counts(gaps, y_bool.astype(int)),
    }
    if e2e:
        ctx["e2e"] = e2e
    return ctx


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="CSV from build_relink_candidates.py",
    )
    ap.add_argument(
        "--study-dir",
        type=Path,
        required=True,
        help="output study directory (context.json, metrics_*.json/csv)",
    )
    ap.add_argument(
        "--score-field",
        default="bridge_dist",
        help="distance-like column (lower = more likely true link)",
    )
    ap.add_argument(
        "--hard-dist",
        type=float,
        default=1.0,
        help="hard pool: score_field <= this (default 1.0, offline_relink)",
    )
    ap.add_argument(
        "--thresholds",
        default=",".join(str(t) for t in DEFAULT_RELINK_THR_GRID),
        help="comma-separated thr grid for metrics_thr.csv",
    )
    ap.add_argument("--study-id", default="", help="defaults to study-dir name")
    ap.add_argument("--commit", default="", help="defaults to git rev-parse --short")
    ap.add_argument(
        "--preset",
        default="mamba_whole_graph_m",
        help="recorded in context (not used to run eval)",
    )
    ap.add_argument("--detector", default="SDP")
    ap.add_argument(
        "--mot-dir",
        default="",
        help="substrate MOT dir (recorded in context.substrate)",
    )
    ap.add_argument(
        "--relink",
        default="off",
        choices=["off", "on"],
        help="substrate flag for context only",
    )
    ap.add_argument(
        "--interpolate",
        default="off",
        choices=["off", "on"],
        help="substrate flag for context only",
    )
    ap.add_argument(
        "--double-buffer",
        default="true",
        choices=["true", "false"],
    )
    ap.add_argument(
        "--e2e-json",
        type=Path,
        default=None,
        help="optional JSON with IDF1/IDs/… for context.e2e",
    )
    ap.add_argument(
        "--copy-pairs",
        action="store_true",
        help="copy --pairs into study-dir/pairs.csv",
    )
    ap.add_argument(
        "--min-n",
        type=int,
        default=10,
        help="min rows per pool for AUC (default 10)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pairs_path = args.pairs.resolve()
    if not pairs_path.is_file():
        raise SystemExit(f"pairs CSV not found: {pairs_path}")

    study_dir = args.study_dir.resolve()
    study_dir.mkdir(parents=True, exist_ok=True)
    study_id = args.study_id or study_dir.name
    commit = args.commit or _git_commit_short()
    hard_rule = f"{args.score_field}<={args.hard_dist:g}"

    with pairs_path.open(newline="", encoding="utf-8") as f:
        n_rows_raw = sum(1 for _ in f) - 1  # exclude header
    if n_rows_raw < 0:
        n_rows_raw = 0

    valid, _fields = load_valid_pairs(pairs_path, score_field=args.score_field)
    if not valid:
        raise SystemExit("no gt_valid pairs after filter")

    y = np.array([1 if r["gt_match"] else 0 for r in valid], dtype=np.int32)
    scores = np.array([r["score"] for r in valid], dtype=np.float64)
    gaps = np.array([r["gap"] for r in valid], dtype=np.int32)
    hard_list = apply_hard_pool_mask(scores.tolist(), hard_rule)
    hard_mask = np.array(hard_list, dtype=bool)

    thr = [float(x) for x in args.thresholds.split(",") if x.strip()]
    if not thr:
        thr = list(DEFAULT_RELINK_THR_GRID)

    e2e = None
    if args.e2e_json is not None:
        e2e = json.loads(args.e2e_json.read_text(encoding="utf-8"))

    substrate = {
        "mot_dir": args.mot_dir,
        "relink": args.relink,
        "interpolate": args.interpolate,
        "double_buffer": args.double_buffer == "true",
        "notes": "flags recorded for provenance; not verified by this script",
    }

    context = build_context(
        study_id=study_id,
        pairs_path=pairs_path,
        n_rows_raw=n_rows_raw,
        score_field=args.score_field,
        hard_pool_rule=hard_rule,
        hard_dist=float(args.hard_dist),
        y=y,
        scores=scores,
        gaps=gaps,
        hard_mask=hard_mask,
        substrate=substrate,
        e2e=e2e,
        commit=commit,
        preset=args.preset,
        detector=args.detector,
    )

    y_bool = [bool(v) for v in y.tolist()]
    scores_list = scores.tolist()
    auc_payload = auc_full_and_hard_pool(
        scores_list,
        y_bool,
        hard_list,
        lower_is_better=True,
        min_n=int(args.min_n),
    )
    auc_payload.update(
        {
            "score_field": args.score_field,
            "hard_pool_rule": hard_rule,
            "commit": commit,
            "study_id": study_id,
        }
    )

    thr_full = thr_rows_for_pool(scores_list, y_bool, thr, pool="full")
    thr_hard = thr_rows_for_pool(
        scores[hard_mask].tolist(),
        [bool(v) for v in y[hard_mask].tolist()],
        thr,
        pool="hard",
    )
    thr_all = thr_full + thr_hard

    # Write artifacts
    (study_dir / CONTEXT_FILENAME).write_text(
        json.dumps(context, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (study_dir / METRICS_AUC_FILENAME).write_text(
        json.dumps(auc_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    thr_path = study_dir / METRICS_THR_FILENAME
    with thr_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "pool",
                "threshold",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
            ],
        )
        w.writeheader()
        w.writerows(thr_all)

    if args.copy_pairs:
        dest = study_dir / "pairs.csv"
        dest.write_bytes(pairs_path.read_bytes())

    # Stdout summary (color_relink style)
    full = auc_payload["full"]
    hard = auc_payload["hard"]
    print(f"Wrote B1 artifacts → {study_dir}")
    for name in B1_OUTPUT_FILES:
        print(f"  {name}")
    print(
        f"\nPool full: {full['n']} | pos {full['n_pos']} "
        f"({100 * full['base_rate']:.1f}%) | AUC={full['auc']:.4f}"
        f" {full.get('skipped_reason') or ''}"
    )
    print(
        f"Pool hard ({hard_rule}): {hard['n']} | pos {hard['n_pos']} "
        f"({100 * hard['base_rate']:.1f}%) | AUC={hard['auc']:.4f}"
        f" {hard.get('skipped_reason') or ''}"
    )
    print(f"citation_ok={auc_payload.get('citation_ok')}")
    print(
        f"score_dist pos median={context['score_dist']['pos']['median']:.4f} "
        f"neg median={context['score_dist']['neg']['median']:.4f}"
    )


if __name__ == "__main__":
    main()
