#!/usr/bin/env python3
# mypy: ignore-errors
"""
Unified ablation runner for scripts/eval/mot17.py.

This keeps the tuning environment centered on mot17.py itself instead of
spreading category studies across many small entry points.

Categories mirror the grouped knobs in mot17.py:
- detection
- association
- geometry
- reid
- semantic
- trigger
- lifecycle

Usage:
    uv run python scripts/eval/ablation_mot17.py --category detection
    uv run python scripts/eval/ablation_mot17.py --category detection,geometry
    uv run python scripts/eval/ablation_mot17.py --category all --detector SDP
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

import numpy as np  # noqa: E402

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
import motmetrics as mm  # noqa: E402


_METRICS = [
    "idf1",
    "recall",
    "precision",
    "mota",
    "num_switches",
    "num_false_positives",
    "num_misses",
]
_DISPLAY = ["IDF1", "Rcll", "Prcn", "MOTA", "IDs", "FP", "FN"]
_PCT = {"idf1", "recall", "precision", "mota"}
_CATEGORY_ORDER = [
    "detection",
    "association",
    "geometry",
    "reid",
    "semantic",
    "trigger",
    "lifecycle",
    "combination",
    "combination_fine",
    "mot17_a",
    "mot17_b",
    "mot17_c",
    "a2",
    "a3",
    "a6",
    "regression",
]

_A2_BEST = [
    "--cross-tile-merge",
    "--match-thresh", "0.78",
    "--semantic-w-sim-base", "0.8012",
    "--semantic-w-iou-base", "0.3423",
    "--semantic-w-maha-base", "0.3117",
    "--semantic-shift-ambiguity", "0.3425",
    "--semantic-shift-lost-age", "0.1778",
    "--semantic-clean-score-threshold", "0.65",
    "--semantic-strict-sim-threshold", "0.0",
    "--appearance-bank-high-quality-min-score", "0.75",
]

_CATEGORY_EXPERIMENTS: dict[str, list[tuple[str, str, list[str]]]] = {
    "detection": [
        ("conf=0.03", "conf003", ["--conf-threshold", "0.03"]),
        ("conf=0.10", "conf010", ["--conf-threshold", "0.10"]),
        ("letterbox only", "letterbox_only", ["--preprocess", "letterbox"]),
        ("letterbox+gamma", "letterbox_gamma", ["--preprocess", "letterbox,gamma"]),
        ("contrast=1.0", "contrast100", ["--contrast", "1.0"]),
        ("cross-tile merge", "cross_tile_merge", ["--cross-tile-merge"]),
        ("nms=0.60", "nms060", ["--nms-iou-threshold", "0.60"]),
    ],
    "association": [
        ("high=0.45", "high045", ["--high-thresh", "0.45"]),
        ("high=0.55", "high055", ["--high-thresh", "0.55"]),
        ("new-track=0.35", "newtrack035", ["--new-track-thresh", "0.35"]),
        ("new-track=0.50", "newtrack050", ["--new-track-thresh", "0.50"]),
        ("match=0.75", "match075", ["--match-thresh", "0.75"]),
        ("match=0.85", "match085", ["--match-thresh", "0.85"]),
        ("confirm=2", "confirm2", ["--confirm-streak", "2"]),
        ("adaptive confirm", "adaptive_confirm", ["--adaptive-confirmation"]),
        ("no gmc", "no_gmc", ["--no-gmc"]),
    ],
    "geometry": [
        ("geometry mid-scale", "mid_scale", ["--geometry-mid-scale"]),
        (
            "geometry tight",
            "geometry_tight",
            ["--geometry-min-scale", "0.92", "--geometry-max-scale", "1.12"],
        ),
        (
            "geometry loose",
            "geometry_loose",
            ["--geometry-min-scale", "0.80", "--geometry-max-scale", "1.35"],
        ),
        ("no id stability", "no_id_stability", ["--no-id-stability-filter"]),
        (
            "strict id stability",
            "strict_id_stability",
            [
                "--id-stability-min-iou",
                "0.10",
                "--id-stability-max-center-shift",
                "1.5",
            ],
        ),
        ("no person prior", "no_person_prior", ["--no-person-geometry-prior"]),
        ("min height 0.024", "min_height_024", ["--person-min-height-ratio", "0.024"]),
    ],
    "reid": [
        ("reid off", "reid_off", ["--reid-mode", "off"]),
        ("tracker mode", "tracker_mode", ["--reid-mode", "tracker"]),
        ("hybrid mode", "hybrid_mode", ["--reid-mode", "hybrid"]),
        ("transreid", "transreid", ["--reid-model", "transreid"]),
        ("osnet", "osnet", ["--reid-model", "osnet"]),
        ("cos=0.88", "cos088", ["--reid-cos-threshold", "0.88"]),
        ("cos=0.92", "cos092", ["--reid-cos-threshold", "0.92"]),
    ],
    "semantic": [
        ("thr=0.88", "thr088", ["--semantic-threshold", "0.88"]),
        ("thr=0.92", "thr092", ["--semantic-threshold", "0.92"]),
        ("ttl=90", "ttl090", ["--semantic-ttl", "90"]),
        ("gate=0.18", "gate018", ["--semantic-spatial-gate", "0.18"]),
        ("gate=0.22", "gate022", ["--semantic-spatial-gate", "0.22"]),
        ("bank inject off", "no_bank_inject", ["--no-semantic-bank-inject"]),
        ("margin=0.05", "margin005", ["--semantic-reciprocal-margin", "0.05"]),
    ],
    "trigger": [
        (
            "fixed interval",
            "fixed_interval",
            ["--no-need-reid", "--reid-interval", "16"],
        ),
        ("count jump", "count_jump", ["--reid-trigger-mode", "count_jump"]),
        ("event persist", "event_persist", ["--reid-trigger-mode", "event_persist"]),
        ("event memory", "event_memory", ["--reid-trigger-mode", "event_memory"]),
        ("score ema", "score_ema", ["--reid-trigger-mode", "score_ema"]),
        (
            "score ema strict",
            "score_ema_strict",
            [
                "--reid-trigger-mode",
                "score_ema",
                "--reid-score-threshold",
                "2.5",
                "--reid-cooldown-frames",
                "8",
            ],
        ),
        (
            "score ema hysteresis",
            "score_ema_hyst",
            [
                "--reid-trigger-mode",
                "score_ema",
                "--reid-score-threshold",
                "2.0",
                "--reid-score-threshold-low",
                "1.5",
            ],
        ),
    ],
    "lifecycle": [
        ("nsa kalman", "nsa_kalman", ["--nsa-kalman"]),
        ("min len 2", "min_len_2", ["--min-tracklet-len", "2"]),
        ("min score 0.10", "min_score_010", ["--min-tracklet-score", "0.10"]),
        ("lifecycle merge", "lifecycle_merge", ["--lifecycle-merge"]),
        (
            "post merge ttl60",
            "postmerge_t60",
            ["--post-lifecycle-merge", "--post-lifecycle-ttl", "60"],
        ),
        (
            "post merge app gate",
            "postmerge_app",
            [
                "--post-lifecycle-merge",
                "--post-lifecycle-appearance-gate",
                "--post-lifecycle-appearance-threshold",
                "0.90",
            ],
        ),
        (
            "qf2 + post merge",
            "qf2_postmerge",
            [
                "--min-tracklet-len",
                "2",
                "--post-lifecycle-merge",
                "--post-lifecycle-ttl",
                "60",
            ],
        ),
    ],
    "combination": [
        (
            "cross-tile + match=0.75",
            "cross_tile_match075",
            ["--cross-tile-merge", "--match-thresh", "0.75"],
        ),
        (
            "cross-tile + semantic thr=0.92",
            "cross_tile_thr092",
            ["--cross-tile-merge", "--semantic-threshold", "0.92"],
        ),
        (
            "cross-tile + match=0.75 + semantic thr=0.92",
            "cross_tile_match075_thr092",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.75",
                "--semantic-threshold",
                "0.92",
            ],
        ),
    ],
    "combination_fine": [
        (
            "cross-tile + match=0.72 + semantic thr=0.91",
            "cross_tile_match072_thr091",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.72",
                "--semantic-threshold",
                "0.91",
            ],
        ),
        (
            "cross-tile + match=0.72 + semantic thr=0.92",
            "cross_tile_match072_thr092",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.72",
                "--semantic-threshold",
                "0.92",
            ],
        ),
        (
            "cross-tile + match=0.72 + semantic thr=0.93",
            "cross_tile_match072_thr093",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.72",
                "--semantic-threshold",
                "0.93",
            ],
        ),
        (
            "cross-tile + match=0.75 + semantic thr=0.91",
            "cross_tile_match075_thr091",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.75",
                "--semantic-threshold",
                "0.91",
            ],
        ),
        (
            "cross-tile + match=0.75 + semantic thr=0.92",
            "cross_tile_match075_thr092",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.75",
                "--semantic-threshold",
                "0.92",
            ],
        ),
        (
            "cross-tile + match=0.75 + semantic thr=0.93",
            "cross_tile_match075_thr093",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.75",
                "--semantic-threshold",
                "0.93",
            ],
        ),
        (
            "cross-tile + match=0.78 + semantic thr=0.91",
            "cross_tile_match078_thr091",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.91",
            ],
        ),
        (
            "cross-tile + match=0.78 + semantic thr=0.92",
            "cross_tile_match078_thr092",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.92",
            ],
        ),
        (
            "cross-tile + match=0.78 + semantic thr=0.93",
            "cross_tile_match078_thr093",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.93",
            ],
        ),
    ],
    # ── MOT17-a: semantic relink ambiguity suppression ─────────────────────────
    # Tuple format: (label, slug, extra_args[, path_override])
    # path_override reuses an existing output dir (must already have metrics.json).
    "mot17_a": [
        (
            "best [m=0.78 thr=0.91]",
            "best_m78_t091",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.91",
            ],
            "combination_fine/cross_tile_match078_thr091",  # reuse cached result
        ),
        (
            "thr=0.92 no fix",
            "t092_no_fix",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.92",
            ],
            "combination_fine/cross_tile_match078_thr092",  # reuse cached result
        ),
        (
            "crowd=0.05 thr=0.91",
            "crowd05_t091",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.91",
                "--semantic-dynamic-margin-crowd",
                "0.05",
            ],
        ),
        (
            "crowd=0.03 thr=0.92",
            "crowd03_t092",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.92",
                "--semantic-dynamic-margin-crowd",
                "0.03",
            ],
        ),
        (
            "crowd=0.05 thr=0.92",
            "crowd05_t092",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.92",
                "--semantic-dynamic-margin-crowd",
                "0.05",
            ],
        ),
        (
            "crowd=0.08 thr=0.92",
            "crowd08_t092",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.92",
                "--semantic-dynamic-margin-crowd",
                "0.08",
            ],
        ),
        (
            "age=0.05 thr=0.92",
            "age05_t092",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.92",
                "--semantic-dynamic-margin-age",
                "0.05",
            ],
        ),
        (
            "iou=0.10 thr=0.92",
            "iou10_t092",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.92",
                "--semantic-iou-weight",
                "0.10",
            ],
        ),
        (
            "crowd=0.05+age=0.03 thr=0.92",
            "crowd05_age03_t092",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.92",
                "--semantic-dynamic-margin-crowd",
                "0.05",
                "--semantic-dynamic-margin-age",
                "0.03",
            ],
        ),
        (
            "crowd=0.05+iou=0.10 thr=0.92",
            "crowd05_iou10_t092",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.92",
                "--semantic-dynamic-margin-crowd",
                "0.05",
                "--semantic-iou-weight",
                "0.10",
            ],
        ),
        (
            "py_relinker thr=0.92 no-iou",
            "py_t092_noiou",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.92",
                "--force-python-relinker",
            ],
        ),
    ],
    # ── MOT17-b: cross-tile score penalty ──────────────────────────────────────
    "mot17_b": [
        (
            "best [m=0.78 thr=0.91]",
            "best_m78_t091",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.91",
            ],
            "combination_fine/cross_tile_match078_thr091",
        ),
        (
            "thr=0.92 no fix",
            "t092_no_fix",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.92",
            ],
            "combination_fine/cross_tile_match078_thr092",
        ),
        (
            "pen=0.95 thr=0.91",
            "pen95_t091",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.91",
                "--cross-tile-score-penalty",
                "0.95",
            ],
        ),
        (
            "pen=0.90 thr=0.91",
            "pen90_t091",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.91",
                "--cross-tile-score-penalty",
                "0.90",
            ],
        ),
        (
            "pen=0.85 thr=0.91",
            "pen85_t091",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.91",
                "--cross-tile-score-penalty",
                "0.85",
            ],
        ),
        (
            "pen=0.95 thr=0.92",
            "pen95_t092",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.92",
                "--cross-tile-score-penalty",
                "0.95",
            ],
        ),
        (
            "pen=0.90 thr=0.92",
            "pen90_t092",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.92",
                "--cross-tile-score-penalty",
                "0.90",
            ],
        ),
        (
            "pen=0.85 thr=0.92",
            "pen85_t092",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.92",
                "--cross-tile-score-penalty",
                "0.85",
            ],
        ),
        (
            "pen=0.90+crowd=0.05 thr=0.92",
            "pen90_crowd05_t092",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.92",
                "--cross-tile-score-penalty",
                "0.90",
                "--semantic-dynamic-margin-crowd",
                "0.05",
            ],
        ),
    ],
    # ── MOT17-c: semantic buffer/rerank default-path verification ──────────────
    "mot17_c": [
        (
            "best [m=0.78 thr=0.91]",
            "best_m78_t091",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.91",
            ],
            "combination_fine/cross_tile_match078_thr091",
        ),
        (
            "buf=2",
            "buf2_t091",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.91",
                "--semantic-buffer-size",
                "2",
            ],
        ),
        (
            "buf=3",
            "buf3_t091",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.91",
                "--semantic-buffer-size",
                "3",
            ],
        ),
        (
            "buf=2 rerank=max",
            "buf2_max_t091",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.91",
                "--semantic-buffer-size",
                "2",
                "--semantic-rerank-mode",
                "max",
            ],
        ),
        (
            "buf=2 cons=0.65",
            "buf2_cons065_t091",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.91",
                "--semantic-buffer-size",
                "2",
                "--semantic-min-consistency",
                "0.65",
            ],
        ),
        (
            "buf=2 top2_mean",
            "buf2_top2_t091",
            [
                "--cross-tile-merge",
                "--match-thresh",
                "0.78",
                "--semantic-threshold",
                "0.91",
                "--semantic-buffer-size",
                "2",
                "--semantic-rerank-mode",
                "top2_mean",
            ],
        ),
    ],
    # ── A2: Reference Quality Gate Sweep (margin + aspect bounds) ─────────────
    # Base: A1+A2 best params (clean_score=0.65, strict_sim=0.0, hq_bank=0.75).
    # Sweeps the two remaining unswept dimensions: frame-edge margin_ratio and
    # h/w aspect ratio bounds [min_aspect, max_aspect].
    "a2": [
        (
            "baseline (defaults)",
            "a2_baseline",
            _A2_BEST,
        ),
        # ── margin_ratio variants ──────────────────────────────────────────────
        (
            "margin=0.02",
            "a2_margin002",
            _A2_BEST + ["--semantic-clean-margin-ratio", "0.02"],
        ),
        (
            "margin=0.05",
            "a2_margin005",
            _A2_BEST + ["--semantic-clean-margin-ratio", "0.05"],
        ),
        (
            "margin=0.08",
            "a2_margin008",
            _A2_BEST + ["--semantic-clean-margin-ratio", "0.08"],
        ),
        # ── min_aspect variants (default=1.2) ──────────────────────────────────
        (
            "min_asp=1.0 (relaxed)",
            "a2_minasp10",
            _A2_BEST + ["--semantic-clean-min-aspect", "1.0"],
        ),
        (
            "min_asp=1.5 (stricter)",
            "a2_minasp15",
            _A2_BEST + ["--semantic-clean-min-aspect", "1.5"],
        ),
        (
            "min_asp=2.0 (strict)",
            "a2_minasp20",
            _A2_BEST + ["--semantic-clean-min-aspect", "2.0"],
        ),
        # ── max_aspect variants (default=4.5) ──────────────────────────────────
        (
            "max_asp=3.5 (tighter)",
            "a2_maxasp35",
            _A2_BEST + ["--semantic-clean-max-aspect", "3.5"],
        ),
        (
            "max_asp=6.0 (looser)",
            "a2_maxasp60",
            _A2_BEST + ["--semantic-clean-max-aspect", "6.0"],
        ),
        # ── aspect combo: typical pedestrian range ─────────────────────────────
        (
            "asp=[1.5,3.5] (tight pedestrian)",
            "a2_asp_tight",
            _A2_BEST + ["--semantic-clean-min-aspect", "1.5", "--semantic-clean-max-aspect", "3.5"],
        ),
        (
            "asp=[1.0,6.0] (loose)",
            "a2_asp_loose",
            _A2_BEST + ["--semantic-clean-min-aspect", "1.0", "--semantic-clean-max-aspect", "6.0"],
        ),
        # ── joint combo: margin + aspect ──────────────────────────────────────
        (
            "margin=0.05 + asp=[1.5,4.5]",
            "a2_margin005_minasp15",
            _A2_BEST + [
                "--semantic-clean-margin-ratio", "0.05",
                "--semantic-clean-min-aspect", "1.5",
            ],
        ),
        (
            "margin=0.05 + asp=[1.5,3.5]",
            "a2_margin005_asp_tight",
            _A2_BEST + [
                "--semantic-clean-margin-ratio", "0.05",
                "--semantic-clean-min-aspect", "1.5",
                "--semantic-clean-max-aspect", "3.5",
            ],
        ),
    ],
    # ── A3: Track-Level Budgeted ReID Sweep ────────────────────────────────────
    "a3": [
        (
            "fixed interval 16 (base)",
            "fixed16",
            _A2_BEST + ["--no-need-reid", "--reid-interval", "16"],
        ),
        (
            "score_ema (unlimited)",
            "score_ema_unlimited",
            _A2_BEST + ["--need-reid", "--reid-trigger-mode", "score_ema", "--reid-budget", "0"],
        ),
        (
            "fixed budget 1",
            "fixed_b1",
            _A2_BEST + ["--no-need-reid", "--reid-interval", "1", "--reid-budget", "1"],
        ),
        (
            "fixed budget 4",
            "fixed_b4",
            _A2_BEST + ["--no-need-reid", "--reid-interval", "1", "--reid-budget", "4"],
        ),
        (
            "fixed budget 8",
            "fixed_b8",
            _A2_BEST + ["--no-need-reid", "--reid-interval", "1", "--reid-budget", "8"],
        ),
        (
            "dynamic budget 1",
            "dyn_b1",
            _A2_BEST + ["--need-reid", "--reid-trigger-mode", "score_ema", "--reid-budget", "1"],
        ),
        (
            "dynamic budget 4",
            "dyn_b4",
            _A2_BEST + ["--need-reid", "--reid-trigger-mode", "score_ema", "--reid-budget", "4"],
        ),
        (
            "dynamic budget 8",
            "dyn_b8",
            _A2_BEST + ["--need-reid", "--reid-trigger-mode", "score_ema", "--reid-budget", "8"],
        ),
        (
            "dynamic ratio 0.2",
            "dyn_r02",
            _A2_BEST + ["--need-reid", "--reid-trigger-mode", "score_ema", "--reid-budget", "0.2"],
        ),
        (
            "dynamic ratio 0.4",
            "dyn_r04",
            _A2_BEST + ["--need-reid", "--reid-trigger-mode", "score_ema", "--reid-budget", "0.4"],
        ),
    ],
    "a6": [
        (
            "bank quality v2 (default weights)",
            "bank_v2_default",
            _A2_BEST + ["--bank-quality-v2"],
        ),
        (
            "detection quality scaling",
            "det_quality_scaling",
            _A2_BEST + ["--detection-quality-scaling"],
        ),
        (
            "bank v2 + det quality",
            "bank_det_quality",
            _A2_BEST + ["--bank-quality-v2", "--detection-quality-scaling"],
        ),
        (
            "bank quality v2 (high det weight)",
            "bank_v2_high_det",
            _A2_BEST + ["--bank-quality-v2", "--bank-quality-w-det", "0.7", "--bank-quality-w-iou", "0.1", "--bank-quality-w-area", "0.05"],
        ),
    ],
    # ── Regression: 46.28% → 43.4% gap isolation ──────────────────────────────
    # Baseline is current defaults (43.40%). Each experiment removes exactly one
    # change that landed after the 46.28% result.  Run in order: the biggest
    # single-factor deltas identify the primary culprits.
    "regression": [
        (
            "budget=0 (unlimited)",
            "budget0",
            ["--reid-budget", "0"],
        ),
        (
            "event_any trigger",
            "event_any",
            ["--reid-trigger-mode", "event_any"],
        ),
        (
            "strict_sim=0 (disabled→uses 0.91)",
            "strict_sim_0",
            ["--semantic-strict-sim-threshold", "0.0"],
        ),
        (
            "no det quality scaling",
            "no_det_quality",
            ["--no-detection-quality-scaling"],
        ),
        (
            "budget=0 + event_any",
            "budget0_event_any",
            ["--reid-budget", "0", "--reid-trigger-mode", "event_any"],
        ),
        (
            "budget=0 + event_any + strict_sim=0",
            "budget0_event_any_strict0",
            ["--reid-budget", "0", "--reid-trigger-mode", "event_any",
             "--semantic-strict-sim-threshold", "0.0"],
        ),
        (
            "all-old approx (46.28% era config)",
            "all_old",
            [
                "--cross-tile-merge", "--match-thresh", "0.78",
                "--semantic-threshold", "0.91",
                "--reid-trigger-mode", "event_any",
                "--reid-budget", "0",
                "--semantic-strict-sim-threshold", "0.0",
                "--semantic-clean-score-threshold", "0.60",
                "--semantic-w-sim-base", "0",
                "--semantic-w-iou-base", "0",
                "--semantic-w-maha-base", "0",
                "--semantic-shift-ambiguity", "0",
                "--semantic-shift-lost-age", "0",
                "--no-detection-quality-scaling",
                "--no-bank-quality-v2",
                "--appearance-bank-high-quality-min-score", "0.70",
            ],
        ),
        (
            "all-old + no GMC (exact 46.28% era)",
            "all_old_no_gmc",
            [
                "--cross-tile-merge", "--match-thresh", "0.78",
                "--semantic-threshold", "0.91",
                "--reid-trigger-mode", "event_any",
                "--reid-budget", "0",
                "--semantic-strict-sim-threshold", "0.0",
                "--semantic-clean-score-threshold", "0.60",
                "--semantic-w-sim-base", "0",
                "--semantic-w-iou-base", "0",
                "--semantic-w-maha-base", "0",
                "--semantic-shift-ambiguity", "0",
                "--semantic-shift-lost-age", "0",
                "--no-detection-quality-scaling",
                "--no-bank-quality-v2",
                "--appearance-bank-high-quality-min-score", "0.70",
                "--no-gmc",
            ],
        ),
    ],
}


def is_mot_file(path: str) -> bool:
    name = Path(path).name
    return name.startswith("MOT") and name.endswith(".txt")


def evaluate_dir(results_dir: str, gt_root: str, detector: str | None) -> dict | None:
    import glob
    import json

    cache_path = os.path.join(results_dir, "metrics.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    files = sorted(
        f for f in glob.glob(os.path.join(results_dir, "*.txt")) if is_mot_file(f)
    )
    if detector:
        files = [f for f in files if f"-{detector}" in Path(f).stem]
    if not files:
        return None

    gt_files = glob.glob(os.path.join(gt_root, "*/gt/gt.txt"))
    if detector:
        gt_files = [f for f in gt_files if f"-{detector}" in Path(f).parts[-3]]
    gt = {
        Path(f).parts[-3]: mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1)
        for f in gt_files
    }

    accs, names = [], []
    for f in files:
        name = os.path.splitext(Path(f).name)[0]
        if name not in gt:
            continue
        ts = mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=-1.0)
        accs.append(mm.utils.compare_to_groundtruth(gt[name], ts, "iou", distth=0.5))
        names.append(name)
    if not accs:
        return None

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs, names=names, metrics=_METRICS, generate_overall=True
    )
    results = {m: summary.loc["OVERALL", m] for m in _METRICS}
    
    # Cast to float/int for JSON serialization
    results = {
        m: (float(v) if m in _PCT or "num" not in m else int(v))
        for m, v in results.items()
    }

    # Save to cache
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def fmt(val, metric: str) -> str:
    if metric in _PCT:
        return f"{val * 100:.1f}%"
    return f"{int(val)}"


def delta_str(base, val, metric: str) -> str:
    if metric in _PCT:
        d = (val - base) * 100
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}pp"
    d = int(val) - int(base)
    sign = "+" if d >= 0 else ""
    return f"{sign}{d}"


def run_eval(
    label: str, output_dir: str, extra_args: list[str], base_args: list[str], dry: bool
) -> bool:
    cmd = (
        ["uv", "run", "python", "scripts/eval/mot17.py", "--output", output_dir]
        + base_args
        + extra_args
    )
    env = os.environ.copy()
    pythonpath_entries = [str(project_root)]
    if src_path.exists():
        pythonpath_entries.append(str(src_path))
    if build_path.exists():
        pythonpath_entries.append(str(build_path))
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    print(f"\n{'=' * 68}")
    print(f"  Running: {label}")
    print(f"  Cmd:     {' '.join(cmd)}")
    print(f"{'=' * 68}")
    if dry:
        return True
    result = subprocess.run(cmd, cwd=str(project_root), env=env)
    return result.returncode == 0


def parse_categories(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names or names == ["all"]:
        return list(_CATEGORY_ORDER)
    unknown = [name for name in names if name not in _CATEGORY_EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unsupported categories: {', '.join(unknown)}")
    return [name for name in _CATEGORY_ORDER if name in names]


def print_table(title: str, results: list[tuple[str, dict | None]]) -> None:
    print(f"\n{'=' * 88}")
    print(f"  {title}")
    print(f"{'=' * 88}")

    base_vals = results[0][1]
    if base_vals is None:
        print("Baseline results not found.")
        return

    lw = 28
    mw = 8
    header = f"{'Config':<{lw}}"
    for d in _DISPLAY:
        header += f" {d:>{mw}}"
    header += "  IDF1 D  IDs D  MOTA D"
    print(header)
    print("-" * len(header))

    for label, r in results:
        if r is None:
            print(f"{label:<{lw}}  (no results)")
            continue
        row = f"{label:<{lw}}"
        for metric in _METRICS:
            row += f" {fmt(r[metric], metric):>{mw}}"
        row += (
            f"  {delta_str(base_vals['idf1'], r['idf1'], 'idf1'):>7}"
            f"  {delta_str(base_vals['num_switches'], r['num_switches'], 'num_switches'):>5}"
            f"  {delta_str(base_vals['mota'], r['mota'], 'mota'):>7}"
        )
        print(row)
    print(f"{'=' * 88}")


def run_optuna_a1(args, base_args):
    import optuna
    
    def objective(trial):
        w_sim = trial.suggest_float("w_sim_base", 0.0, 1.0)
        w_iou = trial.suggest_float("w_iou_base", 0.0, 1.0)
        w_maha = trial.suggest_float("w_maha_base", 0.0, 1.0)
        shift_ambiguity = trial.suggest_float("shift_ambiguity", 0.0, 0.5)
        shift_lost_age = trial.suggest_float("shift_lost_age", 0.0, 0.5)
        
        extra_args = [
            "--semantic-w-sim-base", str(w_sim),
            "--semantic-w-iou-base", str(w_iou),
            "--semantic-w-maha-base", str(w_maha),
            "--semantic-shift-ambiguity", str(shift_ambiguity),
            "--semantic-shift-lost-age", str(shift_lost_age),
        ]
        
        out_dir = f"{args.output_root}/optuna/trial_{trial.number}"
        run_eval(f"Optuna A1 Trial {trial.number}", out_dir, extra_args, base_args, args.dry_run)
        
        metrics = evaluate_dir(out_dir, args.gt_root, args.detector)
        if not metrics:
            raise optuna.TrialPruned()
        
        # Maximize IDF1 primarily
        return metrics["idf1"]
        
    study = optuna.create_study(direction="maximize", study_name="A1_Unified_Score")
    study.optimize(objective, n_trials=args.optuna_trials)
    
    print(f"{'=' * 88}")
    print("Optuna A1 Unified Score Sweep Complete")
    try:
        best_trial = study.best_trial
        print(f"Best trial (IDF1: {best_trial.value:.4f}):")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value:.4f}")
    except ValueError:
        print("No trials completed successfully.")
    print(f"{'=' * 88}")


def run_optuna_a2(args, base_args):
    import optuna
    
    # Use A1 best params as fixed base for A2 sweep
    a1_base = [
        "--semantic-w-sim-base", "0.8012",
        "--semantic-w-iou-base", "0.3423",
        "--semantic-w-maha-base", "0.3117",
        "--semantic-shift-ambiguity", "0.3425",
        "--semantic-shift-lost-age", "0.1778",
    ]
    
    def objective(trial):
        clean_score = trial.suggest_float("clean_score_threshold", 0.50, 0.90)
        strict_sim = trial.suggest_float("strict_sim_threshold", 0.55, 0.95)
        hq_bank_score = trial.suggest_float("high_quality_min_score", 0.60, 0.95)
        margin_ratio = trial.suggest_float("clean_margin_ratio", 0.0, 0.10)
        min_aspect = trial.suggest_float("clean_min_aspect", 0.8, 2.0)
        max_aspect = trial.suggest_float("clean_max_aspect", 3.0, 7.0)

        extra_args = a1_base + [
            "--semantic-clean-score-threshold", str(clean_score),
            "--semantic-strict-sim-threshold", str(strict_sim),
            "--appearance-bank-high-quality-min-score", str(hq_bank_score),
            "--semantic-clean-margin-ratio", str(margin_ratio),
            "--semantic-clean-min-aspect", str(min_aspect),
            "--semantic-clean-max-aspect", str(max_aspect),
        ]
        
        out_dir = f"{args.output_root}/optuna_a2/trial_{trial.number}"
        run_eval(f"Optuna A2 Trial {trial.number}", out_dir, extra_args, base_args, args.dry_run)
        
        metrics = evaluate_dir(out_dir, args.gt_root, args.detector)
        if not metrics:
            raise optuna.TrialPruned()
        
        # Maximize IDF1
        return metrics["idf1"]
        
    study = optuna.create_study(direction="maximize", study_name="A2_Reference_Quality")
    study.optimize(objective, n_trials=args.optuna_trials)
    
    print(f"{'=' * 88}")
    print("Optuna A2 Reference Quality Sweep Complete")
    try:
        best_trial = study.best_trial
        print(f"Best trial (IDF1: {best_trial.value:.4f}):")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value:.4f}")
    except ValueError:
        print("No trials completed successfully.")
    print(f"{'=' * 88}")


def run_optuna_a3(args, base_args):
    import optuna

    # Use A2 best params as fixed base for A3 sweep
    a2_base = _A2_BEST

    def objective(trial):
        budget = trial.suggest_categorical("reid_budget", [1, 2, 4, 8, 12, 16])
        trigger_mode = trial.suggest_categorical(
            "reid_trigger_mode", ["score_ema", "event_any"]
        )

        extra_args = a2_base + [
            "--need-reid",
            "--reid-trigger-mode",
            trigger_mode,
            "--reid-budget",
            str(budget),
        ]

        out_dir = f"{args.output_root}/optuna_a3/trial_{trial.number}"
        run_eval(
            f"Optuna A3 Trial {trial.number}", out_dir, extra_args, base_args, args.dry_run
        )

        metrics = evaluate_dir(out_dir, args.gt_root, args.detector)
        if not metrics:
            raise optuna.TrialPruned()

        # Maximize IDF1
        return metrics["idf1"]

    study = optuna.create_study(direction="maximize", study_name="A3_Budgeted_ReID")
    study.optimize(objective, n_trials=args.optuna_trials)

    print(f"{'=' * 88}")
    print("Optuna A3 Budgeted ReID Sweep Complete")
    try:
        best_trial = study.best_trial
        print(f"Best trial (IDF1: {best_trial.value:.4f}):")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
    except ValueError:
        print("No trials completed successfully.")
    print(f"{'=' * 88}")


def run_optuna_a6(args, base_args):
    import optuna

    # Use A2/A3 best params as fixed base for A6 sweep
    a3_base = _A2_BEST + ["--reid-budget", "0.2", "--reid-trigger-mode", "score_ema"]

    def objective(trial):
        w_det = trial.suggest_float("w_det", 0.3, 0.7)
        w_iou = trial.suggest_float("w_iou", 0.1, 0.4)
        w_aspect = trial.suggest_float("w_aspect", 0.05, 0.25)
        w_center = trial.suggest_float("w_center", 0.05, 0.20)
        w_area = trial.suggest_float("w_area", 0.05, 0.20)
        
        # Normalize weights
        sum_w = w_det + w_iou + w_aspect + w_center + w_area
        w_det /= sum_w
        w_iou /= sum_w
        w_aspect /= sum_w
        w_center /= sum_w
        w_area /= sum_w
        
        extra_args = a3_base + [
            "--bank-quality-v2",
            "--bank-quality-w-det", f"{w_det:.4f}",
            "--bank-quality-w-iou", f"{w_iou:.4f}",
            "--bank-quality-w-aspect", f"{w_aspect:.4f}",
            "--bank-quality-w-center", f"{w_center:.4f}",
            "--bank-quality-w-area", f"{w_area:.4f}",
        ]
        
        out_dir = f"{args.output_root}/optuna_a6/trial_{trial.number}"
        run_eval(f"Optuna A6 Trial {trial.number}", out_dir, extra_args, base_args, args.dry_run)
        
        metrics = evaluate_dir(out_dir, args.gt_root, args.detector)
        if not metrics:
            raise optuna.TrialPruned()
            
        return metrics["idf1"]

    study = optuna.create_study(direction="maximize", study_name="A6_Bank_Quality")
    study.optimize(objective, n_trials=args.optuna_trials)

    print(f"{'=' * 88}")
    print("Optuna A6 Bank Quality Sweep Complete")
    try:
        best_trial = study.best_trial
        print(f"Best trial (IDF1: {best_trial.value:.4f}):")
        # Normalize best params for printing
        p = best_trial.params
        s = sum(p.values())
        for key, value in p.items():
            print(f"  --bank-quality-{key.replace('_', '-')}: {value/s:.4f}")
    except ValueError:
        print("No trials completed successfully.")
    print(f"{'=' * 88}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category", default="all", help="Comma-separated categories or 'all'."
    )
    parser.add_argument("--detector", choices=["SDP", "DPM", "FRCNN"], default="SDP")
    parser.add_argument("--sequences", default="", help="Comma-separated sequence names.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--gt-root", default="datasets/MOT17/train")
    parser.add_argument("--output-root", default="scripts/eval/output/ablation_mot17")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--optuna",
        choices=["a1", "a2", "a3", "a6"],
        default=None,
        help="Run Bayesian optimization for a specific module.",
    )
    parser.add_argument(
        "--optuna-trials", type=int, default=20, help="Number of trials for Optuna."
    )
    args = parser.parse_args()

    base_args = ["--detector", args.detector]
    if args.sequences:
        base_args += ["--sequences", args.sequences]
    if args.max_frames:
        base_args += ["--max-frames", str(args.max_frames)]

    if args.optuna == "a1":
        run_optuna_a1(args, base_args)
        return
    if args.optuna == "a2":
        run_optuna_a2(args, base_args)
        return
    if args.optuna == "a3":
        run_optuna_a3(args, base_args)
        return
    if args.optuna == "a6":
        run_optuna_a6(args, base_args)
        return

    categories = parse_categories(args.category)

    baseline_dir = f"{args.output_root}/baseline"
    if not args.skip_run:
        run_eval("Baseline", baseline_dir, [], base_args, args.dry_run)

    baseline_metrics = evaluate_dir(baseline_dir, args.gt_root, args.detector)

    for category in categories:
        # Expand experiment tuples: (label, slug, extra[, path_override])
        # path_override points to an existing output dir to reuse (skips run if
        # metrics.json already present there).
        expanded: list[tuple[str, str, list[str]]] = []
        for entry in _CATEGORY_EXPERIMENTS[category]:
            label, slug, extra = entry[0], entry[1], entry[2]
            path_override: str | None = entry[3] if len(entry) > 3 else None  # type: ignore[misc]
            if path_override:
                out_dir = f"{args.output_root}/{path_override}"
            else:
                out_dir = f"{args.output_root}/{category}/{slug}"
            expanded.append((label, out_dir, extra))

        experiments: list[tuple[str, str, list[str]]] = [
            ("Baseline", baseline_dir, [])
        ] + expanded

        if not args.skip_run:
            for label, out_dir, extra in experiments[1:]:
                # Skip if cached — supports both resume and path_override reuse.
                metrics_cache = os.path.join(out_dir, "metrics.json")
                if os.path.exists(metrics_cache):
                    print(f"  [CACHED] {category} / {label}  ({out_dir})")
                    continue
                ok = run_eval(
                    f"{category}: {label}", out_dir, extra, base_args, args.dry_run
                )
                if not ok:
                    print(f"  [WARN] {category} / {label} failed, continuing...")

        results: list[tuple[str, dict | None]] = [("Baseline", baseline_metrics)]
        for label, out_dir, _ in experiments[1:]:
            results.append((label, evaluate_dir(out_dir, args.gt_root, args.detector)))
        print_table(f"MOT17 Ablation - {category} - detector={args.detector}", results)


if __name__ == "__main__":
    main()
