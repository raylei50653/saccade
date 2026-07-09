#!/usr/bin/env python3
"""Static guard for the headline tracker-decision contract (no GPU).

Checks:

1. Production presets (YAML):
     configs/presets/mamba_whole_graph.yaml      (s)
     configs/presets/mamba_whole_graph_m.yaml    (m)
2. Inject-map sanity (source text, C8):
     pipeline.py remains the production set_* site
     kalman_r_scale → r_scale remap still present
     private continuation remains det-set policy (not a tracker setter)

Against the locked active contract in
``docs/research/tracker-decision/README.md`` and the C1–C8 matrix in
``docs/research/tracker-decision/audit/active_contract_healthcheck.md``.

What this does **not** do:
  - change presets / schema / kernels
  - merge dual stability knobs
  - run MOT17 / 7-seq
  - read env vars at runtime (dual-stability bid is a printed NOTE only)
  - execute or import the eval stack (text/regex only for C8)

Exit 0 if all hard checks pass; exit 1 on any failure.

Usage:
  uv run python scripts/tools/check_headline_decision_contract.py
  uv run python scripts/tools/check_headline_decision_contract.py --quiet
  uv run python scripts/tools/check_headline_decision_contract.py --skip-inject
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESET_S = REPO_ROOT / "configs" / "presets" / "mamba_whole_graph.yaml"
PRESET_M = REPO_ROOT / "configs" / "presets" / "mamba_whole_graph_m.yaml"
PIPELINE_PY = REPO_ROOT / "src" / "saccade" / "perception" / "eval" / "pipeline.py"
DETECTION_FILTERS_PY = (
    REPO_ROOT / "src" / "saccade" / "perception" / "eval" / "detection_filters.py"
)
TRACKER_GPU_PY = (
    REPO_ROOT / "src" / "saccade" / "perception" / "tracking" / "tracker_gpu.py"
)

# --- Locked contract values (update with README active contract) ---

OCC_STATE_EXPECTED: dict[str, Any] = {
    "occ_state_enabled": True,
    "occ_iou_thresh": 0.45,
    "occ_foot_gap": 0.15,
    "occ_ttl": 4,
    "occ_cost_weight": 0.50,
}

# Keys that must be present and equal on s and m.
SHARED_EQUAL_KEYS: tuple[str, ...] = (
    "match_thresh",
    "new_track_thresh",
    "confirm_streak",
    "confirm_score_thresh",
    "oao_tau",
    "oao_ramp_frames",
    "multiplicative_cost",
    "sinkhorn_lambda",
    "stability_cost_w",
    "private_continuation_enabled",
    "private_candidate_nms_iou",
    "private_prior_iou_threshold",
    "private_min_score",
    "private_max_candidates",
    "private_selection_mode",
    "relink_bridge_enabled",
    "relink_bridge_margin",
    "relink_bridge_spatial_gate",
    "reid_mode",
    "gmc",
    "gmc_downscale",
    "gmc_fg_mask",
    *OCC_STATE_EXPECTED.keys(),
)

SHARED_EXPECTED: dict[str, Any] = {
    "match_thresh": 0.50,
    "new_track_thresh": 0.28,
    "confirm_streak": 3,
    "confirm_score_thresh": 0.50,
    "oao_tau": 0.50,
    "oao_ramp_frames": 25,
    "multiplicative_cost": True,
    "sinkhorn_lambda": 10,
    "stability_cost_w": 0.20,
    "private_continuation_enabled": True,
    "private_candidate_nms_iou": 0.70,
    "private_prior_iou_threshold": 0.30,
    "private_min_score": 0.10,
    "private_max_candidates": 50,
    "private_selection_mode": "global",
    "relink_bridge_enabled": True,
    "relink_bridge_margin": 0.05,
    "relink_bridge_spatial_gate": 0.0,
    "reid_mode": "off",
    "gmc": True,
    "gmc_downscale": 4,
    "gmc_fg_mask": False,
    **OCC_STATE_EXPECTED,
}

# Intentional s vs m decision deltas (motion + bridge gates).
M_DELTAS: dict[str, tuple[Any, Any]] = {
    # key: (s_expected, m_expected)
    "kalman_r_scale": (2.8, 3.5),
    "relink_bridge_px": (0.25, 0.4),
    "relink_bridge_h_lo": (0.75, 0.6),
    "relink_bridge_h_hi": (1.33, 1.7),
    "relink_bridge_dir_bonus": (0.8, 0.0),
}

# Headline must keep these off / inert (NO-GO or PRESET-OFF contract).
# Missing key is OK (schema default) unless require_present=True.
NO_GO_OR_OFF: dict[str, Any] = {
    "fuse_score_weight": 0.0,
    "nsa_kalman": False,
    "gmc_fg_mask": False,
    "person_geometry_prior": False,
    "detection_quality_scaling": False,
    "geometry_suspect_support": False,
    "id_stability_filter": False,
    "track_person_only": False,
    "per_seq_adapt": False,
    "relink_enabled": False,  # bank ReID relink
    "association_scoring_mode": "baseline",
    "vel_dir_weight": 0.0,
    # OAO spatial family — inert when present
    "oao_contest_thresh": -1,
    "oao_score_w": -1,
    "oao_occ_mode": 0,
    "oao_crowd_radius": 0,
    "oao_height_gate": 0,
    "oao_foot_gate": 0,
}

# Must be written explicitly (not omitted) so silence ≠ accidental off.
REQUIRE_EXPLICIT: tuple[str, ...] = (
    *OCC_STATE_EXPECTED.keys(),
    "private_continuation_enabled",
    "relink_bridge_dir_bonus",
    "kalman_r_scale",
    "stability_cost_w",
    "multiplicative_cost",
    "fuse_score_weight",
)


def load_preset(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"preset missing: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"preset is not a mapping: {path}")
    return data


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) is bool(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def _fmt(v: Any) -> str:
    return repr(v)


def check_presets(
    s: dict[str, Any],
    m: dict[str, Any],
    *,
    s_label: str = "s",
    m_label: str = "m",
) -> tuple[list[str], list[str]]:
    """Return (failures, notes). Pure function for unit tests."""
    failures: list[str] = []
    notes: list[str] = []

    def need(cfg: dict[str, Any], key: str, label: str) -> Any | None:
        if key not in cfg:
            failures.append(
                f"[{label}] missing required key {key!r} (must be explicit)"
            )
            return None
        return cfg[key]

    # C1 — occ_state explicit + expected on both
    for label, cfg in ((s_label, s), (m_label, m)):
        for key, exp in OCC_STATE_EXPECTED.items():
            got = need(cfg, key, label)
            if got is None:
                continue
            if not _eq(got, exp):
                failures.append(
                    f"[{label}] {key}={_fmt(got)}, expected {_fmt(exp)} (C1 occ_state)"
                )

    # Explicit keys (silence risk)
    for label, cfg in ((s_label, s), (m_label, m)):
        for key in REQUIRE_EXPLICIT:
            if key not in cfg:
                failures.append(
                    f"[{label}] {key!r} not written in preset "
                    f"(must be explicit for reviewability)"
                )

    # C2 + fixed shared values
    for key in SHARED_EQUAL_KEYS:
        sv = s.get(key, _MISSING)
        mv = m.get(key, _MISSING)
        if sv is _MISSING and mv is _MISSING:
            # some shared keys are also in REQUIRE_EXPLICIT / OCC; already flagged
            if key not in REQUIRE_EXPLICIT and key not in OCC_STATE_EXPECTED:
                failures.append(
                    f"[s/m] shared key {key!r} missing on both presets (C2)"
                )
            continue
        if sv is _MISSING:
            failures.append(f"[{s_label}] missing shared key {key!r} (C2)")
            continue
        if mv is _MISSING:
            failures.append(f"[{m_label}] missing shared key {key!r} (C2)")
            continue
        if not _eq(sv, mv):
            failures.append(
                f"[s/m] {key} differs: s={_fmt(sv)} m={_fmt(mv)} (must be equal — C2)"
            )
        if key in SHARED_EXPECTED and not _eq(sv, SHARED_EXPECTED[key]):
            failures.append(
                f"[s/m] {key}={_fmt(sv)}, contract expects {_fmt(SHARED_EXPECTED[key])} "
                f"(C2 shared value)"
            )

    # C3 — m deltas
    for key, (s_exp, m_exp) in M_DELTAS.items():
        sv = need(s, key, s_label)
        mv = need(m, key, m_label)
        if sv is not None and not _eq(sv, s_exp):
            failures.append(
                f"[{s_label}] {key}={_fmt(sv)}, expected {_fmt(s_exp)} (C3 m-delta s side)"
            )
        if mv is not None and not _eq(mv, m_exp):
            failures.append(
                f"[{m_label}] {key}={_fmt(mv)}, expected {_fmt(m_exp)} (C3 m-delta m side)"
            )

    # C4 — private continuation enabled (value also in SHARED_EXPECTED)
    for label, cfg in ((s_label, s), (m_label, m)):
        if cfg.get("private_continuation_enabled") is not True:
            if "private_continuation_enabled" in cfg:
                failures.append(
                    f"[{label}] private_continuation_enabled="
                    f"{_fmt(cfg.get('private_continuation_enabled'))}, expected True (C4)"
                )

    # C5 covered by M_DELTAS dir_bonus

    # C6 — dual stability NOTE only (do not auto-merge / fail on env)
    scw = s.get("stability_cost_w")
    notes.append(
        "NOTE [C6 dual stability] cost-side stability_cost_w="
        f"{_fmt(scw)} (YAML ACTIVE). Bid-side SACCADE_STABILITY_W is env-only "
        "(default ~0.1 in tracker_gpu.cu) — not checked here; do not merge knobs "
        "without dual_stability_cleanup.md A/B/C decision + eval."
    )

    # C7 — NO-GO / PRESET-OFF not enabled
    for label, cfg in ((s_label, s), (m_label, m)):
        for key, off_val in NO_GO_OR_OFF.items():
            if key not in cfg:
                continue  # absent → rely on schema; not a YAML regression
            got = cfg[key]
            if not _eq(got, off_val):
                # Special-case: fuse_score must be ~0
                if (
                    key == "fuse_score_weight"
                    and isinstance(got, (int, float))
                    and float(got) == 0.0
                ):
                    continue
                failures.append(
                    f"[{label}] {key}={_fmt(got)}, headline contract requires "
                    f"{_fmt(off_val)} (C7 NO-GO/PRESET-OFF)"
                )

    # reid_mode must be off string
    for label, cfg in ((s_label, s), (m_label, m)):
        rm = cfg.get("reid_mode")
        if rm is not None and str(rm).strip().lower() not in ("off", "none", "false"):
            failures.append(
                f"[{label}] reid_mode={_fmt(rm)}, headline contract requires 'off' (C7)"
            )

    return failures, notes


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"source missing: {path}")
    return path.read_text(encoding="utf-8")


def check_inject_map(
    pipeline_src: str,
    detection_filters_src: str,
    tracker_gpu_src: str,
    *,
    pipeline_label: str = "pipeline.py",
    filters_label: str = "detection_filters.py",
    tracker_label: str = "tracker_gpu.py",
) -> tuple[list[str], list[str]]:
    """C8 inject-map sanity on source text. Pure function for unit tests.

    Does not import modules — only static patterns that lock the production
    call map documented in tracker-decision callpoints / native_bridge.
    """
    failures: list[str] = []
    notes: list[str] = []

    # --- pipeline.py is production inject site for tracker setters ---
    for method in ("set_params", "set_occ_params", "set_relink_params"):
        # Prefer the real call shape: detector.tracker.set_* (
        call_pat = re.compile(
            rf"detector\.tracker\.{re.escape(method)}\s*\(",
            re.MULTILINE,
        )
        if not call_pat.search(pipeline_src):
            # Fallback: any .set_* (still on tracker path in this file)
            loose = re.compile(rf"\.{re.escape(method)}\s*\(", re.MULTILINE)
            if not loose.search(pipeline_src):
                failures.append(
                    f"[{pipeline_label}] missing production call "
                    f"detector.tracker.{method}(...) (C8 inject site)"
                )

    # kalman_r_scale → r_scale remap at inject
    if not re.search(r"r_scale\s*=\s*cfg\.geometry\.kalman_r_scale", pipeline_src):
        failures.append(
            f"[{pipeline_label}] missing r_scale=cfg.geometry.kalman_r_scale "
            f"remap (C8 name remap)"
        )

    # occ_state fields must flow through set_occ_params kwargs
    for field in (
        "occ_state_enabled",
        "occ_iou_thresh",
        "occ_foot_gap",
        "occ_ttl",
        "occ_cost_weight",
    ):
        if f"cfg.geometry.{field}" not in pipeline_src:
            failures.append(
                f"[{pipeline_label}] set_occ path must reference "
                f"cfg.geometry.{field} (C8 occ inject)"
            )

    # stability cost still injected when multiplicative path is used
    if "set_stability_cost_w" not in pipeline_src:
        failures.append(f"[{pipeline_label}] missing set_stability_cost_w inject (C8)")
    if "set_multiplicative_cost" not in pipeline_src:
        failures.append(
            f"[{pipeline_label}] missing set_multiplicative_cost inject (C8)"
        )

    # --- private continuation is det-set policy, not a tracker setter ---
    if "def _append_private_continuation_candidates" not in detection_filters_src:
        failures.append(
            f"[{filters_label}] missing _append_private_continuation_candidates "
            f"(C8 private is det-set policy)"
        )

    # Score clamp below birth threshold (continue ≠ birth)
    if "birth_ceiling" not in detection_filters_src:
        failures.append(
            f"[{filters_label}] missing birth_ceiling "
            f"(C8 private score clamp below new_track_thresh)"
        )
    if not re.search(r"frame_new_track_thresh|new_track_thresh", detection_filters_src):
        failures.append(
            f"[{filters_label}] private continuation must reference "
            f"new_track_thresh for birth ceiling (C8)"
        )

    # Must not become a GPUByteTracker setter API
    forbidden_tracker_private = (
        "set_private_continuation",
        "private_continuation_enabled=",  # kwargs on set_params-style
    )
    # Only flag if set_params / tracker methods grow a private_continuation API
    if re.search(
        r"def set_params\([\s\S]*?private_continuation",
        tracker_gpu_src,
    ):
        failures.append(
            f"[{tracker_label}] set_params must not take private_continuation "
            f"(C8 private is not a tracker setter)"
        )
    for token in forbidden_tracker_private:
        if token in tracker_gpu_src:
            failures.append(
                f"[{tracker_label}] unexpected {token!r} — private continuation "
                f"must remain det-set policy (C8)"
            )

    # pipeline must not inject private via a tracker setter either
    if re.search(
        r"detector\.tracker\.set_private_continuation|"
        r"set_params\([\s\S]{0,800}?private_continuation",
        pipeline_src,
    ):
        failures.append(
            f"[{pipeline_label}] private_continuation must not be injected via "
            f"tracker setters (C8 det-set policy)"
        )

    # facade still exposes r_scale (not only kalman_r_scale) on set_params
    if not re.search(
        r"def set_params\([\s\S]*?\br_scale\b",
        tracker_gpu_src,
    ):
        failures.append(
            f"[{tracker_label}] set_params signature should accept r_scale "
            f"(C8 remap target)"
        )

    notes.append(
        "NOTE [C8 inject map] production setters: pipeline.py "
        "set_params / set_occ_params / set_relink_params; "
        "kalman_r_scale→r_scale; private continuation stays in "
        "detection_filters (det-set), not GPUByteTracker."
    )
    return failures, notes


def check_inject_map_from_repo(
    *,
    pipeline_path: Path = PIPELINE_PY,
    filters_path: Path = DETECTION_FILTERS_PY,
    tracker_path: Path = TRACKER_GPU_PY,
) -> tuple[list[str], list[str]]:
    return check_inject_map(
        _read_text(pipeline_path),
        _read_text(filters_path),
        _read_text(tracker_path),
        pipeline_label=str(pipeline_path.relative_to(REPO_ROOT))
        if pipeline_path.is_relative_to(REPO_ROOT)
        else str(pipeline_path),
        filters_label=str(filters_path.relative_to(REPO_ROOT))
        if filters_path.is_relative_to(REPO_ROOT)
        else str(filters_path),
        tracker_label=str(tracker_path.relative_to(REPO_ROOT))
        if tracker_path.is_relative_to(REPO_ROOT)
        else str(tracker_path),
    )


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only print failures / final line",
    )
    parser.add_argument(
        "--s-preset",
        type=Path,
        default=PRESET_S,
        help="override s preset path (tests)",
    )
    parser.add_argument(
        "--m-preset",
        type=Path,
        default=PRESET_M,
        help="override m preset path (tests)",
    )
    parser.add_argument(
        "--skip-inject",
        action="store_true",
        help="skip C8 inject-map source checks (YAML only)",
    )
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=PIPELINE_PY,
        help="override pipeline.py path (tests)",
    )
    parser.add_argument(
        "--detection-filters",
        type=Path,
        default=DETECTION_FILTERS_PY,
        help="override detection_filters.py path (tests)",
    )
    parser.add_argument(
        "--tracker-gpu",
        type=Path,
        default=TRACKER_GPU_PY,
        help="override tracker_gpu.py path (tests)",
    )
    args = parser.parse_args(argv)

    try:
        s = load_preset(args.s_preset)
        m = load_preset(args.m_preset)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"✗ failed to load presets: {exc}")
        return 1

    failures, notes = check_presets(s, m)

    if not args.skip_inject:
        try:
            inj_f, inj_n = check_inject_map_from_repo(
                pipeline_path=args.pipeline,
                filters_path=args.detection_filters,
                tracker_path=args.tracker_gpu,
            )
        except OSError as exc:
            print(f"✗ failed to load inject sources: {exc}")
            return 1
        failures.extend(inj_f)
        notes.extend(inj_n)

    if not args.quiet:
        print("Headline decision contract check")
        print(f"  s: {args.s_preset}")
        print(f"  m: {args.m_preset}")
        if not args.skip_inject:
            print(f"  inject: {args.pipeline.name} + private det-set path")
        for note in notes:
            print(f"  {note}")

    if failures:
        print(f"✗ {len(failures)} headline decision contract violation(s):")
        for f in failures:
            print(f"  - {f}")
        print(
            "See docs/research/tracker-decision/audit/active_contract_healthcheck.md "
            "and no_go_guardrails.md"
        )
        return 1

    print(
        "✓ headline decision contract OK (YAML C1–C7 + inject-map C8)"
        if not args.skip_inject
        else "✓ headline decision contract OK (YAML only; inject skipped)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
