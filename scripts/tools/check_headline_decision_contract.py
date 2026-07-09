#!/usr/bin/env python3
"""Static guard for the headline tracker-decision contract (no GPU).

Checks production presets:

  configs/presets/mamba_whole_graph.yaml      (s)
  configs/presets/mamba_whole_graph_m.yaml    (m)

Against the locked active contract in
``docs/research/tracker-decision/README.md`` and the C1–C7 matrix in
``docs/research/tracker-decision/audit/active_contract_healthcheck.md``.

What this does **not** do:
  - change presets / schema / kernels
  - merge dual stability knobs
  - run MOT17 / 7-seq
  - read env vars at runtime (dual-stability bid is a printed NOTE only)

Exit 0 if all hard checks pass; exit 1 on any failure.

Usage:
  uv run python scripts/tools/check_headline_decision_contract.py
  uv run python scripts/tools/check_headline_decision_contract.py --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESET_S = REPO_ROOT / "configs" / "presets" / "mamba_whole_graph.yaml"
PRESET_M = REPO_ROOT / "configs" / "presets" / "mamba_whole_graph_m.yaml"

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
    args = parser.parse_args(argv)

    try:
        s = load_preset(args.s_preset)
        m = load_preset(args.m_preset)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"✗ failed to load presets: {exc}")
        return 1

    failures, notes = check_presets(s, m)

    if not args.quiet:
        print("Headline decision contract check")
        print(f"  s: {args.s_preset}")
        print(f"  m: {args.m_preset}")
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
        "✓ headline decision contract OK "
        "(shared keys + m deltas + occ_state + private + NO-GO surface)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
