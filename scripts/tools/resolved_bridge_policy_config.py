#!/usr/bin/env python3
"""Single authority for `resolved_bridge_policy_config_v1` fingerprints.

A declaration that freezes a bridge policy must pin the *resolved* configuration
it will run under, not the handful of knobs its review table happens to print.
The short table is a review aid; this fingerprint is the identity.

The resolution order is the runtime's own (`scripts/eval/mot17.py`):

    argparse defaults < config file < module YAMLs < preset < CLI flags

so a fingerprint is taken over the sole preset invocation with no module or CLI
override, exactly as a headline run resolves it.

Canonical form, per the H0 declaration § 2: UTF-8 JSON over exactly `FIELDS`,
lexicographic keys, no insignificant whitespace, JSON `true`/`false`, finite
numbers. This module reproduces the fingerprint already declared for the `s`
preset (`b1b78318…`), which is what licenses it to compute new ones.

Usage:
    python scripts/tools/resolved_bridge_policy_config.py --preset mamba_whole_graph_m
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Final

_REPO: Final[Path] = Path(__file__).resolve().parents[2]
_PRESET_DIR: Final[Path] = _REPO / "configs" / "presets"

# `mot17_args` is a flat module under scripts/eval, not an installed package.
if str(_REPO / "scripts" / "eval") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts" / "eval"))

import yaml  # noqa: E402
from mot17_args import build_parser  # noqa: E402

# The frozen field set of `resolved_bridge_policy_config_v1`. Adding or removing
# a field changes every fingerprint, so it is a versioned contract, not a list
# to tidy: a new field means a new `_v2` and a re-declaration, never an in-place
# edit here.
FIELDS: Final[tuple[str, ...]] = (
    "reid_mode",
    "relink_enabled",
    "relink_bank_cap",
    "relink_sim_thresh",
    "relink_lambda",
    "relink_spatial_gate",
    "relink_max_age",
    "relink_bridge_enabled",
    "relink_bridge_px",
    "relink_bridge_at",
    "relink_bridge_min_lost",
    "relink_bridge_ttl",
    "relink_bridge_max_speed",
    "relink_bridge_person_height",
    "relink_bridge_fps",
    "relink_bridge_margin",
    "relink_bridge_spatial_gate",
    "relink_bridge_anchor",
    "relink_bridge_anchor_rate",
    "relink_bridge_h_lo",
    "relink_bridge_h_hi",
    "relink_bridge_dir_bonus",
    "relink_bridge_occ_gate_cover",
    "relink_bridge_occ_gap_min",
    "relink_bridge_occ_expand_px",
    "relink_bridge_occ_expand_cover",
    "relink_bridge_app_veto",
)

SCHEMA_ID: Final[str] = "resolved_bridge_policy_config_v1"


def preset_path(preset: str) -> Path:
    path = _PRESET_DIR / f"{preset}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no such preset: {path}")
    return path


def preset_sha256(preset: str) -> str:
    """Byte hash of the preset file itself (the declaration's `headline preset`)."""
    return hashlib.sha256(preset_path(preset).read_bytes()).hexdigest()


def resolve(preset: str) -> dict[str, Any]:
    """Resolve `preset` through the runtime parser, no module or CLI override."""
    parser = build_parser()
    with preset_path(preset).open(encoding="utf-8") as handle:
        overrides = yaml.safe_load(handle) or {}
    parser.set_defaults(**overrides)
    resolved = vars(parser.parse_args([]))

    missing = [field for field in FIELDS if field not in resolved]
    if missing:
        raise KeyError(
            f"{SCHEMA_ID} fields absent from the resolved namespace: {missing}"
        )
    return {field: resolved[field] for field in FIELDS}


def canonical_json(policy: dict[str, Any]) -> str:
    return json.dumps(policy, sort_keys=True, separators=(",", ":"))


def fingerprint(preset: str) -> str:
    return hashlib.sha256(canonical_json(resolve(preset)).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", required=True, help="preset stem under configs/presets/"
    )
    parser.add_argument(
        "--json", action="store_true", help="print the canonical JSON payload"
    )
    args = parser.parse_args()

    policy = resolve(args.preset)
    if args.json:
        print(canonical_json(policy))
    print(f"preset            {args.preset}.yaml")
    print(f"preset sha256     {preset_sha256(args.preset)}")
    print(f"{SCHEMA_ID}  {fingerprint(args.preset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
