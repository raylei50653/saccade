#!/usr/bin/env python3
"""Non-authoritative launch-hygiene pre-authorization gate for H0 Phase A.

Both prior owner-authorized re-entries (#209 and #224/#227) consumed their
single exactly-once authorization only to fail the controller's launch preflight
on a pre-existing ``build/h0_phase_a`` tree, terminating at
``H0_PROVENANCE_INVALID`` before any capture checkpoint ran.  The registry's
``future_reentry_precondition`` (docs/research/contracts/claim_state_registry.md)
requires that, before the next exactly-once authorization is granted, launch
hygiene be a *machine-checked, non-authoritative* gate that reuses the
controller's real preflight predicate and fail-closed rejects that hazard.

This tool is that gate.  It is deliberately **not** a controller mode: it never
discovers a freeze, reads a research sequence, writes any H0 evidence root,
produces an H0 terminal, or consumes an authorization.  It calls the controller's
own ``assert_no_preexisting_build_tree`` predicate — the single source of the
``build/h0_phase_a exists at controller launch`` terminal — so a green gate here
means the controller's own verdict on that predicate is clear.  A red gate means
the operator must clean the workspace before spending an authorization on it.
"""
# status: stable

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "scripts/tools"
if TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, TOOLS.as_posix())

import run_h0_phase_a as controller  # noqa: E402

SCHEMA = "h0_launch_hygiene_gate_v1"
# The controller predicate this gate reuses.  Named as a string only for the
# report; the call below binds to the live function so the two cannot diverge.
PREDICATE_SOURCE = "run_h0_phase_a.assert_no_preexisting_build_tree"


def evaluate(root: Path) -> dict[str, Any]:
    """Run the reused controller predicate and return a canonical verdict.

    Never raises for the hazard it screens: a pre-existing build tree is a
    ``rejected`` verdict, not a crash, because the whole point is to report the
    hazard to the operator before an authorization is spent.
    """
    resolved = root.resolve(strict=True)
    report: dict[str, Any] = {
        "authority": "non_authoritative",
        "authorization_consumed": False,
        "capture": "forbidden",
        "checked_subtree": controller.AUTHORITATIVE_BUILD_SUBTREE,
        "predicate_source": PREDICATE_SOURCE,
        "reason": None,
        "repository_root": resolved.as_posix(),
        "schema": SCHEMA,
        "terminal_claim": "forbidden",
    }
    try:
        controller.assert_no_preexisting_build_tree(resolved)
    except controller.ContractError as exc:
        report["result"] = "rejected"
        report["reason"] = str(exc)
    else:
        report["result"] = "clear"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root to screen (default: this checkout)",
    )
    args = parser.parse_args(argv)
    report = evaluate(args.root)
    json.dump(report, sys.stdout, ensure_ascii=False, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0 if report["result"] == "clear" else 1


if __name__ == "__main__":
    raise SystemExit(main())
