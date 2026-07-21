#!/usr/bin/env python3
"""Check association recovery tools registry (R) against disk (D) and NO-GO (N).

Implements Step 2 of:
  docs/modules/semantic/research/association_recovery_info_source_contract_20260709.md

This script **checks / prints only**. It must never invent doors, roles, verdicts,
metrics, or promotion decisions.

Checks (errors fail; warnings print but exit 0 unless --strict-warn):

  R schema     required fields, allowed enums, forbidden payload keys
  path_exists  every tool.path exists on D                          (error)
  fact_owner   every fact_owner path exists on D                    (warn)
  no_go_id     every cited no_go_ids[] resolves in no_go_registry   (error)
  recipe_steps recipe steps reference known tool ids                (error)
  wrapper      role wrapper requires canonical_id in R              (error)
  redirect     wrapper path appears to target declared canonical    (warn)
  stale_R      (same as path_exists)
  missing_R    AssA-ish scripts on disk not present in R            (warn)

Modes:

  (default)          run checks
  --list             print tools grouped by door (from R only)
  --print-recipe ID  print recipe steps + paths (from R only; no exec)

Usage:
  uv run python3 scripts/tools/check_association_tools.py
  uv run python3 scripts/tools/check_association_tools.py --list
  uv run python3 scripts/tools/check_association_tools.py --print-recipe R-A
"""
# status: diagnostic

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = (
    REPO_ROOT / "docs" / "modules" / "semantic" / "research" / "association_tools.yaml"
)
DEFAULT_NO_GO = REPO_ROOT / "docs" / "reference" / "no_go_registry.md"

# Conservative globs for "looks AssA-related" missing-R warnings (D only).
# Do not use these to invent R entries or doors.
ASSA_GLOBS = (
    "scripts/tools/*relink*.py",
    "scripts/tools/depth_ordering_*.py",
    "scripts/tools/*occ*.py",
    "scripts/tools/gap_occupancy_features.py",
    "scripts/tools/bench_bank_scatter.py",
    "scripts/tools/add_occlusion_to_seq.py",
    "scripts/eval/diagnostics/*.py",
    "scripts/eval/appearance/reid_id_benchmark.py",
    "scripts/eval/appearance/cheb_gr_osnet_gate.py",
    "scripts/eval/experiments/oracle_occlusion_hold.py",
    "scripts/eval/probe_occ*.py",
    "scripts/eval/probe_assoc_appearance_veto.py",
    "scripts/eval/probe_lowiou_occ_gate.py",
    "scripts/eval/run_offline_handover_ablation.py",
    "scripts/eval/run_occ_audit_offline.py",
    "scripts/eval/analyze_occlusion_events.py",
    "scripts/eval/analyze_occ_size.py",
    "scripts/eval/occ_rank.py",
    "scripts/eval/occ_tune.py",
    "scripts/eval/reconnect_rate.py",
    "scripts/eval/reid_id_benchmark.py",
    "scripts/eval/analyze_crossing_swaps.py",
    "scripts/eval/oracle_occlusion_hold.py",
    "scripts/eval/cheb_gr_osnet_gate.py",
    "scripts/eval/mot17.py",
    "scripts/train/reid_domain_probe.py",
)

NO_GO_ID_RE = re.compile(r"^#(\d+)$")
NO_GO_ANCHOR_RE = re.compile(r'<a id="(\d+)"></a>')
WRAPPER_TARGET_RE = re.compile(r"""run_eval_script\(\s*["']([^"']+)["']\s*\)""")
TOOL_ALLOWED_KEYS = {
    "id",
    "path",
    "door",
    "roles",
    "priority",
    "fact_owner",
    "no_go_ids",
    "recipes",
    "expected_artifacts",
    "canonical_id",
    "notes",
}
RECIPE_ALLOWED_KEYS = {
    "id",
    "title",
    "door",
    "purpose",
    "steps",
    "fact_owner",
    "expected_artifacts",
    "notes",
}


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_registry(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"registry root must be a mapping: {_rel(path)}")
    return data


def load_no_go_ids(path: Path) -> set[str]:
    """Return set of ids like '#39' present as anchors in the NO-GO master."""
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8")
    return {f"#{m.group(1)}" for m in NO_GO_ANCHOR_RE.finditer(text)}


def discover_assa_paths() -> set[str]:
    found: set[str] = set()
    for pattern in ASSA_GLOBS:
        for p in REPO_ROOT.glob(pattern):
            if p.is_file() and p.suffix == ".py":
                found.add(_rel(p))
    return found


def parse_wrapper_target(wrapper_path: Path) -> str | None:
    """Best-effort: extract run_eval_script('relative') target under scripts/eval/."""
    try:
        text = wrapper_path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = WRAPPER_TARGET_RE.search(text)
    if not m:
        return None
    rel = m.group(1).lstrip("./")
    return f"scripts/eval/{rel}"


def check_registry(
    data: dict[str, Any],
    *,
    no_go_ids: set[str],
    no_go_path: Path,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    forbidden = set(data.get("forbidden_fields") or [])
    allowed_doors = set((data.get("allowed_doors") or {}).keys())
    allowed_roles = set(data.get("allowed_roles") or [])
    allowed_priorities = set(data.get("allowed_priorities") or [])

    tools = data.get("tools")
    recipes = data.get("recipes")
    if not isinstance(tools, list):
        errors.append("R: tools must be a list")
        tools = []
    if not isinstance(recipes, list):
        errors.append("R: recipes must be a list")
        recipes = []

    tool_by_id: dict[str, dict[str, Any]] = {}
    registered_paths: set[str] = set()

    for i, tool in enumerate(tools):
        prefix = f"tools[{i}]"
        if not isinstance(tool, dict):
            errors.append(f"{prefix}: must be a mapping")
            continue

        for key in tool:
            if key in forbidden:
                errors.append(f"{prefix}: forbidden field '{key}'")
            elif key not in TOOL_ALLOWED_KEYS:
                warnings.append(f"{prefix}: unknown field '{key}' (not in schema)")

        tid = tool.get("id")
        if not tid or not isinstance(tid, str):
            errors.append(f"{prefix}: missing id")
            continue
        if tid in tool_by_id:
            errors.append(f"R: duplicate tool id '{tid}'")
        tool_by_id[tid] = tool

        path = tool.get("path")
        if not path or not isinstance(path, str):
            errors.append(f"tool {tid}: missing path")
        else:
            registered_paths.add(path)
            abs_path = REPO_ROOT / path
            if not abs_path.is_file():
                errors.append(f"path_exists (D): missing {path} (tool {tid})")

        door = tool.get("door")
        if door not in allowed_doors:
            errors.append(f"tool {tid}: door {door!r} not in allowed_doors")

        roles = tool.get("roles")
        if not isinstance(roles, list) or not roles:
            errors.append(f"tool {tid}: roles must be non-empty list")
            roles = []
        for role in roles:
            if role not in allowed_roles:
                errors.append(f"tool {tid}: role {role!r} not allowed")

        pri = tool.get("priority")
        if pri not in allowed_priorities:
            errors.append(f"tool {tid}: priority {pri!r} not allowed")

        fact_owner = tool.get("fact_owner")
        if not fact_owner or not isinstance(fact_owner, str):
            errors.append(f"tool {tid}: missing fact_owner")
        else:
            if not (REPO_ROOT / fact_owner).is_file():
                warnings.append(
                    f"fact_owner_exists (D): missing {fact_owner} (tool {tid})"
                )

        cited = tool.get("no_go_ids") or []
        if cited is None:
            cited = []
        if not isinstance(cited, list):
            errors.append(f"tool {tid}: no_go_ids must be a list")
            cited = []
        for nid in cited:
            if not isinstance(nid, str) or not NO_GO_ID_RE.match(nid):
                errors.append(f"tool {tid}: no_go id {nid!r} must match #<number>")
                continue
            if nid not in no_go_ids:
                errors.append(
                    f"no_go_id_exists (N): {nid} not in {_rel(no_go_path)} (tool {tid})"
                )

        if "wrapper" in roles and not tool.get("canonical_id"):
            errors.append(f"tool {tid}: role wrapper requires canonical_id")

        recs = tool.get("recipes") or []
        if recs is not None and not isinstance(recs, list):
            errors.append(f"tool {tid}: recipes must be a list")

    # Second pass: wrapper canonical + redirect after all ids known
    for tid, tool in tool_by_id.items():
        roles = tool.get("roles") or []
        if "wrapper" not in roles:
            continue
        can = tool.get("canonical_id")
        if can and can not in tool_by_id:
            errors.append(f"tool {tid}: canonical_id {can!r} not found in R tools")
        path = tool.get("path")
        if not path or not isinstance(path, str):
            continue
        wrapper_abs = REPO_ROOT / path
        target = parse_wrapper_target(wrapper_abs)
        if can and can in tool_by_id:
            expected = tool_by_id[can].get("path")
            if target and expected and target != expected:
                warnings.append(
                    f"redirect_matches_R: wrapper {tid} targets "
                    f"{target} but R canonical {can} is {expected}"
                )
        elif target is None and wrapper_abs.is_file():
            warnings.append(f"redirect_matches_R: could not parse redirect in {path}")

    recipe_ids: set[str] = set()
    for i, recipe in enumerate(recipes):
        prefix = f"recipes[{i}]"
        if not isinstance(recipe, dict):
            errors.append(f"{prefix}: must be a mapping")
            continue
        for key in recipe:
            if key in forbidden:
                errors.append(f"{prefix}: forbidden field '{key}'")
            elif key not in RECIPE_ALLOWED_KEYS:
                warnings.append(f"{prefix}: unknown field '{key}'")

        rid = recipe.get("id")
        if not rid or not isinstance(rid, str):
            errors.append(f"{prefix}: missing id")
            continue
        if rid in recipe_ids:
            errors.append(f"R: duplicate recipe id '{rid}'")
        recipe_ids.add(rid)

        if not recipe.get("title"):
            errors.append(f"recipe {rid}: missing title")
        if recipe.get("door") not in allowed_doors:
            errors.append(f"recipe {rid}: door {recipe.get('door')!r} invalid")
        if not recipe.get("purpose"):
            errors.append(f"recipe {rid}: missing purpose")

        fo = recipe.get("fact_owner")
        if not fo:
            errors.append(f"recipe {rid}: missing fact_owner")
        elif not (REPO_ROOT / fo).is_file():
            warnings.append(f"fact_owner_exists (D): missing {fo} (recipe {rid})")

        steps = recipe.get("steps")
        if not isinstance(steps, list) or not steps:
            errors.append(f"recipe {rid}: steps must be non-empty list")
            steps = []
        for step in steps:
            if step not in tool_by_id:
                errors.append(f"recipe {rid}: step {step!r} not a known tool id")

    # Tool recipe refs exist
    for tid, tool in tool_by_id.items():
        for rid in tool.get("recipes") or []:
            if rid not in recipe_ids:
                warnings.append(
                    f"tool {tid}: recipes ref {rid!r} not defined in R recipes"
                )

    # missing_R: AssA-ish paths on D not in R
    on_disk = discover_assa_paths()
    for path in sorted(on_disk - registered_paths):
        warnings.append(f"missing_R_entry (D): AssA-related path not in R: {path}")

    return errors, warnings


def print_list(data: dict[str, Any]) -> None:
    tools = data.get("tools") or []
    by_door: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in tools:
        if isinstance(t, dict):
            by_door[str(t.get("door", "?"))].append(t)

    status = data.get("registry_status", "?")
    print(f"association tools registry ({status}) — {len(tools)} tools")
    print("source_class: R (curated); paths checked separately with default mode")
    for door in sorted(by_door):
        door_meta = (data.get("allowed_doors") or {}).get(door) or {}
        name = door_meta.get("name", "")
        print(f"\n## Door {door} — {name}")
        for t in sorted(
            by_door[door], key=lambda x: (x.get("priority", ""), x.get("id", ""))
        ):
            roles = ",".join(t.get("roles") or [])
            print(
                f"  [{t.get('priority')}] {t.get('id'):40s}  {t.get('path')}  ({roles})"
            )


def print_recipe(data: dict[str, Any], recipe_id: str) -> int:
    recipes = {
        r["id"]: r
        for r in (data.get("recipes") or [])
        if isinstance(r, dict) and "id" in r
    }
    tools = {
        t["id"]: t
        for t in (data.get("tools") or [])
        if isinstance(t, dict) and "id" in t
    }
    recipe = recipes.get(recipe_id)
    if not recipe:
        print(f"unknown recipe id: {recipe_id}", file=sys.stderr)
        known = ", ".join(sorted(recipes)) or "(none)"
        print(f"known: {known}", file=sys.stderr)
        return 2

    print(f"# {recipe['id']} — {recipe.get('title', '')}")
    print(f"# door: {recipe.get('door')}  fact_owner: {recipe.get('fact_owner')}")
    print(f"# purpose: {recipe.get('purpose')}")
    print("# print-only skeleton (R); do not treat as auto-exec GO gate")
    print()
    for i, step in enumerate(recipe.get("steps") or [], start=1):
        tool = tools.get(step, {})
        path = tool.get("path", f"<missing tool {step}>")
        print(f"# step {i}: {step}")
        print(f"uv run python {path}  # see --help / fact_owner for flags")
        print()
    notes = recipe.get("notes")
    if notes:
        print("# notes:")
        for line in str(notes).strip().splitlines():
            print(f"#   {line.strip()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="path to association_tools.yaml (R)",
    )
    parser.add_argument(
        "--no-go",
        type=Path,
        default=DEFAULT_NO_GO,
        help="path to no_go_registry.md (N)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print tools from R grouped by door (no disk invent)",
    )
    parser.add_argument(
        "--print-recipe",
        metavar="ID",
        help="print recipe command skeleton from R (no execution)",
    )
    parser.add_argument(
        "--strict-warn",
        action="store_true",
        help="exit non-zero if any warnings are present",
    )
    args = parser.parse_args(argv)

    reg_path = args.registry
    if not reg_path.is_file():
        print(f"error: registry not found: {_rel(reg_path)}", file=sys.stderr)
        return 2

    try:
        data = load_registry(reg_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: failed to load registry: {exc}", file=sys.stderr)
        return 2

    if args.list:
        print_list(data)
        return 0

    if args.print_recipe:
        return print_recipe(data, args.print_recipe)

    no_go_path = args.no_go
    no_go_ids = load_no_go_ids(no_go_path)
    if not no_go_ids:
        print(
            f"warning: no NO-GO anchors parsed from {_rel(no_go_path)}",
            file=sys.stderr,
        )

    errors, warnings = check_registry(data, no_go_ids=no_go_ids, no_go_path=no_go_path)

    n_tools = len(data.get("tools") or [])
    n_recipes = len(data.get("recipes") or [])
    status = data.get("registry_status", "?")

    if errors:
        print(
            f"✗ association tools check failed "
            f"({len(errors)} error(s), {len(warnings)} warning(s); "
            f"R status={status}, tools={n_tools}, recipes={n_recipes})"
        )
        for e in errors:
            print(f"  ERROR  {e}")
        for w in warnings:
            print(f"  WARN   {w}")
        return 1

    if warnings:
        print(
            f"✓ association tools check OK with warnings "
            f"({len(warnings)} warning(s); "
            f"R status={status}, tools={n_tools}, recipes={n_recipes})"
        )
        for w in warnings:
            print(f"  WARN   {w}")
        return 1 if args.strict_warn else 0

    print(
        f"✓ association tools check OK "
        f"(R status={status}, tools={n_tools}, recipes={n_recipes}; "
        f"paths D, no_go N, schema R)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
