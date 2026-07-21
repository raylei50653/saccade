#!/usr/bin/env python3
# mypy: ignore-errors
"""
Validate that pipeline_contribution.py profiles correctly map to pipeline stages
described in docs/DATAFLOW.md and docs/pipeline_flow.md.

This script performs three levels of validation:
  1. Structure: profiles are cumulative (each adds one new module)
  2. CLI flags: each profile's flags correctly enable/disable modules
  3. Documentation: profiles match stage descriptions in DATAFLOW.md

Usage:
    uv run python scripts/eval/validate_profiles.py
    uv run python scripts/eval/validate_profiles.py --check-docs
    uv run python scripts/eval/validate_profiles.py --check-flags
    uv run python scripts/eval/validate_profiles.py --check-stages
    uv run python scripts/eval/validate_profiles.py --dry-run
    uv run python scripts/eval/validate_profiles.py --all
"""
# status: diagnostic

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent


@dataclass
class ProfileInfo:
    name: str
    description: str
    args: tuple[str, ...]
    enabled_modules: list[str] = field(default_factory=list)
    disabled_modules: list[str] = field(default_factory=list)


def parse_profiles_from_source() -> list[ProfileInfo]:
    """Parse build_profiles() from pipeline_contribution.py to extract profile definitions."""
    source_file = project_root / "scripts" / "eval" / "pipeline_contribution.py"
    source_code = source_file.read_text()
    tree = ast.parse(source_code)

    profiles: list[ProfileInfo] = []

    # Find all Profile(...) calls anywhere in the tree
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name == "Profile":
                args = node.args
                if len(args) >= 3:
                    name_node, desc_node, args_node = args[0], args[1], args[2]
                    name = (
                        ast.literal_eval(name_node)
                        if isinstance(name_node, ast.Constant)
                        else ""
                    )
                    desc = (
                        ast.literal_eval(desc_node)
                        if isinstance(desc_node, ast.Constant)
                        else ""
                    )

                    # Extract string args from the tuple
                    arg_list = []
                    if isinstance(args_node, ast.Tuple):
                        for elt in args_node.elts:
                            if isinstance(elt, ast.Constant):
                                arg_list.append(elt.value)
                            elif isinstance(elt, ast.JoinedStr):
                                # Handle f-string like f"{semantic_threshold_full:.2f}"
                                arg_list.append("f-string")

                    profile = ProfileInfo(
                        name=name,
                        description=desc,
                        args=tuple(arg_list),
                    )
                    profiles.append(profile)

    return profiles


def analyze_profile_flags(profile: ProfileInfo) -> dict[str, list[str]]:
    """Analyze which modules are enabled/disabled based on CLI flags."""
    flags = profile.args
    enabled = []
    disabled = []

    for flag in flags:
        if flag == "--gmc":
            enabled.append("gmc")
        elif flag == "--no-gmc":
            disabled.append("gmc")
        elif flag == "--reid-mode":
            idx = list(flags).index(flag)
            if idx + 1 < len(flags):
                val = flags[idx + 1]
                if val == "semantic":
                    enabled.append("semantic_relink")
                elif val == "off":
                    disabled.append("semantic_relink")
        elif flag == "--appearance-bank":
            enabled.append("appearance_bank")
        elif flag == "--no-appearance-bank":
            disabled.append("appearance_bank")
        elif flag == "--semantic-bank-inject":
            enabled.append("semantic_bank_inject")
        elif flag == "--no-semantic-bank-inject":
            disabled.append("semantic_bank_inject")
        elif flag == "--async-reid":
            enabled.append("async_reid")
        elif flag == "--pipeline-relink":
            enabled.append("pipeline_relink")
        elif flag == "--lifecycle-merge":
            enabled.append("lifecycle_merge")
        elif flag == "--no-lifecycle-merge":
            disabled.append("lifecycle_merge")
        elif flag == "--post-lifecycle-merge":
            enabled.append("post_lifecycle_merge")
        elif flag == "--no-post-lifecycle-merge":
            disabled.append("post_lifecycle_merge")
        elif flag == "--pose-engine":
            enabled.append("pose_engine")

    profile.enabled_modules = enabled
    profile.disabled_modules = disabled
    return {"enabled": enabled, "disabled": disabled}


def validate_cumulative_property(profiles: list[ProfileInfo]) -> list[str]:
    """Validate that each profile adds exactly one new module on top of the previous."""
    errors = []
    expected_additions: dict[str, list[str]] = {
        "tracker_core": [],  # baseline
        "tracker_core_gmc": ["gmc"],
        "semantic_core": ["semantic_relink"],
        "semantic_bank": ["appearance_bank", "semantic_bank_inject"],
        "full_default": ["async_reid", "pipeline_relink"],
        "pose_sidecar": ["pose_engine"],
    }

    for i, profile in enumerate(profiles):
        expected = expected_additions.get(profile.name, [])
        actual_enabled = set(profile.enabled_modules)

        if profile.name == "tracker_core":
            expected_off = {
                "semantic_relink",
                "appearance_bank",
                "semantic_bank_inject",
                "async_reid",
                "pipeline_relink",
                "lifecycle_merge",
            }
            unexpected = actual_enabled & expected_off
            if unexpected:
                errors.append(
                    f"  {profile.name}: unexpected enabled modules: {unexpected}"
                )
        elif i > 0:
            prev_profile = profiles[i - 1]
            prev_enabled = set(prev_profile.enabled_modules)
            new_modules = actual_enabled - prev_enabled
            lost_modules = prev_enabled - actual_enabled

            if lost_modules:
                errors.append(
                    f"  {profile.name}: lost modules from previous: {lost_modules}"
                )
            if new_modules != set(expected):
                errors.append(
                    f"  {profile.name}: expected to add {expected}, but added {new_modules}"
                )

    return errors


def validate_flag_consistency(profiles: list[ProfileInfo]) -> list[str]:
    """Validate that flags are consistent with documented module requirements."""
    errors = []

    for profile in profiles:
        flags = set(profile.args)

        if "--appearance-bank" in flags and "--no-appearance-bank" in flags:
            errors.append(
                f"  {profile.name}: contradictory --appearance-bank / --no-appearance-bank"
            )
        if "--semantic-bank-inject" in flags and "--no-semantic-bank-inject" in flags:
            errors.append(
                f"  {profile.name}: contradictory --semantic-bank-inject / --no-semantic-bank-inject"
            )

    return errors


def _parse_stage_range(stage_str: str) -> tuple[int, int] | None:
    """Parse [N-M] or [N] into (start, end) or None if invalid."""
    match = re.match(r"^\[(\d+)(?:-(\d+))?\]$", stage_str.strip())
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else start
    return (start, end)


def _stage_in_range(stage: str, range_str: str) -> bool:
    """Check if stage [N] or [N-M] is covered by range_str."""
    s = _parse_stage_range(stage)
    r = _parse_stage_range(range_str)
    if s is None or r is None:
        return False
    return s[0] >= r[0] and s[1] <= r[1]


def validate_stage_descriptions(profiles: list[ProfileInfo]) -> list[str]:
    """Validate that each profile's description mentions the correct stages.

    This function checks that:
    1. All expected stage references are present in the description
    2. No unexpected stage references are present (except subset ranges)
    """
    errors = []
    expected_stage_refs: dict[str, set[str]] = {
        "tracker_core": {"[1-4]", "[10]", "[11-12]"},
        "tracker_core_gmc": {"[1-4]", "[10]", "[11-12]"},
        # semantic_core mentions [5] because it's SKIPPED (bank not enabled)
        # and [6], [7], [8] as individual stages within the [6-8] range
        "semantic_core": {"[1-4]", "[10]", "[6-8]", "[14]", "[5]"},
        "semantic_bank": {"[1-4]", "[10]", "[5-8]", "[14]"},
        "full_default": {"[1-4]", "[10]", "[5-8]", "[13]", "[14]"},
        "pose_sidecar": {"[1-4]", "[10]", "[5-8]", "[13]", "[14]"},
    }

    for profile in profiles:
        expected = expected_stage_refs.get(profile.name, set())
        desc = profile.description

        # Extract all stage refs from description
        found_stages = re.findall(r"\[(?:\d+-\d+|\d+)\]", desc)
        found_set = set(found_stages)

        # Check required refs are present
        for stage_ref in expected:
            if stage_ref not in found_set:
                # Also check if covered by a range
                is_covered = any(
                    _stage_in_range(stage_ref, found) for found in found_stages
                )
                if not is_covered:
                    errors.append(
                        f"  {profile.name}: description missing stage reference {stage_ref}"
                    )

        # Check for unexpected stage references
        unexpected = []
        for found in found_set:
            if found not in expected:
                # Check if it's a subset of any expected range
                is_subset = False
                for exp in expected:
                    if _stage_in_range(found, exp):
                        is_subset = True
                        break
                if not is_subset:
                    unexpected.append(found)
        if unexpected:
            errors.append(f"  {profile.name}: unexpected stage refs: {unexpected}")

    return errors


def check_dataflow_documentation() -> list[str]:
    """Check that DATAFLOW.md matches the expected profile structure."""
    errors = []
    dataflow_path = project_root / "docs" / "DATAFLOW.md"

    if not dataflow_path.exists():
        errors.append("DATAFLOW.md not found")
        return errors

    content = dataflow_path.read_text()

    required_profiles = [
        "tracker_core",
        "tracker_core_gmc",
        "semantic_core",
        "semantic_bank",
        "full_default",
    ]
    for profile in required_profiles:
        if profile not in content:
            errors.append(f"DATAFLOW.md missing profile: {profile}")

    if "[5]" not in content or "[10]" not in content:
        errors.append("DATAFLOW.md missing stage references")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate pipeline_contribution.py profiles against DATAFLOW.md"
    )
    parser.add_argument(
        "--check-docs", action="store_true", help="Check DATAFLOW.md consistency"
    )
    parser.add_argument(
        "--check-flags", action="store_true", help="Check flag consistency"
    )
    parser.add_argument(
        "--check-stages",
        action="store_true",
        help="Check stage description consistency",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only parse, don't validate"
    )
    parser.add_argument("--all", action="store_true", help="Run all checks")
    args = parser.parse_args()

    run_docs = args.check_docs or args.all
    run_flags = args.check_flags or args.all
    run_stages = args.check_stages or args.all

    print("=" * 70)
    print("Pipeline Profile Validation")
    print("=" * 70)

    # Parse profiles
    profiles = parse_profiles_from_source()
    print(f"\nParsed {len(profiles)} profiles from pipeline_contribution.py:\n")

    for profile in profiles:
        flag_analysis = analyze_profile_flags(profile)
        print(f"  {profile.name:20s}")
        print(f"    Enabled: {flag_analysis['enabled']}")
        print(f"    Disabled: {flag_analysis['disabled']}")
        print()

    # Validation checks
    all_errors: list[str] = []

    if not args.dry_run:
        # Check cumulative property
        print("Checking cumulative property...")
        cum_errors = validate_cumulative_property(profiles)
        if cum_errors:
            print("  ✗ Cumulative property violated:")
            all_errors.extend(cum_errors)
            for e in cum_errors:
                print(e)
        else:
            print("  ✓ Cumulative property OK\n")

        # Check flag consistency
        if run_flags:
            print("Checking flag consistency...")
            flag_errors = validate_flag_consistency(profiles)
            if flag_errors:
                print("  ✗ Flag inconsistencies:")
                all_errors.extend(flag_errors)
                for e in flag_errors:
                    print(e)
            else:
                print("  ✓ Flag consistency OK\n")

        # Check documentation
        if run_docs:
            print("Checking DATAFLOW.md...")
            doc_errors = check_dataflow_documentation()
            if doc_errors:
                print("  ✗ Documentation mismatches:")
                all_errors.extend(doc_errors)
                for e in doc_errors:
                    print(e)
            else:
                print("  ✓ DATAFLOW.md references OK\n")

        # Check stage descriptions
        if run_stages:
            print("Checking stage descriptions...")
            stage_errors = validate_stage_descriptions(profiles)
            if stage_errors:
                print("  ✗ Stage description mismatches:")
                all_errors.extend(stage_errors)
                for e in stage_errors:
                    print(e)
            else:
                print("  ✓ Stage descriptions OK\n")

    # Summary
    if all_errors:
        print("=" * 70)
        print(f"VALIDATION FAILED: {len(all_errors)} error(s)")
        print("=" * 70)
        return 1
    else:
        print("=" * 70)
        print("ALL CHECKS PASSED")
        print("=" * 70)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
