#!/usr/bin/env python3
"""Warn-only documentation structure / research index coverage checks.

Companion to ``check_doc_links.py``, ``check_doc_stale_paths.py``, and
``check_doc_freshness.py``. Implements the machine side of the Doc Structure
Contract (``docs/ownership/doc_structure_contract.md`` § C4 / C9):

  S1  Every ``docs/modules/<m>/research/*.md`` must be referenced (by basename)
      in ``docs/modules/<m>/README.md``.
  S2  Every note under ``docs/research/{pipeline,eval,training,reid,threads}/*.md``
      (except README.md) must be referenced by basename in the subdir README
      if it exists, else in ``docs/research/README.md``.
  S3  Every ``docs/modules/<m>/`` directory must contain README.md and TODO.md.

This checker is **warn-only** by default for index-coverage findings (exit 0
even with findings). ``--strict`` exits non-zero on C6.4 lifecycle violations
(L1–L4); ``scripts/pre_push.sh`` uses that mode. Index coverage remains
warn-only in either mode.

Index detection currently uses basename substring match against the owning
README body. It remains warn-only hygiene; Markdown link parsing would reduce
false positives (basename mentioned only in prose or stale paths).

Usage: uv run python3 scripts/tools/check_doc_structure.py [--strict]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MODULES_ROOT = REPO_ROOT / "docs" / "modules"
RESEARCH_ROOT = REPO_ROOT / "docs" / "research"
RESEARCH_SUBDIRS = ("pipeline", "eval", "training", "reid", "threads")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _notes_in(dir_path: Path) -> list[Path]:
    if not dir_path.is_dir():
        return []
    return sorted(
        p for p in dir_path.glob("*.md") if p.name != "README.md" and p.is_file()
    )


def check_module_packages() -> list[str]:
    warnings: list[str] = []
    if not MODULES_ROOT.is_dir():
        return [f"[S3] missing modules root: {MODULES_ROOT.relative_to(REPO_ROOT)}"]

    for mod_dir in sorted(p for p in MODULES_ROOT.iterdir() if p.is_dir()):
        rel = mod_dir.relative_to(REPO_ROOT).as_posix()
        for required in ("README.md", "TODO.md"):
            if not (mod_dir / required).is_file():
                warnings.append(f"[S3] {rel}: missing {required}")
    return warnings


def check_module_research_indexes() -> list[str]:
    warnings: list[str] = []
    if not MODULES_ROOT.is_dir():
        return warnings

    for mod_dir in sorted(p for p in MODULES_ROOT.iterdir() if p.is_dir()):
        research = mod_dir / "research"
        if not research.is_dir():
            continue
        readme = mod_dir / "README.md"
        body = _read(readme)
        mod_rel = mod_dir.relative_to(REPO_ROOT).as_posix()
        if not body:
            warnings.append(
                f"[S1] {mod_rel}/research: parent README.md missing or empty; "
                "cannot verify index coverage"
            )
            continue
        for note in _notes_in(research):
            if note.name not in body:
                note_rel = note.relative_to(REPO_ROOT).as_posix()
                warnings.append(
                    f"[S1] {note_rel}: not referenced in {mod_rel}/README.md "
                    "(Doc Structure C4 — add index row)"
                )
    return warnings


def check_global_research_indexes() -> list[str]:
    warnings: list[str] = []
    top_readme = RESEARCH_ROOT / "README.md"
    top_body = _read(top_readme)

    for sub in RESEARCH_SUBDIRS:
        sub_dir = RESEARCH_ROOT / sub
        if not sub_dir.is_dir():
            continue
        sub_readme = sub_dir / "README.md"
        if sub_readme.is_file():
            index_body = _read(sub_readme)
            index_label = sub_readme.relative_to(REPO_ROOT).as_posix()
        else:
            index_body = top_body
            index_label = top_readme.relative_to(REPO_ROOT).as_posix()

        if not index_body:
            warnings.append(
                f"[S2] docs/research/{sub}: index file missing/empty ({index_label})"
            )
            continue

        for note in _notes_in(sub_dir):
            if note.name not in index_body:
                note_rel = note.relative_to(REPO_ROOT).as_posix()
                warnings.append(
                    f"[S2] {note_rel}: not referenced in {index_label} "
                    "(Doc Structure C4 — add index row)"
                )
    return warnings


# --- C6.4 lifecycle enforcement (fail-closed under --strict) -----------------
#
# A note whose own header declares ``doc-status: closed`` has finished its unit;
# leaving it in an active path and an active index is what makes docs grow
# monotonically. C6.3 makes closing part of the accepting PR, and these rules
# make it a merge condition rather than a matter of discipline.
#
# Legacy notes predating C6.3 are exempted below: backfill is cleanup, and
# cleanup must not block the mainline. New violations are not exempt.

LEGACY_CLOSED_IN_ACTIVE = frozenset(
    {
        "docs/modules/semantic/research/m_b1_stage1_online_hook_final_20260710.md",
        "docs/modules/semantic/research/m_b1_5_stage2_d_online_final_20260710.md",
        "docs/modules/semantic/research/m_b1_repaired_eps0_loo_pass_candidate_20260709.md",
        "docs/modules/semantic/research/m_b1_portable_or_tail_hook_contract_20260709.md",
        "docs/modules/semantic/research/composition_grammar_t0_region_interpretation_20260710.md",
        "docs/modules/semantic/research/m_b1_research_history_20260709_20260710.md",
        "docs/modules/semantic/research/composition_grammar_safe_region_coverage_audit_20260710.md",
    }
)

# docs/research/contracts/ is the decision/rule layer (C0.1): rules + one state
# registry, never prose. Anything else added there is a violation.
CONTRACTS_ALLOWED = frozenset(
    {
        "README.md",
        "claim_state_registry.md",
        "statistical_robust_feasible_set_estimation_under_asymmetric_loss.md",
        "runtime_quantity_fidelity_protocol.md",
        "signal_table_schema.md",
        "boolean_composition_semantics_contract.md",
        "safe_region_asset_contract.md",
    }
)

HEADER_LINES = 12  # a doc's own status marker lives in its header, not its prose


def _declared_status(path: Path) -> str:
    """Status this doc declares for *itself* (header only).

    Scanning the whole body would misread contracts that merely document the
    vocabulary (``doc-status: closed`` appears in C6's own table).
    """
    for line in _read(path).splitlines()[:HEADER_LINES]:
        if "doc-status:" in line:
            value = line.split("doc-status:", 1)[1]
            for token in ("proposed", "active", "parked", "closed", "archived"):
                if token in value:
                    return token
    return ""


def _active_section_body(readme: Path) -> str:
    """Body of the README's Active block(s) only."""
    body = _read(readme)
    if not body:
        return ""
    chunks: list[str] = []
    current: list[str] | None = None
    for line in body.splitlines():
        if line.startswith("#"):
            heading = line.lower()
            if current is not None:
                chunks.append("\n".join(current))
                current = None
            if "active" in heading and "inactive" not in heading:
                current = []
        elif current is not None:
            current.append(line)
    if current is not None:
        chunks.append("\n".join(current))
    return "\n".join(chunks)


def _check_thread_status_projection() -> list[str]:
    """L4 — the threads index projects thread state; it must not restate it wrongly.

    A projection maintained by hand drifts. This is the one status surface worth
    a checker, because agreement between the card's ``wip-role`` and the index
    row is a mechanical equality, not a judgement.
    """
    violations: list[str] = []
    threads_dir = RESEARCH_ROOT / "threads"
    index = threads_dir / "README.md"
    if not index.is_file():
        return violations

    # Parse index into sections and rows
    current_section = ""
    # Map card name (e.g. "foo.md") -> list of (section_name, row_text)
    card_occurrences: dict[str, list[tuple[str, str]]] = {}
    import re

    for line in _read(index).splitlines():
        if line.startswith("##"):
            heading = line.lower()
            if "active" in heading:
                current_section = "active"
            elif "parked" in heading:
                current_section = "parked"
            elif "proposed" in heading:
                current_section = "proposed"
            elif "closed" in heading:
                current_section = "closed"
            else:
                current_section = ""
        elif current_section and line.strip().startswith("|"):
            if ":--" in line or "---" in line:
                continue
            # Extract link e.g. [foo.md](path/to/foo.md) or similar
            match = re.search(r"\[([^\]]+\.md)\]", line)
            if match:
                card_name = match.group(1)
                card_occurrences.setdefault(card_name, []).append(
                    (current_section, line)
                )

    all_cards = list(threads_dir.glob("*.md")) + list(
        (threads_dir / "closed").glob("*.md")
    )
    for card in sorted(all_cards):
        if card.name == "README.md":
            continue
        status = ""
        role = ""
        for line in _read(card).splitlines()[:HEADER_LINES]:
            if line.startswith("doc-status:"):
                status = line.split(":", 1)[1].strip().lower()
            elif line.startswith("wip-role:"):
                role = line.split(":", 1)[1].strip().lower()

        if not status:
            continue

        # Card-level status and wip-role consistency
        if status == "parked" and role != "parked":
            rel = card.relative_to(REPO_ROOT).as_posix()
            violations.append(
                f"[L4] {rel}: card doc-status is 'parked' but wip-role is '{role}' (must be 'parked')"
            )
        elif role == "parked" and status != "parked":
            rel = card.relative_to(REPO_ROOT).as_posix()
            violations.append(
                f"[L4] {rel}: card wip-role is 'parked' but doc-status is '{status}' (must be 'parked')"
            )
        elif status == "proposed" and role != "non-wip":
            rel = card.relative_to(REPO_ROOT).as_posix()
            violations.append(
                f"[L4] {rel}: card doc-status is 'proposed' but wip-role is '{role}' (must be 'non-wip')"
            )

        occurrences = card_occurrences.get(card.name, [])
        if len(occurrences) > 1:
            rel = card.relative_to(REPO_ROOT).as_posix()
            sections_str = ", ".join(f"'{sec}'" for sec, _ in occurrences)
            violations.append(
                f"[L4] {rel}: card is projected multiple times in the index README (found in sections: {sections_str}; must appear exactly once)"
            )

        for indexed_section, row in occurrences:
            # 1. Section validation
            if indexed_section != status:
                rel = card.relative_to(REPO_ROOT).as_posix()
                violations.append(
                    f"[L4] {rel}: card declares doc-status '{status}' but is listed in the "
                    f"'{indexed_section.capitalize()}' section of the threads index"
                )

            # 2. WIP role validation (only for non-closed)
            if status in ("active", "parked", "proposed") and role:
                parts = [p.strip() for p in row.split("|")]
                if len(parts) >= 3:
                    role_cell = parts[2]
                    expected_role_token = f"**{role}**"
                    if expected_role_token.lower() not in role_cell.lower():
                        rel = card.relative_to(REPO_ROOT).as_posix()
                        violations.append(
                            f"[L4] {rel}: card declares wip-role '{role}' but the threads index "
                            f"row WIP role cell '{role_cell}' does not match (expected '{expected_role_token}')"
                        )
    return violations


def check_lifecycle() -> list[str]:
    """C6.4 L1–L4 — violations, not warnings."""
    violations: list[str] = []

    for note in sorted(REPO_ROOT.joinpath("docs").rglob("*.md")):
        rel = note.relative_to(REPO_ROOT).as_posix()
        parts = note.relative_to(REPO_ROOT).parts
        if "closed" in parts or "archive" in parts:
            continue
        if _declared_status(note) != "closed":
            continue
        if rel in LEGACY_CLOSED_IN_ACTIVE:
            continue

        violations.append(
            f"[L1] {rel}: doc-status is closed but the note still sits in an "
            "active path (Doc Structure C6.1/C6.3 — git mv it into closed/ in "
            "the same PR that accepted the terminal)"
        )

        if "modules" in parts:
            owner_readme = note.parent.parent / "README.md"
        elif "research" in parts:
            subdir_readme = note.parent / "README.md"
            if subdir_readme.is_file():
                owner_readme = subdir_readme
            else:
                owner_readme = note.parent.parent / "README.md"
        else:
            owner_readme = note.parent / "README.md"

        if owner_readme.is_file() and note.name in _active_section_body(owner_readme):
            owner_rel = owner_readme.relative_to(REPO_ROOT).as_posix()
            violations.append(
                f"[L2] {rel}: closed note still indexed in the Active section of "
                f"{owner_rel} (Doc Structure C6.3 — leave one pointer row, not an "
                "active row)"
            )

    violations.extend(_check_thread_status_projection())

    contracts_dir = RESEARCH_ROOT / "contracts"
    if contracts_dir.is_dir():
        for doc in sorted(contracts_dir.glob("*.md")):
            if doc.name not in CONTRACTS_ALLOWED:
                rel = doc.relative_to(REPO_ROOT).as_posix()
                violations.append(
                    f"[L3] {rel}: the decision/rule layer carries rules and state, "
                    "not prose (Doc Structure C0.1 — put explanation in a research "
                    "note, navigation in a thread)"
                )

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on C6.4 lifecycle violations (index coverage stays warn-only)",
    )
    args = parser.parse_args()

    warnings: list[str] = []
    warnings.extend(check_module_packages())
    warnings.extend(check_module_research_indexes())
    warnings.extend(check_global_research_indexes())
    violations = check_lifecycle()

    if warnings:
        print(f"doc structure: {len(warnings)} warning(s)")
        for w in warnings:
            print(f"  {w}")
    else:
        print("doc structure: ok (no index coverage warnings)")

    if violations:
        print(f"doc structure: {len(violations)} lifecycle violation(s) [C6.4]")
        for v in violations:
            print(f"  {v}")
        if args.strict:
            return 1
    else:
        print("doc structure: ok (no lifecycle violations)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
