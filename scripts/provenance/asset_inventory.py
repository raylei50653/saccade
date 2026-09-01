"""Workspace-local projection of which artifact directories are accounted for.

ADR 021 (AP-3).  Answers one question about the four artifact roots — for each
run directory, is anything able to account for it? — and answers it from the
workspace as it is right now.

**This is a projection, not a committed authority.**  ``runs/``, ``results/``,
``out/`` and ``output/`` are all gitignored, so a clean CI clone holds none of
them.  Committing the rendered view and diffing it for freshness — the pattern
``scripts_inventory`` and ``tests_inventory`` use — would therefore compare a
82 GB workspace against an empty one and call the difference drift.  The
generated view would become a second truth that cannot be reproduced anywhere
except the machine that wrote it.

So the split is:

* ``--emit`` renders the view for a human, to a **gitignored** path outside
  ``docs/``.  Emitting into ``docs/`` is refused outright: it is a committed
  surface that ``build_master_map`` scans with ``rglob``, so an untracked
  document there would break the checked-in master map on the very machine
  that generated it, and leaving that as a convention rather than a check is
  how it would happen anyway.
* ``--check`` is what CI runs.  It validates rather than compares: an empty
  inventory on a clean clone is a correct answer, and the only fail-closed
  condition is a manifest that exists but is not valid.  ``pre_push`` does
  **not** run it: ``scripts/pre_push.sh`` is a protected path on the
  ``identity_semantics`` axis (ADR 021 §4.3), so the workspace that actually
  holds the artifacts is the one with no automatic hook, and the check must be
  run by hand there until a lawful re-attestation carries it.

An invalid manifest is never quietly downgraded to ``orphan``.  Orphan means
"nothing accounts for this"; a corrupt manifest means "something tried to and
we cannot read it", and folding the second into the first would hide a broken
producer inside a routine backlog number.

The views carry no age, no deletion eligibility, and no recommendation.
Disposal is AP-5, behind owner approval.
"""

# status: stable

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve()
if str(_HERE.parents[2]) not in sys.path:
    sys.path.insert(0, str(_HERE.parents[2]))

from scripts.provenance.run_manifest import (  # noqa: E402
    MANIFEST_FILENAME,
    ManifestError,
    read_manifest,
)

# The four roots ADR 021 §1.1 measured.  All gitignored.
ASSET_ROOTS = ("runs", "results", "out", "output")

DEFAULT_OUTPUT = Path(".provenance/asset_inventory.generated.md")

# Generated views must never feed the citation corpus.  Without this the tool
# is self-fulfilling: render every unit name into a document once, and the next
# scan reports the entire workspace as cited.
GENERATED_SUFFIX = ".generated.md"


class InventoryError(RuntimeError):
    """The workspace cannot be inventoried, or holds a manifest that is not valid."""


@dataclass(frozen=True)
class Unit:
    """One inventory unit: an immediate child directory of an asset root.

    Deliberately not recursive.  ``results/<run>/_per_seq/<seq>`` and a training
    run's checkpoint directories belong to their run; treating them as units
    would invent thousands of orphans out of one accounted-for run.
    """

    path: str
    root: str
    manifest_state: str  # "valid" | "invalid" | "absent"
    cited: bool
    detail: str = ""

    @property
    def manifested(self) -> bool:
        return self.manifest_state == "valid"

    @property
    def orphan(self) -> bool:
        """Not cited, and carrying no manifest at all.  A query view, never a verdict.

        Note the third state.  ``not manifested`` is *not* the same as "no
        manifest": an invalid manifest is also not manifested, and defining
        orphan as its complement would sweep every broken producer into the
        backlog count — the exact silent downgrade this module refuses. Invalid
        is its own state, and it fails closed instead of being classified.
        """
        return not self.cited and self.manifest_state == "absent"


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def citation_corpus(repo_root: Path) -> list[Path]:
    """Tracked documents under ``docs/``, excluding generated views.

    Tracked-only is the load-bearing half: this tool's own projection is
    untracked, so it can never cite anything, no matter where it is written.
    """
    listing = _git(repo_root, "ls-files", "--", "docs")
    if listing is None:
        raise InventoryError(
            f"cannot list tracked documents under {repo_root}/docs; "
            "the citation corpus is defined as tracked files and cannot be guessed"
        )
    corpus = []
    for line in listing.splitlines():
        if not line.strip():
            continue
        if line.endswith(GENERATED_SUFFIX):
            continue
        path = repo_root / line
        if path.is_file():
            corpus.append(path)
    return corpus


def cited_names(repo_root: Path, names: set[str]) -> set[str]:
    """Which unit names appear literally in the corpus.

    Literal substring matching, the same method ADR 021 §1.1 measured with, and
    with the same stated limit: it under-counts semantic references. A unit
    that no document names cannot be reached by name from any document — that
    is all this establishes.
    """
    if not names:
        return set()
    found: set[str] = set()
    for path in citation_corpus(repo_root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name in names - found:
            if name in text:
                found.add(name)
        if found == names:
            break
    return found


def _manifest_state(directory: Path) -> tuple[str, str]:
    if not (directory / MANIFEST_FILENAME).exists():
        return "absent", ""
    try:
        read_manifest(directory)
    except ManifestError as exc:
        return "invalid", str(exc)
    return "valid", ""


def scan(repo_root: str | os.PathLike[str]) -> tuple[Unit, ...]:
    """Inventory every unit under the asset roots that exist in this workspace."""
    root = Path(repo_root).resolve()
    discovered: list[tuple[str, Path]] = []
    for asset_root in ASSET_ROOTS:
        base = root / asset_root
        if not base.is_dir():
            continue  # a clean clone has none of these; that is not an error
        for child in sorted(base.iterdir()):
            if child.is_dir():
                discovered.append((asset_root, child))

    names = {path.name for _, path in discovered}
    cited = cited_names(root, names)

    units = []
    for asset_root, path in discovered:
        state, detail = _manifest_state(path)
        units.append(
            Unit(
                path=path.relative_to(root).as_posix(),
                root=asset_root,
                manifest_state=state,
                cited=path.name in cited,
                detail=detail,
            )
        )
    return tuple(units)


def invalid_manifests(units: tuple[Unit, ...]) -> tuple[Unit, ...]:
    return tuple(unit for unit in units if unit.manifest_state == "invalid")


def loose_entries(repo_root: str | os.PathLike[str]) -> tuple[str, ...]:
    """Files sitting directly under an asset root.

    Not units — a file cannot carry a manifest — but counted so that they are
    visibly out of scope rather than silently dropped.
    """
    root = Path(repo_root).resolve()
    loose = []
    for asset_root in ASSET_ROOTS:
        base = root / asset_root
        if not base.is_dir():
            continue
        loose.extend(
            (base / item.name).relative_to(root).as_posix()
            for item in sorted(base.iterdir())
            if item.is_file()
        )
    return tuple(loose)


def render(units: tuple[Unit, ...], loose: tuple[str, ...], *, repo_root: Path) -> str:
    cited = [unit for unit in units if unit.cited]
    manifested = [unit for unit in units if unit.manifested]
    orphan = [unit for unit in units if unit.orphan]

    lines = [
        "<!-- Generated by scripts/provenance/asset_inventory.py; "
        "workspace-local projection, not committed, not an authority. -->",
        "# Asset inventory (workspace projection)",
        "",
        f"Workspace: `{repo_root}`",
        "",
        "The asset roots are gitignored, so this view describes **this machine "
        "only** and is reproducible only against this workspace. The accounting "
        "authorities are the manifests themselves and the tracked documents that "
        "cite them; this file is a reading of them, not a record.",
        "",
        "Views are queries and may overlap. `orphan` is exactly "
        "`not cited and no manifest at all` — a manifest that exists but does "
        "not validate is a separate, fail-closed state, never an orphan. "
        "Orphan carries no age, no deletion "
        "eligibility, and no recommendation — disposal is AP-5, behind owner "
        "approval.",
        "",
        "`cited` is a literal substring match, so a short unit name can match "
        "text that was not about it. The error runs toward calling a unit "
        "accounted for, which shortens the orphan list rather than lengthening "
        "it — the safe direction for anything feeding an approval queue, and a "
        "reason `orphan` still needs a human before it means anything.",
        "",
        "## Summary",
        "",
        "| View | Units |",
        "|:--|--:|",
        f"| total units | {len(units)} |",
        f"| cited | {len(cited)} |",
        f"| manifested | {len(manifested)} |",
        f"| orphan | {len(orphan)} |",
        f"| loose files (not units) | {len(loose)} |",
        "",
    ]

    broken = invalid_manifests(units)
    if broken:
        lines += [
            "## Invalid manifests (fail-closed)",
            "",
            "A manifest that exists but does not validate is a broken producer, "
            "not a backlog item, and is never counted as `orphan`.",
            "",
        ]
        lines += [f"- `{unit.path}` — {unit.detail}" for unit in broken]
        lines.append("")

    lines += [
        "## Units",
        "",
        "| Unit | Root | Manifest | Cited | View |",
        "|:--|:--|:--|:--|:--|",
    ]
    for unit in units:
        view = (
            "orphan"
            if unit.orphan
            else ", ".join(
                filter(
                    None,
                    [
                        "cited" if unit.cited else "",
                        "manifested" if unit.manifested else "",
                    ],
                )
            )
        )
        lines.append(
            f"| `{unit.path}` | {unit.root} | {unit.manifest_state} | "
            f"{'yes' if unit.cited else 'no'} | {view} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Project which artifact directories are accounted for. "
            "--check validates the workspace; --emit renders the view."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail closed on manifests that exist but are not valid. An empty "
        "inventory (clean clone, no local assets) is a correct answer.",
    )
    parser.add_argument(
        "--emit",
        type=Path,
        nargs="?",
        const=DEFAULT_OUTPUT,
        help=f"Render the view to a gitignored path (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args(argv)

    root = args.repo_root.resolve()
    try:
        units = scan(root)
    except InventoryError as exc:
        print(f"asset inventory: {exc}", file=sys.stderr)
        return 2

    if args.emit is not None:
        target = args.emit if args.emit.is_absolute() else root / args.emit
        target = target.resolve()
        docs_root = (root / "docs").resolve()
        if target == docs_root or docs_root in target.parents:
            print(
                f"asset inventory: refusing to emit into {docs_root} — build_master_map "
                "collects documents with rglob rather than git, so an untracked view "
                "there fails the checked-in master map on this machine while CI stays "
                f"green (ADR 021 §3 AP-3). Emit outside docs/, e.g. {DEFAULT_OUTPUT}.",
                file=sys.stderr,
            )
            return 2
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            render(units, loose_entries(root), repo_root=root), encoding="utf-8"
        )
        print(f"asset inventory: wrote {target}")

    broken = invalid_manifests(units)
    if broken:
        print(
            f"asset inventory: {len(broken)} invalid manifest(s) — a manifest that "
            "exists but does not validate is a broken producer, not an orphan:",
            file=sys.stderr,
        )
        for unit in broken:
            print(f"  {unit.path}: {unit.detail}", file=sys.stderr)
        return 1

    if args.check:
        orphan = sum(1 for unit in units if unit.orphan)
        print(
            f"asset inventory: ok ({len(units)} unit(s), {orphan} orphan, "
            "0 invalid manifests)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
