"""Compare repeated MOT eval outputs for silent run-to-run divergence.

Issue #363: under a fixed configuration, ``scripts/eval/mot17.py`` can exit 0
and still write different MOT files.  This module is the fail-closed detector
of that symptom.  Pass/fail is raw MOT identity, including track IDs.

Lives under ``scripts/tools/``, not ``src/``: it is a check, not production
eval, and must not move the published implementation identity axis.

It does not attribute a CUDA-level mechanism.  ``classify_first_diff`` only
says whether the first differing line is already a box/score change
(``geometry_or_score``) or an ID-only change (``identity_only``).
"""
# status: stable

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Sequence

from saccade.perception.eval.decimal_hash import canonicalize_mot_lines, decimal_hash

MOT_GLOB = "MOT17-*.txt"
MISSING = "MISSING"
EMPTY = "EMPTY"

KIND_IDENTICAL = "identical"
KIND_GEOMETRY_OR_SCORE = "geometry_or_score"
KIND_IDENTITY_ONLY = "identity_only"
KIND_EMPTY = "empty"
KIND_MISSING = "missing"


@dataclass(frozen=True)
class FirstDiff:
    """First raw-line difference between two MOT files of one sequence."""

    kind: str
    line_index: int | None
    frame: int | None
    n_ref_lines: int
    n_other_lines: int
    n_ref_frame: int | None
    n_other_frame: int | None
    ref_line: str | None
    other_line: str | None
    decimal_hash_equal: bool | None


@dataclass(frozen=True)
class SequenceReport:
    """Per-sequence identity across a list of run directories."""

    sequence: str
    n_runs: int
    n_distinct: int
    hashes: tuple[str, ...]
    reference_run: int | None
    divergent_runs: tuple[int, ...]
    first_diffs: tuple[FirstDiff | None, ...]
    ok: bool


@dataclass(frozen=True)
class RepeatReport:
    """Identity of every MOT sequence across the compared run directories."""

    n_runs: int
    sequences: tuple[str, ...]
    reports: tuple[SequenceReport, ...]
    ok: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def mot_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def list_mot_files(run_dir: Path) -> dict[str, Path]:
    files = sorted(run_dir.glob(MOT_GLOB))
    return {path.stem: path for path in files}


def _nonempty_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def _frame_of(line: str) -> int:
    return int(line.split(",", 1)[0])


def classify_first_diff(ref_text: str, other_text: str) -> FirstDiff:
    """Classify the first raw difference between two MOT file bodies.

    ``geometry_or_score`` means the ID-free canonical records of that frame
    already differ (box, score, or count).  ``identity_only`` means those
    records match and only track IDs (or line order of identical records)
    differ.
    """

    ref_lines = _nonempty_lines(ref_text)
    other_lines = _nonempty_lines(other_text)
    if not ref_lines and not other_lines:
        return FirstDiff(
            kind=KIND_IDENTICAL,
            line_index=None,
            frame=None,
            n_ref_lines=0,
            n_other_lines=0,
            n_ref_frame=None,
            n_other_frame=None,
            ref_line=None,
            other_line=None,
            decimal_hash_equal=True,
        )
    if not ref_lines or not other_lines:
        return FirstDiff(
            kind=KIND_EMPTY,
            line_index=0,
            frame=None,
            n_ref_lines=len(ref_lines),
            n_other_lines=len(other_lines),
            n_ref_frame=None,
            n_other_frame=None,
            ref_line=ref_lines[0] if ref_lines else None,
            other_line=other_lines[0] if other_lines else None,
            decimal_hash_equal=False,
        )

    limit = min(len(ref_lines), len(other_lines))
    index: int | None = None
    for i in range(limit):
        if ref_lines[i] != other_lines[i]:
            index = i
            break
    if index is None:
        if len(ref_lines) == len(other_lines):
            recs_equal = decimal_hash(
                canonicalize_mot_lines(ref_lines)
            ) == decimal_hash(canonicalize_mot_lines(other_lines))
            return FirstDiff(
                kind=KIND_IDENTICAL,
                line_index=None,
                frame=None,
                n_ref_lines=len(ref_lines),
                n_other_lines=len(other_lines),
                n_ref_frame=None,
                n_other_frame=None,
                ref_line=None,
                other_line=None,
                decimal_hash_equal=recs_equal,
            )
        index = limit

    ref_line = ref_lines[index] if index < len(ref_lines) else None
    other_line = other_lines[index] if index < len(other_lines) else None
    frame_line = ref_line or other_line
    assert frame_line is not None
    frame = _frame_of(frame_line)
    ref_frame = [line for line in ref_lines if _frame_of(line) == frame]
    other_frame = [line for line in other_lines if _frame_of(line) == frame]
    ref_recs = canonicalize_mot_lines(ref_frame)
    other_recs = canonicalize_mot_lines(other_frame)
    kind = KIND_GEOMETRY_OR_SCORE if ref_recs != other_recs else KIND_IDENTITY_ONLY
    file_hash_equal = decimal_hash(canonicalize_mot_lines(ref_lines)) == decimal_hash(
        canonicalize_mot_lines(other_lines)
    )
    return FirstDiff(
        kind=kind,
        line_index=index,
        frame=frame,
        n_ref_lines=len(ref_lines),
        n_other_lines=len(other_lines),
        n_ref_frame=len(ref_frame),
        n_other_frame=len(other_frame),
        ref_line=ref_line,
        other_line=other_line,
        decimal_hash_equal=file_hash_equal,
    )


def _hash_mot_path(path: Path | None) -> str:
    if path is None or not path.is_file():
        return MISSING
    if path.stat().st_size == 0:
        return EMPTY
    text = path.read_text(encoding="utf-8")
    if not _nonempty_lines(text):
        return EMPTY
    return mot_md5(path)


def _mode_index(hashes: Sequence[str]) -> int | None:
    countable = [item for item in hashes if item not in (MISSING, EMPTY)]
    if not countable:
        return None
    counts: dict[str, int] = {}
    first_index: dict[str, int] = {}
    for index, item in enumerate(hashes):
        if item in (MISSING, EMPTY):
            continue
        counts[item] = counts.get(item, 0) + 1
        first_index.setdefault(item, index)
    mode = max(counts, key=lambda key: (counts[key], -first_index[key]))
    return first_index[mode]


def compare_run_dirs(run_dirs: Sequence[Path]) -> RepeatReport:
    """Fail-closed identity of MOT files across run directories.

    A run is divergent when any sequence's MOT bytes differ, a required MOT
    file is missing, or a MOT file is empty.  Matching directories pass.
    """

    resolved = [Path(path) for path in run_dirs]
    if not resolved:
        return RepeatReport(
            n_runs=0,
            sequences=(),
            reports=(),
            ok=False,
            reasons=("no run directories",),
        )

    per_dir = [list_mot_files(path) for path in resolved]
    sequences = tuple(sorted({name for mapping in per_dir for name in mapping}))
    if not sequences:
        return RepeatReport(
            n_runs=len(resolved),
            sequences=(),
            reports=(),
            ok=False,
            reasons=("no MOT17-*.txt files in any run directory",),
        )

    reports: list[SequenceReport] = []
    reasons: list[str] = []
    all_ok = True
    for sequence in sequences:
        paths = [mapping.get(sequence) for mapping in per_dir]
        hashes = tuple(_hash_mot_path(path) for path in paths)
        reference_run = _mode_index(hashes)
        distinct = {item for item in hashes if item not in (MISSING, EMPTY)}
        n_distinct = len(distinct)
        missing = [i for i, item in enumerate(hashes) if item == MISSING]
        empty = [i for i, item in enumerate(hashes) if item == EMPTY]
        first_diffs: list[FirstDiff | None] = []
        if reference_run is not None:
            ref_path = paths[reference_run]
            assert ref_path is not None
            ref_text = ref_path.read_text(encoding="utf-8")
            for index, path in enumerate(paths):
                if index == reference_run:
                    first_diffs.append(None)
                    continue
                if path is None or hashes[index] in (MISSING, EMPTY):
                    first_diffs.append(
                        FirstDiff(
                            kind=KIND_MISSING
                            if hashes[index] == MISSING
                            else KIND_EMPTY,
                            line_index=None,
                            frame=None,
                            n_ref_lines=0,
                            n_other_lines=0,
                            n_ref_frame=None,
                            n_other_frame=None,
                            ref_line=None,
                            other_line=None,
                            decimal_hash_equal=None,
                        )
                    )
                    continue
                if hashes[index] == hashes[reference_run]:
                    first_diffs.append(None)
                    continue
                first_diffs.append(
                    classify_first_diff(ref_text, path.read_text(encoding="utf-8"))
                )
        else:
            first_diffs = [None] * len(paths)

        if reference_run is None:
            divergent = tuple(range(len(hashes)))
        else:
            reference_hash = hashes[reference_run]
            divergent = tuple(
                i for i, item in enumerate(hashes) if item != reference_hash
            )
        ok = (
            n_distinct == 1
            and not missing
            and not empty
            and all(item not in (MISSING, EMPTY) for item in hashes)
        )
        if not ok:
            all_ok = False
            if missing:
                reasons.append(f"{sequence}: missing in runs {missing}")
            if empty:
                reasons.append(f"{sequence}: empty in runs {empty}")
            if n_distinct > 1:
                reasons.append(
                    f"{sequence}: {n_distinct} distinct MOT hashes across {len(hashes)} runs"
                )
            if n_distinct == 0:
                reasons.append(f"{sequence}: no non-empty MOT file")
        reports.append(
            SequenceReport(
                sequence=sequence,
                n_runs=len(hashes),
                n_distinct=n_distinct,
                hashes=hashes,
                reference_run=reference_run,
                divergent_runs=divergent,
                first_diffs=tuple(first_diffs),
                ok=ok,
            )
        )

    return RepeatReport(
        n_runs=len(resolved),
        sequences=sequences,
        reports=tuple(reports),
        ok=all_ok,
        reasons=tuple(reasons),
    )


def format_report(report: RepeatReport) -> str:
    lines = [
        f"runs={report.n_runs} sequences={len(report.sequences)} "
        f"{'PASS' if report.ok else 'FAIL'}"
    ]
    for item in report.reports:
        status = "PASS" if item.ok else "FAIL"
        lines.append(
            f"  {item.sequence}: {status} distinct={item.n_distinct}/{item.n_runs} "
            f"divergent={list(item.divergent_runs)}"
        )
        if item.reference_run is not None:
            ref_hash = item.hashes[item.reference_run]
            lines.append(f"    reference_run={item.reference_run} md5={ref_hash}")
        for run_index, diff in enumerate(item.first_diffs):
            if diff is None:
                continue
            lines.append(
                f"    run {run_index}: kind={diff.kind} frame={diff.frame} "
                f"line={diff.line_index} md5={item.hashes[run_index]}"
            )
            if diff.ref_line is not None:
                lines.append(f"      ref  {diff.ref_line}")
            if diff.other_line is not None:
                lines.append(f"      other {diff.other_line}")
    for reason in report.reasons:
        lines.append(f"  reason: {reason}")
    return "\n".join(lines)
