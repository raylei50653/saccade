"""Which eval sequences run, and which may be skipped (#537 F06).

``sequences`` names them explicitly (``MOT17-02-SDP,MOT17-04-SDP``) or asks for
discovery: empty, or only detector suffixes (``SDP``, ``SDP,FRCNN``), which
selects every sequence directory under ``<data_root>/<split>`` (ending in
``-<suffix>``). :func:`requested_sequences` is that rule; ``parse_eval_config``
uses it for ``EvalConfig.seqs``.

:func:`select_sequences` checks every requested sequence for ``seqinfo.ini``
before an eval runs:

* an explicitly named sequence without it is an error;
* a discovered one is skipped, with the reason logged;
* zero valid sequences is an error either way.

Contract: docs/architecture/failure_degradation_537.md §26.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DETECTOR_SUFFIXES = frozenset({"SDP", "DPM", "FRCNN"})


class SequenceSelectionError(ValueError):
    """The requested sequences cannot run; nothing was evaluated."""


@dataclass(frozen=True)
class SequenceSelection:
    explicit: bool
    requested: tuple[str, ...]
    valid: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]  # (sequence, reason); discovery only


def requested_sequences(
    data_root: str | Path, split: str, sequences: str | None
) -> tuple[bool, list[str]]:
    """``(explicit, names)`` in run order; discovery lists the split directory."""
    if sequences:
        filters = [s.strip() for s in sequences.split(",")]
        if not all(f in DETECTOR_SUFFIXES for f in filters):
            return True, filters
        found = _sequence_dirs(data_root, split)
        return False, [d for d in found if any(d.endswith(f"-{f}") for f in filters)]
    return False, _sequence_dirs(data_root, split)


def select_sequences(
    data_root: str | Path,
    split: str,
    sequences: str | None,
    *,
    log: Callable[[str], None] = print,
) -> SequenceSelection:
    """Validate the requested sequences; raise :class:`SequenceSelectionError`.

    Empty names (``"A,,B"``, a trailing comma) are not sequences and are
    dropped without a message.
    """
    root = Path(data_root) / split
    if not root.is_dir():
        raise SequenceSelectionError(f"split directory not found: {root}")
    explicit, names = requested_sequences(data_root, split, sequences)
    valid: list[str] = []
    skipped: list[tuple[str, str]] = []
    for name in names:
        if not name:
            continue
        if (root / name / "seqinfo.ini").exists():
            valid.append(name)
        else:
            skipped.append((name, "no seqinfo.ini"))
    if explicit and skipped:
        missing = ", ".join(name for name, _ in skipped)
        raise SequenceSelectionError(
            f"explicitly requested sequence(s) without seqinfo.ini under {root}: {missing}"
        )
    for name, reason in skipped:
        log(f"⚠️  Skipping discovered sequence {name}: {reason} ({root / name})")
    if not valid:
        raise SequenceSelectionError(
            f"no valid sequence under {root} (sequences={sequences!r}; "
            f"{len(skipped)} discovered, all without seqinfo.ini)"
            if skipped
            else f"no valid sequence under {root} (sequences={sequences!r}; none found)"
        )
    return SequenceSelection(
        explicit=explicit,
        requested=tuple(n for n in names if n),
        valid=tuple(valid),
        skipped=tuple(skipped),
    )


def _sequence_dirs(data_root: str | Path, split: str) -> list[str]:
    return sorted(d.name for d in (Path(data_root) / split).iterdir() if d.is_dir())
