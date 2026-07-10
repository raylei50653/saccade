"""ID-free hashes for final, serialized MOT decimal records.

This module intentionally operates on the evaluator's final text output.  It
does not make claims about pre-serialization tensors or internal float drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import struct
from typing import Iterable, Sequence


FIELDS = (
    "frame",
    "x_centipixel",
    "y_centipixel",
    "w_centipixel",
    "h_centipixel",
    "score_1e4",
)
_SCALES = (100, 100, 100, 100, 10_000)


@dataclass(frozen=True)
class CanonicalRecord:
    frame: int
    x_centipixel: int
    y_centipixel: int
    w_centipixel: int
    h_centipixel: int
    score_1e4: int

    @property
    def values(self) -> tuple[int, int, int, int, int]:
        return (
            self.x_centipixel,
            self.y_centipixel,
            self.w_centipixel,
            self.h_centipixel,
            self.score_1e4,
        )


def _fixed_integer(text: str, scale: int, *, line_number: int, field: str) -> int:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(
            f"invalid {field} on MOT line {line_number}: {text!r}"
        ) from exc
    if not value.is_finite():
        raise ValueError(f"non-finite {field} on MOT line {line_number}: {text!r}")
    scaled = value * scale
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise ValueError(
            f"{field} on MOT line {line_number} is not aligned to serialized scale 1/{scale}: {text!r}"
        )
    return int(integral)


def canonicalize_mot_lines(lines: Iterable[str]) -> list[CanonicalRecord]:
    """Parse final MOT text, exclude IDs, and sort by ``frame,x,y,w,h,score``.

    Geometry is represented as centipixels and score as 1e-4 units, matching
    the evaluator's ``.2f`` / ``.4f`` final serialization contract.
    """
    records: list[CanonicalRecord] = []
    for line_number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        columns = [part.strip() for part in text.split(",")]
        if len(columns) < 7:
            raise ValueError(f"MOT line {line_number} has fewer than 7 columns")
        try:
            frame = int(columns[0])
        except ValueError as exc:
            raise ValueError(
                f"invalid frame on MOT line {line_number}: {columns[0]!r}"
            ) from exc
        values = tuple(
            _fixed_integer(item, scale, line_number=line_number, field=field)
            for item, scale, field in zip(columns[2:7], _SCALES, FIELDS[1:])
        )
        records.append(CanonicalRecord(frame, *values))
    return sorted(records, key=lambda record: (record.frame, *record.values))


def decimal_hash(records: Sequence[CanonicalRecord]) -> str:
    """Hash fixed little-endian ``frame + x100,y100,w100,h100,score10000``."""
    payload = bytearray()
    for record in records:
        payload.extend(struct.pack("<6q", record.frame, *record.values))
    return hashlib.sha256(payload).hexdigest()


def record_as_dict(record: CanonicalRecord) -> dict[str, int]:
    return {"frame": record.frame, **dict(zip(FIELDS[1:], record.values))}
