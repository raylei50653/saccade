"""Per-sequence applicability analysis for occ-exit audit (#55 WP2).

This module **classifies evidence only**. It does not enable sequence gates,
change production defaults, or rewrite tracker outputs.

Inputs
------
- Rows from ``_occ_audit.csv`` (probe-off or probe-on).
- Optional per-seq metric deltas (control vs treatment).

Outputs
-------
Per-seq recommendation in
``{enable_candidate, abstain, harmful, insufficient_evidence}``
plus flag_delta tallies and rollups by MOT17 scene type.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "CLASSIFICATIONS",
    "MOT17_SEQ_TYPE",
    "SeqEvidence",
    "Thresholds",
    "aggregate_occ_audit_rows",
    "build_applicability_table",
    "classify_seq",
    "load_metrics_json",
    "load_occ_audit_csv",
    "render_applicability_md",
    "rollup_by_seq_type",
]

CLASSIFICATIONS = (
    "enable_candidate",
    "abstain",
    "harmful",
    "insufficient_evidence",
)

# MOT17 train SDP sequences → coarse scene type for rollup (not a gate).
# Types are descriptive labels for the applicability map, not runtime policy.
MOT17_SEQ_TYPE: dict[str, str] = {
    "MOT17-02": "crowded_static",
    "MOT17-04": "crowded_static",
    "MOT17-05": "moving_low",
    "MOT17-09": "static_low",
    "MOT17-10": "moving",
    "MOT17-11": "indoor_static",
    "MOT17-13": "moving_night",
    # bare forms + detector suffixes resolve via seq_type()
}


@dataclass(frozen=True)
class Thresholds:
    """Conservative classification thresholds (analysis defaults)."""

    min_audited: int = 5
    min_useful_flags: int = 2
    idf1_noise_pp: float = 0.15  # |ΔIDF1| within this band = noise
    idf1_harm_pp: float = 0.30  # ΔIDF1 ≤ -this → harmful
    ids_material: int = 5  # |ΔIDs| ≥ this is material (IDs: lower better)
    chebgr_only_domination: float = 0.70  # share of disagreements that are chebgr_only


@dataclass
class SeqEvidence:
    seq: str
    seq_type: str = "unknown"
    audited: int = 0
    cosine_flags: int = 0
    chebgr_flags: int = 0
    flag_delta_same: int = 0
    flag_delta_same_flagged: int = 0  # both cosine + chebgr flag true
    flag_delta_cosine_only: int = 0
    flag_delta_chebgr_only: int = 0
    has_chebgr_columns: bool = False
    # Metric deltas: treatment − control. IDF1 in percentage points.
    # IDs: positive means more switches (worse).
    idf1_delta: float | None = None
    ids_delta: int | None = None
    has_metrics: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def useful_flags(self) -> int:
        """Episodes where at least one path flags (agreement-flagged + either-only)."""
        if self.has_chebgr_columns:
            return (
                self.flag_delta_same_flagged
                + self.flag_delta_cosine_only
                + self.flag_delta_chebgr_only
            )
        return self.cosine_flags

    @property
    def disagreement_n(self) -> int:
        return self.flag_delta_cosine_only + self.flag_delta_chebgr_only


def seq_type(seq: str) -> str:
    """Map ``MOT17-XX[-DET]`` to a coarse scene type label."""
    key = seq.strip()
    if key in MOT17_SEQ_TYPE:
        return MOT17_SEQ_TYPE[key]
    # MOT17-04-SDP → MOT17-04
    parts = key.split("-")
    if len(parts) >= 2:
        bare = f"{parts[0]}-{parts[1]}"
        if bare in MOT17_SEQ_TYPE:
            return MOT17_SEQ_TYPE[bare]
    return "unknown"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_occ_audit_csv(path: Path | str) -> list[dict[str, str]]:
    """Load ``_occ_audit.csv``; empty list if missing."""
    p = Path(path)
    if not p.is_file():
        return []
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


def aggregate_occ_audit_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, SeqEvidence]:
    """Aggregate decision-log rows into per-seq evidence.

    Tolerates probe-off logs (missing ``chebgr_*`` / ``flag_delta`` columns).
    """
    by_seq: dict[str, SeqEvidence] = {}
    for raw in rows:
        seq = str(raw.get("seq") or raw.get("sequence") or "").strip()
        if not seq:
            continue
        ev = by_seq.get(seq)
        if ev is None:
            ev = SeqEvidence(seq=seq, seq_type=seq_type(seq))
            by_seq[seq] = ev
        ev.audited += 1
        cosine_flag = _as_bool(raw.get("cosine_flag", raw.get("flagged")))
        if cosine_flag:
            ev.cosine_flags += 1

        has_chebgr = "chebgr_flag" in raw or "flag_delta" in raw
        if has_chebgr:
            ev.has_chebgr_columns = True
            chebgr_flag = _as_bool(raw.get("chebgr_flag"))
            if chebgr_flag:
                ev.chebgr_flags += 1
            delta = str(raw.get("flag_delta") or "").strip().lower()
            if delta not in {"same", "cosine_only", "chebgr_only"}:
                # Derive delta if column missing/blank but both flags present.
                if cosine_flag == chebgr_flag:
                    delta = "same"
                elif cosine_flag:
                    delta = "cosine_only"
                else:
                    delta = "chebgr_only"
            if delta == "same":
                ev.flag_delta_same += 1
                if cosine_flag and chebgr_flag:
                    ev.flag_delta_same_flagged += 1
            elif delta == "cosine_only":
                ev.flag_delta_cosine_only += 1
            elif delta == "chebgr_only":
                ev.flag_delta_chebgr_only += 1
    return by_seq


def load_metrics_json(path: Path | str) -> dict[str, dict[str, float | int | None]]:
    """Load optional per-seq metrics JSON.

    Accepted shapes (any combination of keys)::

        {
          "MOT17-04-SDP": {
            "idf1_delta": 0.4,          # pp, treatment − control
            "ids_delta": -3,            # treatment − control (lower IDs better)
            "idf1_control": 80.1,
            "idf1_treatment": 80.5,
            "ids_control": 40,
            "ids_treatment": 37
          },
          ...
        }

    Or wrapped as ``{"per_sequence": {...}}``.
    """
    import json

    p = Path(path)
    if not p.is_file():
        return {}
    data = json.loads(p.read_text())
    if isinstance(data, dict) and "per_sequence" in data:
        data = data["per_sequence"]
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, float | int | None]] = {}
    for seq, row in data.items():
        if not isinstance(row, Mapping):
            continue
        idf1_delta = _as_float(row.get("idf1_delta"))
        ids_delta_f = _as_float(row.get("ids_delta"))
        if idf1_delta is None:
            c = _as_float(row.get("idf1_control"))
            t = _as_float(row.get("idf1_treatment"))
            if c is not None and t is not None:
                # Accept either fraction [0,1] or percentage points.
                if max(abs(c), abs(t)) <= 1.5:
                    idf1_delta = (t - c) * 100.0
                else:
                    idf1_delta = t - c
        ids_delta: int | None
        if ids_delta_f is not None:
            ids_delta = int(round(ids_delta_f))
        else:
            c_ids = _as_float(row.get("ids_control"))
            t_ids = _as_float(row.get("ids_treatment"))
            if c_ids is not None and t_ids is not None:
                ids_delta = int(round(t_ids - c_ids))
            else:
                ids_delta = None
        if idf1_delta is None and ids_delta is None:
            continue
        out[str(seq)] = {"idf1_delta": idf1_delta, "ids_delta": ids_delta}
    return out


def attach_metrics(
    by_seq: dict[str, SeqEvidence],
    metrics: Mapping[str, Mapping[str, float | int | None]],
) -> None:
    """Mutate evidence with metric deltas when present."""
    for seq, m in metrics.items():
        ev = by_seq.get(seq)
        if ev is None:
            ev = SeqEvidence(seq=seq, seq_type=seq_type(seq))
            by_seq[seq] = ev
            ev.notes.append("metrics_without_audit_rows")
        idf1 = m.get("idf1_delta")
        ids = m.get("ids_delta")
        if idf1 is not None:
            ev.idf1_delta = float(idf1)
            ev.has_metrics = True
        if ids is not None:
            ev.ids_delta = int(ids)
            ev.has_metrics = True


def classify_seq(
    ev: SeqEvidence,
    thresholds: Thresholds | None = None,
) -> str:
    """Return one of :data:`CLASSIFICATIONS` for a single sequence.

    Conservative rules (analysis recommendation only — not a runtime gate):

    * **insufficient_evidence** — no / too few audited episodes, or no metrics
      and no strong chebgr-only domination signal.
    * **harmful** — clear IDF1 drop or material ID regression, or chebgr_only
      dominates disagreements without metric support for benefit.
    * **enable_candidate** — non-negative IDF1, IDs not material-worse, and
      useful flags are not isolated one-offs.
    * **abstain** — mixed / low-count / within noise band without clear harm.
    """
    th = thresholds or Thresholds()

    if ev.audited <= 0:
        return "insufficient_evidence"
    if ev.audited < th.min_audited:
        return "insufficient_evidence"

    disagree = ev.disagreement_n
    chebgr_dom = (
        disagree > 0
        and (ev.flag_delta_chebgr_only / disagree) >= th.chebgr_only_domination
    )

    if not ev.has_metrics:
        # Without a metric pair we do not promote enable_candidate.
        if chebgr_dom and ev.flag_delta_chebgr_only >= th.min_useful_flags:
            return "harmful"
        return "insufficient_evidence"

    idf1 = ev.idf1_delta if ev.idf1_delta is not None else 0.0
    ids_d = ev.ids_delta if ev.ids_delta is not None else 0

    # Clear harm on metrics.
    if idf1 <= -th.idf1_harm_pp:
        return "harmful"
    if ids_d >= th.ids_material and idf1 < th.idf1_noise_pp:
        return "harmful"

    # Suspicious chebgr-only mass with no metric upside.
    if chebgr_dom and idf1 < th.idf1_noise_pp and ids_d >= 0:
        return "harmful"

    ids_ok = ids_d <= th.ids_material  # allow small worsen within band if IDF1 good
    ids_not_material_worse = ids_d < th.ids_material
    metric_nonneg = idf1 >= 0.0 and ids_not_material_worse
    metric_positive = idf1 >= th.idf1_noise_pp or ids_d <= -th.ids_material
    useful = ev.useful_flags if ev.has_chebgr_columns else ev.cosine_flags
    not_one_off = useful >= th.min_useful_flags

    if metric_nonneg and not_one_off and (metric_positive or idf1 >= 0.0):
        # Require at least non-negative IDF1 and non-material ID regression.
        if ids_ok and idf1 >= 0.0:
            return "enable_candidate"

    # Within noise or mixed → abstain (not insufficient: we *have* evidence).
    if abs(idf1) < th.idf1_noise_pp and abs(ids_d) < th.ids_material:
        return "abstain"
    if not not_one_off:
        return "abstain"
    return "abstain"


def build_applicability_table(
    by_seq: Mapping[str, SeqEvidence],
    thresholds: Thresholds | None = None,
) -> list[dict[str, Any]]:
    """Build sorted per-seq recommendation rows."""
    th = thresholds or Thresholds()
    rows: list[dict[str, Any]] = []
    for seq in sorted(by_seq):
        ev = by_seq[seq]
        rec = classify_seq(ev, th)
        rows.append(
            {
                "seq": ev.seq,
                "seq_type": ev.seq_type,
                "recommendation": rec,
                "audited": ev.audited,
                "cosine_flags": ev.cosine_flags,
                "chebgr_flags": ev.chebgr_flags,
                "flag_delta_same": ev.flag_delta_same,
                "flag_delta_same_flagged": ev.flag_delta_same_flagged,
                "flag_delta_cosine_only": ev.flag_delta_cosine_only,
                "flag_delta_chebgr_only": ev.flag_delta_chebgr_only,
                "has_chebgr_columns": ev.has_chebgr_columns,
                "has_metrics": ev.has_metrics,
                "idf1_delta": ev.idf1_delta,
                "ids_delta": ev.ids_delta,
                "useful_flags": ev.useful_flags,
                "notes": list(ev.notes),
            }
        )
    return rows


def rollup_by_seq_type(table: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Count recommendations per scene type."""
    out: dict[str, dict[str, Any]] = {}
    for row in table:
        st = str(row.get("seq_type") or "unknown")
        bucket = out.setdefault(
            st,
            {
                "seq_type": st,
                "n_seq": 0,
                "recommendations": {c: 0 for c in CLASSIFICATIONS},
                "seqs": [],
            },
        )
        bucket["n_seq"] += 1
        rec = str(row.get("recommendation") or "insufficient_evidence")
        if rec not in bucket["recommendations"]:
            bucket["recommendations"][rec] = 0
        bucket["recommendations"][rec] += 1
        bucket["seqs"].append(row["seq"])
    return out


def render_applicability_md(
    table: list[dict[str, Any]],
    *,
    title: str = "occ-exit audit sequence conditioning (WP2)",
    thresholds: Thresholds | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> str:
    """Render a markdown applicability report (analysis only)."""
    th = thresholds or Thresholds()
    lines: list[str] = [
        f"# {title}",
        "",
        "**Objective:** RESEARCH + DEBUG — applicability map only.",
        "",
        "This document does **not** enable a sequence gate, change headline",
        "presets, or promote live critical-path behavior.",
        "",
        "## Classification thresholds",
        "",
        f"- `min_audited`: {th.min_audited}",
        f"- `min_useful_flags`: {th.min_useful_flags}",
        f"- `idf1_noise_pp`: {th.idf1_noise_pp}",
        f"- `idf1_harm_pp`: {th.idf1_harm_pp}",
        f"- `ids_material`: {th.ids_material}",
        f"- `chebgr_only_domination`: {th.chebgr_only_domination}",
        "",
    ]
    if provenance:
        lines.append("## Provenance")
        lines.append("")
        for k, v in provenance.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")

    lines.extend(
        [
            "## Per-sequence applicability",
            "",
            "| seq | type | recommendation | audited | cos_flags | chebgr_flags "
            "| same | cos_only | chebgr_only | ΔIDF1 | ΔIDs |",
            "|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
        ]
    )
    for row in table:
        idf1 = row["idf1_delta"]
        ids = row["ids_delta"]
        idf1_s = f"{idf1:+.2f}" if idf1 is not None else "—"
        ids_s = f"{ids:+d}" if ids is not None else "—"
        lines.append(
            f"| {row['seq']} | {row['seq_type']} | `{row['recommendation']}` "
            f"| {row['audited']} | {row['cosine_flags']} | {row['chebgr_flags']} "
            f"| {row['flag_delta_same']} | {row['flag_delta_cosine_only']} "
            f"| {row['flag_delta_chebgr_only']} | {idf1_s} | {ids_s} |"
        )

    rollup = rollup_by_seq_type(table)
    lines.extend(["", "## Rollup by scene type", ""])
    if not rollup:
        lines.append("_No sequences._")
    else:
        lines.append(
            "| type | n_seq | enable_candidate | abstain | harmful | insufficient |"
        )
        lines.append("|:--|--:|--:|--:|--:|--:|")
        for st in sorted(rollup):
            r = rollup[st]["recommendations"]
            lines.append(
                f"| {st} | {rollup[st]['n_seq']} "
                f"| {r.get('enable_candidate', 0)} "
                f"| {r.get('abstain', 0)} "
                f"| {r.get('harmful', 0)} "
                f"| {r.get('insufficient_evidence', 0)} |"
            )

    counts: dict[str, int] = defaultdict(int)
    for row in table:
        counts[str(row["recommendation"])] += 1
    lines.extend(
        [
            "",
            "## Summary counts",
            "",
        ]
    )
    for c in CLASSIFICATIONS:
        lines.append(f"- `{c}`: {counts.get(c, 0)}")
    lines.extend(
        [
            "",
            "## Non-goals of this map",
            "",
            "- No default-on sequence gate in evaluator / lifecycle",
            "- No production preset / headline YAML change",
            "- No live critical path promotion",
            "- Gate / promotion deferred to WP3",
            "",
        ]
    )
    return "\n".join(lines)


def evidence_to_dict(ev: SeqEvidence) -> dict[str, Any]:
    return asdict(ev)
