#!/usr/bin/env python3
"""Exploratory #465 study (r2): at which stage does TF32-off T's drift first become structural?

Implements ``docs/research/studies/tf32_off_head_drift_stage_465_r2/`` (contract
§20.11, tier ``exploratory``; the result is not citable by a formal chain). It
supersedes ``tf32_off_head_drift_stage_465`` (r1), whose only attempt was
invalid on ``V_FORMAT`` because the head emits inverted xyxy boxes (x2 < x1).
The question, f*, IoU 0.9, 5-of-7 and the terminals are unchanged; r2 only
accepts inverted boxes as well formed and compares boxes by inversion
signature plus IoU of the canonicalized rectangle. It
reads only the frozen r2 localization packet and the PR-2 / PR-2R parity
packets, through :func:`research_study.open_frozen_study`, and runs no model.

For each r2 arm X in (R_T, R_E, H_M, H_V) against R_C, per sequence:

* per probe stage (``detector_output``, ``post_nms``, ``tracker_input``): the
  first frame whose rows differ bit for bit, and the first frame whose rows
  differ *structurally* (different count, or no one-to-one pairing using only
  eligible pairs: same class, same inversion signature, canonical IoU >=
  ``IOU_MIN``; scores are ignored);
* tracker output: the first bit divergence and the first structural divergence
  f* (different id multiset, or a same-id box pair that is not eligible);
* the f* label: ``different`` if ``tracker_input`` differs structurally at
  f*, else ``same``. It describes the tracker boundary at f* only, not where
  the divergence entered (f* may follow earlier, re-converged input changes).

The terminal is a rule over R_T's seven f* labels (declaration §4). Every
other number is report-only, including the TF32-on/off comparison of the final
txt layer of the PR-2 and PR-2R packets and the inverted-box census.

Usage:
  .venv/bin/python scripts/eval/diagnostics/tf32_off_head_drift_stage_r2.py
"""

# status: experiment

from __future__ import annotations

import io
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "scripts" / "tools"))

from research_study import StudyBinding, open_frozen_study  # noqa: E402

# --- frozen by the declaration (§2–§5) --------------------------------------
STUDY_ID = "tf32_off_head_drift_stage_465_r2"
BINDING = StudyBinding(
    study_id=STUDY_ID,
    runner_file=__file__,
    freeze_tag=f"freeze/{STUDY_ID}/1",
    pinned_blobs={
        f"docs/research/studies/{STUDY_ID}/study.yaml": "7125c9796d9dd6c165e55faeec15bd42bbcb4955",
        f"docs/research/studies/{STUDY_ID}/declaration.md": "04d4a42f23c4477cbaa2c3a6705eb4372aaef6aa",
    },
)

SEQUENCE_FRAMES: dict[str, int] = {
    "MOT17-02-SDP": 600,
    "MOT17-04-SDP": 1050,
    "MOT17-05-SDP": 837,
    "MOT17-09-SDP": 525,
    "MOT17-10-SDP": 654,
    "MOT17-11-SDP": 900,
    "MOT17-13-SDP": 750,
}
REF_ARM = "R_C"
ARMS = ("R_T", "R_E", "H_M", "H_V")
PRIMARY_ARM = "R_T"
RUNS = (1, 2)
PROBE_STAGES = ("detector_output", "post_nms", "tracker_input")
FSTAR_STAGE = "tracker_input"
IOU_MIN = 0.9
SUPPORT_MIN = 5  # 5-of-7 support threshold (stronger than a 4/7 strict majority)
TERMINAL_BY_LABEL = {
    "same": "TRACKER_INPUT_SAME_AT_FSTAR",
    "different": "TRACKER_INPUT_DIFFERENT_AT_FSTAR",
}
SPLIT = "SPLIT"
# Validity criteria in the order they are evaluated (declaration §3).
VALIDITY_ORDER = ("V_COMPLETE", "V_FORMAT", "V_REF_SELF", "V_RUN_REPRO", "V_RUNNER")

# Report-only txt pairs (declaration §5): (label, (input, member dir), (input, member dir)).
TXT_PAIRS: tuple[tuple[str, tuple[str, str], tuple[str, str]], ...] = (
    ("tf32_on_T_vs_C", ("pr2", "l2/A_C_1"), ("pr2", "l2/A_T_1")),
    ("tf32_off_T_vs_C", ("pr2r", "l2/A_C_1"), ("pr2r", "l2/A_T_1")),
    ("T_tf32_on_vs_off", ("pr2r", "l2/A_T_1"), ("pr2", "l2/A_T_1")),
    ("C_pr2_vs_pr2r", ("pr2r", "l2/A_C_1"), ("pr2", "l2/A_C_1")),
)

_NPZ_KEYS = (
    "rows",
    *(f"{s}_{k}" for s in PROBE_STAGES for k in ("frames", "counts", "rows")),
    "tracker_frames",
    "tracker_counts",
    "tracker_ids",
    "tracker_boxes",
)

Frames = dict[int, np.ndarray]
Tracks = dict[int, tuple[np.ndarray, np.ndarray]]


class Invalid(Exception):
    """A predeclared validity criterion failed; ``criterion`` names it."""

    def __init__(self, criterion: str, detail: str) -> None:
        super().__init__(f"{criterion}: {detail}")
        self.criterion = criterion
        self.detail = detail


# --------------------------------------------------------------------------
# pure logic (unit-tested on synthetic arrays)
# --------------------------------------------------------------------------
def ragged_by_frame(
    frames: np.ndarray, counts: np.ndarray, rows: np.ndarray
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    offset = 0
    for frame, count in zip(frames.tolist(), counts.tolist(), strict=True):
        if frame in out:
            raise ValueError(f"frame {frame} recorded twice")
        out[int(frame)] = rows[offset : offset + count]
        offset += count
    if offset != len(rows):
        raise ValueError(f"counts cover {offset} rows, body has {len(rows)}")
    return out


def stage_view(ev: Mapping[str, np.ndarray], stage: str, n_frames: int) -> Frames:
    """Rows per frame 1..n; a frame the probe never saw is zero rows (§2)."""
    got = ragged_by_frame(
        ev[f"{stage}_frames"], ev[f"{stage}_counts"], ev[f"{stage}_rows"]
    )
    if any(f < 1 or f > n_frames for f in got):
        raise ValueError(f"{stage} frame outside 1..{n_frames}")
    empty = np.zeros((0, 6), dtype=np.float32)
    return {f: got.get(f, empty) for f in range(1, n_frames + 1)}


def tracker_view(ev: Mapping[str, np.ndarray], n_frames: int) -> Tracks:
    """(ids, xyxy boxes) per frame 1..n; a frame with no emit has no tracks (§2)."""
    ids = ragged_by_frame(ev["tracker_frames"], ev["tracker_counts"], ev["tracker_ids"])
    boxes = ragged_by_frame(
        ev["tracker_frames"], ev["tracker_counts"], ev["tracker_boxes"]
    )
    if any(f < 1 or f > n_frames for f in ids):
        raise ValueError(f"tracker frame outside 1..{n_frames}")
    none = (np.zeros((0,), np.int64), np.zeros((0, 4), np.float32))
    return {f: (ids[f], boxes[f]) if f in ids else none for f in range(1, n_frames + 1)}


def canonical_xyxy(boxes: np.ndarray) -> np.ndarray:
    """``[min x, min y, max x, max y]`` of each box (float64)."""
    b = boxes.astype(np.float64)
    return np.stack(
        [
            np.minimum(b[:, 0], b[:, 2]),
            np.minimum(b[:, 1], b[:, 3]),
            np.maximum(b[:, 0], b[:, 2]),
            np.maximum(b[:, 1], b[:, 3]),
        ],
        axis=1,
    ).reshape(-1, 4)


def inversion_signature(boxes: np.ndarray) -> np.ndarray:
    """(x2 < x1, y2 < y1) per box, as an (n, 2) bool array (§2)."""
    b = boxes.astype(np.float64)
    return np.stack([b[:, 2] < b[:, 0], b[:, 3] < b[:, 1]], axis=1).reshape(-1, 2)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every canonicalized box in ``a`` against every one in ``b``.

    Boxes are canonicalized first, so an inverted box has the area of its
    rectangle. Two identical canonical rectangles have IoU 1 even when their
    area is zero (§2); any other pair with an empty union has IoU 0.
    """
    a = canonical_xyxy(a)
    b = canonical_xyxy(b)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    identical = np.all(a[:, None, :] == b[None, :, :], axis=2)
    return np.where(identical, 1.0, iou)


def eligible_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Box pairs with the same inversion signature and canonical IoU >= IOU_MIN."""
    same_sig = np.all(
        inversion_signature(a)[:, None, :] == inversion_signature(b)[None, :, :],
        axis=2,
    )
    return same_sig & (iou_matrix(a, b) >= IOU_MIN)


def rows_bit_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def rows_structurally_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Same count and a perfect matching over eligible pairs exists.

    Rows are (x1, y1, x2, y2, score, class); the score is not compared. A pair
    is eligible when the classes match, the inversion signatures match and the
    canonical IoU >= IOU_MIN; the question is
    whether *some* one-to-one pairing uses only eligible pairs (declaration
    §2), not whether the max-total-IoU pairing happens to.
    """
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    eligible = (a[:, None, 5] == b[None, :, 5]) & eligible_pairs(a[:, :4], b[:, :4])
    r, c = linear_sum_assignment((~eligible).astype(np.float64))
    return bool(np.all(eligible[r, c]))


def tracks_bit_equal(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> bool:
    return rows_bit_equal(a[0], b[0]) and rows_bit_equal(a[1], b[1])


def tracks_structurally_equal(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> bool:
    """Same id multiset and every same-id box pair eligible (§2)."""
    ids_a, boxes_a = a
    ids_b, boxes_b = b
    if len(ids_a) != len(ids_b):
        return False
    oa, ob = np.argsort(ids_a, kind="stable"), np.argsort(ids_b, kind="stable")
    if not np.array_equal(ids_a[oa], ids_b[ob]):
        return False
    if len(ids_a) == 0:
        return True
    return bool(np.all(np.diagonal(eligible_pairs(boxes_a[oa], boxes_b[ob]))))


def first_divergence(n_frames: int, same: Callable[[int], bool]) -> int | None:
    for frame in range(1, n_frames + 1):
        if not same(frame):
            return frame
    return None


def compare_sequence(
    ref: Mapping[str, np.ndarray], other: Mapping[str, np.ndarray], n_frames: int
) -> dict[str, Any]:
    """Stage-wise divergence of ``other`` against ``ref`` for one sequence (§2)."""
    out: dict[str, Any] = {"stages": {}}
    views = {}
    for stage in PROBE_STAGES:
        r, o = stage_view(ref, stage, n_frames), stage_view(other, stage, n_frames)
        views[stage] = (r, o)
        struct_diff = [
            f for f in range(1, n_frames + 1) if not rows_structurally_equal(r[f], o[f])
        ]
        out["stages"][stage] = {
            "first_bit": first_divergence(
                n_frames, lambda f: rows_bit_equal(r[f], o[f])
            ),
            "first_structural": struct_diff[0] if struct_diff else None,
            "structural_frames": len(struct_diff),
            "inversion_mismatch_frames": sum(
                1
                for f in range(1, n_frames + 1)
                if inversion_counts(r[f][:, :4]) != inversion_counts(o[f][:, :4])
            ),
        }
    tr, to = tracker_view(ref, n_frames), tracker_view(other, n_frames)
    f_star = first_divergence(
        n_frames, lambda f: tracks_structurally_equal(tr[f], to[f])
    )
    out["tracker"] = {
        "first_bit": first_divergence(
            n_frames, lambda f: tracks_bit_equal(tr[f], to[f])
        ),
        "first_structural": f_star,
    }
    if f_star is None:
        out["label_at_f_star"] = None
        out["at_f_star"] = None
        return out
    at = {
        stage: not rows_structurally_equal(r[f_star], o[f_star])
        for stage, (r, o) in views.items()
    }
    ti_ref, ti_oth = views[FSTAR_STAGE]
    out["at_f_star"] = {
        "structural_differs": at,
        "tracker_input_rows": [int(len(ti_ref[f_star])), int(len(ti_oth[f_star]))],
        "tracker_input_structural_frames_before": sum(
            1
            for f in range(1, f_star)
            if not rows_structurally_equal(ti_ref[f], ti_oth[f])
        ),
    }
    out["label_at_f_star"] = "different" if at[FSTAR_STAGE] else "same"
    return out


def inversion_counts(boxes: np.ndarray) -> tuple[int, int, int]:
    """Multiset of inversion signatures as (x only, y only, both) counts."""
    sig = inversion_signature(boxes)
    x, y = sig[:, 0], sig[:, 1]
    return (int(np.sum(x & ~y)), int(np.sum(~x & y)), int(np.sum(x & y)))


def inversion_census(boxes: np.ndarray) -> dict[str, int]:
    """Report-only (§5): inverted rows of one arm/run/sequence/stage."""
    sig = inversion_signature(boxes)
    return {
        "rows": int(len(sig)),
        "x_inverted_rows": int(np.sum(sig[:, 0])),
        "y_inverted_rows": int(np.sum(sig[:, 1])),
        "either_inverted_rows": int(np.sum(sig[:, 0] | sig[:, 1])),
    }


def decide(labels: Mapping[str, str | None]) -> str:
    """§4: a label held by >= SUPPORT_MIN of the seven sequences names the terminal."""
    for label, terminal in TERMINAL_BY_LABEL.items():
        if sum(1 for e in labels.values() if e == label) >= SUPPORT_MIN:
            return terminal
    return SPLIT


def parse_mot_txt(data: bytes) -> Tracks:
    """MOT rows ``frame,id,left,top,w,h,...`` as (ids, xyxy) per frame."""
    per: dict[int, list[tuple[int, list[float]]]] = {}
    for lineno, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        cols = line.split(",")
        if len(cols) < 6:
            raise ValueError(f"line {lineno} has {len(cols)} columns")
        frame, tid = int(float(cols[0])), int(float(cols[1]))
        left, top, w, h = (float(c) for c in cols[2:6])
        if not all(np.isfinite([left, top, w, h])):
            raise ValueError(f"line {lineno} has a non-finite box")
        per.setdefault(frame, []).append((tid, [left, top, left + w, top + h]))
    return {
        f: (
            np.asarray([t for t, _ in rows], np.int64),
            np.asarray([b for _, b in rows], np.float64).reshape(-1, 4),
        )
        for f, rows in per.items()
    }


def compare_txt(ref: Tracks, other: Tracks, n_frames: int) -> dict[str, int | None]:
    none = (np.zeros((0,), np.int64), np.zeros((0, 4), np.float64))
    get_r = lambda f: ref.get(f, none)  # noqa: E731
    get_o = lambda f: other.get(f, none)  # noqa: E731
    return {
        "first_bit": first_divergence(
            n_frames, lambda f: tracks_bit_equal(get_r(f), get_o(f))
        ),
        "first_structural": first_divergence(
            n_frames, lambda f: tracks_structurally_equal(get_r(f), get_o(f))
        ),
    }


def boxes_well_formed(boxes: np.ndarray) -> bool:
    """Finite coordinates; inverted boxes are well formed (§3 V_FORMAT)."""
    return bool(np.all(np.isfinite(boxes.astype(np.float64))))


def comparison_signature(result: Mapping[str, Any]) -> Any:
    """Everything V_RUN_REPRO requires run 1 and run 2 to agree on."""
    return json.loads(json.dumps(result, sort_keys=True))


# --------------------------------------------------------------------------
# frozen-data driver
# --------------------------------------------------------------------------
def _load_evidence(study: Any, arm: str, run: int, seq: str) -> dict[str, np.ndarray]:
    member = f"l2/{arm}_{run}.evidence/{seq}.npz"
    try:
        raw = study.read_input("r2", member)
    except Exception as exc:  # missing from the manifest is incompleteness
        raise Invalid("V_COMPLETE", f"{member}: {exc}") from exc
    with np.load(io.BytesIO(raw)) as z:
        missing = [k for k in _NPZ_KEYS if k not in z.files]
        if missing:
            raise Invalid("V_COMPLETE", f"{member} lacks {missing}")
        ev = {k: z[k] for k in _NPZ_KEYS}
    if len(ev["rows"]) != SEQUENCE_FRAMES[seq]:
        raise Invalid(
            "V_COMPLETE",
            f"{member} has {len(ev['rows'])} frames, expected {SEQUENCE_FRAMES[seq]}",
        )
    return ev


def _check_format(label: str, ev: Mapping[str, np.ndarray]) -> None:
    for stage in PROBE_STAGES:
        rows = ev[f"{stage}_rows"]
        if rows.ndim != 2 or rows.shape[1] != 6 or not boxes_well_formed(rows[:, :4]):
            raise Invalid(
                "V_FORMAT",
                f"{label} {stage} rows are not 6 wide with finite coordinates",
            )
    boxes = ev["tracker_boxes"]
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not boxes_well_formed(boxes):
        raise Invalid(
            "V_FORMAT", f"{label} tracker boxes are not 4 wide with finite coordinates"
        )


Evidence = dict[tuple[str, int, str], dict[str, np.ndarray]]
TxtKey = tuple[str, str, str]  # (input, member dir, sequence)


def _txt_keys() -> list[TxtKey]:
    keys: list[TxtKey] = []
    for _, ref, other in TXT_PAIRS:
        for name, directory in (ref, other):
            for seq in SEQUENCE_FRAMES:
                if (name, directory, seq) not in keys:
                    keys.append((name, directory, seq))
    return keys


def _load_inputs(study: Any) -> tuple[Evidence, dict[TxtKey, bytes]]:
    """Phase V_COMPLETE: every declared member, for every input, before anything else."""
    evidence: Evidence = {}
    for arm in (REF_ARM, *ARMS):
        for run in RUNS:
            for seq in SEQUENCE_FRAMES:
                evidence[(arm, run, seq)] = _load_evidence(study, arm, run, seq)
    raw_txt: dict[TxtKey, bytes] = {}
    for name, directory, seq in _txt_keys():
        member = f"{directory}/{seq}.txt"
        try:
            raw_txt[(name, directory, seq)] = study.read_input(name, member)
        except Exception as exc:
            raise Invalid("V_COMPLETE", f"{name}:{member}: {exc}") from exc
    return evidence, raw_txt


def _check_inputs(
    evidence: Evidence, raw_txt: Mapping[TxtKey, bytes]
) -> dict[TxtKey, Tracks]:
    """Phase V_FORMAT: all probe stages, tracker output and txt, before any comparison."""
    for (arm, run, seq), ev in evidence.items():
        _check_format(f"{arm}_{run}/{seq}", ev)
        try:
            for stage in PROBE_STAGES:
                stage_view(ev, stage, SEQUENCE_FRAMES[seq])
            tracker_view(ev, SEQUENCE_FRAMES[seq])
        except ValueError as exc:
            raise Invalid("V_FORMAT", f"{arm}_{run}/{seq}: {exc}") from exc
    parsed: dict[TxtKey, Tracks] = {}
    for (name, directory, seq), raw in raw_txt.items():
        member = f"{name}:{directory}/{seq}.txt"
        try:
            tracks = parse_mot_txt(raw)
        except ValueError as exc:
            raise Invalid("V_FORMAT", f"{member}: {exc}") from exc
        n = SEQUENCE_FRAMES[seq]
        if any(f < 1 or f > n for f in tracks):
            raise Invalid("V_FORMAT", f"{member}: frame outside 1..{n}")
        parsed[(name, directory, seq)] = tracks
    return parsed


def _stage_comparisons(evidence: Evidence) -> dict[str, Any]:
    """Phases V_REF_SELF then V_RUN_REPRO; returns run-1 comparisons per arm."""
    for seq, n in SEQUENCE_FRAMES.items():
        self_cmp = compare_sequence(
            evidence[(REF_ARM, 1, seq)], evidence[(REF_ARM, 2, seq)], n
        )
        diverged = [s for s, v in self_cmp["stages"].items() if v["first_bit"]]
        if diverged or self_cmp["tracker"]["first_bit"] is not None:
            raise Invalid(
                "V_REF_SELF", f"{seq}: R_C_1 vs R_C_2 differ at {diverged or 'tracker'}"
            )

    by_arm: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        by_arm[arm] = {}
        for seq, n in SEQUENCE_FRAMES.items():
            runs = [
                compare_sequence(
                    evidence[(REF_ARM, r, seq)], evidence[(arm, r, seq)], n
                )
                for r in RUNS
            ]
            if comparison_signature(runs[0]) != comparison_signature(runs[1]):
                raise Invalid("V_RUN_REPRO", f"{arm}/{seq}: run 1 and run 2 disagree")
            by_arm[arm][seq] = runs[0]
    return by_arm


def _txt_comparisons(parsed: Mapping[TxtKey, Tracks]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, (ref_in, ref_dir), (oth_in, oth_dir) in TXT_PAIRS:
        out[label] = {
            seq: compare_txt(
                parsed[(ref_in, ref_dir, seq)], parsed[(oth_in, oth_dir, seq)], n
            )
            for seq, n in SEQUENCE_FRAMES.items()
        }
    return out


def _census(evidence: Evidence) -> dict[str, Any]:
    """Report-only inverted-box census per arm/run/sequence and stage (§5)."""
    out: dict[str, Any] = {}
    for (arm, run, seq), ev in evidence.items():
        entry = {s: inversion_census(ev[f"{s}_rows"][:, :4]) for s in PROBE_STAGES}
        entry["tracker"] = inversion_census(ev["tracker_boxes"])
        out.setdefault(f"{arm}_{run}", {})[seq] = entry
    return out


def evaluate(study: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The validity phases in declared order (§3), then the report-only layers."""
    evidence, raw_txt = _load_inputs(study)
    parsed = _check_inputs(evidence, raw_txt)
    by_arm = _stage_comparisons(evidence)
    return by_arm, _txt_comparisons(parsed), _census(evidence)


def _report_md(
    by_arm: Mapping[str, Any],
    txt: Mapping[str, Any],
    census: Mapping[str, Any],
    terminal: str,
) -> str:
    lines = [
        f"# {STUDY_ID} attempt report (generated; exploratory, not citable)",
        "",
        f"terminal: **{terminal}** (rule over {PRIMARY_ARM} f* labels, declaration §4)",
        "",
        "## Stage divergence against R_C (run 1; run 2 identical by V_RUN_REPRO)",
        "",
        "first bit / first structural frame per stage; `—` = never. "
        "`n` = structurally divergent frames over the sequence.",
        "",
        "| arm | seq | det_out bit/struct (n) | post_nms bit/struct (n) "
        "| tracker_input bit/struct (n) | tracker bit/struct | tracker_input at f* "
        "| ti struct frames before f* | inv-sig mismatch frames det/nms/ti |",
        "|:--|:--|:--|:--|:--|:--|:--|--:|:--|",
    ]
    fmt = lambda v: "—" if v is None else str(v)  # noqa: E731
    for arm, seqs in by_arm.items():
        for seq, r in seqs.items():
            cells = [
                f"{fmt(r['stages'][s]['first_bit'])}/{fmt(r['stages'][s]['first_structural'])}"
                f" ({r['stages'][s]['structural_frames']})"
                for s in PROBE_STAGES
            ]
            before = (
                r["at_f_star"]["tracker_input_structural_frames_before"]
                if r["at_f_star"]
                else None
            )
            lines.append(
                f"| {arm} | {seq[6:8]} | {' | '.join(cells)} "
                f"| {fmt(r['tracker']['first_bit'])}/{fmt(r['tracker']['first_structural'])} "
                f"| {fmt(r['label_at_f_star'])} | {fmt(before)} "
                f"| {'/'.join(str(r['stages'][s]['inversion_mismatch_frames']) for s in PROBE_STAGES)} |"
            )
    lines += [
        "",
        "## Final txt layer, TF32 on vs off (report-only)",
        "",
        "| pair | " + " | ".join(s[6:8] for s in SEQUENCE_FRAMES) + " |",
        "|:--|" + "--:|" * len(SEQUENCE_FRAMES),
    ]
    for label, seqs in txt.items():
        lines.append(
            f"| {label} | "
            + " | ".join(
                f"{fmt(v['first_bit'])}/{fmt(v['first_structural'])}"
                for v in seqs.values()
            )
            + " |"
        )
    lines += [
        "",
        "## Inverted boxes (report-only; either-inverted rows / rows)",
        "",
        "| arm_run | seq | detector_output | post_nms | tracker_input | tracker |",
        "|:--|:--|--:|--:|--:|--:|",
    ]
    for arm_run, seqs in census.items():
        for seq, entry in seqs.items():
            lines.append(
                f"| {arm_run} | {seq[6:8]} | "
                + " | ".join(
                    f"{v['either_inverted_rows']}/{v['rows']}" for v in entry.values()
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    study = open_frozen_study(BINDING)  # no data path exists before this passes
    payload = study.payload_dir()
    try:
        by_arm, txt, census = evaluate(study)
    except Exception as exc:  # noqa: BLE001 -- every failure is recorded, never dropped
        criterion, detail = (
            (exc.criterion, exc.detail)
            if isinstance(exc, Invalid)
            else ("V_RUNNER", f"{type(exc).__name__}: {exc}")
        )
        (payload / "invalid.json").write_text(
            json.dumps({"criterion": criterion, "detail": detail}, indent=2) + "\n",
            encoding="utf-8",
        )
        record = study.record("invalid", invalid_criterion=criterion)
        print(f"INVALID {criterion}: {detail}; recorded {record}")
        return 1
    labels = {seq: r["label_at_f_star"] for seq, r in by_arm[PRIMARY_ARM].items()}
    terminal = decide(labels)
    result = {
        "study_id": STUDY_ID,
        "terminal": terminal,
        "primary_labels_at_f_star": labels,
        "iou_min": IOU_MIN,
        "stages": by_arm,
        "txt_report_only": txt,
        "inversion_census_report_only": census,
    }
    (payload / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (payload / "report.md").write_text(
        _report_md(by_arm, txt, census, terminal), encoding="utf-8"
    )
    record = study.record("valid", terminal=terminal)
    print(f"VALID terminal={terminal}; recorded {record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
