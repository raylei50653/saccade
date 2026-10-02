#!/usr/bin/env python3
"""Exploratory #465 study: in which tracker block does R_T first diverge structurally from R_C?

Implements ``docs/research/studies/tracker_block_divergence_465/`` (contract
§20.11, tier ``exploratory``; the result is not citable by a formal chain). It
follows ``tf32_off_head_drift_stage_465_r2``, whose terminal was
``TRACKER_INPUT_SAME_AT_FSTAR`` (7/7): at R_T's first structural tracker-output
divergence f*, the detections entering the tracker were structurally the same
as R_C's. This study asks where inside the tracker the first structural
divergence appears, at or before f*.

Replay (declaration §2). For each replay (``R_C`` and ``R_T`` twice each,
``R_E`` once) one unmodified ``scripts/eval/mot17.py`` run (headline preset, the
same argv as the r2 localization) is driven by a worker in which exactly two
things are replaced, both outside production code:

* ``evaluator._run_track`` receives the arm's captured r2 ``tracker_input`` rows
  for that frame instead of the live detector's (the live detector still runs;
  its detections reach nothing the tracker reads);
* ``GraphedTrackerUpdate._capture`` installs an eager callable that makes the
  exact ``update_into`` call the graph captures (same buffers, ``num_dets =
  max_assoc``, no embeddings, ``light_factor 0``, ``mid_thresh_scale 1``) after
  the same warm-up, so the tracker's existing ``SACCADE_ASSOC_DUMP`` debug
  output can run (it does host I/O, which graph capture forbids).

``V_REPLAY`` requires every replay to reproduce its arm's captured r2 tracker
output bit for bit on every frame; otherwise the attempt is invalid.

Blocks (§2), in execution order within a tracker step for frame f:

* ``A`` association -- per active track, ``trk_to_det`` after all auction
  stages (dump ``TRK`` rows at f);
* ``P`` state transition -- state update, spawn and bridge, read as the
  multiset of (id, lifecycle state, age) of active tracks at the end of step f,
  from the tracker's read-only ``get_state_snapshots`` /
  ``get_tentative_candidates`` called right after the update returns;
* ``E`` emission -- the tracker output, compared as in r2 (its first
  divergence is r2's f*).

The terminal is a rule over the block of R_T's first divergence in the seven
sequences (§4). Everything else is report-only.

Usage (the one formal attempt; clean tree at the freeze tag, under a lease)::

  .venv/bin/python tools/resctl.py run gpu0 -- \\
      .venv/bin/python scripts/eval/diagnostics/tracker_block_divergence_465.py \\
      --raw-out <raw record directory named in declaration §8>
"""

# status: experiment

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import runpy
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "tools"))

from research_study import StudyBinding, open_frozen_study  # noqa: E402


def _load_stage_r2() -> Any:
    """The r2 stage study's comparator (frozen with that study; reused unchanged)."""
    path = (
        project_root
        / "scripts"
        / "eval"
        / "diagnostics"
        / "tf32_off_head_drift_stage_r2.py"
    )
    spec = importlib.util.spec_from_file_location("tf32_off_head_drift_stage_r2", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s2 = _load_stage_r2()

# --- frozen by the declaration (§2–§5) --------------------------------------
STUDY_ID = "tracker_block_divergence_465"
BINDING = StudyBinding(
    study_id=STUDY_ID,
    runner_file=__file__,
    freeze_tag=f"freeze/{STUDY_ID}/1",
    pinned_blobs={
        f"docs/research/studies/{STUDY_ID}/study.yaml": "c0a1cb34b82a13e802ba50a293b1d964d58fbdfc",
        f"docs/research/studies/{STUDY_ID}/declaration.md": "49e3eca0880cd9b87223846e3e5645d7a7c6776d",
    },
)

SEQUENCE_FRAMES: dict[str, int] = dict(s2.SEQUENCE_FRAMES)
IOU_MIN = s2.IOU_MIN  # 0.9, the r2 structural definition
SUPPORT_MIN = 5
PRESET_NAME = "mamba_whole_graph"
# Replay runs in execution order: (run label, r2 arm/run whose tracker_input is injected).
REPLAYS: tuple[tuple[str, str], ...] = (
    ("R_C#1", "R_C_1"),
    ("R_T#1", "R_T_1"),
    ("R_E", "R_E_1"),
    ("R_C#2", "R_C_1"),
    ("R_T#2", "R_T_1"),
)
REF_RUN = "R_C#1"
# V_REPEAT: every terminal-driving observation (dump and step-end state) of
# the reference and the primary arm must replay identically (§3).
REPEAT_PAIRS: tuple[tuple[str, str], ...] = (("R_C#1", "R_C#2"), ("R_T#1", "R_T#2"))
COMPARED = ("R_T#1", "R_E")
PRIMARY = "R_T#1"
BLOCKS = ("A", "P", "E")
TERMINAL_BY_BLOCK = {
    "A": "FIRST_DIVERGENCE_ASSOCIATION",
    "P": "FIRST_DIVERGENCE_STATE_TRANSITION",
    "E": "FIRST_DIVERGENCE_EMISSION",
}
SPLIT = "SPLIT"
VALIDITY_ORDER = (
    "V_COMPLETE",
    "V_FORMAT",
    "V_REPLAY",
    "V_RECORD",
    "V_REPEAT",
    "V_RUNNER",
)
LEASE_RESOURCES = ("gpu0", "machine-bench")
TENTATIVE, CONFIRMED = 1, 2  # tracker_gpu.cu TRACK_TENTATIVE / TRACK_CONFIRMED
DUMP_ENV = "SACCADE_ASSOC_DUMP"


class Invalid(Exception):
    """A predeclared validity criterion failed; ``criterion`` names it."""

    def __init__(self, criterion: str, detail: str) -> None:
        super().__init__(f"{criterion}: {detail}")
        self.criterion = criterion
        self.detail = detail


# --------------------------------------------------------------------------
# pure logic (unit-tested on synthetic data)
# --------------------------------------------------------------------------
def parse_segment(text: str) -> dict[str, Any]:
    """One tracker call's dump text -> its GMC row and per-track records.

    The first call of a sequence also holds the graph warm-up's block (it runs
    inside the first replay call), so only the block after the last ``GMC``
    line belongs to the frame. Raises ``ValueError`` on any malformed line.
    """
    lines = [ln for ln in text.splitlines() if ln]
    starts = [i for i, ln in enumerate(lines) if ln.startswith("GMC,")]
    if not starts:
        raise ValueError("segment has no GMC line")
    body = lines[starts[-1] :]
    gmc = body[0].split(",")
    if len(gmc) != 5:
        raise ValueError(f"malformed GMC line {body[0]!r}")
    tracks: list[dict[str, Any]] = []
    for ln in body[1:]:
        parts = ln.split(",")
        if parts[0] == "TRK":
            if len(parts) != 12:
                raise ValueError(f"malformed TRK line {ln!r}")
            tracks.append(
                {
                    "id": int(parts[2]),
                    "state": int(parts[3]),
                    "age": int(parts[4]),
                    "pred_cxcywh": [float(v) for v in parts[5:9]],
                    "n_cands": int(parts[9]),
                    "best_cost": float(parts[10]),
                    "t2d": int(parts[11]),
                    "cands": [],
                }
            )
        elif parts[0] == "CND":
            if len(parts) != 9:
                raise ValueError(f"malformed CND line {ln!r}")
            if not tracks or tracks[-1]["id"] != int(parts[2]):
                raise ValueError(f"CND line not under its TRK line: {ln!r}")
            tracks[-1]["cands"].append((int(parts[3]), float(parts[4])))
        else:
            raise ValueError(f"unknown dump line {ln!r}")
    for t in tracks:
        if t["t2d"] < -1:
            raise ValueError(f"track {t['id']} has t2d {t['t2d']}")
    return {"gmc": [float(v) for v in gmc[2:5]], "tracks": tracks}


def discrete_state(
    tracks: Sequence[Mapping[str, Any]],
) -> Counter[tuple[int, int, int]]:
    """Multiset of (id, lifecycle state, age) over active tracks."""
    return Counter((int(t["id"]), int(t["state"]), int(t["age"])) for t in tracks)


def step_end_rows(
    active: Sequence[tuple[int, int, int, int]],
    tentative: Sequence[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int, int]]:
    """Join the two read-only snapshots into (id, state, age, uid, generation) rows.

    ``active`` and ``tentative`` hold (id, age, uid, generation) per active /
    active-tentative slot. Each tentative entry must consume exactly one equal
    active entry; those rows get state 1 and the remaining active rows state 2
    (an active slot is only ever written 1 or 2: spawn and the post-update
    kernel). Raises ``ValueError`` when the join fails.
    """
    pool = Counter(tuple(int(v) for v in a) for a in active)
    tent = Counter(tuple(int(v) for v in t) for t in tentative)
    if tent - pool:
        raise ValueError(
            f"tentative entries without an active slot: {sorted(tent - pool)}"
        )
    rows = [(k[0], TENTATIVE, k[1], k[2], k[3]) for k in tent.elements()]
    rows += [(k[0], CONFIRMED, k[1], k[2], k[3]) for k in (pool - tent).elements()]
    return sorted(rows)


def state_multiset(
    rows: Sequence[Sequence[int]],
) -> Counter[tuple[int, int, int]]:
    """Block P observation: multiset of (id, lifecycle state, age) at step end."""
    return Counter((int(r[0]), int(r[1]), int(r[2])) for r in rows)


def _row_or_pad(rows: np.ndarray, d: int) -> np.ndarray | int:
    """The tracker_input row a det index names, or the index itself for padding."""
    return rows[d] if 0 <= d < len(rows) else d


def det_equal(rows_a: np.ndarray, da: int, rows_b: np.ndarray, db: int) -> bool:
    """Whether two det indices name structurally equal detections (§2).

    -1 (unmatched) equals only -1. A real row equals a real row iff the pair is
    eligible (same class, same inversion signature, canonical IoU >= IOU_MIN);
    an index past the frame's real rows (a zero padding slot of the graph's
    fixed-size buffer) equals only the same padding index.
    """
    if da < 0 or db < 0:
        return da == db
    a, b = _row_or_pad(rows_a, da), _row_or_pad(rows_b, db)
    if isinstance(a, int) or isinstance(b, int):
        return isinstance(a, int) and isinstance(b, int) and a == b
    if a[5] != b[5]:
        return False
    return bool(s2.eligible_pairs(a[None, :4], b[None, :4])[0, 0])


def association_equal(
    tracks_a: Sequence[Mapping[str, Any]],
    rows_a: np.ndarray,
    tracks_b: Sequence[Mapping[str, Any]],
    rows_b: np.ndarray,
) -> bool | None:
    """Block A at one frame; ``None`` when the pre-step states already differ.

    Tracks are grouped by id; within an id group, a perfect matching must pair
    entries with the same (state, age) whose ``trk_to_det`` name structurally
    equal detections (so duplicate-id order is not structure).
    """
    if discrete_state(tracks_a) != discrete_state(tracks_b):
        return None
    for tid in {int(t["id"]) for t in tracks_a}:
        ga = [t for t in tracks_a if int(t["id"]) == tid]
        gb = [t for t in tracks_b if int(t["id"]) == tid]
        eligible = np.array(
            [
                [
                    x["state"] == y["state"]
                    and x["age"] == y["age"]
                    and det_equal(rows_a, int(x["t2d"]), rows_b, int(y["t2d"]))
                    for y in gb
                ]
                for x in ga
            ],
            dtype=bool,
        )
        if not s2.perfect_matching_exists(eligible):
            return False
    return True


def candidates_equal(
    ta: Mapping[str, Any], rows_a: np.ndarray, tb: Mapping[str, Any], rows_b: np.ndarray
) -> bool:
    """Report-only: the two candidate lists name structurally equal det sets."""
    ca, cb = [d for d, _ in ta["cands"]], [d for d, _ in tb["cands"]]
    if len(ca) != len(cb):
        return False
    eligible = np.array(
        [[det_equal(rows_a, x, rows_b, y) for y in cb] for x in ca], dtype=bool
    ).reshape(len(ca), len(cb))
    return s2.perfect_matching_exists(eligible)


def ladder(
    n_frames: int,
    dump_ref: Mapping[int, Mapping[str, Any]],
    dump_oth: Mapping[int, Mapping[str, Any]],
    state_ref: Mapping[int, Sequence[Sequence[int]]],
    state_oth: Mapping[int, Sequence[Sequence[int]]],
    ti_ref: Mapping[int, np.ndarray],
    ti_oth: Mapping[int, np.ndarray],
    out_ref: Mapping[int, Any],
    out_oth: Mapping[int, Any],
) -> dict[str, Any]:
    """Per-block structural equality over one sequence and the first divergence.

    ``A`` at frame f is read from the dump of step f; ``P`` at frame f from the
    step-end state rows of step f; ``E`` from the output of step f. The first
    divergence is the earliest (frame, block) in the order A, P, E per frame (§2).
    """
    a_same: dict[int, bool | None] = {}
    p_same: dict[int, bool] = {}
    e_same: dict[int, bool] = {}
    for f in range(1, n_frames + 1):
        a_same[f] = association_equal(
            dump_ref[f]["tracks"], ti_ref[f], dump_oth[f]["tracks"], ti_oth[f]
        )
        p_same[f] = state_multiset(state_ref[f]) == state_multiset(state_oth[f])
        e_same[f] = s2.tracks_structurally_equal(out_ref[f], out_oth[f])
    first: tuple[int, str] | None = None
    for f in range(1, n_frames + 1):
        if a_same[f] is False:
            first = (f, "A")
        elif not p_same[f]:
            first = (f, "P")
        elif not e_same[f]:
            first = (f, "E")
        if first is not None:
            break
    e_frames = [f for f in range(1, n_frames + 1) if not e_same[f]]
    return {
        "first": None if first is None else {"frame": first[0], "block": first[1]},
        "first_by_block": {
            "A": next((f for f in range(1, n_frames + 1) if a_same[f] is False), None),
            "P": next((f for f in range(1, n_frames + 1) if not p_same[f]), None),
            "E": e_frames[0] if e_frames else None,
        },
        "divergent_frames": {
            "A": sum(1 for v in a_same.values() if v is False),
            "A_undefined": sum(1 for v in a_same.values() if v is None),
            "P": sum(1 for v in p_same.values() if not v),
            "E": len(e_frames),
        },
    }


def decide(blocks: Mapping[str, str | None]) -> str:
    """§4: the block of R_T's first divergence in >= SUPPORT_MIN of 7 sequences."""
    counts = Counter(b for b in blocks.values() if b is not None)
    for block, terminal in TERMINAL_BY_BLOCK.items():
        if counts.get(block, 0) >= SUPPORT_MIN:
            return terminal
    return SPLIT


def _row_json(rows: np.ndarray, d: int) -> Any:
    if d < 0:
        return None
    r = _row_or_pad(rows, d)
    if isinstance(r, int):
        return {"padding_index": r}
    return {
        "index": d,
        "xyxy": [float(v) for v in r[:4]],
        "score": float(r[4]),
        "class": int(r[5]),
    }


def explain_first(
    first: Mapping[str, Any],
    dump_ref: Mapping[int, Mapping[str, Any]],
    dump_oth: Mapping[int, Mapping[str, Any]],
    state_ref: Mapping[int, Sequence[Sequence[int]]],
    state_oth: Mapping[int, Sequence[Sequence[int]]],
    ti_ref: Mapping[int, np.ndarray],
    ti_oth: Mapping[int, np.ndarray],
    out_ref: Mapping[int, Any],
    out_oth: Mapping[int, Any],
) -> dict[str, Any]:
    """Report-only description of the first divergence (§5); decides nothing."""
    f, block = int(first["frame"]), str(first["block"])
    out: dict[str, Any] = {"frame": f, "block": block}
    out["tracker_input"] = {
        "structurally_equal": s2.rows_structurally_equal(ti_ref[f], ti_oth[f]),
        "bit_equal": s2.rows_bit_equal(ti_ref[f], ti_oth[f]),
        "rows": [len(ti_ref[f]), len(ti_oth[f])],
        "structural_frames_before": sum(
            1
            for g in range(1, f)
            if not s2.rows_structurally_equal(ti_ref[g], ti_oth[g])
        ),
    }
    if block == "A":
        tr, to = dump_ref[f]["tracks"], dump_oth[f]["tracks"]
        records = []
        for tid in sorted({int(t["id"]) for t in tr}):
            ga = [t for t in tr if int(t["id"]) == tid]
            gb = [t for t in to if int(t["id"]) == tid]
            if len(ga) == 1 and len(gb) == 1:
                x, y = ga[0], gb[0]
                if det_equal(ti_ref[f], x["t2d"], ti_oth[f], y["t2d"]):
                    continue
                records.append(
                    {
                        "id": tid,
                        "state": x["state"],
                        "age": x["age"],
                        "ref": {
                            "t2d": _row_json(ti_ref[f], x["t2d"]),
                            "cands": [
                                {"det": _row_json(ti_ref[f], d), "cost": c}
                                for d, c in x["cands"]
                            ],
                            "pred_cxcywh": x["pred_cxcywh"],
                        },
                        "other": {
                            "t2d": _row_json(ti_oth[f], y["t2d"]),
                            "cands": [
                                {"det": _row_json(ti_oth[f], d), "cost": c}
                                for d, c in y["cands"]
                            ],
                            "pred_cxcywh": y["pred_cxcywh"],
                        },
                        "candidate_sets_equal": candidates_equal(
                            x, ti_ref[f], y, ti_oth[f]
                        ),
                    }
                )
            else:
                records.append({"id": tid, "duplicate_id_group": [len(ga), len(gb)]})
        out["association"] = records
    elif block == "P":
        sa = state_multiset(state_ref[f])
        sb = state_multiset(state_oth[f])
        ids_a = {k[0] for k in sa}
        ids_b = {k[0] for k in sb}
        out["state_transition"] = {
            "only_ref": sorted([list(k) for k in (sa - sb).elements()]),
            "only_other": sorted([list(k) for k in (sb - sa).elements()]),
            "ids_only_ref": sorted(ids_a - ids_b),
            "ids_only_other": sorted(ids_b - ids_a),
            "association_equal_at_frame": association_equal(
                dump_ref[f]["tracks"], ti_ref[f], dump_oth[f]["tracks"], ti_oth[f]
            ),
        }
    else:
        (ia, ba), (ib, bb) = out_ref[f], out_oth[f]
        ca, cb = Counter(ia.tolist()), Counter(ib.tolist())
        out["emission"] = {
            "ids_only_ref": sorted((ca - cb).elements()),
            "ids_only_other": sorted((cb - ca).elements()),
            "same_id_ineligible": sorted(
                int(t)
                for t in set(ca) & set(cb)
                if not s2.perfect_matching_exists(
                    s2.eligible_pairs(ba[ia == t], bb[ib == t])
                )
            ),
        }
    return out


# --------------------------------------------------------------------------
# replay worker (one mot17.py run; executed in a child process)
# --------------------------------------------------------------------------
def mot17_argv(out: str) -> list[str]:
    return [
        "scripts/eval/mot17.py",
        "--preset",
        PRESET_NAME,
        "--detector",
        "SDP",
        "--double-buffer",
        "--sequences",
        ",".join(SEQUENCE_FRAMES),
        "--output",
        out,
    ]


def run_worker(inputs_json: Path, out_dir: Path) -> int:
    """Run mot17.py once with the arm's tracker_input injected; write the record.

    ``inputs_json`` maps sequence -> path of a private, read-only copy of the
    arm's r2 evidence npz (made by the parent through the frozen handle).
    """
    build_path = Path(os.environ.get("SACCADE_BUILD_PATH", project_root / "build"))
    if build_path.exists():
        sys.path.insert(0, str(build_path))
    sys.path.insert(0, str(project_root / "src"))
    dump = out_dir / "assoc_dump.csv"
    if DUMP_ENV in os.environ:
        raise RuntimeError(f"{DUMP_ENV} must not be set by the caller")
    os.environ[DUMP_ENV] = str(dump)  # read once by the tracker, at its first update

    import saccade.perception.detector_trt  # noqa: F401  (must precede torchvision)
    import torch

    import saccade.perception.eval.evaluator as evaluator_module
    from saccade.perception.eval import stages as stages_module
    from saccade.perception.tracking import tracker_gpu as tracker_module

    paths = json.loads(inputs_json.read_text(encoding="utf-8"))
    injected: dict[str, dict[int, np.ndarray]] = {}
    for seq, n in SEQUENCE_FRAMES.items():
        ev = np.load(paths[seq])
        injected[seq] = s2.stage_view(ev, "tracker_input", n)
    calls: list[dict[str, Any]] = []
    emits: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]] = []
    states: list[tuple[str, int, list[Any], list[Any]]] = []

    original_track = evaluator_module._run_track
    original_capture = tracker_module.GraphedTrackerUpdate._capture
    original_stages_emit = stages_module._run_emit
    original_evaluator_emit = evaluator_module._run_emit

    def run_track(state: Any, **kwargs: Any) -> Any:
        seq, frame = str(state.seq), int(state.current_frame_id)
        rows = injected[seq][frame]  # KeyError: a frame outside 1..N
        dev = kwargs["fused_boxes"].device
        kwargs["fused_boxes"] = torch.from_numpy(np.ascontiguousarray(rows[:, :4])).to(
            dev
        )
        kwargs["fused_scores"] = torch.from_numpy(np.ascontiguousarray(rows[:, 4])).to(
            dev
        )
        kwargs["fused_classes"] = torch.from_numpy(rows[:, 5].astype(np.int64)).to(dev)
        start = dump.stat().st_size if dump.exists() else 0
        result = original_track(state, **kwargs)
        end = dump.stat().st_size if dump.exists() else 0
        # Step-end state (§2 block P): both reads only copy device arrays to
        # host on the update's stream and synchronize it; neither writes
        # tracker state.
        tracker = state.detector.tracker
        active = [
            (s.obj_id, s.age, s.track_uid, s.generation)
            for s in tracker.get_state_snapshots()
        ]
        tentative = [
            (c.obj_id, c.age, c.track_uid, c.generation)
            for c in tracker.get_tentative_candidates()
        ]
        states.append((seq, frame, active, tentative))
        calls.append(
            {"seq": seq, "frame": frame, "start": start, "end": end, "rows": len(rows)}
        )
        return result

    def capture_eager(self: Any) -> None:
        if self._graphed_callable is not None:
            return
        self._warmup()

        def eager(boxes, scores, classes, d_gmc, ob, osc, oid, ocl, odi, ocn):  # type: ignore[no-untyped-def]
            stream = torch.cuda.current_stream().cuda_stream
            self._tracker.tracker.update_into(
                boxes.data_ptr(),
                scores.data_ptr(),
                classes.data_ptr(),
                self._max_assoc,
                stream,
                ob.data_ptr(),
                osc.data_ptr(),
                oid.data_ptr(),
                ocl.data_ptr(),
                odi.data_ptr(),
                ocn.data_ptr(),
                0,
                d_gmc.data_ptr(),
                0.0,
                1.0,
                self._max_objs,
            )
            return (ob, osc, oid, ocl, odi, ocn)

        self._graphed_callable = eager

    def to_np(t: Any, n: int, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
        if t is None:
            return np.zeros(shape, dtype)
        return t.detach().cpu().numpy()[:n].astype(dtype)

    def observing_emit(original: Any) -> Any:
        def wrapped(state_: Any, **kwargs: Any) -> Any:
            tr = kwargs.get("track_results") or {}
            raw = tr.get("count", 0)
            n = int(raw.item() if hasattr(raw, "item") else raw)
            emits.append(
                (
                    str(state_.seq),
                    int(kwargs["frame_id"]),
                    to_np(tr.get("ids"), n, np.int64, (0,)),
                    to_np(tr.get("det_idx"), n, np.int64, (0,)),
                    to_np(tr.get("boxes"), n, np.float32, (0, 4)),
                )
            )
            return original(state_, **kwargs)

        return wrapped

    evaluator_module._run_track = run_track
    tracker_module.GraphedTrackerUpdate._capture = capture_eager
    stages_module._run_emit = observing_emit(original_stages_emit)
    evaluator_module._run_emit = observing_emit(original_evaluator_emit)
    sys.path.insert(0, str(project_root / "scripts" / "eval"))
    argv = mot17_argv(str(out_dir / "mot"))
    sys.argv = list(argv)
    exit_code = 0
    try:
        runpy.run_path(argv[0], run_name="__main__")
    except SystemExit as exc:
        exit_code = (
            exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        )
    finally:
        evaluator_module._run_track = original_track
        tracker_module.GraphedTrackerUpdate._capture = original_capture
        stages_module._run_emit = original_stages_emit
        evaluator_module._run_emit = original_evaluator_emit
    (out_dir / "calls.json").write_text(json.dumps(calls) + "\n", encoding="utf-8")
    arrays: dict[str, np.ndarray] = {}
    for seq in SEQUENCE_FRAMES:
        rows = [e for e in emits if e[0] == seq]
        arrays[f"{seq}__frames"] = np.array([e[1] for e in rows], np.int64)
        arrays[f"{seq}__counts"] = np.array([len(e[2]) for e in rows], np.int64)
        arrays[f"{seq}__ids"] = np.concatenate(
            [e[2] for e in rows] or [np.zeros(0, np.int64)]
        )
        arrays[f"{seq}__det_idx"] = np.concatenate(
            [e[3] for e in rows] or [np.zeros(0, np.int64)]
        )
        arrays[f"{seq}__boxes"] = np.concatenate(
            [e[4] for e in rows] or [np.zeros((0, 4), np.float32)]
        ).reshape(-1, 4)
    np.savez(out_dir / "emits.npz", **arrays)
    # Raw snapshots, joined by the parent (a failed join is V_RECORD there).
    # Columns: id, age, generation (int64) and uid (uint64, its native type).
    st: dict[str, np.ndarray] = {}
    for seq in SEQUENCE_FRAMES:
        rows_ = [e for e in states if e[0] == seq]
        st[f"{seq}__frames"] = np.array([e[1] for e in rows_], np.int64)
        for k, key in ((2, "active"), (3, "tentative")):
            flat = [r for e in rows_ for r in e[k]]
            st[f"{seq}__{key}_counts"] = np.array([len(e[k]) for e in rows_], np.int64)
            st[f"{seq}__{key}_iag"] = np.array(
                [(r[0], r[1], r[3]) for r in flat], np.int64
            ).reshape(-1, 3)
            st[f"{seq}__{key}_uid"] = np.array([r[2] for r in flat], np.uint64)
    np.savez(out_dir / "states.npz", **st)
    return int(exit_code)


# --------------------------------------------------------------------------
# evaluation (parent process)
# --------------------------------------------------------------------------
def _ragged(
    frames: np.ndarray, counts: np.ndarray, body: np.ndarray
) -> dict[int, np.ndarray]:
    return s2.ragged_by_frame(frames, counts, body)


def load_replay(
    run_dir: Path, ti: Mapping[str, Mapping[int, np.ndarray]]
) -> dict[str, Any]:
    """Check one replay's call record (V_REPLAY part 1); parse its records (V_RECORD)."""
    calls = json.loads((run_dir / "calls.json").read_text(encoding="utf-8"))
    raw = (run_dir / "assoc_dump.csv").read_bytes()
    segments: dict[str, dict[int, bytes]] = {seq: {} for seq in SEQUENCE_FRAMES}
    order: list[tuple[str, int]] = []
    for c in calls:
        seq, frame = c["seq"], int(c["frame"])
        if seq not in segments or frame in segments[seq]:
            raise Invalid(
                "V_REPLAY",
                f"{run_dir.name}: tracker call {seq} f{frame} repeated or foreign",
            )
        if int(c["rows"]) != len(ti[seq][frame]):
            raise Invalid(
                "V_REPLAY", f"{run_dir.name}: {seq} f{frame} injected {c['rows']} rows"
            )
        segments[seq][frame] = raw[int(c["start"]) : int(c["end"])]
        order.append((seq, frame))
    for seq, n in SEQUENCE_FRAMES.items():
        if [f for s, f in order if s == seq] != list(range(1, n + 1)):
            raise Invalid(
                "V_REPLAY",
                f"{run_dir.name}: {seq} tracker calls are not frames 1..{n} in order",
            )
    if sum(len(v) for v in segments.values()) and (
        sum(int(c["end"]) - int(c["start"]) for c in calls) != len(raw)
    ):
        raise Invalid("V_RECORD", f"{run_dir.name}: dump bytes outside tracker calls")
    dumps: dict[str, dict[int, dict[str, Any]]] = {}
    for seq, by_frame in segments.items():
        dumps[seq] = {}
        for frame, data in by_frame.items():
            try:
                dumps[seq][frame] = parse_segment(data.decode("ascii"))
            except (ValueError, UnicodeDecodeError) as exc:
                raise Invalid(
                    "V_RECORD", f"{run_dir.name}: {seq} f{frame}: {exc}"
                ) from exc
    states: dict[str, dict[int, list[tuple[int, int, int, int, int]]]] = {}
    try:
        z = np.load(run_dir / "states.npz")
        for seq, n in SEQUENCE_FRAMES.items():
            frames = z[f"{seq}__frames"]
            if frames.tolist() != list(range(1, n + 1)):
                raise ValueError(f"{seq} step-end states are not frames 1..{n}")
            raw_rows = {}
            for key in ("active", "tentative"):
                iag = _ragged(frames, z[f"{seq}__{key}_counts"], z[f"{seq}__{key}_iag"])
                uid = _ragged(frames, z[f"{seq}__{key}_counts"], z[f"{seq}__{key}_uid"])
                raw_rows[key] = {
                    f: [
                        (int(a[0]), int(a[1]), int(u), int(a[2]))
                        for a, u in zip(iag[f], uid[f], strict=True)
                    ]
                    for f in iag
                }
            states[seq] = {
                f: step_end_rows(raw_rows["active"][f], raw_rows["tentative"][f])
                for f in range(1, n + 1)
            }
    except (OSError, KeyError, ValueError) as exc:
        raise Invalid("V_RECORD", f"{run_dir.name}: {exc}") from exc
    return {"segments": segments, "dumps": dumps, "states": states}


def load_emits(
    run_dir: Path,
) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    z = np.load(run_dir / "emits.npz")
    out: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for seq, n in SEQUENCE_FRAMES.items():
        fr, ct = z[f"{seq}__frames"], z[f"{seq}__counts"]
        ids = _ragged(fr, ct, z[f"{seq}__ids"])
        det = _ragged(fr, ct, z[f"{seq}__det_idx"])
        box = _ragged(fr, ct, z[f"{seq}__boxes"])
        none = (
            np.zeros(0, np.int64),
            np.zeros(0, np.int64),
            np.zeros((0, 4), np.float32),
        )
        out[seq] = {
            f: (ids[f], det[f], box[f]) if f in ids else none for f in range(1, n + 1)
        }
    return out


def r2_tracker_full(
    ev: Mapping[str, np.ndarray], n: int
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    fr, ct = ev["tracker_frames"], ev["tracker_counts"]
    ids = _ragged(fr, ct, ev["tracker_ids"])
    det = _ragged(fr, ct, ev["tracker_det_idx"])
    box = _ragged(fr, ct, ev["tracker_boxes"])
    if any(f < 1 or f > n for f in ids):
        raise ValueError(f"tracker frame outside 1..{n}")
    none = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 4), np.float32))
    return {f: (ids[f], det[f], box[f]) if f in ids else none for f in range(1, n + 1)}


def emits_bit_equal(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...]) -> bool:
    return (
        np.array_equal(a[0].astype(np.int64), b[0].astype(np.int64))
        and np.array_equal(a[1].astype(np.int64), b[1].astype(np.int64))
        and a[2].astype(np.float32).tobytes() == b[2].astype(np.float32).tobytes()
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def held_lease() -> dict[str, Any] | None:
    """The resctl lease whose owner is this process's parent (the ``resctl run``)."""
    out = subprocess.run(
        [sys.executable, "tools/resctl.py", "status", "--json"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        return None
    for row in json.loads(out.stdout).get("leases", []):
        owner = row.get("owner") or {}
        if row.get("state") == "BUSY" and owner.get("pid") == os.getppid():
            return {"resource": row.get("resource"), "pid": owner.get("pid")}
    return None


def evaluate(study: Any, raw_out: Path) -> dict[str, Any]:
    # V_COMPLETE / V_FORMAT over every input before any replay launches (§3).
    members = set(study.input_members("r2"))
    need = {arm for _, arm in REPLAYS}
    for arm in sorted(need):
        for seq in SEQUENCE_FRAMES:
            if f"l2/{arm}.evidence/{seq}.npz" not in members:
                raise Invalid(
                    "V_COMPLETE", f"r2 member l2/{arm}.evidence/{seq}.npz missing"
                )
    ti: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    r2_out: dict[str, dict[str, Any]] = {}
    paths: dict[str, dict[str, str]] = {}
    for arm in sorted(need):
        ti[arm], r2_out[arm], paths[arm] = {}, {}, {}
        for seq, n in SEQUENCE_FRAMES.items():
            member = f"l2/{arm}.evidence/{seq}.npz"
            try:
                path = study.input_file("r2", member)
                ev = np.load(path)
            except Exception as exc:  # noqa: BLE001
                raise Invalid("V_COMPLETE", f"{member}: {exc}") from exc
            try:
                ti[arm][seq] = s2.stage_view(ev, "tracker_input", n)
                r2_out[arm][seq] = r2_tracker_full(ev, n)
            except (KeyError, ValueError) as exc:
                raise Invalid("V_FORMAT", f"{member}: {exc}") from exc
            for f, rows in ti[arm][seq].items():
                if (
                    rows.ndim != 2
                    or rows.shape[1] != 6
                    or not np.all(np.isfinite(rows))
                ):
                    raise Invalid(
                        "V_FORMAT", f"{member} f{f}: tracker_input rows malformed"
                    )
            paths[arm][seq] = str(path)

    replays: dict[str, dict[str, Any]] = {}
    for run, arm in REPLAYS:
        run_dir = raw_out / run.replace("#", "_")
        run_dir.mkdir(parents=True, exist_ok=False)
        inputs_json = run_dir / "inputs.json"
        inputs_json.write_text(
            json.dumps(paths[arm], indent=2) + "\n", encoding="utf-8"
        )
        log = run_dir / "stdout.log"
        with log.open("wb") as fh:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-inputs",
                    str(inputs_json),
                    "--worker-out",
                    str(run_dir),
                ],
                cwd=project_root,
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if proc.returncode != 0:
            raise Invalid(
                "V_REPLAY", f"{run}: worker exit {proc.returncode} (see {log.name})"
            )
        rec = load_replay(run_dir, ti[arm])
        emits = load_emits(run_dir)
        for seq, n in SEQUENCE_FRAMES.items():
            for f in range(1, n + 1):
                if not emits_bit_equal(emits[seq][f], r2_out[arm][seq][f]):
                    raise Invalid(
                        "V_REPLAY",
                        f"{run}: {seq} f{f} tracker output differs from {arm}'s r2 capture",
                    )
        rec["emits"] = emits
        replays[run] = rec

    for first_run, second_run in REPEAT_PAIRS:
        a, b = replays[first_run], replays[second_run]
        for seq, n in SEQUENCE_FRAMES.items():
            for f in range(1, n + 1):
                if a["segments"][seq][f] != b["segments"][seq][f]:
                    raise Invalid(
                        "V_REPEAT",
                        f"{seq} f{f}: dump differs, {first_run} vs {second_run}",
                    )
                if a["states"][seq][f] != b["states"][seq][f]:
                    raise Invalid(
                        "V_REPEAT",
                        f"{seq} f{f}: step-end state differs, {first_run} vs {second_run}",
                    )

    ref_arm = dict(REPLAYS)[REF_RUN]
    by_run: dict[str, dict[str, Any]] = {}
    for run in COMPARED:
        arm = dict(REPLAYS)[run]
        by_run[run] = {}
        for seq, n in SEQUENCE_FRAMES.items():
            dr, do = replays[REF_RUN]["dumps"][seq], replays[run]["dumps"][seq]
            outr = {f: (v[0], v[2]) for f, v in replays[REF_RUN]["emits"][seq].items()}
            outo = {f: (v[0], v[2]) for f, v in replays[run]["emits"][seq].items()}
            sr, so = replays[REF_RUN]["states"][seq], replays[run]["states"][seq]
            views = (dr, do, sr, so, ti[ref_arm][seq], ti[arm][seq], outr, outo)
            lad = ladder(n, *views)
            lad["first_detail"] = (
                explain_first(lad["first"], *views)
                if lad["first"] is not None
                else None
            )
            gmc_diff = [f for f in range(1, n + 1) if dr[f]["gmc"] != do[f]["gmc"]]
            lad["gmc_row_mismatch_frames"] = len(gmc_diff)
            by_run[run][seq] = lad
    return by_run


def _report_md(by_run: Mapping[str, Any], terminal: str) -> str:
    lines = [
        f"# {STUDY_ID} attempt report (generated; exploratory, not citable)",
        "",
        f"terminal: **{terminal}** (rule over R_T first-divergence blocks, declaration §4)",
        "",
        "## First structural divergence against R_C",
        "",
        "`first` = earliest (frame, block); per-block first frame and divergent-frame counts follow. "
        "`ti@first` = tracker_input structural / bit equality at that frame.",
        "",
        "| run | seq | first | A first (n) | P first (n) | E first = f* (n) | ti@first struct/bit | ti struct frames before | GMC row mismatch |",
        "|:--|:--|:--|:--|:--|:--|:--|--:|--:|",
    ]
    for run, seqs in by_run.items():
        for seq, r in seqs.items():
            first = r["first"]
            fb, dv = r["first_by_block"], r["divergent_frames"]
            det = r["first_detail"] or {}
            tid = det.get("tracker_input") or {}
            lines.append(
                "| {run} | {seq} | {first} | {a} ({na}) | {p} ({np_}) | {e} ({ne}) | {ts} | {tb} | {g} |".format(
                    run=run,
                    seq=seq.split("-")[1],
                    first="—"
                    if first is None
                    else f"{first['frame']}/{first['block']}",
                    a=fb["A"] if fb["A"] is not None else "—",
                    na=dv["A"],
                    p=fb["P"] if fb["P"] is not None else "—",
                    np_=dv["P"],
                    e=fb["E"] if fb["E"] is not None else "—",
                    ne=dv["E"],
                    ts=(
                        f"{'same' if tid.get('structurally_equal') else 'diff'}/"
                        f"{'same' if tid.get('bit_equal') else 'diff'}"
                        if tid
                        else "—"
                    ),
                    tb=tid.get("structural_frames_before", "—"),
                    g=r["gmc_row_mismatch_frames"],
                )
            )
    lines += [
        "",
        "Per-divergence details (association records, state-transition keys, emission ids) are in result.json.",
    ]
    return "\n".join(lines) + "\n"


def write_raw_manifest(raw_out: Path) -> Path:
    files = {
        p.relative_to(raw_out).as_posix(): _sha256(p)
        for p in sorted(raw_out.rglob("*"))
        if p.is_file() and p.name != "MANIFEST.json"
    }
    path = raw_out / "MANIFEST.json"
    path.write_text(
        json.dumps({"files": files}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-out", help="new raw record directory (declaration §8)")
    parser.add_argument("--worker-inputs", help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_inputs or args.worker_out:
        if not (args.worker_inputs and args.worker_out):
            parser.error("internal worker flags are malformed")
        return run_worker(Path(args.worker_inputs), Path(args.worker_out))
    if not args.raw_out:
        parser.error("--raw-out is required")
    lease = held_lease()
    if lease is None or lease["resource"] not in LEASE_RESOURCES:
        print(
            f"refused: run under `resctl run` holding one of {LEASE_RESOURCES}",
            file=sys.stderr,
        )
        return 2  # before the freeze opens: no attempt is consumed
    if DUMP_ENV in os.environ:
        print(
            f"refused: {DUMP_ENV} is set in the caller's environment", file=sys.stderr
        )
        return 2

    study = open_frozen_study(BINDING)  # no data path exists before this passes
    raw_out = (project_root / args.raw_out).resolve()
    from scripts.provenance.run_manifest import open_run

    open_run(
        raw_out,
        produced_by="diagnostic",
        preset=PRESET_NAME,
        detector="SDP",
        cmdline=[
            Path(__file__).relative_to(project_root).as_posix(),
            "--raw-out",
            args.raw_out,
        ],
    )
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        # The attempt directory is created only after the replays, so every
        # mot17.py child sees the clean freeze tree.
        by_run = evaluate(study, raw_out)
    except Exception as exc:  # noqa: BLE001 -- every failure is recorded, never dropped
        payload = study.payload_dir()
        criterion, detail = (
            (exc.criterion, exc.detail)
            if isinstance(exc, Invalid)
            else ("V_RUNNER", f"{type(exc).__name__}: {exc}")
        )
        manifest = write_raw_manifest(raw_out)
        (payload / "invalid.json").write_text(
            json.dumps(
                {
                    "criterion": criterion,
                    "detail": detail,
                    "raw_out": args.raw_out,
                    "raw_manifest_sha256": _sha256(manifest),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        record = study.record("invalid", invalid_criterion=criterion)
        print(f"INVALID {criterion}: {detail}; recorded {record}")
        return 1
    blocks = {
        seq: (r["first"]["block"] if r["first"] is not None else None)
        for seq, r in by_run[PRIMARY].items()
    }
    terminal = decide(blocks)
    manifest = write_raw_manifest(raw_out)
    payload = study.payload_dir()
    result = {
        "study_id": STUDY_ID,
        "terminal": terminal,
        "primary_first_blocks": blocks,
        "iou_min": IOU_MIN,
        "support_min": SUPPORT_MIN,
        "lease": lease,
        "started_utc": started,
        "raw_out": args.raw_out,
        "raw_manifest_sha256": _sha256(manifest),
        "runs": by_run,
    }
    (payload / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (payload / "report.md").write_text(_report_md(by_run, terminal), encoding="utf-8")
    record = study.record("valid", terminal=terminal)
    print(f"VALID terminal={terminal}; recorded {record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
