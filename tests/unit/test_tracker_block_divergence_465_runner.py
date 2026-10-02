"""The tracker block-divergence runner implements its exploratory declaration.

``scripts/eval/diagnostics/tracker_block_divergence_465.py`` implements
``docs/research/studies/tracker_block_divergence_465/``. These tests pin (a) the
runner to its study (study.yaml names it, the binding pins the committed
study.yaml and declaration, the static freeze check passes), (b) the dump
parser, (c) the §2 block comparisons and the first-divergence order on
synthetic data, (d) the §4 terminal rule, (e) the §3 validity phases on a fake
replay (no GPU, no frame), and (f) that without a lease or a freeze the runner
stops before any data path exists.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "eval" / "diagnostics" / "tracker_block_divergence_465.py"
STUDY_DIR = REPO / "docs" / "research" / "studies" / "tracker_block_divergence_465"


def _load():
    spec = importlib.util.spec_from_file_location(
        "tracker_block_divergence_465", RUNNER
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load()
rs = sys.modules["research_study"]


# ------------------------------------------------------------------ binding


def test_study_names_this_runner_and_passes_static_check():
    study = rs.parse_study((STUDY_DIR / "study.yaml").read_bytes())
    assert study["runner"] == RUNNER.relative_to(REPO).as_posix()
    assert rs.study_schema_problems(study, m.STUDY_ID) == []
    assert rs.derive_tier(study["section_20_2"]) == "exploratory"
    assert rs.runner_source_problems(RUNNER.read_text(encoding="utf-8"), study) == []
    assert m.BINDING.freeze_tag == rs.freeze_tag_name(m.STUDY_ID, 1)
    assert m.BINDING.study_id == study["study_id"] == STUDY_DIR.name


def test_binding_pins_the_committed_study_and_declaration():
    for rel, blob in m.BINDING.pinned_blobs.items():
        frozen = subprocess.run(
            ["git", "-C", str(REPO), "cat-file", "blob", blob],
            capture_output=True,
            check=True,
        ).stdout
        current = (REPO / rel).read_bytes()
        if rel.endswith("study.yaml"):
            assert current == frozen, rel
        else:
            assert current.startswith(frozen), rel


def test_declared_validity_and_terminals_match_study():
    study = rs.parse_study((STUDY_DIR / "study.yaml").read_bytes())
    assert tuple(study["validity_criteria"]) == m.VALIDITY_ORDER
    assert set(study["section_20_2"]["mainline_transition"]) == {
        *m.TERMINAL_BY_BLOCK.values(),
        m.SPLIT,
    }


def test_manifest_is_the_r2_stage_study_manifest():
    """Same packet, same bytes: the r2 stage study's manifest is reused unchanged."""
    ours = (STUDY_DIR / "inputs_r2.json").read_bytes()
    theirs = (
        REPO / "docs/research/studies/tf32_off_head_drift_stage_465_r2/inputs_r2.json"
    ).read_bytes()
    assert ours == theirs
    files = json.loads(ours)["files"]
    for _, arm in m.REPLAYS:
        for seq in m.SEQUENCE_FRAMES:
            assert f"l2/{arm}.evidence/{seq}.npz" in files


# ------------------------------------------------------------------- parser


def _trk(frame, tid, state=2, age=1, t2d=-1, nc=0):
    return f"TRK,{frame},{tid},{state},{age},10.0,20.0,5.0,10.0,{nc},0.1000,{t2d}"


def test_parse_segment_takes_the_block_after_the_last_gmc_line():
    text = "\n".join(
        [
            "GMC,0,0.000,0.000,0.000",  # graph warm-up inside the first call
            "GMC,1,1.000,2.000,2.236",
            _trk(1, 7, t2d=0, nc=2),
            "CND,1,7,0,0.1000,1.0,2.0,3.0,4.0",
            "CND,1,7,3,0.4000,1.0,2.0,3.0,4.0",
            _trk(1, 8),
        ]
    )
    seg = m.parse_segment(text)
    assert seg["gmc"] == [1.0, 2.0, 2.236]
    assert [t["id"] for t in seg["tracks"]] == [7, 8]
    assert seg["tracks"][0]["cands"] == [(0, 0.1), (3, 0.4)]
    assert seg["tracks"][0]["t2d"] == 0 and seg["tracks"][1]["t2d"] == -1


@pytest.mark.parametrize(
    "text",
    [
        "",
        _trk(1, 7),
        "GMC,1,0,0",
        "GMC,1,0.0,0.0,0.0\nCND,1,7,0,0.1,1,2,3,4",
        "GMC,1,0.0,0.0,0.0\n" + _trk(1, 8) + "\nCND,1,7,0,0.1,1,2,3,4",
        "GMC,1,0.0,0.0,0.0\n" + _trk(1, 8, t2d=-2),
        "GMC,1,0.0,0.0,0.0\nXYZ,1",
    ],
)
def test_parse_segment_rejects_malformed_dumps(text):
    with pytest.raises(ValueError):
        m.parse_segment(text)


# -------------------------------------------------------------- comparisons


def _row(x, y, w=10.0, h=20.0, score=0.9, cls=0.0):
    return [x, y, x + w, y + h, score, cls]


def _rows(*rs_):
    return np.array(rs_, dtype=np.float32).reshape(-1, 6)


def _t(tid, state=2, age=1, t2d=-1, cands=()):
    return {
        "id": tid,
        "state": state,
        "age": age,
        "t2d": t2d,
        "cands": list(cands),
        "pred_cxcywh": [0.0, 0.0, 1.0, 1.0],
    }


def test_det_equal_rules():
    a = _rows(_row(0, 0), _row(100, 0))
    b = _rows(_row(100.2, 0, score=0.1), _row(0.1, 0))  # reordered, scores differ
    assert m.det_equal(a, -1, b, -1)
    assert not m.det_equal(a, -1, b, 1)
    assert m.det_equal(a, 0, b, 1)  # eligible: same class, IoU >= 0.9
    assert not m.det_equal(a, 0, b, 0)
    c = _rows(_row(0, 0, cls=1.0))
    assert not m.det_equal(a, 0, c, 0)  # class differs
    assert m.det_equal(a, 5, b, 5)  # same padding slot
    assert not m.det_equal(a, 5, b, 6)
    assert not m.det_equal(a, 0, b, 5)  # real row vs padding


def test_association_compares_detections_not_indices():
    a = _rows(_row(0, 0), _row(100, 0))
    b = _rows(_row(100, 0), _row(0, 0))
    ta = [_t(1, t2d=0), _t(2, t2d=1), _t(3)]
    tb = [_t(1, t2d=1), _t(2, t2d=0), _t(3)]
    assert m.association_equal(ta, a, tb, b) is True
    tb_swap = [_t(1, t2d=0), _t(2, t2d=1), _t(3)]
    assert m.association_equal(ta, a, tb_swap, b) is False
    tb_lost = [_t(1, t2d=1), _t(2, t2d=-1), _t(3)]
    assert m.association_equal(ta, a, tb_lost, b) is False


def test_association_undefined_when_pre_step_state_differs():
    a = _rows(_row(0, 0))
    assert m.association_equal([_t(1, age=1)], a, [_t(1, age=2)], a) is None
    assert m.association_equal([_t(1)], a, [_t(1), _t(2)], a) is None


def test_duplicate_ids_match_as_a_group():
    a = _rows(_row(0, 0), _row(100, 0))
    ta = [_t(4, t2d=0), _t(4, t2d=1)]
    tb = [_t(4, t2d=1), _t(4, t2d=0)]  # order within the id group is not structure
    assert m.association_equal(ta, a, tb, a) is True
    tb_bad = [_t(4, t2d=0), _t(4, t2d=0)]
    assert m.association_equal(ta, a, tb_bad, a) is False


def test_candidate_sets_report():
    a = _rows(_row(0, 0), _row(100, 0))
    b = _rows(_row(100, 0), _row(0, 0))
    x = _t(1, cands=[(0, 0.1), (1, 0.5)])
    y = _t(1, cands=[(1, 0.2), (0, 0.6)])
    assert m.candidates_equal(x, a, y, b)
    assert not m.candidates_equal(x, a, _t(1, cands=[(1, 0.2)]), b)


# ---------------------------------------------------------- step-end state


def test_step_end_rows_label_tentative_by_join():
    active = [(1, 0, 11, 0), (2, 3, 12, 0), (2, 0, 13, 1)]
    tentative = [(2, 0, 13, 1)]
    rows = m.step_end_rows(active, tentative)
    assert rows == [(1, 2, 0, 11, 0), (2, 1, 0, 13, 1), (2, 2, 3, 12, 0)]
    assert m.state_multiset(rows) == m.Counter([(1, 2, 0), (2, 1, 0), (2, 2, 3)])


def test_step_end_rows_reject_orphan_tentative():
    with pytest.raises(ValueError):
        m.step_end_rows([(1, 0, 11, 0)], [(1, 0, 99, 0)])
    with pytest.raises(ValueError):  # one active slot cannot be tentative twice
        m.step_end_rows([(1, 0, 11, 0)], [(1, 0, 11, 0), (1, 0, 11, 0)])


# --------------------------------------------------------------- the ladder

N = 4
TI = {f: _rows(_row(0, 0), _row(100, 0)) for f in range(1, N + 1)}


def _dumps(per_frame):
    return {f: {"gmc": [0.0, 0.0, 0.0], "tracks": per_frame[f]} for f in per_frame}


def _out(ids):
    ids = np.array(ids, dtype=np.int64)
    return (ids, np.array([[0, 0, 10, 20]] * len(ids), dtype=np.float32).reshape(-1, 4))


def _base():
    """Track 1 is born at step 1, matched at step 2, coasting afterwards."""
    dump = {1: [], 2: [_t(1, age=1)], 3: [_t(1, age=1)], 4: [_t(1, age=2)]}
    dump[2][0]["t2d"] = 0
    state = {
        1: [(1, 2, 0, 1, 0)],
        2: [(1, 2, 0, 1, 0)],
        3: [(1, 2, 1, 1, 0)],
        4: [(1, 2, 2, 1, 0)],
    }
    out = {1: _out([1]), 2: _out([1]), 3: _out([1]), 4: _out([1])}
    return dump, state, out


def _ladder(a, b, ti_b=TI):
    (da, sa, oa), (db, sb, ob) = a, b
    return m.ladder(N, _dumps(da), _dumps(db), sa, sb, TI, ti_b, oa, ob)


def test_identical_replays_have_no_divergence():
    base = _base()
    r = _ladder(base, base)
    assert r["first"] is None
    assert r["divergent_frames"] == {"A": 0, "A_undefined": 0, "P": 0, "E": 0}


def test_association_divergence_comes_first_within_its_frame():
    d, st, o = _base()
    db, sb, ob = copy.deepcopy(d), copy.deepcopy(st), dict(o)
    db[2][0]["t2d"] = 1  # frame 2: matched to the other detection
    sb[2] = [(1, 2, 1, 1, 0)]  # so the step-end state differs too
    ob[2] = _out([])  # and the output too
    r = _ladder((d, st, o), (db, sb, ob))
    assert r["first"] == {"frame": 2, "block": "A"}
    assert r["first_by_block"] == {"A": 2, "P": 2, "E": 2}


def test_state_transition_is_the_step_end_snapshot():
    d, st, o = _base()
    sb = copy.deepcopy(st)
    sb[3] = sb[3] + [(9, 1, 0, 2, 0)]  # a tentative birth at step 3, not emitted
    r = _ladder((d, st, o), (d, sb, o))
    assert r["first"] == {"frame": 3, "block": "P"}
    assert r["first_by_block"]["E"] is None


def test_step_end_difference_hidden_from_the_next_dump_is_still_p():
    """Review round 1: a track that differs at step end but is deactivated by the
    next predict (age >= max_age) leaves the next dumps identical; P must still
    see it at its own step."""
    d, st, o = _base()
    sb = copy.deepcopy(st)
    sb[2] = sb[2] + [(7, 2, 29, 5, 0)]  # only in the other arm; expires next predict
    r = _ladder((d, st, o), (d, sb, o))  # dumps identical on every frame
    assert r["first"] == {"frame": 2, "block": "P"}
    assert r["divergent_frames"]["A_undefined"] == 0


def test_emission_only_divergence():
    d, st, o = _base()
    ob = dict(o)
    ob[3] = _out([1, 5])
    r = _ladder((d, st, o), (d, st, ob))
    assert r["first"] == {"frame": 3, "block": "E"}


def test_association_undefined_after_a_state_divergence():
    d, st, o = _base()
    db, sb = copy.deepcopy(d), copy.deepcopy(st)
    sb[2] = [(1, 2, 0, 1, 0), (6, 1, 0, 3, 0)]
    db[3] = [_t(1, age=1), _t(6, state=1, age=1)]
    r = _ladder((d, st, o), (db, sb, o))
    assert r["first"] == {"frame": 2, "block": "P"}
    assert r["divergent_frames"]["A_undefined"] == 1


def test_explain_first_reports_without_deciding():
    d, st, o = _base()
    db = copy.deepcopy(d)
    db[2][0]["t2d"] = 1
    lad = _ladder((d, st, o), (db, st, o))
    ex = m.explain_first(lad["first"], _dumps(d), _dumps(db), st, st, TI, TI, o, o)
    assert ex["block"] == "A" and ex["tracker_input"]["structurally_equal"]
    (rec,) = ex["association"]
    assert rec["id"] == 1 and rec["ref"]["t2d"]["index"] == 0
    assert rec["other"]["t2d"]["index"] == 1


def test_explain_first_state_transition_keys():
    d, st, o = _base()
    sb = copy.deepcopy(st)
    sb[3] = sb[3] + [(9, 1, 0, 2, 0)]
    lad = _ladder((d, st, o), (d, sb, o))
    ex = m.explain_first(lad["first"], _dumps(d), _dumps(d), st, sb, TI, TI, o, o)
    assert ex["state_transition"]["only_other"] == [[9, 1, 0]]
    assert ex["state_transition"]["association_equal_at_frame"] is True


def test_decide_five_of_seven_support():
    seqs = [f"s{i}" for i in range(7)]
    assert m.decide(dict.fromkeys(seqs, "A")) == "FIRST_DIVERGENCE_ASSOCIATION"
    five_p = {s: ("P" if i < 5 else "E") for i, s in enumerate(seqs)}
    assert m.decide(five_p) == "FIRST_DIVERGENCE_STATE_TRANSITION"
    four = {s: ("E" if i < 4 else "A") for i, s in enumerate(seqs)}
    assert m.decide(four) == m.SPLIT
    assert m.decide({s: None for s in seqs}) == m.SPLIT


# ------------------------------------------------- validity on a fake replay

SEQ = "MOT17-02-SDP"
FN = 3


def _evidence(ti_rows, out_ids):
    ev = {
        "tracker_input_frames": np.arange(1, FN + 1, dtype=np.int32),
        "tracker_input_counts": np.array([len(r) for r in ti_rows], np.int32),
        "tracker_input_rows": np.concatenate(ti_rows).astype(np.float32).reshape(-1, 6),
        "tracker_frames": np.arange(1, FN + 1, dtype=np.int32),
        "tracker_counts": np.array([len(i) for i in out_ids], np.int32),
        "tracker_ids": np.concatenate([np.array(i, np.int64) for i in out_ids]),
        "tracker_det_idx": np.concatenate(
            [np.zeros(len(i), np.int64) for i in out_ids]
        ),
        "tracker_boxes": np.concatenate(
            [np.zeros((len(i), 4), np.float32) for i in out_ids]
        ).reshape(-1, 4),
    }
    return ev


ARM_OUT = {
    "R_C_1": [[], [1], [1]],
    "R_T_1": [[], [1], [2]],
    "R_E_1": [[], [1], [1]],
}


class _FakeStudy:
    def __init__(self, tmp, corrupt=None):
        self._paths = {}
        for arm, out in ARM_OUT.items():
            ev = _evidence([_rows(_row(0, 0))] * FN, out)
            if corrupt and corrupt[0] == arm:
                corrupt[1](ev)
            p = tmp / f"{arm}.npz"
            np.savez(p, **ev)
            self._paths[f"l2/{arm}.evidence/{SEQ}.npz"] = p

    def input_members(self, name):
        return sorted(self._paths)

    def input_file(self, name, member=None):
        return self._paths[member]


def _fake_worker(mutate=None, mutate_state=None):
    """Stand-in for the replay child: writes calls.json, the dump and emits.npz."""

    def run(cmd, **kwargs):
        out_dir = Path(cmd[cmd.index("--worker-out") + 1])
        inputs = json.loads(Path(cmd[cmd.index("--worker-inputs") + 1]).read_text())
        arm = Path(inputs[SEQ]).stem
        ids = ARM_OUT[arm]
        texts = {
            1: "GMC,0,0.000,0.000,0.000\nGMC,1,0.000,0.000,0.000\n",
            2: "GMC,2,0.000,0.000,0.000\n" + _trk(2, 1, state=1, age=1, t2d=0) + "\n",
            3: "GMC,3,0.000,0.000,0.000\n" + _trk(3, 1, age=1, t2d=0) + "\n",
        }
        calls = [{"seq": SEQ, "frame": f, "rows": 1} for f in range(1, FN + 1)]
        if mutate:
            mutate(out_dir.name, texts, calls)
        raw, pos = b"", 0
        for c in calls:
            data = texts.get(c["frame"], "").encode()
            c.update(start=pos, end=pos + len(data))
            raw += data
            pos += len(data)
        (out_dir / "assoc_dump.csv").write_bytes(raw)
        (out_dir / "calls.json").write_text(json.dumps(calls))
        active = {1: [], 2: [(1, 0, 1, 0)], 3: [(1, 0, 1, 0)]}
        tentative = {1: [], 2: [(1, 0, 1, 0)], 3: []}
        if mutate_state:
            mutate_state(out_dir.name, active, tentative)
        st = {f"{SEQ}__frames": np.arange(1, FN + 1)}
        for key, src in (("active", active), ("tentative", tentative)):
            flat = [r for f in range(1, FN + 1) for r in src[f]]
            st[f"{SEQ}__{key}_counts"] = np.array(
                [len(src[f]) for f in range(1, FN + 1)]
            )
            st[f"{SEQ}__{key}_iag"] = np.array(
                [(r[0], r[1], r[3]) for r in flat], np.int64
            ).reshape(-1, 3)
            st[f"{SEQ}__{key}_uid"] = np.array([r[2] for r in flat], np.uint64)
        np.savez(out_dir / "states.npz", **st)
        np.savez(
            out_dir / "emits.npz",
            **{
                f"{SEQ}__frames": np.arange(1, FN + 1),
                f"{SEQ}__counts": np.array([len(i) for i in ids]),
                f"{SEQ}__ids": np.concatenate([np.array(i, np.int64) for i in ids]),
                f"{SEQ}__det_idx": np.concatenate(
                    [np.zeros(len(i), np.int64) for i in ids]
                ),
                f"{SEQ}__boxes": np.zeros((sum(map(len, ids)), 4), np.float32),
            },
        )
        return subprocess.CompletedProcess(cmd, 0)

    return run


def _evaluate(monkeypatch, tmp_path, mutate=None, corrupt=None, mutate_state=None):
    monkeypatch.setattr(m, "SEQUENCE_FRAMES", {SEQ: FN})
    monkeypatch.setattr(m.subprocess, "run", _fake_worker(mutate, mutate_state))
    raw = tmp_path / "raw"
    raw.mkdir()
    return m.evaluate(_FakeStudy(tmp_path, corrupt), raw)


def test_fake_replay_runs_end_to_end(monkeypatch, tmp_path):
    by_run = _evaluate(monkeypatch, tmp_path)
    assert by_run["R_T#1"][SEQ]["first"] == {"frame": 3, "block": "E"}
    assert by_run["R_E"][SEQ]["first"] is None
    assert by_run["R_T#1"][SEQ]["gmc_row_mismatch_frames"] == 0


def test_missing_member_is_v_complete(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "SEQUENCE_FRAMES", {SEQ: FN})
    study = _FakeStudy(tmp_path)
    study._paths.pop(f"l2/R_E_1.evidence/{SEQ}.npz")
    with pytest.raises(m.Invalid) as exc:
        m.evaluate(study, tmp_path)
    assert exc.value.criterion == "V_COMPLETE"


def test_broken_ragged_counts_are_v_format(monkeypatch, tmp_path):
    def corrupt(ev):
        ev["tracker_input_counts"] = ev["tracker_input_counts"] + 1

    with pytest.raises(m.Invalid) as exc:
        _evaluate(monkeypatch, tmp_path, corrupt=("R_T_1", corrupt))
    assert exc.value.criterion == "V_FORMAT"


def test_replay_output_mismatch_is_v_replay(monkeypatch, tmp_path):
    monkeypatch.setitem(ARM_OUT, "R_E_1", [[], [1], [1]])
    real = m.r2_tracker_full

    def shifted(ev, n):
        out = real(ev, n)
        ids, det, box = out[3]
        out[3] = (ids, det, box + 1.0)
        return out

    monkeypatch.setattr(m, "r2_tracker_full", shifted)
    with pytest.raises(m.Invalid) as exc:
        _evaluate(monkeypatch, tmp_path)
    assert exc.value.criterion == "V_REPLAY"


def test_missing_tracker_call_is_v_replay(monkeypatch, tmp_path):
    def drop(run, texts, calls):
        calls.pop()

    with pytest.raises(m.Invalid) as exc:
        _evaluate(monkeypatch, tmp_path, mutate=drop)
    assert exc.value.criterion == "V_REPLAY"


def test_malformed_dump_is_v_record(monkeypatch, tmp_path):
    def garble(run, texts, calls):
        texts[2] = texts[2] + "CND,2,9,0,0.1,1,2,3,4\n"

    with pytest.raises(m.Invalid) as exc:
        _evaluate(monkeypatch, tmp_path, mutate=garble)
    assert exc.value.criterion == "V_RECORD"


def test_orphan_tentative_snapshot_is_v_record(monkeypatch, tmp_path):
    def orphan(run, active, tentative):
        tentative[3] = [(4, 0, 44, 0)]

    with pytest.raises(m.Invalid) as exc:
        _evaluate(monkeypatch, tmp_path, mutate_state=orphan)
    assert exc.value.criterion == "V_RECORD"


@pytest.mark.parametrize("run", ["R_C_2", "R_T_2"])
def test_reference_and_primary_dumps_must_repeat(monkeypatch, tmp_path, run):
    def drift(run_, texts, calls):
        if run_ == run:
            texts[3] = texts[3].replace("10.0,20.0", "10.1,20.0")

    with pytest.raises(m.Invalid) as exc:
        _evaluate(monkeypatch, tmp_path, mutate=drift)
    assert exc.value.criterion == "V_REPEAT"


@pytest.mark.parametrize("run", ["R_C_2", "R_T_2"])
def test_reference_and_primary_step_end_states_must_repeat(monkeypatch, tmp_path, run):
    """Review round 1: a second R_T path with the same output but a different
    internal state must fail V_REPEAT, not pass on V_REPLAY alone."""

    def drift(run_, active, tentative):
        if run_ == run:
            active[3] = [(1, 0, 1, 0), (5, 1, 9, 0)]

    with pytest.raises(m.Invalid) as exc:
        _evaluate(monkeypatch, tmp_path, mutate_state=drift)
    assert exc.value.criterion == "V_REPEAT"


def test_replays_cover_two_runs_of_reference_and_primary():
    runs = [r for r, _ in m.REPLAYS]
    assert set(m.REPEAT_PAIRS) == {("R_C#1", "R_C#2"), ("R_T#1", "R_T#2")}
    assert all(a in runs and b in runs for a, b in m.REPEAT_PAIRS)
    assert m.PRIMARY == "R_T#1" and m.REF_RUN == "R_C#1"


# ------------------------------------------------------- refusal before data


def test_main_refuses_without_a_lease(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "held_lease", lambda: None)
    attempts = STUDY_DIR / "attempts"
    before = sorted(attempts.iterdir()) if attempts.is_dir() else []
    assert m.main(["--raw-out", str(tmp_path / "raw")]) == 2
    assert not (tmp_path / "raw").exists()
    after = sorted(attempts.iterdir()) if attempts.is_dir() else []
    assert after == before


def test_main_refuses_before_any_data_without_a_freeze(monkeypatch, tmp_path):
    tag = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "rev-parse",
            "--verify",
            "--quiet",
            f"refs/tags/{m.BINDING.freeze_tag}^{{commit}}",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    if tag and tag == head:
        pytest.skip("HEAD is the freeze commit; the refusal path is not reachable here")
    monkeypatch.setattr(m, "held_lease", lambda: {"resource": "gpu0", "pid": 1})
    monkeypatch.delenv(m.DUMP_ENV, raising=False)
    attempts = STUDY_DIR / "attempts"
    before = sorted(attempts.iterdir()) if attempts.is_dir() else []
    with pytest.raises(rs.StudyError):
        m.main(["--raw-out", str(tmp_path / "raw")])
    assert not (tmp_path / "raw").exists()
    after = sorted(attempts.iterdir()) if attempts.is_dir() else []
    assert after == before
