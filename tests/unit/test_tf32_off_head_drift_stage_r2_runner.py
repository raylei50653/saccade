"""The TF32-off stage-localization r2 runner implements its exploratory declaration.

``scripts/eval/diagnostics/tf32_off_head_drift_stage_r2.py`` implements
``docs/research/studies/tf32_off_head_drift_stage_465_r2/``. These tests pin (a)
the runner to its study (study.yaml names it, the binding pins the committed
study.yaml and declaration, the static freeze check passes), (b) the §2
structural comparisons, including the inversion-signature rule that r2 adds
over r1, the f* label and the §4 terminal rule on synthetic arrays, (c) the
txt parser, (d) that inverted boxes pass V_FORMAT, and (e) that without a
freeze the runner stops before any data path exists. No test reads a frame.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "eval" / "diagnostics" / "tf32_off_head_drift_stage_r2.py"
STUDY_DIR = REPO / "docs" / "research" / "studies" / "tf32_off_head_drift_stage_465_r2"


def _load():
    spec = importlib.util.spec_from_file_location(
        "tf32_off_head_drift_stage_r2", RUNNER
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
    assert rs.runner_source_problems(RUNNER.read_text(encoding="utf-8"), study) == []
    assert m.BINDING.freeze_tag == rs.freeze_tag_name(m.STUDY_ID, 1)
    assert m.BINDING.study_id == study["study_id"] == STUDY_DIR.name


def test_binding_pins_the_committed_study_and_declaration():
    """A declaration edit without re-pinning (before the freeze) fails here;
    after the freeze only appends below the pinned body are allowed."""
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


def test_declared_validity_order_matches_study():
    study = rs.parse_study((STUDY_DIR / "study.yaml").read_bytes())
    assert tuple(study["validity_criteria"]) == m.VALIDITY_ORDER
    assert set(study["section_20_2"]["mainline_transition"]) == {
        *m.TERMINAL_BY_LABEL.values(),
        m.SPLIT,
    }


# -------------------------------------------------------------- comparisons


def _row(x, y, w=10.0, h=20.0, score=0.9, cls=0.0):
    return [x, y, x + w, y + h, score, cls]


def test_rows_structural_equality():
    a = np.asarray([_row(0, 0), _row(100, 100)], np.float32)
    assert m.rows_structurally_equal(a, a.copy())
    assert m.rows_structurally_equal(a, a[::-1].copy())  # order is not structure
    b = a.copy()
    b[:, 4] -= 0.5  # scores are not compared
    assert m.rows_structurally_equal(a, b)
    c = a.copy()
    c[0, :4] += [0.1, 0.1, 0.1, 0.1]  # IoU well above 0.9
    assert m.rows_structurally_equal(a, c)
    d = a.copy()
    d[0, 5] = 1.0  # class flip
    assert not m.rows_structurally_equal(a, d)
    e = a.copy()
    e[0, :4] += [5, 5, 5, 5]  # IoU far below 0.9
    assert not m.rows_structurally_equal(a, e)
    assert not m.rows_structurally_equal(a, a[:1])
    empty = np.zeros((0, 6), np.float32)
    assert m.rows_structurally_equal(empty, empty)


def test_structural_equality_is_threshold_feasible_matching_not_max_total_iou():
    """Review counterexample: the diagonal pairing is all >= 0.9 (so the rows
    are structurally equal by the declaration), but the max-total-IoU pairing
    takes a 1.0 pair and leaves a 0.887 pair."""
    a = np.asarray([[0, 0, 100, 100, 0.9, 0], [-5, 0, 95, 100, 0.9, 0]], np.float32)
    b = np.asarray([[1, 0, 101, 100, 0.9, 0], [0, 0, 100, 100, 0.9, 0]], np.float32)
    iou = m.iou_matrix(a[:, :4], b[:, :4])
    assert iou[0, 0] >= 0.9 and iou[1, 1] >= 0.9  # a feasible perfect matching
    assert iou[0, 1] + iou[1, 0] > iou[0, 0] + iou[1, 1]  # max-total picks the other
    assert iou[1, 0] < 0.9
    assert m.rows_structurally_equal(a, b)
    assert m.rows_structurally_equal(b, a)
    b_far = b.copy()
    b_far[1, :4] += [10, 0, 10, 0]  # a2 now has no eligible partner at all
    assert not m.rows_structurally_equal(a, b_far)


def test_inversion_signature_must_match():
    """Declaration §2 examples: sorting alone would equate a normal box with
    its left-right flip, which the tracker sees as zero width."""
    normal = np.asarray([[0, 0, 10, 10, 0.9, 0]], np.float32)
    flipped = np.asarray([[10, 0, 0, 10, 0.9, 0]], np.float32)
    assert m.iou_matrix(normal[:, :4], flipped[:, :4])[0, 0] == 1.0  # canonical
    assert not m.rows_structurally_equal(normal, flipped)
    assert m.rows_structurally_equal(flipped, flipped.copy())
    assert m.inversion_signature(flipped[:, :4]).tolist() == [[True, False]]
    y_flip = np.asarray([[0, 10, 10, 0, 0.9, 0]], np.float32)
    assert not m.rows_structurally_equal(flipped, y_flip)
    near = flipped.copy()
    near[0, :4] += [0.1, 0, 0.1, 0]  # same signature, canonical IoU > 0.9
    assert m.rows_structurally_equal(flipped, near)


def test_identical_zero_area_boxes_are_eligible():
    line = np.asarray([[5, 0, 5, 10, 0.9, 0]], np.float32)  # zero width
    assert m.iou_matrix(line[:, :4], line[:, :4])[0, 0] == 1.0
    assert m.rows_structurally_equal(line, line.copy())
    shifted = line.copy()
    shifted[0, [0, 2]] += 1
    assert m.iou_matrix(line[:, :4], shifted[:, :4])[0, 0] == 0.0
    assert not m.rows_structurally_equal(line, shifted)


def test_inverted_tracks_compare_by_signature():
    ids = np.asarray([1], np.int64)
    normal = np.asarray([[0, 0, 10, 20]], np.float32)
    flipped = np.asarray([[10, 0, 0, 20]], np.float32)
    assert not m.tracks_structurally_equal((ids, normal), (ids, flipped))
    assert m.tracks_structurally_equal((ids, flipped), (ids, flipped.copy()))


def test_inversion_census_and_counts():
    boxes = np.asarray(
        [[0, 0, 10, 10], [10, 0, 0, 10], [0, 10, 10, 0], [10, 10, 0, 0]], np.float32
    )
    assert m.inversion_census(boxes) == {
        "rows": 4,
        "x_inverted_rows": 2,
        "y_inverted_rows": 2,
        "either_inverted_rows": 3,
    }
    assert m.inversion_counts(boxes) == (1, 1, 1)
    assert m.inversion_census(np.zeros((0, 4), np.float32))["rows"] == 0


def test_duplicate_track_ids_match_per_id_not_by_position():
    """Review blocker: an id emitted twice in one frame is matched as a k x k
    problem; box order within the id group is not structure."""
    ids = np.asarray([4, 4, 1], np.int64)
    boxes = np.asarray(
        [[0, 0, 10, 20], [100, 0, 110, 20], [50, 50, 60, 70]], np.float32
    )
    swapped = boxes[[1, 0, 2]]
    assert m.tracks_structurally_equal((ids, boxes), (ids, swapped))
    moved = swapped.copy()
    moved[0] += [0, 40, 0, 40]  # one id-4 box has no eligible partner any more
    assert not m.tracks_structurally_equal((ids, boxes), (ids, moved))
    flipped = swapped.copy()
    flipped[1, [0, 2]] = flipped[1, [2, 0]]  # same rectangle, other signature
    assert not m.tracks_structurally_equal((ids, boxes), (ids, flipped))


def test_tracks_structural_equality():
    ids = np.asarray([3, 1], np.int64)
    boxes = np.asarray([[0, 0, 10, 20], [50, 50, 60, 70]], np.float32)
    assert m.tracks_structurally_equal((ids, boxes), (ids[::-1], boxes[::-1]))
    assert not m.tracks_structurally_equal((ids, boxes), (np.asarray([3, 2]), boxes))
    moved = boxes.copy()
    moved[1] += 8
    assert not m.tracks_structurally_equal((ids, boxes), (ids, moved))
    assert not m.tracks_bit_equal((ids, boxes), (ids[::-1], boxes[::-1]))


def test_ragged_and_absent_frames():
    rows = np.asarray([_row(0, 0), _row(1, 1), _row(2, 2)], np.float32)
    got = m.ragged_by_frame(np.asarray([2, 4]), np.asarray([1, 2]), rows)
    assert [len(got[2]), len(got[4])] == [1, 2]
    with pytest.raises(ValueError):
        m.ragged_by_frame(np.asarray([2]), np.asarray([1]), rows)
    ev = {"s_frames": np.asarray([2]), "s_counts": np.asarray([1]), "s_rows": rows[:1]}
    view = m.stage_view(ev, "s", 3)
    assert [len(view[f]) for f in (1, 2, 3)] == [0, 1, 0]


def _evidence(n, ti_rows, tracks):
    """Synthetic arm evidence: every probe stage carries ``ti_rows[f]``."""
    ev = {"rows": np.zeros((n, 1, 6), np.float32)}
    frames = sorted(ti_rows)
    body = [np.asarray(ti_rows[f], np.float32).reshape(-1, 6) for f in frames]
    for stage in m.PROBE_STAGES:
        ev[f"{stage}_frames"] = np.asarray(frames, np.int32)
        ev[f"{stage}_counts"] = np.asarray([len(b) for b in body], np.int32)
        ev[f"{stage}_rows"] = (
            np.concatenate(body) if body else np.zeros((0, 6), np.float32)
        )
    tf = sorted(tracks)
    ev["tracker_frames"] = np.asarray(tf, np.int32)
    ev["tracker_counts"] = np.asarray([len(tracks[f][0]) for f in tf], np.int32)
    ev["tracker_ids"] = np.concatenate([np.asarray(tracks[f][0], np.int64) for f in tf])
    ev["tracker_boxes"] = np.concatenate(
        [np.asarray(tracks[f][1], np.float32).reshape(-1, 4) for f in tf]
    )
    return ev


N = 4
DETS = {f: [_row(0, 0), _row(100, 100)] for f in range(1, N + 1)}
TRACKS = {f: ([1, 2], [[0, 0, 10, 20], [100, 100, 110, 120]]) for f in range(1, N + 1)}


def test_label_same_when_inputs_match_structurally():
    other_dets = {
        f: [[v + 0.01 if i < 4 else v for i, v in enumerate(r)] for r in d]
        for f, d in DETS.items()
    }
    other_tracks = dict(TRACKS)
    other_tracks[3] = ([1, 5], TRACKS[3][1])  # id switch at frame 3
    r = m.compare_sequence(
        _evidence(N, DETS, TRACKS), _evidence(N, other_dets, other_tracks), N
    )
    assert r["stages"]["tracker_input"]["first_bit"] == 1
    assert r["stages"]["tracker_input"]["first_structural"] is None
    assert r["tracker"]["first_structural"] == 3
    assert r["label_at_f_star"] == "same"
    assert r["at_f_star"]["tracker_input_structural_frames_before"] == 0


def test_label_different_when_tracker_input_changes_at_f_star():
    other_dets = dict(DETS)
    other_dets[2] = DETS[2][:1]
    other_tracks = dict(TRACKS)
    other_tracks[2] = ([1], [TRACKS[2][1][0]])
    r = m.compare_sequence(
        _evidence(N, DETS, TRACKS), _evidence(N, other_dets, other_tracks), N
    )
    assert r["tracker"]["first_structural"] == 2
    assert r["at_f_star"]["structural_differs"]["tracker_input"] is True
    assert r["label_at_f_star"] == "different"


def test_no_structural_divergence_has_no_label():
    r = m.compare_sequence(_evidence(N, DETS, TRACKS), _evidence(N, DETS, TRACKS), N)
    assert r["tracker"]["first_bit"] is None
    assert r["label_at_f_star"] is None and r["at_f_star"] is None


def test_decide_five_of_seven_support():
    seqs = list(m.SEQUENCE_FRAMES)
    five_same = dict.fromkeys(seqs, "same")
    five_same.update({seqs[0]: "different", seqs[1]: None})
    assert m.decide(five_same) == "TRACKER_INPUT_SAME_AT_FSTAR"
    five_different = dict.fromkeys(seqs, "different")
    five_different.update({seqs[0]: "same", seqs[1]: "same"})
    assert m.decide(five_different) == "TRACKER_INPUT_DIFFERENT_AT_FSTAR"
    four = dict.fromkeys(seqs[:4], "same") | dict.fromkeys(seqs[4:], "different")
    assert m.decide(four) == m.SPLIT
    assert m.decide(dict.fromkeys(seqs)) == m.SPLIT


def test_parse_mot_txt_and_compare():
    ref = m.parse_mot_txt(b"1,7,10,20,5,10,1,-1,-1,-1\n2,7,11,20,5,10,1,-1,-1,-1\n")
    assert ref[1][0].tolist() == [7]
    assert ref[1][1].tolist() == [[10, 20, 15, 30]]
    other = m.parse_mot_txt(b"1,7,10,20,5,10,1,-1,-1,-1\n2,8,11,20,5,10,1,-1,-1,-1\n")
    assert m.compare_txt(ref, other, 3) == {"first_bit": 2, "first_structural": 2}
    assert m.compare_txt(ref, ref, 3) == {"first_bit": None, "first_structural": None}
    with pytest.raises(ValueError):
        m.parse_mot_txt(b"1,7,10,20\n")
    flipped = m.parse_mot_txt(b"1,7,10,20,-5,10\n")  # negative w is legal in r2
    assert flipped[1][1].tolist() == [[10, 20, 5, 30]]
    with pytest.raises(ValueError):
        m.parse_mot_txt(b"1,7,10,20,nan,10\n")


# ------------------------------------------------------------ freeze first


def test_main_refuses_before_any_data_without_a_freeze(monkeypatch):
    """Without the freeze tag at HEAD the runner raises before a handle exists,
    so no attempt directory is created and no input is read."""
    attempts = STUDY_DIR / "attempts"
    before = sorted(attempts.iterdir()) if attempts.is_dir() else []
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
    with pytest.raises(rs.StudyError):
        m.main()
    after = sorted(attempts.iterdir()) if attempts.is_dir() else []
    assert after == before


# ------------------------------------------------------------ V_FORMAT order


class _FakeStudy:
    def __init__(self, members):
        self._members = members

    def read_input(self, name, member=None):
        return self._members[(name, member)]


def _npz(ev):
    buf = io.BytesIO()
    np.savez(buf, **ev)
    return buf.getvalue()


def _fake_members(monkeypatch, corrupt=None, txt=b"1,1,0,0,10,20,1,-1,-1,-1\n"):
    monkeypatch.setattr(m, "SEQUENCE_FRAMES", {"MOT17-02-SDP": N})
    members = {}
    for arm in (m.REF_ARM, *m.ARMS):
        for run in m.RUNS:
            ev = _evidence(N, DETS, TRACKS)
            if corrupt is not None and corrupt[0] in ((arm, run), arm):
                corrupt[1](ev)
            members[("r2", f"l2/{arm}_{run}.evidence/MOT17-02-SDP.npz")] = _npz(ev)
    for name, directory, seq in m._txt_keys():
        members[(name, f"{directory}/{seq}.txt")] = txt
    return _FakeStudy(members)


def test_stage_comparisons_run_on_well_formed_evidence(monkeypatch):
    by_arm, txt, census = m.evaluate(_fake_members(monkeypatch))
    assert set(by_arm) == set(m.ARMS)
    assert set(txt) == {label for label, _, _ in m.TXT_PAIRS}
    assert by_arm["R_T"]["MOT17-02-SDP"]["label_at_f_star"] is None
    assert set(census) == {f"{a}_{r}" for a in (m.REF_ARM, *m.ARMS) for r in m.RUNS}
    assert census["R_C_1"]["MOT17-02-SDP"]["tracker_input"]["either_inverted_rows"] == 0


def test_inverted_boxes_pass_v_format_and_are_counted(monkeypatch):
    """r1 failed V_FORMAT on inverted xyxy; r2 declares them well formed."""

    def invert(ev):
        for stage in m.PROBE_STAGES:
            rows = ev[f"{stage}_rows"].copy()
            rows[0, [0, 2]] = rows[0, [2, 0]]
            ev[f"{stage}_rows"] = rows

    by_arm, _, census = m.evaluate(_fake_members(monkeypatch, ("R_T", invert)))
    for run in m.RUNS:
        entry = census[f"R_T_{run}"]["MOT17-02-SDP"]
        assert entry["detector_output"]["x_inverted_rows"] == 1
    assert census["R_C_1"]["MOT17-02-SDP"]["detector_output"]["x_inverted_rows"] == 0
    stages = by_arm["R_T"]["MOT17-02-SDP"]["stages"]
    assert stages["tracker_input"]["first_structural"] == 1  # flip != normal
    assert stages["tracker_input"]["inverted_signature_count_mismatch_frames"] == 1


def test_non_finite_coordinates_are_v_format(monkeypatch):
    def poison(ev):
        rows = ev["post_nms_rows"].copy()
        rows[0, 0] = np.nan
        ev["post_nms_rows"] = rows

    with pytest.raises(m.Invalid) as err:
        m.evaluate(_fake_members(monkeypatch, (("H_V", 2), poison)))
    assert err.value.criterion == "V_FORMAT"


@pytest.mark.parametrize("stage", m.PROBE_STAGES)
def test_broken_ragged_counts_in_any_probe_stage_are_v_format(monkeypatch, stage):
    def corrupt(ev):
        ev[f"{stage}_counts"] = ev[f"{stage}_counts"].copy()
        ev[f"{stage}_counts"][0] += 1  # counts no longer cover the body

    study = _fake_members(monkeypatch, (("R_T", 1), corrupt))
    with pytest.raises(m.Invalid) as err:
        m.evaluate(study)
    assert err.value.criterion == "V_FORMAT"


def test_validity_phases_run_in_declared_order(monkeypatch):
    """A missing txt is V_COMPLETE even when R_C also fails its self-check,
    and a malformed txt is V_FORMAT even when R_C fails its self-check."""

    def break_ref(ev):
        ev["tracker_boxes"] = ev["tracker_boxes"] + 0.5

    study = _fake_members(monkeypatch, (("R_C", 2), break_ref))
    with pytest.raises(m.Invalid) as err:
        m.evaluate(study)
    assert err.value.criterion == "V_REF_SELF"

    missing = _fake_members(monkeypatch, (("R_C", 2), break_ref))
    del missing._members[next(k for k in missing._members if k[1].endswith(".txt"))]
    with pytest.raises(m.Invalid) as err:
        m.evaluate(missing)
    assert err.value.criterion == "V_COMPLETE"

    bad_txt = _fake_members(monkeypatch, (("R_C", 2), break_ref), txt=b"1,1,0,0\n")
    with pytest.raises(m.Invalid) as err:
        m.evaluate(bad_txt)
    assert err.value.criterion == "V_FORMAT"
