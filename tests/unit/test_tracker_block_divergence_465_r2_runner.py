"""The tracker block-divergence r2 runner implements its exploratory declaration.

``scripts/eval/diagnostics/tracker_block_divergence_465_r2.py`` implements
``docs/research/studies/tracker_block_divergence_465_r2/``. r2 differs from r1
only in ``V_REPEAT``. These tests pin (a) the runner to its study and to the
reused r1 code (binding blobs), (b) that the study and declaration differ from
r1 only where the declaration says, (c) the canonical form on synthetic dump
segments (multiset with multiplicities, full line bytes, never across a TRK or
GMC line), (d) the V_REPEAT phase on a fake replay, including that the cross-arm
comparison still reads the raw dumps, and (e) that without a lease or a freeze
the runner stops before any data path exists.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNER = (
    REPO / "scripts" / "eval" / "diagnostics" / "tracker_block_divergence_465_r2.py"
)
STUDY_DIR = REPO / "docs" / "research" / "studies" / "tracker_block_divergence_465_r2"
R1_DIR = REPO / "docs" / "research" / "studies" / "tracker_block_divergence_465"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load("tracker_block_divergence_465_r2", RUNNER)
rs = sys.modules["research_study"]
# r1's fake replay fixtures (synthetic, no GPU, no frame), reused as-is.
t1 = _load(
    "r1_runner_tests",
    REPO / "tests" / "unit" / "test_tracker_block_divergence_465_runner.py",
)


def _git_blob(rel):
    return subprocess.run(
        ["git", "-C", str(REPO), "hash-object", rel],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# ------------------------------------------------------------------ binding


def test_study_names_this_runner_and_passes_static_check():
    study = rs.parse_study((STUDY_DIR / "study.yaml").read_bytes())
    assert study["runner"] == RUNNER.relative_to(REPO).as_posix()
    assert rs.study_schema_problems(study, m.STUDY_ID) == []
    assert rs.derive_tier(study["section_20_2"]) == "exploratory"
    assert rs.runner_source_problems(RUNNER.read_text(encoding="utf-8"), study) == []
    assert m.BINDING.freeze_tag == rs.freeze_tag_name(m.STUDY_ID, 1)
    assert m.BINDING.study_id == study["study_id"] == STUDY_DIR.name


def test_binding_pins_the_committed_study_declaration_and_reused_code():
    pinned = m.BINDING.pinned_blobs
    assert set(pinned) == {
        f"docs/research/studies/{m.STUDY_ID}/study.yaml",
        f"docs/research/studies/{m.STUDY_ID}/declaration.md",
        m.R1_RUNNER,
        m.S2_RUNNER,
    }
    for rel, blob in pinned.items():
        frozen = subprocess.run(
            ["git", "-C", str(REPO), "cat-file", "blob", blob],
            capture_output=True,
            check=True,
        ).stdout
        current = (REPO / rel).read_bytes()
        if rel.endswith("declaration.md"):
            assert current.startswith(frozen), rel
        else:
            assert current == frozen, rel


def test_r1_is_closed_and_untouched():
    """r1's declaration, study and attempt stay as frozen; it records no terminal."""
    assert (
        _git_blob(R1_DIR / "study.yaml") == "c0a1cb34b82a13e802ba50a293b1d964d58fbdfc"
    )
    assert (
        _git_blob(R1_DIR / "declaration.md")
        == "49e3eca0880cd9b87223846e3e5645d7a7c6776d"
    )
    attempt = json.loads((R1_DIR / "attempts/001/attempt.json").read_text())
    assert attempt["validity"] == "invalid" and attempt["terminal"] is None
    assert attempt["invalid_criterion"] == "V_REPEAT"
    assert sorted(p.name for p in (R1_DIR / "attempts").iterdir()) == ["001"]


def test_study_differs_from_r1_only_in_identity_and_v_repeat():
    ours = rs.parse_study((STUDY_DIR / "study.yaml").read_bytes())
    theirs = rs.parse_study((R1_DIR / "study.yaml").read_bytes())
    changed = {k for k in ours.keys() | theirs.keys() if ours.get(k) != theirs.get(k)}
    assert changed == {"study_id", "runner", "validity_criteria"}
    vo, vt = ours["validity_criteria"], theirs["validity_criteria"]
    assert list(vo) == list(vt) == list(m.VALIDITY_ORDER)
    assert {k for k in vo if vo[k] != vt[k]} == {"V_REPEAT"}


def _sections(text, start, end):
    i = text.index(start)
    return text[i : text.index(end, i)]


@pytest.mark.parametrize(
    "start,end",
    [("## 0. 承接", "## 3. Validity"), ("## 4. Terminal", "## 7. 事前看過的資料")],
)
def test_declaration_keeps_r1_sections_byte_for_byte(start, end):
    ours = (STUDY_DIR / "declaration.md").read_text(encoding="utf-8")
    theirs = (R1_DIR / "declaration.md").read_text(encoding="utf-8")
    assert _sections(ours, start, end) == _sections(theirs, start, end)


def test_manifest_is_r1s_manifest():
    assert (STUDY_DIR / "inputs_r2.json").read_bytes() == (
        R1_DIR / "inputs_r2.json"
    ).read_bytes()


def test_inference_surface_is_r1s():
    assert m.REPLAYS == m.r1.REPLAYS and m.REPEAT_PAIRS == m.r1.REPEAT_PAIRS
    assert m.decide is m.r1.decide and m.Invalid is m.r1.Invalid
    assert (m.IOU_MIN, m.SUPPORT_MIN) == (0.9, 5)
    assert m.TERMINAL_BY_BLOCK == m.r1.TERMINAL_BY_BLOCK


# ----------------------------------------------------------- canonical form

GMC = b"GMC,3,0.000,0.000,0.000"
TRK1 = b"TRK,3,1,2,1,10.0,20.0,5.0,10.0,2,0.1000,0"
TRK2 = b"TRK,3,2,2,1,50.0,20.0,5.0,10.0,1,0.3000,1"
C1A = b"CND,3,1,0,0.1000,1.0,2.0,3.0,4.0"
C1B = b"CND,3,1,1,0.4000,5.0,2.0,3.0,4.0"
C2A = b"CND,3,2,1,0.3000,5.0,2.0,3.0,4.0"


def _seg(*lines):
    return b"\n".join(lines) + b"\n"


def _same(a, b):
    return m.canonical_segment(a) == m.canonical_segment(b)


def test_cnd_order_within_a_track_is_not_structure():
    assert _same(
        _seg(GMC, TRK1, C1A, C1B, TRK2, C2A), _seg(GMC, TRK1, C1B, C1A, TRK2, C2A)
    )


def test_cnd_row_content_is_compared_in_full():
    changed_cost = C1B.replace(b"0.4000", b"0.4001")
    changed_box = C1B.replace(b"5.0,2.0", b"5.1,2.0")
    for other in (changed_cost, changed_box):
        assert not _same(_seg(GMC, TRK1, C1A, C1B), _seg(GMC, TRK1, C1A, other))


def test_cnd_multiplicities_count():
    assert not _same(_seg(GMC, TRK1, C1A, C1A, C1B), _seg(GMC, TRK1, C1A, C1B, C1B))
    assert not _same(_seg(GMC, TRK1, C1A, C1B), _seg(GMC, TRK1, C1A, C1A, C1B))


def test_cnd_rows_never_move_across_a_trk_or_gmc_line():
    a = _seg(GMC, TRK1, C1A, TRK2, C2A, C1B)
    b = _seg(GMC, TRK1, C1A, C1B, TRK2, C2A)
    assert not _same(a, b)
    warm = b"GMC,0,0.000,0.000,0.000"
    assert not _same(_seg(warm, C1A, GMC, C1B), _seg(warm, C1B, GMC, C1A))


def test_other_lines_keep_bytes_and_position():
    assert not _same(_seg(GMC, TRK1, C1A, TRK2), _seg(GMC, TRK2, TRK1, C1A))
    assert not _same(_seg(GMC, TRK1), _seg(GMC, TRK1[:-1] + b"1"))  # trk_to_det
    assert not _same(_seg(GMC, TRK1), _seg(GMC.replace(b"0.000", b"0.001", 1), TRK1))
    assert not _same(_seg(GMC, TRK1), _seg(GMC, TRK1)[:-1])  # trailing remainder


# ---------------------------------------------- V_REPEAT on a fake replay

SEQ, FN = t1.SEQ, t1.FN
CND_FRAME3 = [
    "CND,3,1,0,0.1000,1.0,2.0,3.0,4.0",
    "CND,3,1,0,0.2000,1.0,2.0,3.0,4.0",
]


def _with_cands(order=None, override=None):
    """Give track 1 two candidate rows at frame 3; per-run order or content."""

    def mutate(run, texts, calls):
        rows = list(CND_FRAME3)
        if override and run in override:
            rows = override[run]
        elif order and run in order:
            rows = [rows[i] for i in order[run]]
        texts[3] = texts[3] + "\n".join(rows) + "\n"

    return mutate


def _evaluate(monkeypatch, tmp_path, mutate=None, mutate_state=None):
    monkeypatch.setattr(m, "SEQUENCE_FRAMES", {SEQ: FN})
    monkeypatch.setattr(m.r1, "SEQUENCE_FRAMES", {SEQ: FN})
    monkeypatch.setattr(m.subprocess, "run", t1._fake_worker(mutate, mutate_state))
    raw = tmp_path / "raw"
    raw.mkdir()
    return m.evaluate(t1._FakeStudy(tmp_path), raw)


def test_fake_replay_runs_end_to_end(monkeypatch, tmp_path):
    by_run = _evaluate(monkeypatch, tmp_path, mutate=_with_cands())
    assert by_run["R_T#1"][SEQ]["first"] == {"frame": 3, "block": "E"}
    assert by_run["R_E"][SEQ]["first"] is None


@pytest.mark.parametrize("run", ["R_C_2", "R_T_2"])
def test_cnd_order_within_a_track_passes_v_repeat(monkeypatch, tmp_path, run):
    """Attempt 001's failure shape: only the race-ordered CND slot order differs."""
    by_run = _evaluate(monkeypatch, tmp_path, mutate=_with_cands(order={run: (1, 0)}))
    assert by_run["R_T#1"][SEQ]["first"] == {"frame": 3, "block": "E"}


@pytest.mark.parametrize("run", ["R_C_2", "R_T_2"])
@pytest.mark.parametrize(
    "rows",
    [
        [CND_FRAME3[0], CND_FRAME3[1].replace("0.2000", "0.2001")],  # content
        [CND_FRAME3[0], CND_FRAME3[0]],  # multiplicity
        [CND_FRAME3[0]],  # a row missing
    ],
)
def test_cnd_multiset_change_is_v_repeat(monkeypatch, tmp_path, run, rows):
    with pytest.raises(m.Invalid) as exc:
        _evaluate(monkeypatch, tmp_path, mutate=_with_cands(override={run: rows}))
    assert exc.value.criterion == "V_REPEAT"


@pytest.mark.parametrize("run", ["R_C_2", "R_T_2"])
def test_trk_drift_is_still_v_repeat(monkeypatch, tmp_path, run):
    base = _with_cands()

    def drift(run_, texts, calls):
        base(run_, texts, calls)
        if run_ == run:
            texts[3] = texts[3].replace("10.0,20.0", "10.1,20.0", 1)

    with pytest.raises(m.Invalid) as exc:
        _evaluate(monkeypatch, tmp_path, mutate=drift)
    assert exc.value.criterion == "V_REPEAT"


@pytest.mark.parametrize("run", ["R_C_2", "R_T_2"])
def test_step_end_state_drift_is_still_v_repeat(monkeypatch, tmp_path, run):
    def drift(run_, active, tentative):
        if run_ == run:
            active[3] = [(1, 0, 1, 0), (5, 1, 9, 0)]

    with pytest.raises(m.Invalid) as exc:
        _evaluate(monkeypatch, tmp_path, mutate=_with_cands(), mutate_state=drift)
    assert exc.value.criterion == "V_REPEAT"


def test_cross_arm_comparison_reads_raw_dumps(monkeypatch, tmp_path):
    """The canonical form is used inside V_REPEAT only; the ladder sees raw rows."""
    seen = []
    real = m.r1.ladder

    def spy(n, dump_ref, dump_oth, *rest):
        seen.append(dump_ref[3]["tracks"][0]["cands"])
        return real(n, dump_ref, dump_oth, *rest)

    monkeypatch.setattr(m.r1, "ladder", spy)
    canon_inputs = []
    real_canon = m.canonical_segment
    monkeypatch.setattr(
        m, "canonical_segment", lambda d: canon_inputs.append(d) or real_canon(d)
    )
    # R_C#1 writes cost 0.2 before 0.1; the sorted canonical form would not.
    _evaluate(monkeypatch, tmp_path, mutate=_with_cands(order={"R_C_1": (1, 0)}))
    assert seen and all(c == [(0, 0.2), (0, 0.1)] for c in seen)
    assert canon_inputs  # it ran, for the differing R_C#1/R_C#2 segment


def test_identical_repeats_have_no_problem():
    seg = _seg(GMC, TRK1, C1A)
    rec = {
        "segments": {
            s: {f: seg for f in range(1, n + 1)} for s, n in m.SEQUENCE_FRAMES.items()
        },
        "states": {
            s: {f: [] for f in range(1, n + 1)} for s, n in m.SEQUENCE_FRAMES.items()
        },
    }
    assert m.repeat_problem(rec, rec) is None


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
