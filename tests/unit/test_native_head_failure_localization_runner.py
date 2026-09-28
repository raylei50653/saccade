"""The #465 localization runner implements its frozen declaration and nothing else.

``scripts/eval/diagnostics/native_head_failure_localization.py`` implements
``docs/reference/native_runtime_head_failure_localization_declaration.md``.
These tests pin (a) the declaration blob and the study identity it cites,
(b) the artifact identity and shared helpers to the PR-2R runner, (c) the §5
bounds, decision terminal and mechanism label, (d) the §4 V1 freeze-tag check
by peeled commit SHA, (e) the §3 anchor composition and the §4 V3 checks on
CPU tensors, (f) the §6 analysis on synthetic arrays, and (g) a command line
with no option that could change the study identity. No test reads a frame.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import argparse
import importlib.util
import inspect
import subprocess
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "scripts" / "eval" / "diagnostics"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, DIAG / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("native_head_failure_localization")
PR2R = _load("native_head_parity_tf32_off")
DECL = (REPO / R.DECLARATION).read_text()


# --- (a) declaration identity ----------------------------------------------
def test_declaration_blob_is_the_frozen_one():
    blob = subprocess.run(
        ["git", "hash-object", R.DECLARATION],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert blob == R.FROZEN_DECLARATION_BLOB  # an amendment must move this constant


def test_declared_values_appear_in_the_declaration():
    for value in (
        R.FREEZE_TAG,
        f"refs/tags/{R.FREEZE_TAG}^{{}}",
        R.HEAD_ENGINE_SHA256_PREFIX,
        R.HEAD_LINEAGE_SHA256[:8],
        f"--preset {R.PRESET_NAME} --detector SDP --double-buffer",
        str(R.TOTAL_FRAMES),
        *R.ARMS,
        *R.TERMINALS,
        *R.MECHANISM_LABELS,
        "`tracker_input`",
        "`max_det = 300`",
        "`conf_thr = 0.001`",
    ):
        assert value in DECL, value
    order = ", ".join(f"{a}#{r}" for a, r in R.RUN_ORDER)
    assert f"`{order}`" in DECL
    assert (
        "`|Δ IDF1| > 0.20` 或 `|Δ HOTA| > 0.20` 或 `|Δ MOTA| > 0.20` 或 `|Δ IDs| > 5`"
        in DECL
    )


def test_arm_table_matches_section_3():
    rows = {
        "R_C": "| `R_C` | C | C |",
        "R_T": "| `R_T` | T | T |",
        "H_M": "| `H_M` | **T** | C |",
        "H_V": "| `H_V` | **C** | T |",
        "R_E": "| `R_E` | E | E |",
    }
    for arm, row in rows.items():
        assert row in DECL, arm
    assert R.ARM_SOURCES == {
        "R_C": ("C", "C"),
        "R_T": ("T", "T"),
        "H_M": ("T", "C"),
        "H_V": ("C", "T"),
        "R_E": ("E", "E"),
    }
    assert R.HYBRID_ARMS == ("H_M", "H_V")
    assert R.RUN_ORDER[:5] == tuple((a, 1) for a in R.ARMS)
    assert R.RUN_ORDER[5:] == tuple((a, 2) for a in R.ARMS)


# --- (b) shared with PR-2R --------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "HEAD_ONNX_SHA256",
        "HEAD_STEM",
        "HEAD_ENGINE",
        "HEAD_LINEAGE",
        "HEAD_LINEAGE_SHA256",
        "EXPECTED_PRECISION",
        "EXPECTED_BUILDER_FLAGS",
        "CKPT",
        "CKPT_SHA256",
        "BACKBONE",
        "PRESET_NAME",
        "PRESET",
        "EXPORT_TOOL",
        "DATA_ROOT",
        "SPLIT",
        "SEQUENCES",
        "TOTAL_FRAMES",
        "LEASE",
        "IMG_SIZE",
        "SCORE_FLOOR",
        "L2_METRICS",
        "TOL_FLOOR",
    ],
)
def test_identity_constants_equal_pr2r(name):
    assert getattr(R, name) == getattr(PR2R, name)


@pytest.mark.parametrize(
    "name",
    [
        "metrics_from_counts",
        "first_divergent_frame",
        "score_arm",
        "held_lease",
        "_sha256",
        "_dig",
        "_write_manifest",
    ],
)
def test_shared_functions_equal_pr2r(name):
    assert inspect.getsource(getattr(R, name)) == inspect.getsource(getattr(PR2R, name))


def test_packet_root_is_new():
    assert R.PACKET_ROOT not in (PR2R.PACKET_ROOT,)
    assert R.PACKET_ROOT.startswith("results/")


# --- (c) §5 bounds, terminal, label ----------------------------------------
# zero base: x - c is then exactly the float delta, so ±0.20 hits the floor itself
BASE = {"IDF1": 0.0, "HOTA": 0.0, "MOTA": 0.0, "IDs": 0.0}


@pytest.mark.parametrize(
    ("metric", "delta", "out"),
    [
        ("IDF1", 0.20, False),
        ("IDF1", -0.20, False),
        ("IDF1", 0.2001, True),
        ("HOTA", -0.2001, True),
        ("MOTA", 0.2001, True),
        ("IDs", 5.0, False),
        ("IDs", -5.0, False),
        ("IDs", 6.0, True),
        ("IDs", -6.0, True),
    ],
)
def test_out_of_bounds_is_two_sided_and_strict(metric, delta, out):
    x = dict(BASE)
    x[metric] += delta
    got, detail = R.out_of_bounds(x, BASE)
    assert got is out
    assert detail[metric]["floor"] == R.TOL_FLOOR[metric]


def test_decision_terminal_comes_from_validity_and_r_e_only():
    for d in (True, False):
        for v in (True, False):
            assert R.decide(True, True, False, d, v)[0] == "EAGER_NUMERICS_WITHIN"
            assert R.decide(True, True, True, d, v)[0] == "EAGER_NUMERICS_OUT"
            assert R.decide(False, True, True, d, v) == ("UNRESOLVED", None)
            assert R.decide(True, False, True, d, v) == ("UNRESOLVED", None)  # V4
            assert R.decide(True, None, True, d, v) == ("UNRESOLVED", None)
    assert R.decide(True, True, None, True, True) == ("UNRESOLVED", None)


@pytest.mark.parametrize(
    ("d", "v", "label"),
    [
        (True, False, "DELTA_ANCHOR_SUFFICIENT"),
        (False, True, "COMMON_ANCHOR_VALUES_SUFFICIENT"),
        (True, True, "MIXED"),
        (False, False, "MIXED"),
    ],
)
def test_mechanism_label(d, v, label):
    assert R.mechanism_label(d, v) == label
    assert R.decide(True, True, False, d, v) == ("EAGER_NUMERICS_WITHIN", label)


def test_terminal_and_label_vocabulary():
    assert R.TERMINALS == ("UNRESOLVED", "EAGER_NUMERICS_WITHIN", "EAGER_NUMERICS_OUT")
    assert set(R.MECHANISM_LABELS) == {
        R.mechanism_label(d, v) for d in (True, False) for v in (True, False)
    }
    for old in ("S2_BOUNDARY_LOCALIZED", "HEAD_NUMERICS_LOCALIZED"):
        assert old not in inspect.getsource(R)


# --- (d) §4 V1 freeze point -------------------------------------------------
HEAD = "a" * 40
TAG_OBJ = "b" * 40


def test_freeze_ok():
    assert R.freeze_problems(HEAD, ["p1", "p2"], True, "tag", HEAD, HEAD) == []


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"parents": ["p1"]}, "parent"),
        ({"on_chain": False}, "first-parent"),
        ({"obj_type": "commit"}, "annotated"),
        ({"obj_type": None}, "annotated"),
        ({"local": TAG_OBJ}, "local"),  # tag object SHA is not the commit
        ({"local": None}, "local"),
        ({"remote": TAG_OBJ}, "origin"),
        ({"remote": None}, "origin"),
    ],
)
def test_freeze_problems_each_condition(kwargs, needle):
    args = {
        "parents": ["p1", "p2"],
        "on_chain": True,
        "obj_type": "tag",
        "local": HEAD,
        "remote": HEAD,
    }
    args.update(kwargs)
    problems = R.freeze_problems(
        HEAD,
        args["parents"],
        args["on_chain"],
        args["obj_type"],
        args["local"],
        args["remote"],
    )
    assert len(problems) == 1 and needle in problems[0]


def test_parse_ls_remote_uses_only_the_peeled_line():
    ref = f"refs/tags/{R.FREEZE_TAG}"
    out = f"{TAG_OBJ}\t{ref}\n{HEAD}\t{ref}^{{}}\n"
    assert R.parse_ls_remote_peeled(out) == HEAD
    assert R.parse_ls_remote_peeled(f"{TAG_OBJ}\t{ref}\n") is None
    assert R.parse_ls_remote_peeled("") is None
    assert R.parse_ls_remote_peeled(out + f"{TAG_OBJ}\t{ref}^{{}}\n") is None


def test_v1_and_ls_remote_use_the_declared_commands():
    src = inspect.getsource(R.observed_freeze)
    assert '"ls-remote", FREEZE_REMOTE, f"{ref}^{{}}"' in src
    assert '"cat-file", "-t", ref' in src
    assert 'f"{ref}^{{commit}}"' in src
    assert '"rev-list", "--first-parent", FREEZE_BRANCH' in src
    assert inspect.signature(R.check_v1).parameters == {}  # no smoke relaxation


def test_ls_remote_peeled_on_a_real_annotated_tag(tmp_path):
    def git(*a, cwd):
        return subprocess.run(
            ["git", *a], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

    remote = tmp_path / "remote.git"
    work = tmp_path / "w"
    git("init", "-q", "--bare", str(remote), cwd=tmp_path)
    git("init", "-q", str(work), cwd=tmp_path)
    git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "a",
        cwd=work,
    )
    git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "tag",
        "-a",
        R.FREEZE_TAG,
        "-m",
        "t",
        cwd=work,
    )
    git("push", "-q", str(remote), "HEAD:refs/heads/main", "--tags", cwd=work)
    head = git("rev-parse", "HEAD", cwd=work)
    tag_obj = git("rev-parse", f"refs/tags/{R.FREEZE_TAG}", cwd=work)
    assert tag_obj != head
    out = git("ls-remote", str(remote), f"refs/tags/{R.FREEZE_TAG}^{{}}", cwd=work)
    assert R.parse_ls_remote_peeled(out) == head


# --- (e) §3 composition and §4 V3 ------------------------------------------
def _heads(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(84, R.N_ANCHORS, generator=g) - 8.0  # all anchors below floor
    t = c + 0.05 * torch.randn(84, R.N_ANCHORS, generator=g)
    # force some floor crossings and a class flip among members
    c[0, :40] = 3.0
    t[0, :40] = 3.0
    c[1, 40:60] = -2.9  # sigmoid ≈ 0.052 (member for C)
    t[1, 40:60] = -3.0  # sigmoid ≈ 0.047 (non-member for T)
    c[0, 60:70] = 2.0
    t[5, 60:70] = 4.0  # class flip for both-member anchors
    return c, t


def test_flatten_split_round_trip_uses_p3_p4_p5_order():
    logits = torch.arange(84 * R.N_ANCHORS, dtype=torch.float32).reshape(
        84, R.N_ANCHORS
    )
    cls, reg = R.split_logits(logits)
    assert [tuple(x.shape) for x in cls] == [
        (1, 80, 80, 80),
        (1, 80, 40, 40),
        (1, 80, 20, 20),
    ]
    assert [tuple(x.shape) for x in reg] == [
        (1, 4, 80, 80),
        (1, 4, 40, 40),
        (1, 4, 20, 20),
    ]
    assert torch.equal(R.flatten_logits(cls, reg), logits)
    assert cls[1][0, 0, 0, 0] == logits[0, 6400]  # P4 starts after 80*80 anchors


def test_membership_floor_topk_and_class():
    c, _ = _heads()
    m = R.membership(c)
    assert m["m"][:40].all() and m["m"][40:60].all()
    assert not m["m"][70:].any() or m["s"][70:][m["m"][70:]].min() >= R.SCORE_FLOOR
    assert int(m["m"].sum()) <= R.MAX_DET
    # more than 300 anchors above the floor: rank truncation applies
    many = torch.full((84, R.N_ANCHORS), -10.0)
    many[0, :400] = torch.linspace(0.0, 4.0, 400)
    mm = R.membership(many)
    assert int(mm["above"].sum()) == 400 and int(mm["m"].sum()) == R.MAX_DET
    assert mm["m"][100:400].all() and not mm["m"][:100].any()


def test_delta_set_definition():
    c, t = _heads()
    mc, mt = R.membership(c), R.membership(t)
    delta = R.delta_set(mc, mt)
    assert delta[40:60].all()  # floor crossings
    assert delta[60:70].all()  # class flips among joint members
    assert not delta[:40].any()
    expected = (mc["m"] != mt["m"]) | (mc["m"] & mt["m"] & (mc["cls"] != mt["cls"]))
    assert torch.equal(delta, expected)


@pytest.mark.parametrize("arm", ["H_M", "H_V"])
def test_hybrid_composition_passes_v3(arm):
    c, t = _heads()
    logits = {"C": c, "T": t}
    mem = {k: R.membership(v) for k, v in logits.items()}
    delta = R.delta_set(mem["C"], mem["T"])
    inside, outside = R.ARM_SOURCES[arm]
    comp = R.compose(logits[inside], logits[outside], delta)
    cm = R.membership(comp)
    rt = R.flatten_logits(*R.split_logits(comp))
    assert (
        R.v3_problems(comp, logits[inside], logits[outside], delta, mem[inside], cm, rt)
        == []
    )
    assert torch.equal(cm["m"], mem[inside]["m"])


def test_v3_fails_closed_on_tampering():
    c, t = _heads()
    mem = {"C": R.membership(c), "T": R.membership(t)}
    delta = R.delta_set(mem["C"], mem["T"])
    comp = R.compose(t, c, delta)
    bad = comp.clone()
    bad[80, 0] = torch.nextafter(bad[80, 0], torch.tensor(1e9))  # one ulp in reg
    rt = R.flatten_logits(*R.split_logits(bad))
    problems = R.v3_problems(bad, t, c, delta, mem["T"], R.membership(bad), rt)
    assert any("non-Δ anchors" in p for p in problems)
    # membership reference mismatch (e.g. rank moved by the substitution)
    wrong_ref = dict(mem["T"])
    wrong_ref["m"] = ~mem["T"]["m"]
    rt = R.flatten_logits(*R.split_logits(comp))
    problems = R.v3_problems(comp, t, c, delta, wrong_ref, R.membership(comp), rt)
    assert any("membership" in p for p in problems)


def test_v3_is_bitwise_not_value_equality():
    z = torch.zeros(84, R.N_ANCHORS)
    nz = z.clone()
    nz[0, 0] = -0.0
    delta = torch.zeros(R.N_ANCHORS, dtype=torch.bool)
    m = R.membership(z)
    problems = R.v3_problems(nz, z, z, delta, m, R.membership(nz), nz)
    assert any("non-Δ anchors" in p for p in problems)


def test_pair_census_counts():
    c, t = _heads()
    census = R.pair_census(R.membership(c), R.membership(t))
    assert set(census) == set(R.CENSUS_KEYS)
    assert census["floor_cross"] >= 20 and census["class_flip"] == 10
    assert census["delta"] == census["member_diff"] + census["class_flip"]
    assert census["set_differs"] and census["class_differs"]


# --- (f) §6 analysis ---------------------------------------------------------
def _rows(*rows):
    """rows given as (box, score) or (box, score, cls)."""
    out = []
    for r in rows:
        box, score, *cls = r
        out.append([*box, score, cls[0] if cls else 0.0])
    return np.asarray(out, dtype=np.float32).reshape(-1, 6)


def test_first_rows_divergence_and_missing_observation():
    a = {1: _rows(((0, 0, 1, 1), 0.9)), 2: _rows(((1, 1, 2, 2), 0.8))}
    b = {1: a[1].copy(), 2: a[2].copy()}
    assert R.first_rows_divergence(a, b) is None
    b[2][0, 4] = np.nextafter(np.float32(0.8), np.float32(1))
    assert R.first_rows_divergence(a, b) == 2
    assert R.first_rows_divergence(a, {1: a[1], 2: R.MISSING}) == 2
    assert R.first_rows_divergence({1: R.MISSING}, {1: R.MISSING}) is None


def test_detector_row_status_consistency_and_ties():
    det = _rows(
        ((0, 0, 1, 1), 0.9),
        ((1, 1, 2, 2), 0.8),
        ((3, 3, 4, 4), 0.8),
        ((5, 5, 6, 6), 0.7),
    )
    in_delta = np.array([True, True, False, False])
    ok = np.ones(4, dtype=bool)
    assert R.detector_row_status(det, in_delta, ok) == [
        "delta_anchor",
        "ambiguous",  # tie at 0.8 with a non-Δ row: top-k index order is not guaranteed
        "ambiguous",
        "values",
    ]
    bad = ok.copy()
    bad[0] = False
    assert R.detector_row_status(det, in_delta, bad)[0] == "inconsistent"
    same_flag = np.array([False, True, True, False])
    assert R.detector_row_status(det, same_flag, ok)[1:3] == ["delta_anchor"] * 2


def test_inconsistent_row_gets_no_definitive_attribution():
    det = _rows(((0, 0, 1, 1), 0.9), ((1, 1, 2, 2), 0.8))
    in_delta = np.array([True, True])
    status = R.detector_row_status(det, in_delta, np.zeros(2, dtype=bool))
    counts = R.provenance_counts(R.row_provenance(det, det, status))
    assert counts["delta_anchor"] == 0 and counts["values"] == 0
    assert counts["inconsistent"] == 2
    assert R.attribution_class(counts) == "undetermined"


def test_duplicate_boxes_map_by_full_row_not_first_box_match():
    # same box, different score/class: the tracker row matches only row 1
    det = _rows(((0, 0, 1, 1), 0.9, 0.0), ((0, 0, 1, 1), 0.7, 2.0))
    status = R.detector_row_status(det, np.array([False, True]), np.ones(2, dtype=bool))
    assert R.row_provenance(det[1:2], det, status) == ["delta_anchor"]
    # a box-only match (score changed downstream) is unmapped, not guessed
    moved = det[1:2].copy()
    moved[0, 4] = 0.5
    assert R.row_provenance(moved, det, status) == ["unmapped"]
    # bit-identical duplicate rows with different Δ flags are ambiguous
    dup = _rows(((0, 0, 1, 1), 0.9), ((0, 0, 1, 1), 0.9))
    st = R.detector_row_status(dup, np.array([True, False]), np.ones(2, dtype=bool))
    assert R.row_provenance(dup[:1], dup, st) == ["ambiguous"]


def test_attribution_class_needs_every_row_for_values():
    z = dict.fromkeys(R.PROVENANCE, 0)
    assert (
        R.attribution_class({**z, "delta_anchor": 1, "unmapped": 3}) == "delta_anchor"
    )
    assert R.attribution_class({**z, "values": 2}) == "values"
    assert R.attribution_class({**z, "values": 2, "ambiguous": 1}) == "undetermined"
    assert R.attribution_class(z) == "none"


def test_multiset_excess_counts_multiplicity():
    a = _rows(((0, 0, 1, 1), 0.9), ((0, 0, 1, 1), 0.9), ((1, 1, 2, 2), 0.8))
    b = _rows(((0, 0, 1, 1), 0.9), ((1, 1, 2, 2), 0.8))
    assert R.multiset_excess(a, b) == [1]
    assert R.multiset_excess(b, a) == []


def _status_for(rows, flags):
    return rows, R.detector_row_status(
        rows, np.asarray(flags), np.ones(len(rows), dtype=bool)
    )


def test_permutation_is_explained_not_none():
    det = _rows(((0, 0, 1, 1), 0.9), ((1, 1, 2, 2), 0.8))
    ref, oth = det, det[::-1].copy()
    st = _status_for(det, [False, True])
    exp = R.explain_rows(oth, ref, st, st)
    assert exp["kind"] == "order_only"
    assert exp["displaced_positions"] == [0, 1]
    assert exp["class"] == "delta_anchor"


def test_multiplicity_change_is_rows_changed():
    det = _rows(((0, 0, 1, 1), 0.9), ((1, 1, 2, 2), 0.8))
    st = _status_for(det, [False, False])
    exp = R.explain_rows(det[[0, 0, 1]], det, st, st)
    assert exp["kind"] == "rows_changed"
    assert exp["rows_only_in_other"]["values"] == 1 and exp["class"] == "values"


def test_missing_observation_is_explicitly_unknown():
    det = _rows(((0, 0, 1, 1), 0.9))
    st = _status_for(det, [True])
    exp = R.explain_rows(R.MISSING, det, st, st)
    assert exp == {
        "kind": "missing_observation",
        "missing_side": "other",
        "class": "unknown",
    }


def _evidence(
    det_rows, in_delta, ti, det_idx, ids, *, det_frames=None, consistent=None
):
    """ti / det_idx / ids: per frame, None = no probe / no emit on that frame."""
    f = len(det_rows)
    ti_frames = [k + 1 for k in range(f) if ti[k] is not None]
    trk_frames = [k + 1 for k in range(f) if ids[k] is not None]
    rows = np.stack(det_rows)
    return {
        "rows": rows,
        "row_in_delta": np.stack(in_delta),
        "row_in_delta_e": np.zeros_like(np.stack(in_delta)),
        "row_consistent": np.ones(rows.shape[:2], dtype=bool)
        if consistent is None
        else np.stack(consistent),
        "detector_output_frames": np.arange(1, f + 1, dtype=np.int32)
        if det_frames is None
        else np.asarray(det_frames, dtype=np.int32),
        "tracker_input_frames": np.asarray(ti_frames, dtype=np.int32),
        "tracker_input_counts": np.asarray(
            [len(ti[k - 1]) for k in ti_frames], dtype=np.int32
        ),
        "tracker_input_rows": np.concatenate([ti[k - 1] for k in ti_frames])
        if ti_frames
        else np.zeros((0, 6), np.float32),
        "tracker_frames": np.asarray(trk_frames, dtype=np.int32),
        "tracker_counts": np.asarray(
            [len(ids[k - 1]) for k in trk_frames], dtype=np.int32
        ),
        "tracker_ids": np.concatenate(
            [np.asarray(ids[k - 1], np.int64) for k in trk_frames]
        )
        if trk_frames
        else np.zeros((0,), np.int64),
        "tracker_det_idx": np.concatenate(
            [np.asarray(det_idx[k - 1], np.int64) for k in trk_frames]
        )
        if trk_frames
        else np.zeros((0,), np.int64),
        "tracker_boxes": np.concatenate(
            [rows[k - 1][: len(ids[k - 1]), :4] for k in trk_frames]
        ).astype(np.float32)
        if trk_frames
        else np.zeros((0, 4), np.float32),
    }


def test_delta_flow_and_compare_runs():
    a = _rows(((0, 0, 1, 1), 0.9), ((1, 1, 2, 2), 0.8))
    b = _rows(((0, 0, 1, 1), 0.9), ((1, 1, 2, 3), 0.8))  # Δ anchor moved box
    flags = np.array([False, True])
    ref = _evidence([a, a], [flags, flags], [a, a], [[0, 1], [0, 1]], [[1, 2], [1, 2]])
    oth = _evidence([a, b], [flags, flags], [a, b], [[0, 1], [0, 1]], [[1, 2], [1, 3]])
    flow = R.delta_flow(oth)
    assert flow["tracker_input_rows"]["delta_anchor"] == 2
    assert flow["track_det_idx_refs"]["delta_anchor"] == 2
    assert flow["track_det_idx_refs"]["values"] == 2
    cmp = R.compare_runs(ref, oth, "row_in_delta")
    assert cmp["first_tracker_input_divergence"] == 2
    assert cmp["first_tracker_input_explanation"]["kind"] == "rows_changed"
    assert cmp["first_tracker_input_explanation"]["class"] == "delta_anchor"
    assert cmp["first_tracker_output_divergence"] == 2
    trace = R.trace_output_divergence(2, cmp)
    assert trace["tracker_input_divergence_at_or_before"] is True
    assert trace["tracks_other"] == [{"id": 1, "det_idx": 0}, {"id": 3, "det_idx": 1}]
    assert R.trace_output_divergence(None, cmp) is None
    same = R.compare_runs(ref, ref, "row_in_delta")
    assert same["first_tracker_input_divergence"] is None
    assert same["first_tracker_input_explanation"] is None


def test_inconsistent_rows_do_not_count_in_delta_flow():
    a = _rows(((0, 0, 1, 1), 0.9), ((1, 1, 2, 2), 0.8))
    flags = np.array([True, True])
    ev = _evidence(
        [a], [flags], [a], [[0, 1]], [[1, 2]], consistent=[np.array([False, False])]
    )
    flow = R.delta_flow(ev)
    assert flow["tracker_input_rows"]["delta_anchor"] == 0
    assert flow["tracker_input_rows"]["inconsistent"] == 2
    assert flow["track_det_idx_refs"]["delta_anchor"] == 0
    assert flow["replay_rows_above_floor"]["inconsistent"] == 2


def test_one_sided_verified_skip_is_compared_as_empty():
    a = _rows(((0, 0, 1, 1), 0.9), ((1, 1, 2, 2), 0.8))
    flags = np.array([False, True])
    # ref: frame 1 skipped by the harness (detector_output seen, no probe, no emit)
    ref = _evidence([a], [flags], [None], [None], [None])
    oth = _evidence([a], [flags], [a], [[0, 1]], [[1, 2]])
    assert R.tracker_input_view(ref)[1].shape == (0, 6)
    cmp = R.compare_runs(ref, oth, "row_in_delta")
    assert cmp["first_tracker_input_divergence"] == 1
    exp = cmp["first_tracker_input_explanation"]
    assert exp["kind"] == "rows_changed"
    assert exp["rows_only_in_other"]["delta_anchor"] == 1
    assert exp["rows_only_in_other"]["values"] == 1


def test_one_sided_unexplained_missing_probe_stays_unknown():
    a = _rows(((0, 0, 1, 1), 0.9))
    flags = np.array([True])
    # no tracker_input probe, yet the frame emitted tracks: not the skip path
    ref = _evidence([a], [flags], [None], [[0]], [[1]])
    oth = _evidence([a], [flags], [a], [[0]], [[1]])
    assert R.tracker_input_view(ref)[1] is R.MISSING
    cmp = R.compare_runs(ref, oth, "row_in_delta")
    assert cmp["first_tracker_input_divergence"] == 1
    assert cmp["first_tracker_input_explanation"]["kind"] == "missing_observation"
    assert cmp["first_tracker_input_explanation"]["class"] == "unknown"
    # no detector_output probe either: also missing, never an empty frame
    ref2 = _evidence([a], [flags], [None], [None], [None], det_frames=[])
    assert R.tracker_input_view(ref2)[1] is R.MISSING


# --- (f2) P1: fail fast between workers ---------------------------------------
def _fake_run(arm, rep):
    return {
        "arm": arm,
        "rep": rep,
        "returncode": 0,
        "txt_sha256": dict.fromkeys(R.SEQUENCES, "x"),
        "resolved_env_overrides": {},
    }


def test_a_failed_worker_stops_the_schedule(tmp_path, monkeypatch):
    launched: list[str] = []

    def run_arm(l2_dir, arm, rep):
        launched.append(f"{arm}#{rep}")
        run = _fake_run(arm, rep)
        if (arm, rep) == ("H_M", 1):
            run["returncode"] = 1  # V3 raised inside the worker
        return run

    monkeypatch.setattr(R, "run_arm", run_arm)
    monkeypatch.setattr(R, "arm_manifest_problem", lambda run, head: None)
    monkeypatch.setattr(R, "evidence_problems", lambda run: [])
    record = {"v1": {"git": {"head": HEAD}}}
    inputs, reasons = R._measure(tmp_path, record)
    assert launched == ["R_C#1", "R_T#1", "H_M#1"]
    assert inputs == (None, None, None, None)
    assert [f"{r['arm']}#{r['rep']}" for r in record["runs"]] == launched
    assert record["aborted_after"] == "H_M#1"
    assert reasons and "H_M#1" in reasons[0]
    assert R.decide(not reasons, *inputs) == ("UNRESOLVED", None)


@pytest.mark.parametrize(
    ("fail_at", "hook"),
    [
        (("R_T", 1), "coverage"),
        (("H_V", 1), "manifest"),
        (("R_C", 2), "v2"),
    ],
)
def test_every_per_run_problem_stops_the_schedule(tmp_path, monkeypatch, fail_at, hook):
    launched: list[tuple[str, int]] = []

    def run_arm(l2_dir, arm, rep):
        launched.append((arm, rep))
        run = _fake_run(arm, rep)
        if hook == "v2" and (arm, rep) == fail_at:
            run["txt_sha256"] = dict.fromkeys(R.SEQUENCES, "y")
        return run

    def manifest(run, head):
        return (
            "bad"
            if hook == "manifest" and (run["arm"], run["rep"]) == fail_at
            else None
        )

    def coverage(run):
        return (
            ["V3 gap"]
            if hook == "coverage" and (run["arm"], run["rep"]) == fail_at
            else []
        )

    monkeypatch.setattr(R, "run_arm", run_arm)
    monkeypatch.setattr(R, "arm_manifest_problem", manifest)
    monkeypatch.setattr(R, "evidence_problems", coverage)
    record = {"v1": {"git": {"head": HEAD}}}
    _, reasons = R._measure(tmp_path, record)
    assert launched[-1] == fail_at
    assert launched == list(R.RUN_ORDER[: R.RUN_ORDER.index(fail_at) + 1])
    assert reasons


# --- (g) CLI surface -----------------------------------------------------------
def test_cli_exposes_no_study_identity_option():
    seen: list[str] = []
    real = argparse.ArgumentParser.add_argument

    def spy(self, *names, **kwargs):
        seen.extend(n for n in names if n.startswith("--"))
        return real(self, *names, **kwargs)

    with mock.patch.object(argparse.ArgumentParser, "add_argument", spy):
        with mock.patch("sys.argv", ["x", "--help"]), pytest.raises(SystemExit):
            R.main()
    assert set(seen) - {"--help"} == {"--_arm-worker", "--_out"}


def test_worker_argv_is_the_declared_harness_command():
    argv = R.mot17_argv("results/x/R_C_1", R.SEQUENCES, None)
    assert argv[0] == "scripts/eval/mot17.py"
    assert argv[1:6] == [
        "--preset",
        R.PRESET_NAME,
        "--detector",
        "SDP",
        "--double-buffer",
    ]
    assert "--data-root" not in argv and "--mamba-head-engine" not in argv
    assert "--no-compile" not in argv and "--max-frames" not in argv
    # data_root is reachable only from Python (structural checks), not the CLI
    assert "data_root" in inspect.signature(R.run_arm_worker).parameters
    assert "data_root" not in inspect.getsource(R.main)
