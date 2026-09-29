"""The #465 compiled-head reverse-engineering runner implements its frozen declaration.

``scripts/eval/diagnostics/compiled_head_reverse_engineering.py`` implements
``docs/reference/native_runtime_compiled_head_reverse_engineering_declaration.md``.
These tests pin (a) the frozen declaration blob, its freeze commit and the
PR-2L identities it localizes, (b) the PR-2L L1 frame path carried over
verbatim (decode, histogram, head call), (c) the arm construction (setters for
the Inductor arms, the same wrap sites for the ladder arms), (d) every §5.1
class condition and every §5.2 / §5.3 row, and (e) the fail-closed validity
functions (V1 freeze point, env, V-anchor, R0, R1). No GPU, no MOT17.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "scripts" / "eval" / "diagnostics"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, DIAG / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("compiled_head_reverse_engineering")
P2L = R.P2L
DECL_TEXT = (REPO / R.DECLARATION).read_text()


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def _nested(path: Path, outer: str, inner: str) -> str:
    """ast.dump of a function nested in a top-level function (comments dropped)."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == outer:
            for sub in ast.walk(node):
                if isinstance(sub, ast.FunctionDef) and sub.name == inner:
                    return ast.dump(sub)
    raise AssertionError(f"{outer}.{inner} not found in {path}")


# --- (a) frozen declaration and identities --------------------------------


def test_declaration_blob_is_the_frozen_one():
    blob = _git("hash-object", R.DECLARATION).stdout.strip()
    assert blob == R.FROZEN_DECLARATION_BLOB  # an amendment must move this constant


def test_declaration_freeze_commit_is_the_488_merge():
    out = _git("log", "-1", "--format=%s%n%P", R.DECLARATION_FREEZE_COMMIT)
    if out.returncode != 0:
        pytest.skip("shallow clone without the declaration freeze commit")
    subject, parents = out.stdout.strip().splitlines()
    assert subject.startswith("Merge pull request #488 ")
    assert len(parents.split()) == 2
    blob = _git(
        "rev-parse", f"{R.DECLARATION_FREEZE_COMMIT}:{R.DECLARATION}"
    ).stdout.strip()
    assert blob == R.FROZEN_DECLARATION_BLOB


def test_pr2l_construction_sources_are_pinned():
    for rel, blob in (
        (R.PR2L_RUNNER, R.PR2L_RUNNER_BLOB),
        (R.MAMBA_HEAD_SOURCE, R.MAMBA_HEAD_BLOB),
    ):
        assert _git("hash-object", rel).stdout.strip() == blob
        at_freeze = _git("rev-parse", f"504ab0e2:{rel}")
        if at_freeze.returncode == 0:
            assert at_freeze.stdout.strip() == blob


@pytest.mark.parametrize(
    "value",
    [
        "437bb97e",
        "504ab0e2",
        "ddd4524c",
        "set_head_compile(True)",
        "set_block_compile(True)",
        'torch.compile(m, mode="default")',
        "_forward_eager(feats,",
        R.PR2L_PACKET,
        R.PR2L_PACKET_FILES["packet.json"],
        "aot_eager_decomp_partition",
        "`eager`",
        '`inductor`, `mode="default"`',
        "`cudnn.benchmark=False`, `cudnn.allow_tf32=True`",
        "`cuda.matmul.allow_tf32=False`",
        "5,316",
        "≥ 0.9",
        "`[0.5, 2] × max-abs(R)`",
        "If `K_R` is empty, `K_X` must be empty too.",
    ],
)
def test_declaration_names_every_bound_identity(value):
    assert value in DECL_TEXT


def test_study_constants_equal_pr2l():
    assert R.SEQUENCES == P2L.SEQUENCES
    assert R.TOTAL_FRAMES == P2L.TOTAL_FRAMES == 5316
    assert (R.DATA_ROOT, R.SPLIT, R.IMG_SIZE) == (
        P2L.DATA_ROOT,
        P2L.SPLIT,
        P2L.IMG_SIZE,
    )
    assert R.SCORE_FLOOR == P2L.SCORE_FLOOR == 0.05
    assert R.REPEAT_FRAMES == P2L.V2_FRAMES
    assert R.POLICY == P2L.RUNTIME_REQUIREMENTS
    assert tuple(R.ANCHOR_PAIR.split(",")) in P2L.L1_PAIRS
    assert R.LEASE == "machine-bench"
    assert R.FREEZE_TAG == "freeze/465-compiled-head-re-r2"
    assert R.PACKET_ROOT == "results/compiled_head_reverse_engineering_465"
    assert R.UNCHANGED_SINCE_DECLARATION[0] == "src/saccade"
    assert R.PR2L_RUNNER in R.UNCHANGED_SINCE_DECLARATION


def test_arms_are_the_declared_table():
    assert R.ARM_SPECS == {
        "E": {"head": None, "block": None},
        "C": {"head": "inductor", "block": "inductor"},
        "C_H": {"head": "inductor", "block": None},
        "C_B": {"head": None, "block": "inductor"},
        "D_H": {"head": "eager", "block": None},
        "D_B": {"head": None, "block": "eager"},
        "A_H": {"head": "aot_eager_decomp_partition", "block": None},
        "A_B": {"head": None, "block": "aot_eager_decomp_partition"},
    }
    assert R.REFERENCE == {
        "C_H": "C",
        "C_B": "C",
        "D_H": "C_H",
        "A_H": "C_H",
        "D_B": "C_B",
        "A_B": "C_B",
    }
    assert R.LADDER == {"H": ("D_H", "A_H"), "B": ("D_B", "A_B")}
    assert R.OMITTED_ARMS == ()
    assert (R.OVERLAP_NUM, R.OVERLAP_DEN, R.MAXABS_LO, R.MAXABS_HI) == (9, 10, 0.5, 2.0)


def test_cli_has_no_public_option():
    out = subprocess.run(
        [sys.executable, str(DIAG / "compiled_head_reverse_engineering.py"), "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "--" not in out.split("options:", 1)[1].replace("--help", "")


# --- (b) PR-2L L1 frame path, verbatim -------------------------------------


@pytest.mark.parametrize("inner", ["decode", "hist"])
def test_l1_helpers_are_pr2l_verbatim(inner):
    assert _nested(
        DIAG / "compiled_head_reverse_engineering.py", "run_r1_worker", inner
    ) == _nested(DIAG / "native_head_parity_libtorch.py", "run_l1_worker", inner)


def test_head_call_is_pr2l_run_torch():
    ours = ast.parse((DIAG / "compiled_head_reverse_engineering.py").read_text())
    run_head = next(
        n for n in ours.body if isinstance(n, ast.FunctionDef) and n.name == "run_head"
    )
    theirs = ast.parse((DIAG / "native_head_parity_libtorch.py").read_text())
    run_torch = next(
        s
        for n in theirs.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_l1_worker"
        for s in ast.walk(n)
        if isinstance(s, ast.FunctionDef) and s.name == "run_torch"
    )
    body = [st for st in run_head.body if not isinstance(st, ast.Expr)]  # docstring
    assert [ast.dump(st) for st in body] == [ast.dump(st) for st in run_torch.body]


def test_anchor_accumulation_is_pr2l_verbatim():
    """The V-anchor block is the PR-2L pair loop body with (a, b) = (E, C)."""
    ours = (DIAG / "compiled_head_reverse_engineering.py").read_text()
    theirs = (DIAG / "native_head_parity_libtorch.py").read_text()
    for line in (
        "ds = (sa - sb).abs()",
        "sa.max(0).values >= SCORE_FLOOR,",
        "mask = ma | mb",
        "db = (ba - bb).abs()[mask]",
        '["score_hist"] += hist(ds)',
        '["box_hist"] += hist(db)',
        '["masked_anchors"] += mask.sum()',
        '["crossings"] += (ma ^ mb).sum()',
    ):
        assert line in theirs and line in ours
    assert '(sa, ba), (sb, bb) = dec["E"], dec["C"]' in ours


# --- (c) arm construction ---------------------------------------------------


class _FakeHead:
    """The compile-relevant surface of MambaDetectionHead, with its real setters."""

    def __init__(self):
        import torch.nn as nn

        def seq(c_out):
            return nn.Sequential(
                nn.Conv2d(4, 4, 3, padding=1), nn.SiLU(), nn.Conv2d(4, c_out, 1)
            )

        self.cls_head = nn.ModuleList([seq(80) for _ in range(3)])
        self.reg_head = nn.ModuleList([seq(4) for _ in range(3)])
        self.mamba_blocks = nn.ModuleList(
            [nn.ModuleList([nn.Linear(4, 4)]) for _ in range(3)]
        )
        self._head_compile_enabled = False
        self._block_compile_enabled = False
        self._head_modules_original = {}
        self._block_modules_original = None

    def eval(self):
        return self


@pytest.fixture(scope="module")
def fake_base():
    pytest.importorskip("torch")
    mh = pytest.importorskip("saccade.perception.temporal_yolo.mamba_head")
    cls = mh.MambaDetectionHead
    _FakeHead.set_head_compile = cls.set_head_compile
    _FakeHead.set_block_compile = cls.set_block_compile
    return _FakeHead()


@pytest.mark.parametrize("arm", list(R.ARM_SPECS))
def test_build_arm_installs_the_declared_sites(fake_base, arm):
    import torch

    head, sites = R.build_arm(torch, fake_base, arm)
    assert sites == R.expected_sites(arm)  # the fake has 3 blocks of 1
    assert R.site_problems(arm, sites) == []
    spec = R.ARM_SPECS[arm]
    assert head._head_compile_enabled == (spec["head"] == "inductor")
    assert head._block_compile_enabled == (spec["block"] == "inductor")
    assert head is not fake_base
    assert fake_base._head_compile_enabled is False  # the base is never toggled


def test_build_arm_c_is_the_pr2l_setter_pair(fake_base):
    import torch

    head, _ = R.build_arm(torch, fake_base, "C")
    assert head._head_compile_enabled and head._block_compile_enabled
    assert "cls_head" in head._head_modules_original
    assert head._block_modules_original is not None


# --- (d) §5.1 class, §5.2 scope, §5.3 stage ----------------------------------


def _stats(score_n, score_max, box_n, box_max, cross_n, equal_e=0):
    return {
        "score": {"n": score_n, "maxabs": score_max},
        "box": {"n": box_n, "maxabs": box_max},
        "cross": {"n": cross_n},
        "equal_E_frames": equal_e,
    }


def _pair(score, box, cross, equal_r=0):
    return {"score": score, "box": box, "cross": cross, "equal_R_frames": equal_r}


REF = _stats(1000, 1e-3, 100, 0.4, 5)


def test_identical_sets_are_in_class_and_exact():
    c = R.classify(REF, REF, _pair(1000, 100, 5, 10), 10)
    assert c == {
        "class": "in_class",
        "exact_R": True,
        "checks": dict.fromkeys(
            [
                "score_overlap",
                "score_maxabs",
                "box_overlap",
                "box_maxabs",
                "cross_overlap",
            ],
            True,
        ),
    }


def test_equal_e_takes_precedence():
    arm = _stats(0, 0.0, 0, 0.0, 0, equal_e=10)
    assert R.classify(arm, REF, _pair(0, 0, 0), 10)["class"] == "equal_E"


@pytest.mark.parametrize(
    "overlap,n_arm,ok",
    [
        (900, 1000, True),  # recall 0.9, precision 0.9
        (899, 1000, False),  # recall < 0.9
        (900, 1001, False),  # precision < 0.9
        (1000, 1111, True),  # precision 0.9001
        (1000, 1112, False),
    ],
)
def test_score_overlap_boundary(overlap, n_arm, ok):
    arm = _stats(n_arm, 1e-3, 100, 0.4, 5)
    checks = R.class_checks(arm, REF, _pair(overlap, 100, 5))
    assert checks["score_overlap"] is ok


@pytest.mark.parametrize(
    "m,ok", [(0.5e-3, True), (0.49e-3, False), (2e-3, True), (2.01e-3, False)]
)
def test_maxabs_band_is_closed(m, ok):
    arm = _stats(1000, m, 100, 0.4, 5)
    assert R.class_checks(arm, REF, _pair(1000, 100, 5))["score_maxabs"] is ok


def test_boxes_are_judged_separately():
    arm = _stats(1000, 1e-3, 100, 0.1, 5)  # box max-abs out of band
    checks = R.class_checks(arm, REF, _pair(1000, 100, 5))
    assert checks["score_maxabs"] and not checks["box_maxabs"]
    assert R.classify(arm, REF, _pair(1000, 100, 5), 10)["class"] == "partial"


@pytest.mark.parametrize(
    "k_arm,overlap,ok",
    [
        (5, 5, True),  # K_X = K_R
        (5, 4, False),  # one crossing different: recall 0.8
        (6, 5, False),  # one extra crossing: precision 0.83
        (4, 4, False),  # one missing
    ],
)
def test_five_crossings_require_the_same_support(k_arm, overlap, ok):
    arm = _stats(1000, 1e-3, 100, 0.4, k_arm)
    assert R.class_checks(arm, REF, _pair(1000, 100, overlap))["cross_overlap"] is ok


def test_empty_reference_requires_empty_arm():
    ref = _stats(1000, 1e-3, 0, 0.0, 0)
    ok = R.class_checks(_stats(1000, 1e-3, 0, 0.0, 0), ref, _pair(1000, 0, 0))
    assert ok["box_overlap"] and ok["box_maxabs"] and ok["cross_overlap"]
    bad = R.class_checks(_stats(1000, 1e-3, 3, 0.1, 1), ref, _pair(1000, 0, 0))
    assert not bad["box_overlap"] and not bad["box_maxabs"] and not bad["cross_overlap"]


def test_nonempty_reference_rejects_empty_arm():
    assert R._overlap_ok(0, 5, 0) is False


def _c(cls):
    return {"class": cls, "exact_R": False, "checks": {}}


@pytest.mark.parametrize(
    "h,b,want",
    [
        ("in_class", "equal_E", ("HEAD_SCOPE", ("H",))),
        ("equal_E", "in_class", ("BLOCK_SCOPE", ("B",))),
        ("in_class", "partial", ("BOTH_SCOPES", ("H", "B"))),
        ("partial", "partial", ("BOTH_SCOPES", ("H", "B"))),
        ("in_class", "in_class", ("BOTH_SCOPES", ("H", "B"))),
        ("equal_E", "equal_E", ("SCOPE_INTERACTION", ())),
        ("partial", "equal_E", ("UNRESOLVED", ())),
        ("equal_E", "partial", ("UNRESOLVED", ())),
    ],
)
def test_scope_rows(h, b, want):
    assert R.scope_result(True, _c(h), _c(b)) == want


def test_scope_validity_failure_is_unresolved():
    assert R.scope_result(False, _c("in_class"), _c("equal_E")) == ("UNRESOLVED", ())
    assert R.scope_result(True, None, _c("equal_E")) == ("UNRESOLVED", ())


@pytest.mark.parametrize(
    "d,a,label",
    [
        ("in_class", "equal_E", "DYNAMO_OR_CAPTURE_BOUNDARY"),
        ("in_class", "partial", "DYNAMO_OR_CAPTURE_BOUNDARY"),
        ("partial", "in_class", "UNRESOLVED"),
        ("equal_E", "in_class", "AOT_OR_DECOMPOSITION_BOUNDARY"),
        ("equal_E", "partial", "UNRESOLVED"),
        ("equal_E", "equal_E", "INDUCTOR_LOWERING_BOUNDARY"),
    ],
)
def test_stage_rows(d, a, label):
    assert R.stage_label(_c(d), _c(a))[0] == label


def test_omitted_ladder_arm_is_unresolved():
    assert R.stage_label(None, _c("equal_E"))[0] == "UNRESOLVED"
    assert R.stage_label(_c("equal_E"), None)[0] == "UNRESOLVED"


def _r1(classes):
    """An R1 record whose totals produce the given per-arm classes."""
    frames = 10
    ref = _stats(1000, 1e-3, 100, 0.4, 5)
    arms = {"E": _stats(0, 0.0, 0, 0.0, 0, equal_e=frames), "C": ref}
    pairs = {}
    for x, r in R.REFERENCE.items():
        cls = classes[x]
        if cls == "equal_E":
            arms[x] = _stats(0, 0.0, 0, 0.0, 0, equal_e=frames)
            pairs[f"{x}|{r}"] = _pair(0, 0, 0)
        elif cls == "in_class":
            arms[x] = copy.deepcopy(ref)
            pairs[f"{x}|{r}"] = _pair(1000, 100, 5)
        else:
            arms[x] = _stats(1000, 1e-3, 100, 0.4, 5)
            pairs[f"{x}|{r}"] = _pair(10, 100, 5)
    return {"frames": frames, "arms": arms, "pairs": pairs}


def test_decide_head_scope_inductor():
    classes = dict.fromkeys(R.REFERENCE, "equal_E")
    classes["C_H"] = "in_class"
    out = R.decide(True, _r1(classes))
    assert out["scope"] == "HEAD_SCOPE"
    assert out["stages"]["H"]["label"] == "INDUCTOR_LOWERING_BOUNDARY"
    assert out["stages"]["H"]["reference"] == "C_H"


def test_decide_both_scopes_runs_both_ladders():
    classes = dict.fromkeys(R.REFERENCE, "equal_E")
    classes.update({"C_H": "in_class", "C_B": "partial", "A_B": "in_class"})
    out = R.decide(True, _r1(classes))
    assert out["scope"] == "BOTH_SCOPES"
    assert out["stages"]["H"]["label"] == "INDUCTOR_LOWERING_BOUNDARY"
    assert out["stages"]["B"]["label"] == "AOT_OR_DECOMPOSITION_BOUNDARY"


def test_decide_invalid_is_unresolved():
    assert R.decide(False, None)["scope"] == "UNRESOLVED"


# --- (e) fail-closed validity -------------------------------------------------

H = "a" * 40


def test_parse_ls_remote_reads_this_runners_own_tag():
    # Regression: r1 reused the PR-2L parser, which matches only the PR-2L tag,
    # so the peeled SHA of this runner's tag was always None and V1 never passed.
    head = "a" * 40
    tag_obj = "b" * 40
    ref = f"refs/tags/{R.FREEZE_TAG}"
    out = f"{tag_obj}\t{ref}\n{head}\t{ref}^{{}}\n"
    assert R.parse_ls_remote_peeled(out) == head
    assert R.parse_ls_remote_peeled(f"{head}\t{ref}^{{}}\n") == head
    assert R.parse_ls_remote_peeled(f"{tag_obj}\t{ref}\n") is None
    assert R.parse_ls_remote_peeled("") is None
    assert R.parse_ls_remote_peeled(out + f"{tag_obj}\t{ref}^{{}}\n") is None
    for other in ("freeze/465-compiled-head-re", P2L.FREEZE_TAG):
        assert R.parse_ls_remote_peeled(f"{head}\trefs/tags/{other}^{{}}\n") is None


def test_observed_freeze_uses_this_runners_parser():
    src = inspect.getsource(R.observed_freeze)
    assert "P2L.parse_ls_remote_peeled" not in src
    assert "parse_ls_remote_peeled(ls.stdout)" in src


@pytest.mark.parametrize(
    "kwargs,n_problems",
    [
        ({}, 0),
        ({"parents": ["x"]}, 1),
        ({"on_first_parent_chain": False}, 1),
        ({"tag_object_type": "commit"}, 1),
        ({"tag_local_commit": "b" * 40}, 1),
        ({"tag_remote_peeled": None}, 1),
    ],
)
def test_freeze_problems(kwargs, n_problems):
    args = {
        "head": H,
        "parents": ["p1", "p2"],
        "on_first_parent_chain": True,
        "tag_object_type": "tag",
        "tag_local_commit": H,
        "tag_remote_peeled": H,
        **kwargs,
    }
    assert len(R.freeze_problems(**args)) == n_problems


def test_forbidden_env():
    env = {
        "PATH": "/bin",
        "SACCADE_X": "1",
        "TORCHINDUCTOR_CACHE_DIR": "/t",
        "TORCH_LOGS": "+dynamo",
        "PYTORCH_CUDA_ALLOC_CONF": "x",
        "TRITON_CACHE_DIR": "/t",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    assert set(R.forbidden_env(env)) == {
        "SACCADE_X",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_LOGS",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TRITON_CACHE_DIR",
    }


def _anchor_ref():
    rows = {
        "S1": {
            "pair": "E,C",
            "sequence": "S1",
            "frames": "3",
            "score_maxabs": "0.0009623616933822632",
            "score_p999_upper": "1e-10",
            "box_maxabs_px": "0.2830810546875",
            "box_p999_upper_px": "0.041686938347033464",
            "masked_anchors": "92030",
            "floor_crossings": "1",
            "nonfinite": "0",
        }
    }
    rows["ALL"] = {**rows["S1"], "sequence": "ALL"}
    hists = {"S1": {"score_hist": [5, 1], "box_hist": [2, 0]}}
    return rows, hists


def _anchor_obs(rows):
    return {
        seq: {
            k: (None if v == "" else float(v))
            for k, v in r.items()
            if k not in ("pair", "sequence")
        }
        for seq, r in rows.items()
    }


def test_anchor_exact_passes():
    rows, hists = _anchor_ref()
    assert R.anchor_problems(_anchor_obs(rows), hists, rows, hists) == []


def test_anchor_one_ulp_fails():
    import math

    rows, hists = _anchor_ref()
    obs = _anchor_obs(rows)
    obs["S1"]["score_maxabs"] = math.nextafter(obs["S1"]["score_maxabs"], 1.0)
    assert any("score_maxabs" in p for p in R.anchor_problems(obs, hists, rows, hists))


def test_anchor_histogram_and_coverage_fail():
    rows, hists = _anchor_ref()
    obs_h = {"S1": {"score_hist": [4, 2], "box_hist": [2, 0]}}
    assert R.anchor_problems(_anchor_obs(rows), obs_h, rows, hists)
    obs = _anchor_obs(rows)
    del obs["S1"]
    assert R.anchor_problems(obs, hists, rows, hists)


def test_anchor_reads_the_pinned_pr2l_rows():
    packet = REPO / R.PR2L_PACKET
    if not (packet / "l1" / "l1.json").exists():
        pytest.skip("PR-2L packet is local-only (results/ is untracked)")
    rows, hists = R.read_pr2l_anchor()
    assert set(rows) == set(R.SEQUENCES) | {"ALL"}
    assert rows["ALL"]["floor_crossings"] == "5"
    assert rows["ALL"]["masked_anchors"] == "897492"
    assert set(hists) == set(R.SEQUENCES)
    assert sum(sum(h["score_hist"][1:]) for h in hists.values()) == 578847912


def _ev(arm, code, backend, tag="g"):
    kinds = {k: f"{tag}-{k}" for k in R.BACKEND_KINDS[backend]}
    return {"arm": arm, "code": code, "kinds": kinds}


def _raw(ids, compiled_during=False, unshimmed=0):
    return {
        "executed_ids": ids,
        "compiled_during_hit_pass": compiled_during,
        "unshimmed_calls": unshimmed,
    }


def _ok_provenance(arm, prefix=""):
    """Every compiled call of the arm hits one own graph per compiled scope."""
    events, ids = {}, []
    for scope, n in (
        ("head", R.EXPECTED_HEAD_SITES),
        ("block", sum(R.EXPECTED_BLOCK_LAYOUT)),
    ):
        backend = R.ARM_SPECS[arm][scope]
        if backend is None:
            continue
        cid = f"{prefix}{scope}/0"
        events[cid] = _ev(arm, R.COMPILED_CODES[scope], backend, f"{arm}{scope}")
        ids += [cid] * n
    return R.finalize_provenance(events, {arm: _raw(ids)})


def _shared_ok():
    events, arms = {}, {}
    for arm in R.ARM_ORDER:
        p = _ok_provenance(arm, prefix=f"{arm}:")
        events.update(p["events"])
        arms.update(p["arms"])
    return {"events": events, "arms": arms}


def _r0_ok(arm):
    return {
        "provenance": _ok_provenance(arm),
        "arm": arm,
        "repeat_identical": True,
        "output_sha256": "s" + arm,
        "wrap_sites": R.expected_sites(arm),
        "recompile_limit_hit": [],
        "policy": dict(R.POLICY),
    }


def test_r0_problems_happy_and_failures():
    assert R.r0_problems("C", _r0_ok("C")) == []
    assert R.r0_problems("C", None)
    for key, bad in (
        ("repeat_identical", False),
        ("output_sha256", None),
        ("wrap_sites", {"head": {"backend": "eager", "count": 6}}),
        ("recompile_limit_hit", ["hit"]),
        ("policy", {**R.POLICY, "cudnn_allow_tf32": False}),
        ("arm", "E"),
    ):
        rec = _r0_ok("C")
        rec[key] = bad
        assert R.r0_problems("C", rec), key


def _r1_ok(frames=10):
    return {
        "frames": frames,
        "sequences": ["S1"],
        "frames_per_sequence": {"S1": frames},
        "synthetic_sha256": {a: "s" + a for a in R.ARM_ORDER},
        "wrap_sites": {a: R.expected_sites(a) for a in R.ARM_ORDER},
        "dynamo_counters_before": {"frames": {"total": 3}},
        "dynamo_counters_after": {"frames": {"total": 3}},
        "recompile_limit_hit": [],
        "repeat_frames_checked": min(R.REPEAT_FRAMES, frames),
        "repeat_failures": [],
        "input_mutations": [],
        "arms": {a: {"nonfinite": 0} for a in R.ARM_ORDER},
        "jit_fallback_calls": {"_selective_scan_jit": 0},
        "policy": dict(R.POLICY),
        "driver_runtime": {},
        "artifact_problems": [],
        "provenance": _shared_ok(),
    }


def test_r1_problems_only_driver_on_a_fake_record():
    r0 = {a: _r0_ok(a) for a in R.ARM_ORDER}
    problems = R.r1_problems(_r1_ok(), r0, 10)
    assert problems and all(p.startswith("R1 driver/runtime") for p in problems)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(frames=9),
        lambda r: r["synthetic_sha256"].update(C="other"),
        lambda r: r.update(dynamo_counters_after={"frames": {"total": 4}}),
        lambda r: r.update(dynamo_counters_before=None),
        lambda r: r.update(recompile_limit_hit=["hit"]),
        lambda r: r.update(repeat_failures=[{"arm": "C"}]),
        lambda r: r.update(repeat_frames_checked=0),
        lambda r: r.update(input_mutations=[{"frame_index": 0}]),
        lambda r: r["arms"]["A_B"].update(nonfinite=1),
        lambda r: r.update(jit_fallback_calls={"_selective_scan_jit": 1}),
        lambda r: r.update(policy={**R.POLICY, "matmul_allow_tf32": True}),
        lambda r: r["wrap_sites"].update(D_H=R.expected_sites("A_H")),
        lambda r: r.update(artifact_problems=["x"]),
    ],
)
def test_r1_problems_fail_closed(mutate):
    r0 = {a: _r0_ok(a) for a in R.ARM_ORDER}
    base = [p for p in R.r1_problems(_r1_ok(), r0, 10)]
    rec = _r1_ok()
    mutate(rec)
    assert len(R.r1_problems(rec, r0, 10)) > len(base)


def test_op_inventory_reads_both_graph_levels():
    log = (
        'V0 [0/0] [__graph_code] input_1: "f32" = torch.conv2d(x, w);  x = None\n'
        'V0 [0/0] [__graph_code] input_2: "f32" = torch.nn.functional.silu(input_1)\n'
        "V0 [aot_graphs] convolution = torch.ops.aten.convolution.default(a, b)\n"
        "V0 [aot_graphs] y = torch.ops.saccade.selective_scan_fwd.default(u)\n"
        "unrelated = torch.conv2d(z)\n"
    )
    inv = R.op_inventory(log)
    assert inv["fx"] == {"torch.conv2d": 1, "torch.nn.functional.silu": 1}
    assert inv["ops"] == {"aten.convolution": 1, "saccade.selective_scan_fwd": 1}


# --- (f) graph-construction identity (§4 code identity, §7 fail closed) ------

HEAD_CODE = R.COMPILED_CODES["head"]
BLOCK_CODE = R.COMPILED_CODES["block"]


def _iso_c_h(tag="g"):
    """C_H isolated: a static and a dynamic compile per head Sequential kind;
    its 6 calls hit the two dynamic graphs (cls, reg)."""
    events = {
        "1/0": _ev("C_H", HEAD_CODE, "inductor", tag + "cls-static"),
        "1/1": _ev("C_H", HEAD_CODE, "inductor", tag + "cls-dyn"),
        "1/2": _ev("C_H", HEAD_CODE, "inductor", tag + "reg-static"),
        "1/3": _ev("C_H", HEAD_CODE, "inductor", tag + "reg-dyn"),
    }
    return R.finalize_provenance(events, {"C_H": _raw(["1/1"] * 3 + ["1/3"] * 3)})


def _shared_c_h_reusing_c(tag="g"):
    """R1: C compiled the head graphs; C_H compiles nothing and reuses them."""
    events = {
        "1/0": _ev("C", HEAD_CODE, "inductor", tag + "cls-static"),
        "1/1": _ev("C", HEAD_CODE, "inductor", tag + "cls-dyn"),
        "1/2": _ev("C", HEAD_CODE, "inductor", tag + "reg-static"),
        "1/3": _ev("C", HEAD_CODE, "inductor", tag + "reg-dyn"),
    }
    return R.finalize_provenance(events, {"C_H": _raw(["1/1"] * 3 + ["1/3"] * 3)})


def test_normalize_payload_drops_process_identity_only():
    text = (
        "# AOT ID: ['3_inference']\n"
        "path /tmp/pk/cache/r1/inductor/ab/cab.py\n"
        "__compiled_fn_12_4e631ec6_6186_4dad_90e9_ac16ebedd23c(x)\n"
        "obj at 0x7f00deadbeef\n"
        "triton_poi_fused_silu_0"
    )
    out = R.normalize_payload(text, ["/tmp/pk/cache/r1/inductor"])
    assert "AOT ID" not in out and "<CACHE>/ab/cab.py" in out
    assert "__compiled_fn(x)" in out and "0x7f" not in out
    assert "triton_poi_fused_silu_0" in out  # kernel identity kept


def test_fingerprint_covers_code_and_every_kind():
    base = _ev("C", HEAD_CODE, "inductor")
    fp = R.event_fingerprint(base)
    assert R.event_fingerprint({**base, "arm": "D_H"}) == fp  # arm is not identity
    assert R.event_fingerprint({**base, "code": BLOCK_CODE}) != fp
    for k in R.PROVENANCE_KINDS:
        other = {**base, "kinds": {**base["kinds"], k: "changed"}}
        assert R.event_fingerprint(other) != fp


def test_post_grad_graph_is_recorded_not_fingerprinted():
    assert "inductor_post_grad_graph" not in R.PROVENANCE_KINDS
    assert R.RECORDED_KINDS == ("inductor_post_grad_graph",)
    base = _ev("C", HEAD_CODE, "inductor")
    assert R.event_fingerprint({**base, "recorded": {"x": "y"}}) == R.event_fingerprint(
        base
    )


def test_finalize_attributes_reuse():
    a = _shared_c_h_reusing_c()["arms"]["C_H"]
    assert a["own"] == [] and a["reused_from"] == ["C"]
    assert {x["compiled_by"] for x in a["executed"]} == {"C"}
    unknown = R.finalize_provenance({}, {"C_H": _raw([None, "9/9"])})
    assert [x["fingerprint"] for x in unknown["arms"]["C_H"]["executed"]] == [
        None,
        None,
    ]


@pytest.mark.parametrize(
    "arm,code,backend,ok",
    [
        ("C_H", HEAD_CODE, "inductor", True),
        ("D_H", HEAD_CODE, "eager", True),
        ("A_B", BLOCK_CODE, "aot_eager_decomp_partition", True),
        ("D_H", HEAD_CODE, "inductor", False),  # hit C's inductor graph
        ("A_H", HEAD_CODE, "eager", False),
        ("C_H", BLOCK_CODE, "inductor", False),  # block compiled, block is eager
        ("C", {"file": "x.py", "name": "f"}, "inductor", False),
    ],
)
def test_backend_problems(arm, code, backend, ok):
    assert (R.backend_problems(arm, _ev(arm, code, backend)) == []) is ok


def test_provenance_reuse_of_the_same_graphs_passes():
    assert R.provenance_problems("C_H", _iso_c_h()) == []
    assert R.provenance_problems("C_H", _iso_c_h(), _shared_c_h_reusing_c()) == []


def test_provenance_rejects_different_graphs_behind_equal_outputs():
    """Review P1 regression: same output hash, different executed graph."""
    problems = R.provenance_problems("C_H", _iso_c_h(), _shared_c_h_reusing_c("other"))
    assert any("R1 executed graphs != isolated R0 graphs" in p for p in problems)
    assert any("reused_from ['C']" in p for p in problems)


def test_r1_problems_rejects_graph_mismatch_even_with_equal_output_hash():
    r0 = {a: _r0_ok(a) for a in R.ARM_ORDER}
    r0["C_H"]["provenance"] = _iso_c_h()
    rec = _r1_ok()
    rec["provenance"]["events"].update(_shared_c_h_reusing_c()["events"])
    rec["provenance"]["arms"]["C_H"] = _shared_c_h_reusing_c()["arms"]["C_H"]
    assert rec["synthetic_sha256"]["C_H"] == r0["C_H"]["output_sha256"]
    assert [p for p in R.r1_problems(rec, r0, 10) if "graph identity C_H" in p] == []
    other = _shared_c_h_reusing_c("other")
    rec["provenance"]["events"].update(other["events"])
    rec["provenance"]["arms"]["C_H"] = other["arms"]["C_H"]
    assert [p for p in R.r1_problems(rec, r0, 10) if "graph identity C_H" in p]


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda p: p["arms"]["C_H"]["executed"].pop(), "compiled calls != 6"),
        (
            lambda p: p["arms"]["C_H"]["executed"][0].update(fingerprint=None),
            "not attributable",
        ),
        (
            lambda p: p["arms"]["C_H"].update(compiled_during_hit_pass=True),
            "attribution pass",
        ),
        (lambda p: p["arms"]["C_H"].update(unshimmed_calls=1), "not attributable"),
        (
            lambda p: p["events"]["1/1"].update(code={"file": "a.py", "name": "f"}),
            "not a declared compiled code object",
        ),
    ],
)
def test_provenance_fails_closed(mutate, needle):
    shared = _shared_c_h_reusing_c()
    mutate(shared)
    problems = R.provenance_problems("C_H", _iso_c_h(), shared)
    assert any(needle in p for p in problems), problems


def test_provenance_rejects_a_foreign_own_compile():
    shared = _shared_c_h_reusing_c()
    foreign = _ev("C_H", HEAD_CODE, "inductor", "foreign")
    shared["events"]["1/9"] = {**foreign, "fingerprint": R.event_fingerprint(foreign)}
    shared["arms"]["C_H"]["own"] = ["1/9"]
    problems = R.provenance_problems("C_H", _iso_c_h(), shared)
    assert any("R1 compiled graphs not in R0" in p for p in problems)


def test_isolated_arm_must_execute_only_its_own_graphs():
    iso = _iso_c_h()
    iso["arms"]["C_H"]["executed"][0]["compiled_by"] = "C"
    assert any("another arm's graph" in p for p in R.provenance_problems("C_H", iso))
    iso = _iso_c_h()
    iso["arms"]["C_H"]["own"] = ["1/0", "1/2"]  # the executed dynamic graphs missing
    assert any("did not compile" in p for p in R.provenance_problems("C_H", iso))


def test_eager_arm_has_no_compiled_calls():
    empty = R.finalize_provenance({}, {"E": _raw([])})
    assert R.provenance_problems("E", empty, empty) == []
    assert (R.expected_calls("E"), R.expected_calls("C"), R.expected_calls("D_B")) == (
        0,
        9,
        3,
    )
