"""Contract for wiring ``scripts/eval/mot17.py`` as an AP-2 producer.

``mot17.py`` cannot be imported on CPU CI: module import loads TensorRT. These
tests therefore lock the source-level claim, and exercise the parent/worker
protocol through ``run_manifest`` rather than by running the eval.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import ast
from pathlib import Path

from scripts.provenance.check_producer_coverage import calls_open_run

REPO = Path(__file__).resolve().parents[2]
MOT17 = REPO / "scripts" / "eval" / "mot17.py"


def _source() -> str:
    return MOT17.read_text(encoding="utf-8")


def test_mot17_source_contains_the_parent_aware_claim_call():
    source = _source()
    assert calls_open_run(source)
    assert "claim_or_join_run(" in source
    assert "parent_claim_environ(" in source


def test_mot17_claims_before_any_result_writer():
    """Ordering in source is the stand-in for an import we cannot perform in CI."""
    source = _source()
    claim_at = source.index("claim_or_join_run(")
    assert claim_at < source.index("subprocess.run(")
    assert claim_at < source.index("run_eval(")
    assert claim_at < source.index("run_eval_cpp(")
    assert claim_at < source.index("run_motmetrics_evaluation(")
    assert claim_at < source.index("log_eval_run(")


def test_mot17_does_not_offer_overwrite_or_resume_defaults():
    source = _source()
    assert "overwrite=" not in source
    assert "resume=" not in source
    assert "force=" not in source


def test_multiprocess_children_are_given_the_parent_claim():
    """The launcher must not spawn workers that only 'see a manifest'."""
    source = _source()
    spawn_at = source.index("subprocess.run(")
    claim_env_at = source.index("parent_claim_environ(")
    assert claim_env_at < spawn_at
    assert "env=child_env" in source


def _if_block(test_source: str) -> ast.If:
    tree = ast.parse(_source())
    hits = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and ast.unparse(n.test) == test_source
    ]
    assert len(hits) == 1, test_source
    return hits[0]


def _calls_named(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(name)
    ]


def test_sequence_selection_is_validated_before_any_side_effect():
    """#537 F06: both branches of the ``--detector`` statement validate the
    sequence selection, and the statement precedes the run claim (which
    writes run_manifest.json), the multi-process launcher and every
    detector builder. Discovery sets ``args.sequences`` to the valid names."""
    tree = ast.parse(_source())
    block = _if_block("args.detector and (not args.sequences)")
    assert [ast.unparse(st) for st in block.body if isinstance(st, ast.Assign)] == [
        "args.sequences = ','.join(select_sequences(args.data_root, args.split, "
        "args.detector).valid)"
    ]
    assert [ast.unparse(st) for st in block.orelse if isinstance(st, ast.Expr)] == [
        "select_sequences(args.data_root, args.split, args.sequences)"
    ]
    assert "iterdir" not in ast.unparse(block)
    later = [
        "claim_or_join_run",
        "run_sequence_children",
        "subprocess.run",
        "build_teacher_head_detector",
        "build_mamba_gated_detector",
        "run_eval",
        "run_eval_cpp",
    ]
    for name in later:
        for call in _calls_named(tree, name):
            assert block.lineno < call.lineno, name


def test_other_discovering_eval_scripts_filter_found_sequences():
    """#537 F06: scripts that list sequence directories themselves apply the
    discovery rule (skip and log, zero valid is an error) before run_eval."""
    for rel in ("dancetrack.py", "sportsmot.py", "concurrent_mot17.py"):
        tree = ast.parse((MOT17.parent / rel).read_text(encoding="utf-8"))
        assigns = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and [ast.unparse(t) for t in n.targets] == ["args.sequences"]
        ]
        assert assigns, rel
        for a in assigns:
            assert _calls_named(a.value, "filter_discovered"), (rel, ast.unparse(a))


def test_mot17_all_sdp_filters_discovered_sequences():
    """#537 F06 (Codex r3 NIT): mot17_all_sdp.py hands each discovered SDP
    sequence to a child as an explicit name, so discovery must filter first."""
    tree = ast.parse((MOT17.parent / "mot17_all_sdp.py").read_text(encoding="utf-8"))
    (fn,) = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_discover_sequences"
    ]
    returns = sorted(
        (n for n in ast.walk(fn) if isinstance(n, ast.Return)), key=lambda n: n.lineno
    )
    # explicit selection returns the named list; discovery goes through the filter
    assert [bool(_calls_named(r, "filter_discovered")) for r in returns] == [
        False,
        True,
    ]
