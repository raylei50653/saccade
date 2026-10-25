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


def test_detector_discovery_goes_through_sequence_selection():
    """#537 F06: ``--detector`` discovery skips (and logs) sequences without
    seqinfo.ini and refuses zero valid ones, via select_sequences."""
    block = _if_block("args.detector and (not args.sequences)")
    (call,) = _calls_named(block, "select_sequences")
    assert ast.unparse(call) == (
        "select_sequences(args.data_root, args.split, args.detector)"
    )
    assigns = [
        n
        for n in ast.walk(block)
        if isinstance(n, ast.Assign)
        and [ast.unparse(t) for t in n.targets] == ["args.sequences"]
    ]
    assert len(assigns) == 1
    assert ast.unparse(assigns[0].value) == (
        "','.join(select_sequences(args.data_root, args.split, args.detector).valid)"
    )
    assert "iterdir" not in ast.unparse(block)


def test_multiprocess_parent_validates_before_spawning():
    """#537 F06: explicit sequences without seqinfo.ini fail in the parent,
    before any child is prepared (``parent_claim_environ`` is the spawn
    preparation in every launcher version)."""
    block = _if_block("getattr(args, 'processes', 0) > 0 and args.sequences")
    (check,) = _calls_named(block, "select_sequences")
    assert ast.unparse(check) == (
        "select_sequences(args.data_root, args.split, args.sequences)"
    )
    (prepare,) = _calls_named(block, "parent_claim_environ")
    assert check.lineno < prepare.lineno
