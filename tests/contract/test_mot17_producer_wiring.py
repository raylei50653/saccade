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
    assert claim_at < source.index("run_sequence_children(")
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
    spawn_at = source.index("run_sequence_children(")
    claim_env_at = source.index("parent_claim_environ(")
    assert claim_env_at < spawn_at
    assert "env=child_env" in source


def _launcher_block() -> ast.If:
    """The ``if getattr(args, "processes", 0) > 0 and args.sequences:`` block."""
    tree = ast.parse(_source())
    hits = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test)
        == "getattr(args, 'processes', 0) > 0 and args.sequences"
    ]
    assert len(hits) == 1
    return hits[0]


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(name)
    ]


def test_multiprocess_launch_failure_exits_before_metrics():
    """#537 F01: a failed child must exit non-zero before parent metrics run.

    Structural (AST), because mot17.py cannot be imported on CPU CI: the
    guard is ``if exit_code != 0:`` with ``exit_code`` from
    ``launch_exit_code(results)``, its body exits with that code, the
    KeyboardInterrupt handler exits with EXIT_INTERRUPTED, and both come
    before the parent's metrics. The launcher itself is exercised with real
    child processes in tests/unit/eval/test_sequence_launcher_537.py.
    """
    block = _launcher_block()
    assert "subprocess.run(" not in _source()
    (spawn,) = _calls(block, "run_sequence_children")
    (metrics,) = _calls(block, "run_motmetrics_evaluation")

    assigns = [
        n
        for n in ast.walk(block)
        if isinstance(n, ast.Assign)
        and [ast.unparse(t) for t in n.targets] == ["exit_code"]
    ]
    assert [ast.unparse(a.value) for a in assigns] == [
        "sequence_launcher.launch_exit_code(results)"
    ]
    guards = [
        n
        for n in ast.walk(block)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "exit_code != 0"
    ]
    assert len(guards) == 1
    guard_exits = [ast.unparse(c) for c in _calls(guards[0], "sys.exit")]
    assert guard_exits == ["sys.exit(exit_code)"]
    assert spawn.lineno < assigns[0].lineno < guards[0].lineno < metrics.lineno

    handlers = [
        h
        for n in ast.walk(block)
        if isinstance(n, ast.Try)
        for h in n.handlers
        if h.type is not None and ast.unparse(h.type) == "KeyboardInterrupt"
    ]
    assert len(handlers) == 1
    handler_exits = [ast.unparse(c) for c in _calls(handlers[0], "sys.exit")]
    assert handler_exits == ["sys.exit(sequence_launcher.EXIT_INTERRUPTED)"]
    assert handlers[0].lineno < metrics.lineno
