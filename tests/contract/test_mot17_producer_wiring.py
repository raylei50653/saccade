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


def _stmt_index(body: list[ast.stmt], pred: object) -> int:
    hits = [i for i, st in enumerate(body) if pred(st)]  # type: ignore[operator]
    assert len(hits) == 1, [ast.unparse(body[i])[:60] for i in hits]
    return hits[0]


def _is_exit(stmt: ast.stmt, arg: str) -> bool:
    return isinstance(stmt, ast.Expr) and ast.unparse(stmt) == f"sys.exit({arg})"


def test_multiprocess_launch_failure_exits_before_metrics():
    """#537 F01: a failed child must exit non-zero before parent metrics run.

    Structural (AST), because mot17.py cannot be imported on CPU CI. Only
    direct statements of the multi-process body count, so an exit nested
    under any extra condition fails the check: the launcher ``try`` whose
    KeyboardInterrupt handler ends in ``sys.exit(EXIT_INTERRUPTED)``, then
    ``exit_code = launch_exit_code(results)``, then ``if exit_code != 0:``
    ending in ``sys.exit(exit_code)``, then the parent metrics. The launcher
    itself is exercised with real child processes in
    tests/unit/eval/test_sequence_launcher_537.py.
    """
    assert "subprocess.run(" not in _source()
    outer = _launcher_block()
    (inner,) = [
        st
        for st in outer.body
        if isinstance(st, ast.If) and ast.unparse(st.test) == "len(seq_list) > 1"
    ]
    body = inner.body

    def launcher_try(st: ast.stmt) -> bool:
        return isinstance(st, ast.Try) and bool(_calls(st, "run_sequence_children"))

    try_i = _stmt_index(body, launcher_try)
    launch = body[try_i]
    assert isinstance(launch, ast.Try)
    (handler,) = [
        h
        for h in launch.handlers
        if h.type is not None and ast.unparse(h.type) == "KeyboardInterrupt"
    ]
    assert [ast.unparse(st) for st in handler.body] == [
        "sequence_launcher.report_interrupt()",
        "sys.exit(sequence_launcher.EXIT_INTERRUPTED)",
    ]
    # The launch itself is the try body, not a call nested under a condition.
    assert len(launch.body) == 1
    assert isinstance(launch.body[0], ast.Assign)
    assert ast.unparse(launch.body[0].targets[0]) == "results"
    call = launch.body[0].value
    assert isinstance(call, ast.Call)
    assert ast.unparse(call.func) == "sequence_launcher.run_sequence_children"
    assert [ast.unparse(a) for a in call.args] == [
        "seq_list",
        "lambda seq_name: cmd_base + ['--sequences', seq_name, '--processes', '0']",
    ]
    keywords = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert keywords == {"max_workers": "args.processes", "env": "child_env"}

    assign_i = _stmt_index(
        body,
        lambda st: (
            isinstance(st, ast.Assign)
            and ast.unparse(st)
            == "exit_code = sequence_launcher.launch_exit_code(results)"
        ),
    )
    guard_i = _stmt_index(
        body,
        lambda st: isinstance(st, ast.If) and ast.unparse(st.test) == "exit_code != 0",
    )
    guard = body[guard_i]
    assert isinstance(guard, ast.If) and not guard.orelse
    assert len(guard.body) == 1 and _is_exit(guard.body[0], "exit_code")
    # Reporting is best effort and comes after the code is decided.
    report_i = _stmt_index(
        body,
        lambda st: ast.unparse(st) == "sequence_launcher.report(results, exit_code)",
    )
    assert assign_i < report_i < guard_i
    metrics_i = _stmt_index(
        body,
        lambda st: (
            isinstance(st, ast.Assign) and bool(_calls(st, "run_motmetrics_evaluation"))
        ),
    )
    assert try_i < assign_i < guard_i < metrics_i
