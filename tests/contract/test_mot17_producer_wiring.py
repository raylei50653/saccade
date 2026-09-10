"""Contract for wiring ``scripts/eval/mot17.py`` as an AP-2 producer.

``mot17.py`` cannot be imported on CPU CI: module import loads TensorRT. These
tests therefore lock the source-level claim, and exercise the parent/worker
protocol through ``run_manifest`` rather than by running the eval.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

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
