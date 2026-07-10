"""Unit tests for the routine continuous-chain determinism sentinel.

Covers fixed chain order, first-occurrence reference comparison, exit
semantics, path fail-closed detection, and independent usability of the
2×2 / all-7 forensic tools.  No GPU sequence execution is required.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from saccade.perception.eval._decimal_hash_tools import (  # noqa: E402
    Run,
    compare_chain_to_first_occurrence,
    verdict_from_comparisons,
)
from saccade.perception.eval.decimal_hash import (  # noqa: E402
    CanonicalRecord,
    decimal_hash,
)
from scripts.tools.check_decimal_chain_routine import (  # noqa: E402
    ROUTINE_CHAIN,
    SEQUENCE_A,
    SEQUENCE_B,
    main as routine_main,
)
from scripts.tools import check_decimal_matrix_2x2 as matrix_2x2  # noqa: E402
from scripts.tools import check_decimal_matrix_all7 as matrix_all7  # noqa: E402
from scripts.tools import check_determinism_paths as determ_paths  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_records(
    *triples: tuple[int, float, float, float, float, float],
) -> tuple[CanonicalRecord, ...]:
    records = [
        CanonicalRecord(
            frame=f,
            x_centipixel=int(round(x * 100)),
            y_centipixel=int(round(y * 100)),
            w_centipixel=int(round(w * 100)),
            h_centipixel=int(round(h * 100)),
            score_1e4=int(round(s * 10000)),
        )
        for f, x, y, w, h, s in triples
    ]
    return tuple(sorted(records, key=lambda r: (r.frame, *r.values)))


def _make_run(
    index: int,
    sequence: str,
    records: tuple[CanonicalRecord, ...],
) -> Run:
    return Run(index, sequence, records, decimal_hash(records))


# ── 1. Fixed routine chain order ───────────────────────────────────────────────


def test_routine_chain_order_is_a_a_b_a_b_b() -> None:
    assert ROUTINE_CHAIN == (
        SEQUENCE_A,
        SEQUENCE_A,
        SEQUENCE_B,
        SEQUENCE_A,
        SEQUENCE_B,
        SEQUENCE_B,
    )
    assert SEQUENCE_A == "MOT17-04-SDP"
    assert SEQUENCE_B == "MOT17-02-SDP"
    assert len(ROUTINE_CHAIN) == 6
    assert ROUTINE_CHAIN.count(SEQUENCE_A) == 3
    assert ROUTINE_CHAIN.count(SEQUENCE_B) == 3


# ── 2. First occurrence is reference ──────────────────────────────────────────


def test_first_occurrence_is_reference_for_each_sequence() -> None:
    a_ref = _make_records((1, 10.0, 20.0, 30.0, 40.0, 0.95))
    b_ref = _make_records((1, 50.0, 60.0, 30.0, 40.0, 0.85))
    a_later = _make_records((1, 11.0, 20.0, 30.0, 40.0, 0.95))
    b_later = _make_records((1, 51.0, 60.0, 30.0, 40.0, 0.85))

    # Chain: A, A, B, A, B, B
    runs = [
        _make_run(1, SEQUENCE_A, a_ref),
        _make_run(2, SEQUENCE_A, a_ref),
        _make_run(3, SEQUENCE_B, b_ref),
        _make_run(4, SEQUENCE_A, a_later),
        _make_run(5, SEQUENCE_B, b_ref),
        _make_run(6, SEQUENCE_B, b_later),
    ]
    comparisons = compare_chain_to_first_occurrence(runs)

    assert [c["sequence_occurrence"] for c in comparisons] == [1, 2, 1, 3, 2, 3]
    assert comparisons[0]["reference_run"] == 1
    assert comparisons[1]["reference_run"] == 1
    assert comparisons[2]["reference_run"] == 3
    assert comparisons[3]["reference_run"] == 1
    assert comparisons[4]["reference_run"] == 3
    assert comparisons[5]["reference_run"] == 3
    assert comparisons[0]["reference_hash"] == runs[0].decimal_hash
    assert comparisons[3]["reference_hash"] == runs[0].decimal_hash
    assert comparisons[5]["reference_hash"] == runs[2].decimal_hash


# ── 3 / 4. Divergence exit / exact pass ───────────────────────────────────────


def test_any_later_occurrence_divergence_fails_verdict() -> None:
    a_ref = _make_records((1, 10.0, 20.0, 30.0, 40.0, 0.95))
    b_ref = _make_records((1, 50.0, 60.0, 30.0, 40.0, 0.85))
    a_bad = _make_records((1, 99.0, 20.0, 30.0, 40.0, 0.95))

    runs = [
        _make_run(1, SEQUENCE_A, a_ref),
        _make_run(2, SEQUENCE_A, a_ref),
        _make_run(3, SEQUENCE_B, b_ref),
        _make_run(4, SEQUENCE_A, a_bad),
        _make_run(5, SEQUENCE_B, b_ref),
        _make_run(6, SEQUENCE_B, b_ref),
    ]
    comparisons = compare_chain_to_first_occurrence(runs)
    assert verdict_from_comparisons(comparisons) == "decimal_divergence"
    assert comparisons[3]["classification"] == "decimal_divergence"
    assert comparisons[3]["decimal_hash_equal"] is False
    assert comparisons[3]["observed_hash"] != comparisons[3]["reference_hash"]


def test_structural_divergence_on_record_count_mismatch() -> None:
    a_ref = _make_records((1, 10.0, 20.0, 30.0, 40.0, 0.95))
    a_extra = _make_records(
        (1, 10.0, 20.0, 30.0, 40.0, 0.95),
        (2, 15.0, 25.0, 35.0, 45.0, 0.90),
    )
    runs = [
        _make_run(1, SEQUENCE_A, a_ref),
        _make_run(2, SEQUENCE_A, a_extra),
    ]
    comparisons = compare_chain_to_first_occurrence(runs)
    assert verdict_from_comparisons(comparisons) == "structural_divergence"
    assert comparisons[1]["record_count_ref"] != comparisons[1]["record_count_test"]


def test_all_equal_yields_decimal_exact_pass() -> None:
    a_ref = _make_records((1, 10.0, 20.0, 30.0, 40.0, 0.95))
    b_ref = _make_records((1, 50.0, 60.0, 30.0, 40.0, 0.85))
    runs = [
        _make_run(1, SEQUENCE_A, a_ref),
        _make_run(2, SEQUENCE_A, a_ref),
        _make_run(3, SEQUENCE_B, b_ref),
        _make_run(4, SEQUENCE_A, a_ref),
        _make_run(5, SEQUENCE_B, b_ref),
        _make_run(6, SEQUENCE_B, b_ref),
    ]
    comparisons = compare_chain_to_first_occurrence(runs)
    assert verdict_from_comparisons(comparisons) == "decimal_exact_pass"
    assert all(c["classification"] == "decimal_exact_pass" for c in comparisons)
    assert all(c["decimal_hash_equal"] is True for c in comparisons)


def test_routine_main_nonzero_exit_on_divergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a_ref = _make_records((1, 10.0, 20.0, 30.0, 40.0, 0.95))
    b_ref = _make_records((1, 50.0, 60.0, 30.0, 40.0, 0.85))
    a_bad = _make_records((1, 99.0, 20.0, 30.0, 40.0, 0.95))
    fake_runs = [
        _make_run(1, SEQUENCE_A, a_ref),
        _make_run(2, SEQUENCE_A, a_ref),
        _make_run(3, SEQUENCE_B, b_ref),
        _make_run(4, SEQUENCE_A, a_bad),
        _make_run(5, SEQUENCE_B, b_ref),
        _make_run(6, SEQUENCE_B, b_ref),
    ]

    def _fake_run_sequences(
        sequences: list[str],
        evaluator_output: Path,
        forwarded: list[str],
    ) -> tuple[list[Any], list[Run]]:
        assert sequences == list(ROUTINE_CHAIN)
        return [], fake_runs

    monkeypatch.setattr(
        "scripts.tools.check_decimal_chain_routine.run_sequences",
        _fake_run_sequences,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_decimal_chain_routine.py",
            "--output",
            str(tmp_path / "out"),
            "--preset",
            "mamba_whole_graph_m",
        ],
    )
    assert routine_main() == 1
    summary = (tmp_path / "out" / "summary.json").read_text(encoding="utf-8")
    assert "decimal_divergence" in summary
    assert "reference_hash" in summary
    assert "observed_hash" in summary
    assert "sequence_occurrence" in summary


def test_routine_main_zero_exit_on_exact_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a_ref = _make_records((1, 10.0, 20.0, 30.0, 40.0, 0.95))
    b_ref = _make_records((1, 50.0, 60.0, 30.0, 40.0, 0.85))
    fake_runs = [
        _make_run(1, SEQUENCE_A, a_ref),
        _make_run(2, SEQUENCE_A, a_ref),
        _make_run(3, SEQUENCE_B, b_ref),
        _make_run(4, SEQUENCE_A, a_ref),
        _make_run(5, SEQUENCE_B, b_ref),
        _make_run(6, SEQUENCE_B, b_ref),
    ]

    monkeypatch.setattr(
        "scripts.tools.check_decimal_chain_routine.run_sequences",
        lambda sequences, evaluator_output, forwarded: ([], fake_runs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_decimal_chain_routine.py",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert routine_main() == 0
    summary = (tmp_path / "out" / "summary.json").read_text(encoding="utf-8")
    assert "decimal_exact_pass" in summary


# ── 5. Path detection fail-closed ─────────────────────────────────────────────


def test_determinism_paths_fail_closed_when_git_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> set[str]:
        raise RuntimeError(
            "all git diff commands failed — cannot determine changed files"
        )

    monkeypatch.setattr(determ_paths, "_changed_files", _boom)
    assert determ_paths.main() == 2


def test_determinism_paths_detects_sensitive_and_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        determ_paths,
        "_changed_files",
        lambda: {"docs/readme.md", "src/saccade/perception/eval/metrics.py"},
    )
    assert determ_paths.main() == 1

    monkeypatch.setattr(
        determ_paths,
        "_changed_files",
        lambda: {"scripts/tools/check_decimal_chain_routine.py"},
    )
    with patch("builtins.print") as printed:
        assert determ_paths.main() == 0
    printed.assert_called_with("determinism")


def test_determinism_paths_includes_routine_chain_pattern() -> None:
    joined = "\n".join(determ_paths._SENSITIVE_PATTERNS)
    assert "check_decimal_chain_routine" in joined
    assert "check_decimal_matrix_2x2" in joined
    assert "check_decimal_matrix_all7" in joined


# ── 6. 2×2 and all-7 remain independently usable ──────────────────────────────


def test_matrix_2x2_module_exports_main_and_defaults() -> None:
    assert callable(matrix_2x2.main)
    assert matrix_2x2._DEFAULT_SEQUENCE_A == SEQUENCE_A
    assert matrix_2x2._DEFAULT_SEQUENCE_B == SEQUENCE_B
    assert (
        "forensic" in matrix_2x2.__doc__.lower()
        or "directional" in (matrix_2x2.__doc__ or "").lower()
    )


def test_matrix_all7_module_exports_main_and_order() -> None:
    assert callable(matrix_all7.main)
    assert SEQUENCE_A in matrix_all7._ALL7_SDP
    assert SEQUENCE_B in matrix_all7._ALL7_SDP
    assert len(matrix_all7._ALL7_ORDER) > 6
    assert (
        "release" in (matrix_all7.__doc__ or "").lower()
        or "deep" in (matrix_all7.__doc__ or "").lower()
    )


def test_matrix_2x2_rejects_managed_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_decimal_matrix_2x2.py",
            "--sequences",
            "x",
        ],
    )
    # --sequences is not managed by 2x2 (forwarded); --processes is managed
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_decimal_matrix_2x2.py", "--processes", "2"],
    )
    with pytest.raises(SystemExit):
        matrix_2x2._parse_args()


def test_matrix_all7_rejects_sequences_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_decimal_matrix_all7.py", "--sequences", "MOT17-04-SDP"],
    )
    with pytest.raises(SystemExit):
        matrix_all7._parse_args()
