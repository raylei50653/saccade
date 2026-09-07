"""Fail-closed MOT repeat-identity harness for issue #363.

The live GPU runner is not exercised here.  These tests pin the comparator
contract: silent MOT divergence must fail, matching files must pass, and a
crashed or empty eval must not look like success — including an eval that
writes a complete, identical MOT and then returns non-zero.
"""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import os
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saccade.perception.eval.repeat_identity import (  # noqa: E402
    KIND_GEOMETRY_OR_SCORE,
    KIND_IDENTITY_ONLY,
    classify_first_diff,
    compare_run_dirs,
    format_report,
)
from scripts.tools import check_eval_repeat_identity as harness  # noqa: E402


LINE_A = "15,7,934.74,435.47,36.08,81.58,0.5767,-1,-1,-1"
LINE_A_SCORE = "15,7,934.74,435.47,36.08,81.58,0.5837,-1,-1,-1"
LINE_A_ID = "15,8,934.74,435.47,36.08,81.58,0.5767,-1,-1,-1"
LINE_B = "16,7,935.10,435.50,36.00,81.40,0.5800,-1,-1,-1"


def _write_mot(directory: Path, sequence: str, *lines: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sequence}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_identical_run_dirs_pass(tmp_path: Path) -> None:
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    _write_mot(a, "MOT17-02-SDP", LINE_A, LINE_B)
    _write_mot(b, "MOT17-02-SDP", LINE_A, LINE_B)
    report = compare_run_dirs([a, b])
    assert report.ok
    assert report.reports[0].n_distinct == 1
    assert report.reports[0].divergent_runs == ()


def test_score_divergence_fails_closed(tmp_path: Path) -> None:
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    _write_mot(a, "MOT17-02-SDP", LINE_A, LINE_B)
    _write_mot(b, "MOT17-02-SDP", LINE_A_SCORE, LINE_B)
    report = compare_run_dirs([a, b])
    assert not report.ok
    assert report.reports[0].n_distinct == 2
    diff = report.reports[0].first_diffs[1]
    assert diff is not None
    assert diff.kind == KIND_GEOMETRY_OR_SCORE
    assert diff.frame == 15
    assert "distinct MOT hashes" in report.reasons[0]


def test_identity_only_divergence_still_fails(tmp_path: Path) -> None:
    """Raw MOT identity includes IDs.  An ID offset is the #363 symptom."""

    a = tmp_path / "r1"
    b = tmp_path / "r2"
    _write_mot(a, "MOT17-02-SDP", LINE_A, LINE_B)
    _write_mot(b, "MOT17-02-SDP", LINE_A_ID, LINE_B)
    report = compare_run_dirs([a, b])
    assert not report.ok
    diff = report.reports[0].first_diffs[1]
    assert diff is not None
    assert diff.kind == KIND_IDENTITY_ONLY
    assert diff.decimal_hash_equal is True


def test_empty_mot_file_fails(tmp_path: Path) -> None:
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    _write_mot(a, "MOT17-02-SDP", LINE_A)
    b.mkdir()
    (b / "MOT17-02-SDP.txt").write_text("", encoding="utf-8")
    report = compare_run_dirs([a, b])
    assert not report.ok
    assert any("empty" in reason for reason in report.reasons)


def test_missing_sequence_fails(tmp_path: Path) -> None:
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    _write_mot(a, "MOT17-02-SDP", LINE_A)
    _write_mot(a, "MOT17-04-SDP", LINE_B)
    _write_mot(b, "MOT17-02-SDP", LINE_A)
    report = compare_run_dirs([a, b])
    assert not report.ok
    assert any(
        "MOT17-04-SDP" in reason and "missing" in reason for reason in report.reasons
    )


def test_no_mot_files_fails() -> None:
    report = compare_run_dirs([])
    assert not report.ok
    assert report.reasons == ("no run directories",)


def test_classify_score_only_row() -> None:
    ref = "154,43,527.15,457.73,22.91,66.35,0.3617,-1,-1,-1\n"
    other = "154,43,527.15,457.73,22.91,66.35,0.3784,-1,-1,-1\n"
    diff = classify_first_diff(ref, other)
    assert diff.kind == KIND_GEOMETRY_OR_SCORE
    assert diff.frame == 154
    assert diff.decimal_hash_equal is False


def test_classify_identical_files() -> None:
    text = f"{LINE_A}\n{LINE_B}\n"
    diff = classify_first_diff(text, text)
    assert diff.kind == "identical"
    assert diff.line_index is None


def test_format_report_mentions_fail(tmp_path: Path) -> None:
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    _write_mot(a, "MOT17-02-SDP", LINE_A)
    _write_mot(b, "MOT17-02-SDP", LINE_A_SCORE)
    text = format_report(compare_run_dirs([a, b]))
    assert "FAIL" in text
    assert "geometry_or_score" in text


def test_cli_compare_exit_codes(tmp_path: Path) -> None:
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    c = tmp_path / "r3"
    _write_mot(a, "MOT17-02-SDP", LINE_A)
    _write_mot(b, "MOT17-02-SDP", LINE_A)
    _write_mot(c, "MOT17-02-SDP", LINE_A_SCORE)
    summary = tmp_path / "summary.json"
    assert harness.main(["compare", str(a), str(b), "--summary", str(summary)]) == 0
    assert summary.exists()
    assert harness.main(["compare", str(a), str(c)]) == 1


def test_merge_eval_flags_fills_block_s_defaults() -> None:
    merged = harness.merge_eval_flags([])
    assert "--preset" in merged and "baseline" in merged
    assert "--no-gpu-decode" in merged
    assert "--sequences" in merged and "MOT17-02-SDP" in merged


def test_merge_eval_flags_does_not_override_caller() -> None:
    merged = harness.merge_eval_flags(
        ["--preset", "mamba_whole_graph_m", "--double-buffer"]
    )
    assert merged.count("--preset") == 1
    assert "mamba_whole_graph_m" in merged
    assert "baseline" not in merged
    assert "--double-buffer" in merged
    assert "--no-gpu-decode" in merged


def test_run_fails_when_eval_writes_distinct_mot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies = [LINE_A, LINE_A_SCORE]

    def fake_run_one_eval(**kwargs: object) -> SimpleNamespace:
        out_dir = kwargs["out_dir"]
        assert isinstance(out_dir, Path)
        body = bodies.pop(0)
        _write_mot(out_dir, "MOT17-02-SDP", body)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(harness, "run_one_eval", fake_run_one_eval)
    rc = harness.cmd_run(
        n=2,
        sleep=0.0,
        artifact_dir=tmp_path / "art",
        forwarded=["--preset", "baseline"],
    )
    assert rc == 1


def test_run_fails_when_eval_exits_zero_with_empty_mot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-#363 scratch harness treated this as success."""

    calls = {"n": 0}

    def fake_run_one_eval(**kwargs: object) -> SimpleNamespace:
        out_dir = kwargs["out_dir"]
        assert isinstance(out_dir, Path)
        out_dir.mkdir(parents=True, exist_ok=True)
        calls["n"] += 1
        if calls["n"] == 1:
            _write_mot(out_dir, "MOT17-02-SDP", LINE_A)
        else:
            (out_dir / "MOT17-02-SDP.txt").write_text("", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(harness, "run_one_eval", fake_run_one_eval)
    rc = harness.cmd_run(
        n=2,
        sleep=0.0,
        artifact_dir=tmp_path / "art",
        forwarded=["--max-frames", "10"],
    )
    assert rc == 1


def test_run_fails_when_eval_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_one_eval(**kwargs: object) -> SimpleNamespace:
        out_dir = kwargs["out_dir"]
        assert isinstance(out_dir, Path)
        out_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(harness, "run_one_eval", fake_run_one_eval)
    rc = harness.cmd_run(
        n=2,
        sleep=0.0,
        artifact_dir=tmp_path / "art",
        forwarded=[],
    )
    assert rc == 1


def test_run_fails_when_eval_returns_nonzero_with_identical_mot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete matching MOT does not wash out a non-zero eval exit."""

    calls = {"n": 0}

    def fake_run_one_eval(**kwargs: object) -> SimpleNamespace:
        out_dir = kwargs["out_dir"]
        assert isinstance(out_dir, Path)
        _write_mot(out_dir, "MOT17-02-SDP", LINE_A, LINE_B)
        calls["n"] += 1
        return SimpleNamespace(returncode=1 if calls["n"] == 2 else 0)

    monkeypatch.setattr(harness, "run_one_eval", fake_run_one_eval)
    artifact = tmp_path / "art"
    rc = harness.cmd_run(
        n=2,
        sleep=0.0,
        artifact_dir=artifact,
        forwarded=[],
    )
    assert rc == 1
    exits = (artifact / "eval_exits.json").read_text(encoding="utf-8")
    assert '"had_eval_failure": true' in exits


def test_run_passes_when_all_mot_files_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_one_eval(**kwargs: object) -> SimpleNamespace:
        out_dir = kwargs["out_dir"]
        assert isinstance(out_dir, Path)
        _write_mot(out_dir, "MOT17-02-SDP", LINE_A, LINE_B)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(harness, "run_one_eval", fake_run_one_eval)
    rc = harness.cmd_run(
        n=3,
        sleep=0.0,
        artifact_dir=tmp_path / "art",
        forwarded=[],
    )
    assert rc == 0


def test_run_rejects_managed_output_flag(tmp_path: Path) -> None:
    rc = harness.cmd_run(
        n=2,
        sleep=0.0,
        artifact_dir=tmp_path / "art",
        forwarded=["--output", str(tmp_path / "nope")],
    )
    assert rc == 2


def test_cli_run_n_must_be_at_least_two() -> None:
    assert harness.main(["run", "-n", "1"]) == 2


EVIDENCE = Path(
    os.environ.get(
        "SACCADE_EVAL_REPEAT_EVIDENCE",
        str(
            Path.home()
            / ".local/state/saccade/perf/nogpudecode-reproducibility-20260907"
        ),
    )
)
_EVIDENCE_REF = EVIDENCE / "S_r1" / "MOT17-02-SDP.txt"
_EVIDENCE_DIV = EVIDENCE / "S_r3" / "MOT17-02-SDP.txt"
_EVIDENCE_SAME = EVIDENCE / "S_r2" / "MOT17-02-SDP.txt"


@pytest.mark.skipif(
    not _EVIDENCE_REF.exists(), reason="stored #363 evidence not on host"
)
def test_stored_block_s_divergence_fails_closed() -> None:
    report = compare_run_dirs([_EVIDENCE_REF.parent, _EVIDENCE_DIV.parent])
    assert not report.ok
    diff = report.reports[0].first_diffs[1]
    assert diff is not None
    assert diff.kind == KIND_GEOMETRY_OR_SCORE
    assert diff.frame == 63


@pytest.mark.skipif(
    not _EVIDENCE_SAME.exists(), reason="stored #363 evidence not on host"
)
def test_stored_block_s_matching_pair_passes() -> None:
    report = compare_run_dirs([_EVIDENCE_REF.parent, _EVIDENCE_SAME.parent])
    assert report.ok
