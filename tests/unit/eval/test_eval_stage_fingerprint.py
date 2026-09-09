"""Fail-closed per-stage fingerprints for issue #363.

Device-free: identical inputs must hash identically, a single-stage
mutation must localize that stage, and missing or incomplete logs fail.
A first divergent stage is an observability bound, not a mechanism.
"""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tools.eval_stage_fingerprint import (  # noqa: E402
    ALLOCATOR_RESERVED_MATERIAL_BYTES,
    BUDGET_EXHAUSTED_CLAIM,
    CONDITION2_SUFFICIENT_SESSION_CLAIM,
    CONDITION2_RULES,
    DIVERGENCE_NAME,
    _track_count,
    INSTRUMENTATION_INSUFFICIENT_CLAIM,
    KIND_FIRST_OBSERVABLE,
    KIND_INCOMPLETE,
    KIND_INSUFFICIENT,
    KIND_STAGE_ONLY,
    GPU_DECODE_BUDGET_EXHAUSTED_CLAIM,
    GPU_DECODE_LOCALIZATION_BUDGET_RUNS,
    LOCALIZATION_CONFIG_GPU_DECODE,
    LOCALIZATION_BUDGET_RUNS,
    OBSERVER_EFFECT_BOUNDED_CLAIM,
    OBSERVER_EFFECT_IDENTIFIED_CLAIM,
    OBSERVER_EFFECT_NONE_CLAIM,
    OBSERVER_EFFECT_SITES,
    OE_BOUNDED,
    OE_IDENTIFIED,
    OE_NONE,
    SESSION_BUDGET_EXHAUSTED_IDENTICAL,
    SESSION_IN_PROGRESS,
    SESSION_INVALID,
    SESSION_PAIR_FOUND,
    STAGES,
    VERDICT_CANDIDATE_SUFFICIENT,
    VERDICT_INSUFFICIENT,
    VERDICT_NOT_APPLICABLE,
    VERDICT_SUFFICIENT,
    StageFingerprintCollector,
    compare_stage_fingerprints,
    fingerprint_detections,
    fingerprint_mot_lines,
    format_stage_report,
    install_eval_hooks,
    measure_allocator_reserved_growth,
    read_condition_2,
    read_localization_session,
    read_observer_effect,
    write_fingerprint_log,
    write_first_divergence,
)
from scripts.tools import check_eval_repeat_identity as harness  # noqa: E402


SEQ = "MOT17-02-SDP"
FRAME = 15
LINE = "15,7,934.74,435.47,36.08,81.58,0.5767,-1,-1,-1"
LINE_SCORE = "15,7,934.74,435.47,36.08,81.58,0.5837,-1,-1,-1"
BOX = np.array([[934.74, 435.47, 970.82, 517.05]], dtype=np.float32)
SCORE = np.array([0.5767], dtype=np.float32)
SCORE_MUT = np.array([0.5837], dtype=np.float32)
CLS = np.array([0], dtype=np.int32)
TID = np.array([7], dtype=np.int32)


def _write_mot(directory: Path, sequence: str, *lines: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sequence}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _det(score: np.ndarray = SCORE, ids: np.ndarray | None = None) -> dict[str, Any]:
    return fingerprint_detections(boxes=BOX, scores=score, classes=CLS, ids=ids)


def _mot(line: str = LINE) -> dict[str, Any]:
    return fingerprint_mot_lines([line])


def _records_for_frame(
    *,
    score: np.ndarray = SCORE,
    line: str = LINE,
    mutate_stage: str | None = None,
) -> list[dict[str, Any]]:
    det = _det(score)
    trk = _det(score, ids=TID)
    mot = _mot(line)
    mutated_det = _det(SCORE_MUT)
    mutated_trk = _det(SCORE_MUT, ids=TID)
    mutated_mot = _mot(LINE_SCORE)
    records: list[dict[str, Any]] = []
    for stage in STAGES:
        if stage in ("detector_output", "post_nms", "tracker_input"):
            payload = mutated_det if mutate_stage == stage else det
        elif stage == "tracker_output":
            payload = mutated_trk if mutate_stage == stage else trk
        else:
            payload = mutated_mot if mutate_stage == stage else mot
        records.append(
            {
                "sequence": SEQ,
                "frame": FRAME,
                "stage": stage,
                **payload,
            }
        )
    return records


def _write_fp(
    run_dir: Path,
    records: list[dict[str, Any]],
    *,
    complete: bool | None = None,
    include_payloads: bool = True,
) -> None:
    write_fingerprint_log(
        run_dir / "stage_fingerprint",
        records,
        include_payloads=include_payloads,
        limitations=(),
        complete=complete,
    )


def test_condition2_table_covers_every_stage_in_order() -> None:
    assert tuple(CONDITION2_RULES) == STAGES
    for predecessor, stage in zip((None, *STAGES[:-1]), STAGES):
        rule = CONDITION2_RULES[stage]
        assert rule.first_divergent_stage == stage
        assert rule.last_identical_stage == predecessor


def test_condition2_verdicts_are_frozen() -> None:
    expected = {
        "detector_output": VERDICT_INSUFFICIENT,
        "post_nms": VERDICT_CANDIDATE_SUFFICIENT,
        "tracker_input": VERDICT_INSUFFICIENT,
        "tracker_output": VERDICT_INSUFFICIENT,
        "mot": VERDICT_SUFFICIENT,
        "mot_file": VERDICT_SUFFICIENT,
    }
    assert {
        name: rule.producing_path_verdict for name, rule in CONDITION2_RULES.items()
    } == expected


def test_condition2_claims_are_frozen() -> None:
    assert CONDITION2_RULES["detector_output"].allowed_claim == (
        "divergence 已在 detect_fn output 出現；"
        "decode / preprocess / detector internals 未切開"
    )
    assert CONDITION2_RULES["post_nms"].allowed_claim == (
        "divergence 被界定在 detector output → evaluator NMS"
    )
    assert CONDITION2_RULES["tracker_input"].allowed_claim == (
        "divergence 在 post-NMS → _run_track 前產生；ReID / GMC 路徑仍多義"
    )
    assert CONDITION2_RULES["tracker_output"].allowed_claim == (
        "divergence 在 tracker execution 內產生；Kalman / association / buffer 未切開"
    )
    assert CONDITION2_RULES["mot"].allowed_claim == (
        "divergence 在 global-ID mapping / serialization 路徑產生"
    )
    assert CONDITION2_RULES["mot_file"].allowed_claim == (
        "divergence 在 sequence-level postprocess 路徑產生"
    )


def test_sufficient_does_not_close_the_issue_or_claim_mechanism() -> None:
    for stage in ("mot", "mot_file", "post_nms"):
        reading = read_condition_2(kind=KIND_FIRST_OBSERVABLE, stage=stage)
        assert reading.issue_close is False
        assert reading.mechanism_claim is False


def test_unknown_stage_fails_closed() -> None:
    with pytest.raises(ValueError, match="no condition-2 rule"):
        read_condition_2(kind=KIND_FIRST_OBSERVABLE, stage="decode")


def test_localization_budget_is_frozen() -> None:
    assert LOCALIZATION_BUDGET_RUNS == 16
    assert GPU_DECODE_LOCALIZATION_BUDGET_RUNS == 8
    assert GPU_DECODE_LOCALIZATION_BUDGET_RUNS != LOCALIZATION_BUDGET_RUNS


def test_pair_inside_budget_applies_condition2_rules() -> None:
    reading = read_localization_session(n_runs=4, divergent_pair=True)
    assert reading.kind == SESSION_PAIR_FOUND
    assert reading.apply_condition2_rules is True
    assert reading.condition_2_advanced is False
    assert reading.condition_1_advanced is False
    assert reading.issue_close is False
    assert reading.mechanism_claim is False


def test_identical_runs_before_budget_are_in_progress() -> None:
    reading = read_localization_session(n_runs=8, divergent_pair=False)
    assert reading.kind == SESSION_IN_PROGRESS
    assert reading.apply_condition2_rules is False
    assert reading.allowed_claim == ""


def test_budget_exhausted_identical_does_not_apply_condition2() -> None:
    reading = read_localization_session(
        n_runs=LOCALIZATION_BUDGET_RUNS, divergent_pair=False
    )
    assert reading.kind == SESSION_BUDGET_EXHAUSTED_IDENTICAL
    assert reading.apply_condition2_rules is False
    assert reading.allowed_claim == BUDGET_EXHAUSTED_CLAIM
    assert reading.producing_path_unresolved is True
    assert reading.condition_2_advanced is False
    assert reading.mechanism_claim is False
    assert reading.issue_close is False


def test_over_budget_session_fails_closed() -> None:
    with pytest.raises(ValueError, match="exceeds preregistered"):
        read_localization_session(
            n_runs=LOCALIZATION_BUDGET_RUNS + 1, divergent_pair=False
        )


def test_invalid_session_has_no_localization_reading() -> None:
    reading = read_localization_session(
        n_runs=2,
        divergent_pair=False,
        session_valid=False,
    )
    assert reading.kind == SESSION_INVALID
    assert reading.apply_condition2_rules is False
    assert reading.condition_2_advanced is False
    assert "no localization-session reading" in reading.allowed_claim


def test_gpu_decode_budget_is_not_block_s_budget() -> None:
    reading = read_localization_session(
        n_runs=GPU_DECODE_LOCALIZATION_BUDGET_RUNS,
        divergent_pair=False,
        config=LOCALIZATION_CONFIG_GPU_DECODE,
    )
    assert reading.kind == SESSION_BUDGET_EXHAUSTED_IDENTICAL
    assert reading.budget_runs == 8
    assert reading.allowed_claim == GPU_DECODE_BUDGET_EXHAUSTED_CLAIM
    assert reading.allowed_claim != BUDGET_EXHAUSTED_CLAIM
    assert reading.apply_condition2_rules is False
    assert reading.condition_2_advanced is False
    assert reading.mechanism_claim is False
    assert reading.issue_close is False


def test_gpu_decode_pair_inside_budget_applies_condition2() -> None:
    reading = read_localization_session(
        n_runs=2,
        divergent_pair=True,
        config=LOCALIZATION_CONFIG_GPU_DECODE,
    )
    assert reading.kind == SESSION_PAIR_FOUND
    assert reading.budget_runs == GPU_DECODE_LOCALIZATION_BUDGET_RUNS
    assert reading.apply_condition2_rules is True
    assert reading.condition_2_advanced is False
    assert reading.mechanism_claim is False


def test_sufficient_verdict_is_condition2_evidence_not_issue_close() -> None:
    reading = read_localization_session(
        n_runs=2,
        divergent_pair=True,
        config=LOCALIZATION_CONFIG_GPU_DECODE,
        producing_path_verdict=VERDICT_SUFFICIENT,
    )
    assert reading.kind == SESSION_PAIR_FOUND
    assert reading.condition_2_advanced is True
    assert reading.producing_path_unresolved is False
    assert reading.condition_1_advanced is False
    assert reading.issue_close is False
    assert reading.mechanism_claim is False
    assert reading.allowed_claim == CONDITION2_SUFFICIENT_SESSION_CLAIM


def test_candidate_sufficient_does_not_count_as_condition2_sufficient() -> None:
    reading = read_localization_session(
        n_runs=2,
        divergent_pair=True,
        producing_path_verdict=VERDICT_CANDIDATE_SUFFICIENT,
    )
    assert reading.condition_2_advanced is False
    assert reading.producing_path_unresolved is True
    assert reading.issue_close is False


def test_gpu_decode_over_budget_fails_closed() -> None:
    with pytest.raises(ValueError, match="gpu_decode"):
        read_localization_session(
            n_runs=GPU_DECODE_LOCALIZATION_BUDGET_RUNS + 1,
            divergent_pair=False,
            config=LOCALIZATION_CONFIG_GPU_DECODE,
        )


def test_run_rejects_n_above_localization_budget(tmp_path: Path) -> None:
    rc = harness.cmd_run(
        n=LOCALIZATION_BUDGET_RUNS + 1,
        sleep=0.0,
        artifact_dir=tmp_path / "art",
        forwarded=[],
        stage_fingerprint=True,
    )
    assert rc == 2


def test_run_rejects_n_above_gpu_decode_budget(tmp_path: Path) -> None:
    rc = harness.cmd_run(
        n=GPU_DECODE_LOCALIZATION_BUDGET_RUNS + 1,
        sleep=0.0,
        artifact_dir=tmp_path / "art",
        forwarded=[],
        stage_fingerprint=True,
        localization_config=LOCALIZATION_CONFIG_GPU_DECODE,
    )
    assert rc == 2
    rc_block_s_n = harness.cmd_run(
        n=LOCALIZATION_BUDGET_RUNS,
        sleep=0.0,
        artifact_dir=tmp_path / "art-s",
        forwarded=[],
        stage_fingerprint=True,
        localization_config=LOCALIZATION_CONFIG_GPU_DECODE,
    )
    assert rc_block_s_n == 2


def test_gpu_decode_refuses_no_gpu_decode_flag(tmp_path: Path) -> None:
    rc = harness.cmd_run(
        n=2,
        sleep=0.0,
        artifact_dir=tmp_path / "art",
        forwarded=["--no-gpu-decode"],
        stage_fingerprint=True,
        localization_config=LOCALIZATION_CONFIG_GPU_DECODE,
    )
    assert rc == 2


def test_gpu_decode_fingerprint_stops_on_first_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def fake_run_one_eval(**kwargs: object) -> SimpleNamespace:
        out_dir = kwargs["out_dir"]
        assert isinstance(out_dir, Path)
        calls["n"] += 1
        if calls["n"] == 1:
            _write_mot(out_dir, SEQ, LINE)
            _write_fp(out_dir, _records_for_frame())
        else:
            _write_mot(out_dir, SEQ, LINE_SCORE)
            _write_fp(
                out_dir,
                _records_for_frame(line=LINE_SCORE, mutate_stage="post_nms"),
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(harness, "run_one_eval", fake_run_one_eval)
    artifact = tmp_path / "art"
    rc = harness.cmd_run(
        n=GPU_DECODE_LOCALIZATION_BUDGET_RUNS,
        sleep=0.0,
        artifact_dir=artifact,
        forwarded=[],
        stage_fingerprint=True,
        localization_config=LOCALIZATION_CONFIG_GPU_DECODE,
    )
    assert rc == 1
    assert calls["n"] == 2
    session = json.loads((artifact / "localization_session.json").read_text())
    assert session["kind"] == SESSION_PAIR_FOUND
    assert session["n_runs"] == 2
    assert session["budget_runs"] == 8
    assert session["config"] == LOCALIZATION_CONFIG_GPU_DECODE
    assert session["apply_condition2_rules"] is True
    assert session["mechanism_claim"] is False
    assert session["condition_2_advanced"] is False
    assert session["issue_close"] is False
    first = json.loads((artifact / "first_divergence.json").read_text())
    assert first["last_identical_stage"] == "detector_output"
    assert first["first_divergent_stage"] == "post_nms"


def test_run_without_fingerprint_does_not_use_localization_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_one_eval(**kwargs: object) -> SimpleNamespace:
        out_dir = kwargs["out_dir"]
        assert isinstance(out_dir, Path)
        _write_mot(out_dir, SEQ, LINE)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(harness, "run_one_eval", fake_run_one_eval)
    rc = harness.cmd_run(
        n=LOCALIZATION_BUDGET_RUNS + 1,
        sleep=0.0,
        artifact_dir=tmp_path / "art",
        forwarded=[],
        stage_fingerprint=False,
    )
    assert rc == 0


def test_boundary_doc_copies_the_frozen_table() -> None:
    text = (
        ROOT / "docs/research/eval/eval_repeat_identity_boundary_20260907.md"
    ).read_text(encoding="utf-8")
    for stage, rule in CONDITION2_RULES.items():
        assert f"`{stage}`" in text
        assert f"`{rule.producing_path_verdict}`" in text
        assert rule.allowed_claim in text
    assert INSTRUMENTATION_INSUFFICIENT_CLAIM in text
    assert "live run 前凍結" in text
    assert "localization experiment" in text
    assert str(LOCALIZATION_BUDGET_RUNS) in text
    assert BUDGET_EXHAUSTED_CLAIM in text
    assert GPU_DECODE_BUDGET_EXHAUSTED_CLAIM in text
    assert CONDITION2_SUFFICIENT_SESSION_CLAIM in text
    assert str(GPU_DECODE_LOCALIZATION_BUDGET_RUNS) in text
    assert OBSERVER_EFFECT_IDENTIFIED_CLAIM in text
    assert OBSERVER_EFFECT_BOUNDED_CLAIM in text
    assert OBSERVER_EFFECT_NONE_CLAIM in text
    assert "live 量測後不得移動門檻" in text


def test_identical_detection_inputs_produce_identical_fingerprints() -> None:
    first = fingerprint_detections(boxes=BOX, scores=SCORE, classes=CLS, ids=TID)
    second = fingerprint_detections(
        boxes=BOX.copy(), scores=SCORE.copy(), classes=CLS.copy(), ids=TID.copy()
    )
    assert first["ordered_bit_hash"] == second["ordered_bit_hash"]
    assert first["multiset_canonical_hash"] == second["multiset_canonical_hash"]
    assert first["rows"] == second["rows"]


def test_identical_mot_lines_produce_identical_fingerprints() -> None:
    assert fingerprint_mot_lines([LINE]) == fingerprint_mot_lines([LINE])


def test_score_mutation_changes_bit_hash() -> None:
    first = fingerprint_detections(boxes=BOX, scores=SCORE, classes=CLS)
    second = fingerprint_detections(boxes=BOX, scores=SCORE_MUT, classes=CLS)
    assert first["ordered_bit_hash"] != second["ordered_bit_hash"]
    assert first["id_free_canonical_hash"] != second["id_free_canonical_hash"]


def test_permutation_changes_ordered_hash_not_multiset() -> None:
    boxes = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 8.0, 10.0]], dtype=np.float32)
    scores = np.array([0.4, 0.5], dtype=np.float32)
    classes = np.array([0, 0], dtype=np.int32)
    first = fingerprint_detections(boxes=boxes, scores=scores, classes=classes)
    second = fingerprint_detections(
        boxes=boxes[::-1], scores=scores[::-1], classes=classes[::-1]
    )
    assert first["ordered_bit_hash"] != second["ordered_bit_hash"]
    assert first["multiset_canonical_hash"] == second["multiset_canonical_hash"]


def test_track_count_treats_dummy_emit_as_empty() -> None:
    assert _track_count({"count": 0}) == 0
    assert _track_count({"count": 4}) == 4
    assert _track_count({}) == 0


def test_collector_hashes_cpu_observations_without_cuda_sync() -> None:
    collector = StageFingerprintCollector(Path("/unused"), include_payloads=True)
    collector.observe_detection(SEQ, FRAME, "detector_output", BOX, SCORE, CLS)
    collector.observe_detection(SEQ, FRAME, "post_nms", BOX, SCORE, CLS)
    collector.observe_detection(SEQ, FRAME, "tracker_input", BOX, SCORE, CLS)
    collector.observe_emit(
        SEQ,
        FRAME,
        {
            "count": 1,
            "boxes": BOX,
            "scores": SCORE,
            "classes": CLS,
            "ids": TID,
        },
        [LINE],
    )
    collector.observe_mot_file(SEQ, [LINE])
    assert collector._gpu == []
    assert {item["stage"] for item in collector.records} == set(STAGES)


def test_identical_run_dirs_pass(tmp_path: Path) -> None:
    records = _records_for_frame()
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    _write_fp(a, records)
    _write_fp(b, records)
    report = compare_stage_fingerprints([a, b])
    assert report.ok
    assert report.complete
    assert report.first_divergence is None
    assert DIVERGENCE_NAME not in {path.name for path in tmp_path.iterdir()}


@pytest.mark.parametrize("stage", STAGES)
def test_single_stage_mutation_localizes_that_stage(tmp_path: Path, stage: str) -> None:
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    _write_fp(a, _records_for_frame())
    _write_fp(b, _records_for_frame(mutate_stage=stage))
    report = compare_stage_fingerprints([a, b], mot_diverged=True)
    assert not report.ok
    first = report.first_divergence
    assert first is not None
    assert first.kind == KIND_FIRST_OBSERVABLE
    assert first.sequence == SEQ
    assert first.frame == FRAME
    assert first.stage == stage
    assert first.first_divergent_stage == stage
    rule = CONDITION2_RULES[stage]
    assert first.last_identical_stage == rule.last_identical_stage
    assert first.producing_path_verdict == rule.producing_path_verdict
    assert first.allowed_claim == rule.allowed_claim
    assert first.mechanism_claim is False
    assert first.issue_close is False
    dump = tmp_path / DIVERGENCE_NAME
    write_first_divergence(dump, report)
    payload = json.loads(dump.read_text(encoding="utf-8"))
    assert payload["stage"] == stage
    assert payload["first_divergent_stage"] == stage
    assert payload["last_identical_stage"] == rule.last_identical_stage
    assert payload["producing_path_verdict"] == rule.producing_path_verdict
    assert payload["reference_payload"] != payload["other_payload"]
    assert payload["mechanism_claim"] is False
    assert payload["issue_close"] is False


def test_earlier_stage_wins_over_later_mutation(tmp_path: Path) -> None:
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    ref = _records_for_frame()
    other = _records_for_frame()
    mutated = _det(SCORE_MUT)
    for item in other:
        if item["stage"] in ("post_nms", "tracker_output"):
            item.update(mutated)
            if item["stage"] == "tracker_output":
                item.update(_det(SCORE_MUT, ids=TID))
    _write_fp(a, ref)
    _write_fp(b, other)
    report = compare_stage_fingerprints([a, b], mot_diverged=True)
    assert report.first_divergence is not None
    assert report.first_divergence.stage == "post_nms"


def test_missing_fingerprint_dir_fails(tmp_path: Path) -> None:
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    _write_fp(a, _records_for_frame())
    b.mkdir()
    report = compare_stage_fingerprints([a, b])
    assert not report.ok
    assert not report.complete
    assert report.first_divergence is not None
    assert report.first_divergence.kind == KIND_INCOMPLETE
    assert report.first_divergence.producing_path_verdict == VERDICT_NOT_APPLICABLE
    assert report.first_divergence.issue_close is False
    assert any("missing" in reason for reason in report.reasons)


def test_mismatched_key_coverage_is_incomplete_not_a_stage_boundary(
    tmp_path: Path,
) -> None:
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    _write_fp(a, _records_for_frame())
    _write_fp(b, _records_for_frame()[:-1], complete=True)
    report = compare_stage_fingerprints([a, b], mot_diverged=True)
    assert not report.ok
    assert not report.complete
    assert report.first_divergence is not None
    assert report.first_divergence.kind == KIND_INCOMPLETE
    assert report.first_divergence.producing_path_verdict == VERDICT_NOT_APPLICABLE
    assert "coverage differs" in report.reasons[0]


def test_mixed_fingerprint_schemas_fail_closed(tmp_path: Path) -> None:
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    _write_fp(a, _records_for_frame())
    _write_fp(b, _records_for_frame())
    manifest_path = b / "stage_fingerprint" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "eval_stage_fingerprint_v1"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    report = compare_stage_fingerprints([a, b])
    assert not report.ok
    assert not report.complete
    assert report.first_divergence is not None
    assert report.first_divergence.kind == KIND_INCOMPLETE
    assert "cannot be mixed" in report.reasons[0]


def test_legacy_v1_artifacts_remain_comparable_with_each_other(tmp_path: Path) -> None:
    runs = [tmp_path / "r1", tmp_path / "r2"]
    for run in runs:
        _write_fp(run, _records_for_frame())
        manifest_path = run / "stage_fingerprint" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema"] = "eval_stage_fingerprint_v1"
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    assert compare_stage_fingerprints(runs).ok


def test_incomplete_manifest_fails(tmp_path: Path) -> None:
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    records = _records_for_frame()
    _write_fp(a, records, complete=True)
    _write_fp(b, records, complete=False)
    report = compare_stage_fingerprints([a, b])
    assert not report.ok
    assert not report.complete
    assert any("incomplete" in reason for reason in report.reasons)


def test_empty_log_fails(tmp_path: Path) -> None:
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    _write_fp(a, _records_for_frame())
    _write_fp(b, [], complete=True)
    report = compare_stage_fingerprints([a, b])
    assert not report.ok
    assert not report.complete


def test_mot_divergence_with_matching_stages_is_insufficient(tmp_path: Path) -> None:
    records = _records_for_frame()
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    _write_fp(a, records)
    _write_fp(b, records)
    report = compare_stage_fingerprints([a, b], mot_diverged=True)
    assert not report.ok
    assert report.complete
    assert report.instrumentation_sufficient is False
    assert report.first_divergence is not None
    assert report.first_divergence.kind == KIND_INSUFFICIENT
    assert report.first_divergence.last_identical_stage == "mot_file"
    assert report.first_divergence.first_divergent_stage is None
    assert report.first_divergence.producing_path_verdict == VERDICT_INSUFFICIENT
    assert report.first_divergence.allowed_claim == INSTRUMENTATION_INSUFFICIENT_CLAIM
    assert report.first_divergence.issue_close is False
    assert "not sufficient" in report.reasons[0]


def test_format_report_states_observability_not_mechanism(tmp_path: Path) -> None:
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    _write_fp(a, _records_for_frame())
    _write_fp(b, _records_for_frame(mutate_stage="tracker_input"))
    text = format_stage_report(compare_stage_fingerprints([a, b], mot_diverged=True))
    assert "FAIL" in text
    assert "last_identical_stage=post_nms" in text
    assert "first_divergent_stage=tracker_input" in text
    assert "producing_path_verdict=insufficient" in text
    assert "mechanism_claim=False" in text
    assert "issue_close=False" in text
    assert "ReID / GMC" in text


def test_compare_without_flag_ignores_missing_fingerprints(
    tmp_path: Path,
) -> None:
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    _write_mot(a, SEQ, LINE)
    _write_mot(b, SEQ, LINE)
    assert harness.main(["compare", str(a), str(b)]) == 0


def test_compare_with_flag_fails_closed_when_fingerprints_missing(
    tmp_path: Path,
) -> None:
    a = tmp_path / "r1"
    b = tmp_path / "r2"
    _write_mot(a, SEQ, LINE)
    _write_mot(b, SEQ, LINE)
    assert harness.main(["compare", str(a), str(b), "--stage-fingerprint"]) == 1


def test_stage_only_divergence_does_not_advance_condition2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies = [LINE, LINE]
    mutations = [None, "post_nms"]

    def fake_run_one_eval(**kwargs: object) -> SimpleNamespace:
        out_dir = kwargs["out_dir"]
        assert isinstance(out_dir, Path)
        assert kwargs["stage_fingerprint"] is True
        _write_mot(out_dir, SEQ, bodies.pop(0))
        _write_fp(out_dir, _records_for_frame(mutate_stage=mutations.pop(0)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(harness, "run_one_eval", fake_run_one_eval)
    artifact = tmp_path / "art"
    rc = harness.cmd_run(
        n=2,
        sleep=0.0,
        artifact_dir=artifact,
        forwarded=["--preset", "baseline"],
        stage_fingerprint=True,
    )
    assert rc == 1
    summary = json.loads(
        (artifact / "stage_fingerprint.json").read_text(encoding="utf-8")
    )
    assert summary["first_divergence"]["stage"] == "post_nms"
    assert summary["first_divergence"]["frame"] == FRAME
    dump = json.loads((artifact / "first_divergence.json").read_text(encoding="utf-8"))
    assert dump["stage"] == "post_nms"
    assert dump["kind"] == KIND_STAGE_ONLY
    assert dump["last_identical_stage"] is None
    assert dump["first_divergent_stage"] is None
    assert dump["producing_path_verdict"] == VERDICT_NOT_APPLICABLE
    assert dump["issue_close"] is False
    session = json.loads((artifact / "localization_session.json").read_text())
    assert session["kind"] == SESSION_IN_PROGRESS
    assert session["mot_pair_valid"] is False
    assert session["session_valid"] is True
    assert session["apply_condition2_rules"] is False
    assert session["condition_2_advanced"] is False
    assert session["eval_returncodes"] == [0, 0]
    assert "--no-gpu-decode" in session["eval_flags"]


def test_invalid_mot_output_does_not_stop_localization_as_a_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def fake_run_one_eval(**kwargs: object) -> SimpleNamespace:
        out_dir = kwargs["out_dir"]
        assert isinstance(out_dir, Path)
        calls["n"] += 1
        if calls["n"] == 2:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{SEQ}.txt").write_text("", encoding="utf-8")
        else:
            _write_mot(out_dir, SEQ, LINE)
        _write_fp(out_dir, _records_for_frame())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(harness, "run_one_eval", fake_run_one_eval)
    artifact = tmp_path / "art"
    rc = harness.cmd_run(
        n=3,
        sleep=0.0,
        artifact_dir=artifact,
        forwarded=[],
        stage_fingerprint=True,
    )
    assert rc == 1
    assert calls["n"] == 3
    session = json.loads((artifact / "localization_session.json").read_text())
    assert session["kind"] == SESSION_INVALID
    assert session["mot_pair_valid"] is False
    assert session["session_valid"] is False
    assert session["apply_condition2_rules"] is False


def test_run_default_does_not_require_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_one_eval(**kwargs: object) -> SimpleNamespace:
        out_dir = kwargs["out_dir"]
        assert isinstance(out_dir, Path)
        assert kwargs.get("stage_fingerprint") is False
        _write_mot(out_dir, SEQ, LINE)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(harness, "run_one_eval", fake_run_one_eval)
    rc = harness.cmd_run(
        n=2,
        sleep=0.0,
        artifact_dir=tmp_path / "art",
        forwarded=[],
    )
    assert rc == 0
    assert not (tmp_path / "art" / "first_divergence.json").exists()


def test_observer_effect_sites_are_frozen() -> None:
    assert "detector_output_clone" in OBSERVER_EFFECT_SITES
    assert "post_nms_clone" in OBSERVER_EFFECT_SITES
    assert "tracker_input_clone" in OBSERVER_EFFECT_SITES
    assert "gpu_snapshot_retention" in OBSERVER_EFFECT_SITES
    assert "finalize_synchronize" in OBSERVER_EFFECT_SITES
    assert OBSERVER_EFFECT_SITES["finalize_synchronize"]["before_mot_write"] is False
    assert OBSERVER_EFFECT_SITES["post_nms_clone"]["before_mot_write"] is True
    assert OBSERVER_EFFECT_SITES["mot_file_host_hash"]["before_mot_write"] is False
    assert ALLOCATOR_RESERVED_MATERIAL_BYTES == 2 * 1024 * 1024


def test_observer_effect_identified_does_not_advance_conditions() -> None:
    reading = read_observer_effect(
        extra_device_sync_during_frame_loop=True,
        allocator_reserved_grew_from_snapshots=False,
        producing_path_gpu_clone=True,
        post_eval_synchronize=True,
    )
    assert reading.kind == OE_IDENTIFIED
    assert reading.allowed_claim == OBSERVER_EFFECT_IDENTIFIED_CLAIM
    assert reading.issue_close is False
    assert reading.mechanism_claim is False
    assert reading.condition_1_advanced is False
    assert reading.condition_2_advanced is False
    assert reading.localization_budget_reopened is False


def test_observer_effect_reserved_growth_is_identified() -> None:
    reading = read_observer_effect(
        extra_device_sync_during_frame_loop=False,
        allocator_reserved_grew_from_snapshots=True,
        producing_path_gpu_clone=True,
        post_eval_synchronize=True,
    )
    assert reading.kind == OE_IDENTIFIED


def test_observer_effect_clones_without_join_or_reserved_growth_are_bounded() -> None:
    reading = read_observer_effect(
        extra_device_sync_during_frame_loop=False,
        allocator_reserved_grew_from_snapshots=False,
        producing_path_gpu_clone=True,
        post_eval_synchronize=True,
    )
    assert reading.kind == OE_BOUNDED
    assert reading.allowed_claim == OBSERVER_EFFECT_BOUNDED_CLAIM
    assert reading.condition_2_advanced is False


def test_observer_effect_none_when_inspected_boundaries_match() -> None:
    reading = read_observer_effect(
        extra_device_sync_during_frame_loop=False,
        allocator_reserved_grew_from_snapshots=False,
        producing_path_gpu_clone=False,
        post_eval_synchronize=False,
    )
    assert reading.kind == OE_NONE
    assert reading.allowed_claim == OBSERVER_EFFECT_NONE_CLAIM


def test_allocator_reserved_growth_threshold_is_frozen() -> None:
    assert (
        measure_allocator_reserved_growth(
            uninstrumented_reserved_bytes=10_000_000,
            instrumented_reserved_bytes=10_000_000 + ALLOCATOR_RESERVED_MATERIAL_BYTES,
            n_reserved_increases_on_clone=0,
        )
        is True
    )
    assert (
        measure_allocator_reserved_growth(
            uninstrumented_reserved_bytes=10_000_000,
            instrumented_reserved_bytes=10_000_000
            + ALLOCATOR_RESERVED_MATERIAL_BYTES
            - 1,
            n_reserved_increases_on_clone=0,
        )
        is False
    )
    assert (
        measure_allocator_reserved_growth(
            uninstrumented_reserved_bytes=10_000_000,
            instrumented_reserved_bytes=10_000_000,
            n_reserved_increases_on_clone=1,
        )
        is True
    )


def test_finalize_writes_observer_effect_measured(tmp_path: Path) -> None:
    collector = StageFingerprintCollector(tmp_path / "fp")
    collector.observe_detection(SEQ, FRAME, "detector_output", BOX, SCORE, CLS)
    collector.observe_mot_lines(SEQ, FRAME, "mot", [LINE])
    collector.finalize()
    measured = json.loads(
        (tmp_path / "fp" / "observer_effect_measured.json").read_text(encoding="utf-8")
    )
    assert measured["n_clone_samples"] == 0
    assert measured["n_reserved_increases_on_clone"] == 0
    assert {item["stage"] for item in collector.records} == set(STAGES)


def test_background_emit_observes_completed_lines_not_enqueue_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import saccade.perception.eval.evaluator as evaluator_module
    import saccade.perception.eval.stages as stages_module

    class CompletedFuture:
        def result(self) -> tuple[list[str], set[int], dict[int, int], dict[int, Any]]:
            return [LINE], {7}, {}, {}

    def background_run_emit(state: Any, **kwargs: Any) -> tuple[set[int], list[str]]:
        state.bg_future = CompletedFuture()
        return set(), []

    monkeypatch.setattr(stages_module, "_run_emit", background_run_emit)
    monkeypatch.setattr(evaluator_module, "_run_emit", background_run_emit)
    collector = StageFingerprintCollector(tmp_path / "fp")
    undo = install_eval_hooks(collector)
    state = SimpleNamespace(seq=SEQ, bg_future=None)
    track_results = {
        "count": 1,
        "boxes": BOX,
        "scores": SCORE,
        "classes": CLS,
        "ids": TID,
    }
    try:
        _, enqueue_lines = stages_module._run_emit(
            state,
            frame_id=FRAME,
            track_results=track_results,
        )
        assert enqueue_lines == []
        mot_before = [item for item in collector.records if item["stage"] == "mot"]
        assert mot_before == []
        completed = state.bg_future.result()
        assert completed[0] == [LINE]
    finally:
        undo()
    mot_after = [item for item in collector.records if item["stage"] == "mot"]
    assert len(mot_after) == 1
    assert mot_after[0]["count"] == 1
