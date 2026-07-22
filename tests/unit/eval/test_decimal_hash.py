"""Tests for decimal-hash MOT canonicalization + hashing (perception.eval.decimal_hash)."""

# scope: eval
# function: contract
# lifecycle: active

import pytest

from saccade.perception.eval.decimal_hash import canonicalize_mot_lines, decimal_hash
from saccade.perception.eval._decimal_hash_tools import (
    Run,
    compare_chain_to_first_occurrence,
    diagnose,
    verdict_from_comparisons,
)
from scripts.tools.check_continuous_decimal_hash import _parse_args


def _records(*lines: str):
    return canonicalize_mot_lines(lines)


def test_global_id_is_excluded_from_decimal_hash() -> None:
    first = _records("1,10,1.00,2.00,3.00,4.00,0.5000,-1,-1,-1")
    second = _records("1,9999,1.00,2.00,3.00,4.00,0.5000,-1,-1,-1")

    assert decimal_hash(first) == decimal_hash(second)


def test_record_order_is_canonicalized_without_identity() -> None:
    first = _records(
        "2,20,4.00,2.00,3.00,4.00,0.5000", "1,10,1.00,2.00,3.00,4.00,0.7000"
    )
    second = _records(
        "1,500,1.00,2.00,3.00,4.00,0.7000", "2,1,4.00,2.00,3.00,4.00,0.5000"
    )

    assert decimal_hash(first) == decimal_hash(second)


def test_serialized_decimal_change_changes_hash() -> None:
    first = _records("1,1,1.00,2.00,3.00,4.00,0.5000")
    second = _records("1,1,1.01,2.00,3.00,4.00,0.5000")

    assert decimal_hash(first) != decimal_hash(second)


def test_count_difference_is_structural_divergence() -> None:
    first = _records("1,1,1.00,2.00,3.00,4.00,0.5000")
    second = _records(
        "1,1,1.00,2.00,3.00,4.00,0.5000", "2,2,1.00,2.00,3.00,4.00,0.5000"
    )
    reference = Run(1, "MOT17-04-SDP", tuple(first), decimal_hash(first))
    compared = Run(2, "MOT17-04-SDP", tuple(second), decimal_hash(second))

    assert (
        diagnose(reference, compared, max_records=20)["classification"]
        == "structural_divergence"
    )


def test_structural_diagnostic_groups_multiple_records_by_frame() -> None:
    first = _records(
        "1,1,1.00,2.00,3.00,4.00,0.5000",
        "1,2,2.00,2.00,3.00,4.00,0.5000",
    )
    second = _records("1,1,1.00,2.00,3.00,4.00,0.5000")
    reference = Run(1, "MOT17-04-SDP", tuple(first), decimal_hash(first))
    compared = Run(2, "MOT17-04-SDP", tuple(second), decimal_hash(second))

    diagnosis = diagnose(reference, compared, max_records=20)

    assert diagnosis["first_diff_frame"] == 1
    assert len(diagnosis["frame_multiset_differences"][0]["reference_records"]) == 2


def test_structural_diagnostic_preserves_same_decimal_record_multiplicity() -> None:
    first = _records(
        "1,1,1.00,2.00,3.00,4.00,0.5000",
        "1,2,1.00,2.00,3.00,4.00,0.5000",
    )
    second = _records("1,1,1.00,2.00,3.00,4.00,0.5000")
    reference = Run(1, "MOT17-04-SDP", tuple(first), decimal_hash(first))
    compared = Run(2, "MOT17-04-SDP", tuple(second), decimal_hash(second))

    diagnosis = diagnose(reference, compared, max_records=20)

    assert diagnosis["first_diff_frame"] == 1


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_serialized_values_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _records(f"1,1,-0.00,{value},3.00,4.00,0.5000")


def test_negative_zero_is_canonicalized_to_zero() -> None:
    first = _records("1,1,-0.00,2.00,3.00,4.00,0.5000")
    second = _records("1,2,0.00,2.00,3.00,4.00,0.5000")

    assert first[0].x_centipixel == 0
    assert decimal_hash(first) == decimal_hash(second)


def test_unserialized_precision_is_rejected() -> None:
    with pytest.raises(ValueError, match="serialized scale"):
        _records("1,1,1.000001,2.00,3.00,4.00,0.5000")


@pytest.mark.parametrize("flag", ["--cpp-threads", "--cpp-threads=2"])
def test_cpp_threads_is_rejected_before_evaluator(monkeypatch, flag: str) -> None:
    argv = ["probe", "--sequences", "MOT17-04-SDP", "--output", "out/test", flag]
    if flag == "--cpp-threads":
        argv.append("2")
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit):
        _parse_args()


def test_compare_chain_enriches_first_occurrence_artifacts() -> None:
    first = _records("1,1,1.00,2.00,3.00,4.00,0.5000")
    second = _records("1,1,1.01,2.00,3.00,4.00,0.5000")
    runs = [
        Run(1, "MOT17-04-SDP", tuple(first), decimal_hash(first)),
        Run(2, "MOT17-04-SDP", tuple(second), decimal_hash(second)),
    ]
    comparisons = compare_chain_to_first_occurrence(runs)

    assert comparisons[0]["sequence_occurrence"] == 1
    assert comparisons[1]["sequence_occurrence"] == 2
    assert comparisons[1]["reference_hash"] == runs[0].decimal_hash
    assert comparisons[1]["observed_hash"] == runs[1].decimal_hash
    assert verdict_from_comparisons(comparisons) == "decimal_divergence"
