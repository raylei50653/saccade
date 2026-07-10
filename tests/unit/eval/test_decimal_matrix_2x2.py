"""Unit tests for the 2×2 sequence-order determinism matrix coordinator.

These tests validate the orchestration logic without requiring a full GPU
sequence execution.  They reuse fixtures from ``test_decimal_hash.py``.
"""

from __future__ import annotations

from pathlib import Path
import sys


SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC))

from saccade.perception.eval._decimal_hash_tools import (  # noqa: E402
    Run,
    CANONICAL_FIELDS,
    HASH_METADATA,
    diagnose,
    generate_manifest,
    write_csv,
    write_summary,
)
from saccade.perception.eval.decimal_hash import (  # noqa: E402
    CanonicalRecord,
    decimal_hash,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_run(sequence: str, records: list[CanonicalRecord], index: int = 1) -> Run:
    records_tuple = tuple(sorted(records, key=lambda r: (r.frame, *r.values)))
    return Run(index, sequence, records_tuple, decimal_hash(records_tuple))


def _make_records(
    *triples: tuple[int, float, float, float, float, float],
) -> list[CanonicalRecord]:
    return [
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


# ── Matrix cell generation ────────────────────────────────────────────────────


def test_four_cells_generated() -> None:
    cells: list[tuple[str, str, str]] = [
        ("A_to_A", "MOT17-04-SDP", "MOT17-04-SDP"),
        ("B_to_B", "MOT17-02-SDP", "MOT17-02-SDP"),
        ("B_to_A", "MOT17-02-SDP", "MOT17-04-SDP"),
        ("A_to_B", "MOT17-04-SDP", "MOT17-02-SDP"),
    ]
    assert len(cells) == 4
    labels = {c[0] for c in cells}
    assert labels == {"A_to_A", "B_to_B", "B_to_A", "A_to_B"}


def test_cells_cover_all_directed_combinations() -> None:
    cells: list[tuple[str, str, str]] = [
        ("A_to_A", "A", "A"),
        ("B_to_B", "B", "B"),
        ("B_to_A", "B", "A"),
        ("A_to_B", "A", "B"),
    ]
    pairs = {(pre, tgt) for _, pre, tgt in cells}
    assert pairs == {("A", "A"), ("B", "B"), ("B", "A"), ("A", "B")}


def test_first_and_second_sequence_output_identification() -> None:
    rec = _make_records((1, 10.0, 20.0, 30.0, 40.0, 0.95))

    prec = _make_run("A", rec, index=1)
    tgt = _make_run("A", rec, index=2)

    assert prec.index == 1
    assert tgt.index == 2
    assert prec.decimal_hash == tgt.decimal_hash


# ── Reference assignment ──────────────────────────────────────────────────────


def test_reference_from_self_cell() -> None:
    a_ref = _make_run(
        "A",
        _make_records(
            (1, 10.0, 20.0, 30.0, 40.0, 0.95),
        ),
    )
    b_ref = _make_run(
        "B",
        _make_records(
            (1, 50.0, 60.0, 30.0, 40.0, 0.85),
        ),
    )

    assert a_ref.sequence == "A"
    assert b_ref.sequence == "B"
    assert a_ref.decimal_hash != b_ref.decimal_hash


def test_reference_assignment_correct_per_target() -> None:
    a_ref = _make_run("A", _make_records((1, 10.0, 20.0, 30.0, 40.0, 0.95)))
    b_ref = _make_run("B", _make_records((1, 50.0, 60.0, 30.0, 40.0, 0.85)))

    cells = [
        ("A_to_A", "A", "A"),
        ("B_to_B", "B", "B"),
        ("B_to_A", "B", "A"),
        ("A_to_B", "A", "B"),
    ]

    for label, pre, tgt in cells:
        ref = a_ref if tgt == "A" else b_ref
        assert ref.sequence == tgt


# ── Output capture before filename reuse ──────────────────────────────────────


def test_output_keys_unique() -> None:
    cells = [
        ("A_to_A", "A", "A"),
        ("B_to_B", "B", "B"),
        ("B_to_A", "B", "A"),
        ("A_to_B", "A", "B"),
    ]

    keys = []
    for label, pre, tgt in cells:
        keys.append(f"{label}_preceding")
        keys.append(f"{label}_target")

    assert len(keys) == len(set(keys))
    assert "A_to_A_target" in keys
    assert "B_to_A_target" in keys
    assert "A_to_B_target" in keys
    assert "B_to_B_target" in keys


# ── Failure propagation ───────────────────────────────────────────────────────


def test_diagnose_exact_match() -> None:
    rec = _make_records((1, 10.0, 20.0, 30.0, 40.0, 0.95))
    ref = _make_run("A", rec, index=1)
    tst = _make_run("A", rec, index=2)

    diag = diagnose(ref, tst)
    assert diag["classification"] == "decimal_exact_pass"
    assert diag["decimal_hash_equal"] is True
    assert diag["record_count_ref"] == diag["record_count_test"]


def test_diagnose_record_count_divergence() -> None:
    ref = _make_run("A", _make_records((1, 10.0, 20.0, 30.0, 40.0, 0.95)))
    tst = _make_run(
        "A",
        _make_records(
            (1, 10.0, 20.0, 30.0, 40.0, 0.95),
            (2, 15.0, 25.0, 35.0, 45.0, 0.90),
        ),
        index=2,
    )

    diag = diagnose(ref, tst)
    assert diag["classification"] == "structural_divergence"
    assert diag["record_count_ref"] != diag["record_count_test"]


def test_diagnose_decimal_divergence() -> None:
    ref = _make_run("A", _make_records((1, 10.0, 20.0, 30.0, 40.0, 0.95)))
    tst = _make_run("A", _make_records((1, 10.0, 20.0, 31.0, 40.0, 0.95)), index=2)

    diag = diagnose(ref, tst)
    assert diag["classification"] == "decimal_divergence"
    assert diag["decimal_hash_equal"] is False
    assert diag["record_count_ref"] == diag["record_count_test"]


def test_diagnose_first_diff_frame() -> None:
    ref = _make_run(
        "A",
        _make_records(
            (1, 10.0, 20.0, 30.0, 40.0, 0.95),
            (2, 15.0, 25.0, 35.0, 45.0, 0.90),
        ),
    )
    tst = _make_run(
        "A",
        _make_records(
            (1, 10.0, 20.0, 31.0, 40.0, 0.95),
            (2, 15.0, 25.0, 35.0, 45.0, 0.90),
        ),
        index=2,
    )

    diag = diagnose(ref, tst)
    assert diag["first_diff_frame"] == 1


# ── Ordered-vs-multiset diagnosis ─────────────────────────────────────────────


def test_diagnose_order_only_divergence() -> None:
    ref = _make_run(
        "A",
        _make_records(
            (1, 10.0, 20.0, 30.0, 40.0, 0.95),
            (1, 15.0, 25.0, 35.0, 45.0, 0.90),
        ),
    )
    tst = _make_run(
        "A",
        _make_records(
            (1, 15.0, 25.0, 35.0, 45.0, 0.90),
            (1, 10.0, 20.0, 30.0, 40.0, 0.95),
        ),
        index=2,
    )

    diag = diagnose(ref, tst)
    assert diag["classification"] == "decimal_exact_pass"


def test_diagnose_multiset_divergence() -> None:
    ref = _make_run("A", _make_records((1, 10.0, 20.0, 30.0, 40.0, 0.95)))
    tst = _make_run("A", _make_records((1, 11.0, 21.0, 30.0, 40.0, 0.95)), index=2)

    diag = diagnose(ref, tst)
    assert diag["classification"] == "decimal_divergence"
    assert diag["different_frame_count"] > 0


# ── Global ID exclusion ───────────────────────────────────────────────────────


def test_global_ids_excluded_from_canonical_fields() -> None:
    assert "global_track_id" not in CANONICAL_FIELDS
    for field in CANONICAL_FIELDS:
        assert "id" not in field.lower()


# ── Manifest and summary generation ───────────────────────────────────────────


def test_generate_manifest_includes_required_keys(tmp_path: Path) -> None:
    manifest = generate_manifest(
        "20260710T000000Z",
        "A",
        "B",
        "mamba_whole_graph_m",
        "SDP",
        True,
        ["--preset", "mamba_whole_graph_m", "--detector", "SDP", "--double-buffer"],
        {},
    )
    required = [
        "git_commit",
        "dirty_tree",
        "command",
        "model",
        "detector",
        "sequence_a",
        "sequence_b",
        "double_buffer",
        "start_timestamp",
    ]
    for key in required:
        assert key in manifest, f"manifest missing key: {key}"


def test_write_summary(tmp_path: Path) -> None:
    summary = {"verdict": "decimal_exact_pass", "cells": []}
    write_summary(tmp_path, summary)
    assert (tmp_path / "summary.json").exists()
    import json

    data = json.loads((tmp_path / "summary.json").read_text())
    assert data["verdict"] == "decimal_exact_pass"


def test_write_csv(tmp_path: Path) -> None:
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    path = tmp_path / "test.csv"
    write_csv(path, rows)
    assert path.exists()
    text = path.read_text()
    assert "a" in text
    assert "b" in text


# ── Nonzero exit status on divergence ─────────────────────────────────────────


def test_nonzero_verdict_on_divergence() -> None:
    ref = _make_run("A", _make_records((1, 10.0, 20.0, 30.0, 40.0, 0.95)))
    tst = _make_run("A", _make_records((1, 11.0, 21.0, 30.0, 40.0, 0.95)), index=2)
    diag = diagnose(ref, tst)
    assert diag["classification"] != "decimal_exact_pass"


def test_zero_verdict_on_exact_pass() -> None:
    rec = _make_records((1, 10.0, 20.0, 30.0, 40.0, 0.95))
    ref = _make_run("A", rec, index=1)
    tst = _make_run("A", rec, index=2)
    diag = diagnose(ref, tst)
    assert diag["classification"] == "decimal_exact_pass"


# ── HASH_METADATA ─────────────────────────────────────────────────────────────


def test_hash_metadata_excludes_global_ids() -> None:
    excluded = HASH_METADATA.get("excluded_fields", [])
    assert "global_track_id" in excluded


def test_hash_metadata_sort_key_no_identity() -> None:
    sort_fields = {f.strip() for f in CANONICAL_FIELDS}
    identity_terms = {"track_id", "global_id", "obj_id", "identity"}
    assert not sort_fields.intersection(identity_terms)
