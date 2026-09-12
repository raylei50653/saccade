"""Stage-order contract for the output-layer repair chaining harness.

These tests do not score MOT17 and do not load a ReID engine. They lock the
measurement-only chaining semantics: each stage rebuilds its input from the
previous stage's lines, and stage order is recorded.
"""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path("scripts/eval/experiments/run_output_layer_repair_chaining.py")


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "run_output_layer_repair_chaining", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _line(frame: int, tid: int, x: float = 10.0) -> str:
    return f"{frame},{tid},{x:.1f},10.0,20.0,40.0,0.9,-1,-1,-1"


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


def test_arm_stage_orders_are_explicit(harness):
    assert harness.ARM_STAGES["base"] == ()
    assert harness.ARM_STAGES["handover_only"] == ("handover",)
    assert harness.ARM_STAGES["merge_only"] == ("merge",)
    assert harness.ARM_STAGES["handover_then_merge"] == ("handover", "merge")
    assert harness.ARM_STAGES["merge_then_handover"] == ("merge", "handover")


def test_second_stage_sees_first_stage_output_not_raw_lines(harness):
    raw = [_line(1, 1), _line(2, 1), _line(10, 2), _line(11, 2)]
    seen_inputs: list[tuple[str, tuple[int, ...]]] = []

    def _ids(lines):
        return tuple(sorted(harness.unique_track_ids(lines)))

    def fake_handover(lines, *, seq_img_dir, extractor, params):
        seen_inputs.append(("handover", _ids(lines)))
        # Relabel id 2 -> 1, as a real handover would.
        out = [
            ln.replace(",2,", ",1,") if ln.split(",")[1] == "2" else ln for ln in lines
        ]
        rec = {
            "stage": "handover",
            "input_track_ids": sorted(harness.unique_track_ids(lines)),
            "output_track_ids": sorted(harness.unique_track_ids(out)),
            "stats": {"handovers": 1},
            "diagnostics": {"handovers": 1},
            "accepted_links": 1,
        }
        return out, rec

    def fake_merge(lines, *, seq_img_dir, extractor, params):
        seen_inputs.append(("merge", _ids(lines)))
        rec = {
            "stage": "merge",
            "input_track_ids": sorted(harness.unique_track_ids(lines)),
            "output_track_ids": sorted(harness.unique_track_ids(lines)),
            "stats": {"merges": 0},
            "diagnostics": {"accepted": 0, "no_embedding": 0},
            "accepted_links": 0,
        }
        return list(lines), rec

    out, records = harness.apply_stages(
        raw,
        ("handover", "merge"),
        seq_img_dir="/unused",
        extractor=object(),
        merge_params={},
        handover_params={},
        merge_stage=fake_merge,
        handover_stage=fake_handover,
    )
    assert [rec["stage"] for rec in records] == ["handover", "merge"]
    assert seen_inputs[0] == ("handover", (1, 2))
    assert seen_inputs[1] == ("merge", (1,))
    assert harness.unique_track_ids(out) == {1}
    assert records[0]["stage_index"] == 0
    assert records[1]["stage_index"] == 1


def test_reverse_order_does_not_share_the_raw_track_set(harness):
    raw = [_line(1, 1), _line(2, 1), _line(10, 2), _line(11, 2)]
    seen: list[tuple[str, tuple[int, ...]]] = []

    def fake_merge(lines, *, seq_img_dir, extractor, params):
        seen.append(("merge", tuple(sorted(harness.unique_track_ids(lines)))))
        out = [
            ln.replace(",2,", ",1,") if ln.split(",")[1] == "2" else ln for ln in lines
        ]
        rec = {
            "stage": "merge",
            "input_track_ids": sorted(harness.unique_track_ids(lines)),
            "output_track_ids": sorted(harness.unique_track_ids(out)),
            "stats": {"merges": 1},
            "diagnostics": {"accepted": 1},
            "accepted_links": 1,
        }
        return out, rec

    def fake_handover(lines, *, seq_img_dir, extractor, params):
        seen.append(("handover", tuple(sorted(harness.unique_track_ids(lines)))))
        rec = {
            "stage": "handover",
            "input_track_ids": sorted(harness.unique_track_ids(lines)),
            "output_track_ids": sorted(harness.unique_track_ids(lines)),
            "stats": {"handovers": 0},
            "diagnostics": {"handovers": 0},
            "accepted_links": 0,
        }
        return list(lines), rec

    harness.apply_stages(
        raw,
        ("merge", "handover"),
        seq_img_dir="/unused",
        extractor=object(),
        merge_params={},
        handover_params={},
        merge_stage=fake_merge,
        handover_stage=fake_handover,
    )
    assert seen[0] == ("merge", (1, 2))
    assert seen[1] == ("handover", (1,))


def test_summarize_merge_log_counts_are_diagnostics_not_metrics(harness):
    log = [
        {"kind": "tracklet", "a_id": 1, "b_id": -1, "verdict": "has_embedding"},
        {"kind": "tracklet", "a_id": 2, "b_id": -1, "verdict": "no_embedding"},
        {"kind": "pair", "a_id": 1, "b_id": 3, "verdict": "reject_cost"},
        {"kind": "pair", "a_id": 1, "b_id": 4, "verdict": "accepted"},
        {"kind": "pair", "a_id": 4, "b_id": 5, "verdict": "accepted"},
        {"kind": "pair", "a_id": 6, "b_id": 7, "verdict": "reject_same_component"},
    ]
    diag = harness.summarize_merge_log(log)
    assert diag["accepted"] == 2
    assert diag["no_embedding"] == 1
    assert diag["reject_cost"] == 1
    assert diag["reject_same_component"] == 1
    assert diag["component_count"] == 1  # 1-4-5 is one component; 6-7 was rejected
