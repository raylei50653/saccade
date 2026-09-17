"""Contract tests for production double-buffer critical-path attribution."""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from saccade.perception.eval.assoc_stats_env import assoc_stats_env_enabled
from saccade.perception.eval.evaluator import _double_buffer_eligible
from scripts.benchmarks.production_db_attribution import (
    classify_assoc_stage,
    classify_exposure,
    classify_kernel,
    derive,
    load_assoc_dir,
    load_production_dir,
    nsys_overlap_from_spans,
    removal_ceiling,
)


def test_double_buffer_stays_eligible_with_assoc_stats_env(monkeypatch) -> None:
    monkeypatch.setenv("SACCADE_DOUBLE_BUFFER", "1")
    monkeypatch.setenv("SACCADE_DETECT_BARRIER", "event")
    monkeypatch.setenv("SACCADE_ASSOC_STATS", "1")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    detector = SimpleNamespace(_temporal_T=0, use_whole_graph=True)
    assert _double_buffer_eligible(SimpleNamespace(workbench=False), detector, False)
    assert not _double_buffer_eligible(SimpleNamespace(workbench=False), detector, True)


def test_assoc_stats_env_fail_closed() -> None:
    assert assoc_stats_env_enabled("1")
    assert assoc_stats_env_enabled("true")
    assert assoc_stats_env_enabled("YES")
    assert assoc_stats_env_enabled("on")
    for token in ("", "0", "false", "False", "no", "off", "OFF", "maybe", "2"):
        assert assoc_stats_env_enabled(token) is False


def test_assoc_stats_env_tokens_match_cpp() -> None:
    header = Path("include/saccade/env_flag.hpp").read_text(encoding="utf-8")
    for tok in ("0", "false", "no", "off", "1", "true", "yes", "on"):
        assert f'"{tok}"' in header


def test_classify_exposure_and_kernel_names() -> None:
    assert classify_exposure(1.0, 0.01) == "fully hidden"
    assert classify_exposure(1.0, 0.4) == "partially exposed"
    assert classify_exposure(1.0, 0.99) == "critical-path"
    assert classify_kernel("selective_scan_fwd_kernel") == "scan"
    assert classify_kernel("chw_to_grayscale_downscale_kernel") == "gmc_downscale"
    assert classify_kernel("parallel_auction_shmem_kernel") == "tracker_auction"
    assert classify_kernel("append_private_continuation_kernel") == "private_append"
    assert classify_kernel("accumulate_assoc_frame_kernel") == "tracker_stats"


def test_nsys_overlap_hides_tracker_under_detect() -> None:
    windows = [(0.0, 3.0), (3.0, 6.0)]
    work = {
        "detect_graph": [(0.0, 2.0), (3.0, 5.0)],
        "trt": [(0.0, 1.8), (3.0, 4.8)],
        "tracker_auction": [(0.5, 1.0), (3.5, 4.0)],
        "gmc_downscale": [(1.9, 2.2), (4.9, 5.2)],
    }
    out = nsys_overlap_from_spans(windows, work)
    by_name = {row["stage"]: row for row in out["stages"]}
    assert by_name["tracker_auction"]["class"] == "fully hidden"
    assert by_name["gmc_downscale"]["class"] == "partially exposed"
    assert abs(by_name["gmc_downscale"]["exposed_ms"] - 0.2) < 1e-9
    assert by_name["trt"]["class"] == "fully hidden"


def test_classify_assoc_stage_uses_activity_frequency() -> None:
    assert (
        classify_assoc_stage(
            frames=100,
            frames_with_assignment=90,
            frames_with_valid_topk=95,
            assignments=1500,
        )
        == "常跑 + 有效工作"
    )
    assert (
        classify_assoc_stage(
            frames=100,
            frames_with_assignment=0,
            frames_with_valid_topk=0,
            assignments=0,
        )
        == "常跑 + 幾乎沒工作"
    )
    assert (
        classify_assoc_stage(
            frames=100,
            frames_with_assignment=4,
            frames_with_valid_topk=4,
            assignments=12,
        )
        == "常跑 + 幾乎沒工作"
    )


def test_load_production_dir_and_assoc_dir(tmp_path) -> None:
    prod = tmp_path / "p0"
    prod.mkdir()
    (prod / "_fps_summary.txt").write_text(
        "MOT17-04-SDP\tfps=347.50\tmean_ms=5.74\tframes=1000\n"
        "MOT17-05-SDP\tfps=388.80\tmean_ms=5.13\tframes=787\n"
        "OVERALL\tfps=349.64\tmean_ms=5.71\tframes=4966\n"
    )
    loaded = load_production_dir(prod)
    assert loaded["overall_fps"] == 349.64
    assert abs(loaded["mean_frame_period_ms"] - 1000.0 / 349.64) < 1e-9
    assert loaded["sequences"]["MOT17-04-SDP"]["fps"] == 347.5

    assoc = tmp_path / "d2"
    assoc.mkdir()
    (assoc / "_assoc_workload_MOT17-04-SDP.json").write_text(
        """
        {
          "seq": "MOT17-04-SDP",
          "association": {
            "enabled": true,
            "frames": 100,
            "sum_active": 2500,
            "sum_cand_n": 8000,
            "sum_matched": 2000,
            "sum_num_dets": 4000,
            "stages": [
              {
                "name": "S0",
                "unmatched_tracks_entering": 2500,
                "tracks_with_valid_topk": 1800,
                "assignments": 1500,
                "frames_with_assignment": 90,
                "frames_with_valid_topk": 95
              },
              {
                "name": "S2",
                "unmatched_tracks_entering": 500,
                "tracks_with_valid_topk": 12,
                "assignments": 12,
                "frames_with_assignment": 4,
                "frames_with_valid_topk": 4
              }
            ]
          },
          "private_continuation": {
            "enabled": true,
            "invocations": 100,
            "sum_candidate_count": 2000,
            "sum_added": 50,
            "frames_with_added": 20
          }
        }
        """
    )
    assoc_loaded = load_assoc_dir(assoc)
    seq = assoc_loaded["sequences"]["MOT17-04-SDP"]
    assert seq["per_frame"]["active_tracks"] == 25.0
    assert seq["per_frame"]["private_added"] == 0.5
    by_name = {s["name"]: s for s in seq["stages"]}
    assert by_name["S0"]["class"] == "常跑 + 有效工作"
    assert by_name["S2"]["class"] == "常跑 + 幾乎沒工作"


def test_derive_does_not_treat_detector_container_as_removable() -> None:
    production = {"overall_fps": 347.83, "mean_frame_period_ms": 2.875, "sequences": {}}
    nsys = {
        "gpu_union_busy_ms_per_frame": 3.08,
        "detect_span_mean_ms": 2.65,
        "category_ms_per_frame": {"scan": 0.38},
        "tail_mean_ms": 0.7,
        "tail_other_work_busy_ms": 0.12,
        "exposed_stages": [
            {
                "stage": "scan",
                "duration_ms": 0.38,
                "hidden_ms": 0.38,
                "exposed_ms": 0.0,
                "class": "fully hidden",
            },
            {
                "stage": "tracker_occlusion",
                "duration_ms": 0.165,
                "hidden_ms": 0.102,
                "exposed_ms": 0.063,
                "class": "partially exposed",
            },
            {
                "stage": "tracker_sinkhorn",
                "duration_ms": 0.077,
                "hidden_ms": 0.014,
                "exposed_ms": 0.063,
                "class": "partially exposed",
            },
            {
                "stage": "tracker_auction",
                "duration_ms": 0.039,
                "hidden_ms": 0.018,
                "exposed_ms": 0.021,
                "class": "partially exposed",
            },
            {
                "stage": "tracker_cost",
                "duration_ms": 0.034,
                "hidden_ms": 0.010,
                "exposed_ms": 0.024,
                "class": "partially exposed",
            },
            {
                "stage": "memcpy",
                "duration_ms": 0.164,
                "hidden_ms": 0.058,
                "exposed_ms": 0.107,
                "class": "partially exposed",
            },
        ],
    }
    payload = derive(production, nsys=nsys)
    decomp = payload["period_decomposition"]
    assert "production_bubble_ms" not in decomp
    assert decomp["outside_detect_remainder_ms"] == 0.225
    assert decomp["diagnostic_gpu_union_busy_ms"] == 3.08

    primary = payload["bottlenecks"][0]
    assert primary["rank"] == "Primary"
    assert primary["name"].startswith("detector")
    assert primary["removal_applicable"] is False
    assert primary["removal_upper_bound_ms"] is None
    assert primary["removal_upper_bound_fps"] is None
    slice_ = primary["attackable_slice"]
    assert slice_["name"] == "selective_scan"
    assert slice_["removal_upper_bound_ms"] == 0.38
    assert slice_["removal_upper_bound_fps"] == 400.8
    assert slice_["removal_upper_bound_fps"] < 1000

    secondary = payload["bottlenecks"][1]
    assert secondary["rank"] == "Secondary"
    assert "outside-detect remainder" in secondary["name"]
    assert secondary["exposed_cost_ms"] == 0.225

    tertiary = payload["bottlenecks"][2]
    assert tertiary["rank"] == "Tertiary"
    assert "association" in tertiary["name"]
    assert tertiary["exposed_cost_ms"] == 0.171


def test_committed_json_matches_corrected_derivation() -> None:
    payload = json.loads(
        Path(
            "docs/reference/benchmarks/production_db_critical_path_20260917.json"
        ).read_text(encoding="utf-8")
    )
    decomp = payload["period_decomposition"]
    assert "production_bubble_ms" not in decomp
    assert decomp["outside_detect_remainder_ms"] > 0
    primary = payload["bottlenecks"][0]
    assert primary["removal_applicable"] is False
    assert primary["removal_upper_bound_fps"] is None
    slice_fps = primary["attackable_slice"]["removal_upper_bound_fps"]
    assert slice_fps is not None and 300 < slice_fps < 600
    for seq_data in payload["association"]["sequences"].values():
        for stg in seq_data["stages"]:
            if stg["name"] == "S2":
                assert stg["class"] == "常跑 + 幾乎沒工作"


def test_removal_ceiling_rejects_container_sized_slices() -> None:
    assert removal_ceiling(2.875, 2.65) == (None, None)
    ms, fps = removal_ceiling(2.875, 0.38)
    assert ms == 0.38
    assert fps == 400.8
    assert fps is not None and fps < 1000
