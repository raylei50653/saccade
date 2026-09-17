"""Contract tests for production double-buffer critical-path attribution."""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

from types import SimpleNamespace

from saccade.perception.eval.evaluator import _double_buffer_eligible
from scripts.benchmarks.production_db_attribution import (
    classify_exposure,
    classify_kernel,
    derive,
    load_assoc_dir,
    load_production_dir,
    nsys_overlap_from_spans,
)


def test_double_buffer_stays_eligible_with_assoc_stats_env(monkeypatch) -> None:
    monkeypatch.setenv("SACCADE_DOUBLE_BUFFER", "1")
    monkeypatch.setenv("SACCADE_DETECT_BARRIER", "event")
    monkeypatch.setenv("SACCADE_ASSOC_STATS", "1")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    detector = SimpleNamespace(_temporal_T=0, use_whole_graph=True)
    assert _double_buffer_eligible(SimpleNamespace(workbench=False), detector, False)
    assert not _double_buffer_eligible(SimpleNamespace(workbench=False), detector, True)


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
                "tracks_with_valid_topk": 0,
                "assignments": 0,
                "frames_with_assignment": 0,
                "frames_with_valid_topk": 0
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


def test_derive_ranks_detector_above_hidden_tracker() -> None:
    production = {"overall_fps": 350.0, "mean_frame_period_ms": 2.857, "sequences": {}}
    nsys = {
        "gpu_union_busy_ms_per_frame": 2.4,
        "detect_span_mean_ms": 2.1,
        "tail_mean_ms": 0.7,
        "tail_other_work_busy_ms": 0.12,
        "exposed_stages": [
            {
                "stage": "tracker_auction",
                "duration_ms": 0.4,
                "hidden_ms": 0.4,
                "exposed_ms": 0.0,
                "class": "fully hidden",
            },
            {
                "stage": "gmc_downscale",
                "duration_ms": 0.08,
                "hidden_ms": 0.06,
                "exposed_ms": 0.02,
                "class": "partially exposed",
            },
        ],
    }
    payload = derive(production, nsys=nsys)
    ranks = [b["rank"] for b in payload["bottlenecks"]]
    assert ranks[0] == "Primary"
    assert payload["bottlenecks"][0]["name"].startswith("detector")
    assert payload["period_decomposition"]["production_bubble_ms"] is not None
    assert payload["bottlenecks"][0]["class_label"].startswith("D.")
