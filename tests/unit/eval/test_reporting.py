import csv
from types import SimpleNamespace
from unittest.mock import MagicMock

from saccade.perception.eval.reporting import (
    print_overall_summary,
    print_sequence_summary,
)

TOP_STAGES = ("detect", "reid_extract")
BREAKDOWN_STAGES = ("post_filter",)
SEGMENT_BREAKDOWN_NAMES: tuple[str, ...] = ()
NATIVE_REID_STAGES = ("crop",)
GMC_STAGES = ("gmc_gray_downscale", "gmc_fg_mask", "gmc_phase_corr", "gmc_handoff")


def make_cfg(**kwargs):
    defaults = dict(
        profile_stages=False,
        profile_lazy_reid_candidates=False,
        profile_lazy_reid_embeddings=False,
        tile_diagnostics=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def make_mapper(lines=None):
    m = MagicMock()
    m.dump_lines.return_value = lines or []
    return m


def base_overall_kwargs(output_root, **overrides):
    kw = dict(
        cfg=make_cfg(),
        output_root=output_root,
        fps_summary_lines=[],
        overall_latency_ms=[],
        global_id_mapper=make_mapper(),
        overall_profiled_frames=0,
        top_level_stage_names=TOP_STAGES,
        overall_stage_samples={s: [] for s in TOP_STAGES},
        stage_summary_lines=[],
        breakdown_stage_names=BREAKDOWN_STAGES,
        overall_stage_totals={s: 0.0 for s in BREAKDOWN_STAGES},
        overall_post_counts={},
        gmc_breakdown_names=GMC_STAGES,
        overall_gmc_samples={s: [] for s in GMC_STAGES},
        segment_breakdown_names=SEGMENT_BREAKDOWN_NAMES,
        overall_segment_samples={s: [] for s in SEGMENT_BREAKDOWN_NAMES},
        overall_lazy_reid_frames=0,
        overall_lazy_reid_candidates=0,
        overall_lazy_reid_crops=0,
        overall_lazy_reid_self_sim_sum=0.0,
        overall_lazy_reid_self_pairs=0,
        overall_lazy_reid_self_pass=0,
        overall_lazy_reid_arbiter_checks=0,
        overall_lazy_reid_arbiter_approve=0,
        debug_dump_csv="",
        debug_stage_dump_rows=[],
        debug_birth_csv="",
        debug_birth_rows=[],
    )
    kw.update(overrides)
    return kw


# --- print_overall_summary ---


def test_writes_fps_summary_file(tmp_path):
    print_overall_summary(
        **base_overall_kwargs(
            tmp_path,
            fps_summary_lines=["SEQ1\tfps=30"],
            overall_latency_ms=[33.3, 33.3],
        )
    )
    assert (tmp_path / "_fps_summary.txt").exists()
    content = (tmp_path / "_fps_summary.txt").read_text()
    assert "OVERALL" in content


def test_no_fps_file_when_lines_empty(tmp_path):
    print_overall_summary(**base_overall_kwargs(tmp_path, fps_summary_lines=[]))
    assert not (tmp_path / "_fps_summary.txt").exists()


def test_writes_global_id_map(tmp_path):
    mapper = make_mapper(["1 -> 42", "2 -> 99"])
    print_overall_summary(**base_overall_kwargs(tmp_path, global_id_mapper=mapper))
    assert (tmp_path / "_global_id_map.txt").exists()
    assert "1 -> 42" in (tmp_path / "_global_id_map.txt").read_text()


def test_no_global_id_map_when_empty(tmp_path):
    print_overall_summary(
        **base_overall_kwargs(tmp_path, global_id_mapper=make_mapper([]))
    )
    assert not (tmp_path / "_global_id_map.txt").exists()


def test_writes_stage_profile_when_profiling(tmp_path):
    samples = {s: [10.0, 11.0, 12.0] for s in TOP_STAGES}
    print_overall_summary(
        **base_overall_kwargs(
            tmp_path,
            cfg=make_cfg(profile_stages=True),
            overall_profiled_frames=3,
            overall_stage_samples=samples,
            stage_summary_lines=[],
        )
    )
    assert (tmp_path / "_stage_profile.txt").exists()
    content = (tmp_path / "_stage_profile.txt").read_text()
    assert "detect" in content


def test_writes_debug_csv(tmp_path):
    rows = [
        {
            "seq": "S1",
            "frame": 1,
            "stage": "raw",
            "det_idx": 0,
            "x1": 0.0,
            "y1": 0.0,
            "x2": 10.0,
            "y2": 10.0,
            "w": 10.0,
            "h": 10.0,
            "score": 0.9,
            "cls": 0,
        },
    ]
    csv_path = str(tmp_path / "dump.csv")
    print_overall_summary(
        **base_overall_kwargs(
            tmp_path,
            debug_dump_csv=csv_path,
            debug_stage_dump_rows=rows,
        )
    )
    assert (tmp_path / "dump.csv").exists()
    with open(csv_path) as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 1
    assert reader[0]["seq"] == "S1"


def test_no_csv_when_rows_empty(tmp_path):
    csv_path = str(tmp_path / "dump.csv")
    print_overall_summary(
        **base_overall_kwargs(
            tmp_path,
            debug_dump_csv=csv_path,
            debug_stage_dump_rows=[],
        )
    )
    assert not (tmp_path / "dump.csv").exists()


# --- print_sequence_summary ---


def base_seq_kwargs(**overrides):
    kw = dict(
        cfg=make_cfg(),
        seq="MOT17-02-SDP",
        seq_tile_diag={
            "frames_tiled": 0,
            "pre_merge_seam_boxes": 0,
            "post_merge_seam_boxes": 0,
            "merged_clusters": 0,
            "merged_members": 0,
            "merged_outputs": 0,
        },
        profile_stages=False,
        seq_profiled_frames=0,
        top_level_stage_names=TOP_STAGES,
        seq_stage_samples={s: [] for s in TOP_STAGES},
        overall_stage_totals={s: 0.0 for s in TOP_STAGES + BREAKDOWN_STAGES},
        overall_stage_samples={s: [] for s in TOP_STAGES},
        breakdown_stage_names=BREAKDOWN_STAGES,
        seq_stage_totals={s: 0.0 for s in BREAKDOWN_STAGES},
        native_reid_breakdown_names=NATIVE_REID_STAGES,
        seq_native_reid_samples={s: [] for s in NATIVE_REID_STAGES},
        gmc_breakdown_names=GMC_STAGES,
        seq_gmc_samples={s: [] for s in GMC_STAGES},
        overall_gmc_samples={s: [] for s in GMC_STAGES},
        segment_breakdown_names=SEGMENT_BREAKDOWN_NAMES,
        seq_segment_samples={s: [] for s in SEGMENT_BREAKDOWN_NAMES},
        overall_segment_samples={s: [] for s in SEGMENT_BREAKDOWN_NAMES},
        seq_post_counts={},
        overall_post_counts={},
        seq_lazy_reid_frames=0,
        seq_lazy_reid_candidates=0,
        overall_lazy_reid_candidates=0,
        overall_lazy_reid_frames=0,
        overall_lazy_reid_crops=0,
        overall_lazy_reid_self_pairs=0,
        overall_lazy_reid_self_pass=0,
        overall_lazy_reid_self_sim_sum=0.0,
        overall_lazy_reid_arbiter_checks=0,
        overall_lazy_reid_arbiter_approve=0,
        seq_lazy_reid_crops=0,
        seq_lazy_reid_self_pairs=0,
        seq_lazy_reid_self_pass=0,
        seq_lazy_reid_self_sim_sum=0.0,
        seq_lazy_reid_arbiter_checks=0,
        seq_lazy_reid_arbiter_approve=0,
        overall_profiled_frames=0,
        stage_summary_lines=[],
    )
    kw.update(overrides)
    return kw


def test_seq_summary_appends_stage_lines_when_profiling():
    lines = []
    print_sequence_summary(
        **base_seq_kwargs(
            profile_stages=True,
            seq_profiled_frames=5,
            seq_stage_samples={"detect": [10.0] * 5, "reid_extract": [5.0] * 5},
            stage_summary_lines=lines,
        )
    )
    assert any("detect" in line for line in lines)
    assert any("MOT17-02-SDP" in line for line in lines)


def test_seq_summary_no_profile_no_lines():
    lines = []
    print_sequence_summary(
        **base_seq_kwargs(
            profile_stages=False,
            seq_profiled_frames=0,
            stage_summary_lines=lines,
        )
    )
    assert lines == []


def test_seq_summary_lazy_reid_appended():
    lines = []
    print_sequence_summary(
        **base_seq_kwargs(
            cfg=make_cfg(profile_stages=True, profile_lazy_reid_candidates=True),
            profile_stages=True,
            seq_profiled_frames=5,
            seq_stage_samples={"detect": [10.0] * 5, "reid_extract": [5.0] * 5},
            seq_lazy_reid_frames=5,
            seq_lazy_reid_candidates=20,
            stage_summary_lines=lines,
        )
    )
    assert any("lazy_reid_candidates" in line for line in lines)
