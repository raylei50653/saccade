"""Extended tests for perception/eval/reporting.py.

Covers additional branches not tested in test_reporting.py:
  - _print_stage_waterfall (pure function)
  - print_overall_summary: JSON output, lazy reid, birth CSV, GMC breakdown
  - print_sequence_summary: tile diagnostics, lazy reid, post counts
"""

from __future__ import annotations

import csv
from types import SimpleNamespace
from unittest.mock import MagicMock


from saccade.perception.eval.reporting import (
    _print_stage_waterfall,
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
        all_seq_profile=None,
    )
    kw.update(overrides)
    return kw


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


# ─── _print_stage_waterfall ──────────────────────────────────────────────────


def test_waterfall_empty_stages(capsys):
    """_print_stage_waterfall with no stages prints nothing."""
    _print_stage_waterfall({}, 10.0)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_waterfall_zero_frame_total(capsys):
    """_print_stage_waterfall with zero frame total prints nothing."""
    stages = {"detect": 5.0}
    _print_stage_waterfall(stages, 0.0)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_waterfall_with_stages(capsys):
    """_print_stage_waterfall prints stage breakdown."""
    stages = {"detect": 5.0, "reid": 3.0}
    _print_stage_waterfall(stages, 10.0)
    captured = capsys.readouterr()
    assert "Stage Breakdown" in captured.out
    assert "detect" in captured.out
    assert "reid" in captured.out
    assert "5.00ms" in captured.out
    assert "50.0%" in captured.out


def test_waterfall_sorted_by_time(capsys):
    """Stages are sorted by time descending."""
    stages = {"reid": 2.0, "detect": 7.0, "post": 1.0}
    _print_stage_waterfall(stages, 10.0)
    captured = capsys.readouterr()
    lines = captured.out.split("\n")
    detect_line = next(i for i, line in enumerate(lines) if "detect" in line)
    reid_line = next(i for i, line in enumerate(lines) if "reid" in line)
    post_line = next(i for i, line in enumerate(lines) if "post" in line)
    assert detect_line < reid_line < post_line


def test_waterfall_zero_value_stages_filtered(capsys):
    """Stages with zero time are not shown."""
    stages = {"detect": 5.0, "empty_stage": 0.0}
    _print_stage_waterfall(stages, 10.0)
    captured = capsys.readouterr()
    assert "empty_stage" not in captured.out
    assert "detect" in captured.out


def test_waterfall_unaccounted_time(capsys):
    """Shows [unaccounted] bar when total > accounted."""
    stages = {"detect": 5.0}
    _print_stage_waterfall(stages, 10.0)
    captured = capsys.readouterr()
    assert "[unaccounted]" in captured.out


def test_waterfall_no_unaccounted_when_fit(capsys):
    """Does not show [unaccounted] when stages sum to total."""
    stages = {"detect": 5.0, "reid": 5.0}
    _print_stage_waterfall(stages, 10.0)
    captured = capsys.readouterr()
    # unaccounted threshold is 0.5ms, exact match should not show
    assert "[unaccounted]" not in captured.out


def test_waterfall_small_unaccounted_not_shown(capsys):
    """Does not show [unaccounted] when < 0.5ms."""
    stages = {"detect": 5.0}
    _print_stage_waterfall(stages, 5.4)  # 0.4ms unaccounted < 0.5 threshold
    captured = capsys.readouterr()
    assert "[unaccounted]" not in captured.out


# ─── print_overall_summary: JSON + lazy reid + birth ─────────────────────────


def test_writes_json_profile(tmp_path):
    """print_overall_summary writes JSON profile when profiling."""
    samples = {s: [10.0, 11.0, 12.0] for s in TOP_STAGES}
    print_overall_summary(
        **base_overall_kwargs(
            tmp_path,
            cfg=make_cfg(profile_stages=True),
            overall_profiled_frames=3,
            overall_stage_samples=samples,
            stage_summary_lines=[],
            all_seq_profile=[
                {
                    "seq": "test_seq",
                    "frames": 100,
                    "stages": {
                        "detect": {"mean_ms": 10.0, "std_ms": 1.0},
                        "reid_extract": {"mean_ms": 5.0, "std_ms": 0.5},
                    },
                }
            ],
        )
    )
    json_path = tmp_path / "_stage_profile.json"
    assert json_path.exists()
    import json

    data = json.loads(json_path.read_text())
    assert "meta" in data
    assert "overall" in data
    assert "sequences" in data
    assert "test_seq" in data["sequences"]


def test_json_contains_stage_stats(tmp_path):
    """JSON output contains mean/std/p95/p99 for each stage."""
    samples = {s: [10.0, 10.0, 10.0] for s in TOP_STAGES}
    print_overall_summary(
        **base_overall_kwargs(
            tmp_path,
            cfg=make_cfg(profile_stages=True),
            overall_profiled_frames=3,
            overall_stage_samples=samples,
            stage_summary_lines=[],
            all_seq_profile=None,
        )
    )
    import json

    data = json.loads((tmp_path / "_stage_profile.json").read_text())
    detect_stats = data["overall"]["detect"]
    assert "mean_ms" in detect_stats
    assert "std_ms" in detect_stats
    assert "p95_ms" in detect_stats
    assert "p99_ms" in detect_stats
    reid_stats = data["overall"]["reid_extract"]
    assert "mean_ms" in reid_stats


def test_lazy_reid_appended_to_overall_summary(tmp_path, capsys):
    """print_overall_summary appends lazy reid stats when profiled."""
    samples = {s: [10.0] for s in TOP_STAGES}
    print_overall_summary(
        **base_overall_kwargs(
            tmp_path,
            cfg=make_cfg(profile_stages=True, profile_lazy_reid_candidates=True),
            overall_profiled_frames=5,
            overall_stage_samples=samples,
            stage_summary_lines=[],
            overall_lazy_reid_frames=5,
            overall_lazy_reid_candidates=20,
        )
    )
    captured = capsys.readouterr()
    assert "lazy_reid_candidates" in captured.out


def test_lazy_reid_embeddings_appended(tmp_path, capsys):
    """print_overall_summary appends lazy reid embeddings when profiled."""
    samples = {s: [10.0] for s in TOP_STAGES}
    print_overall_summary(
        **base_overall_kwargs(
            tmp_path,
            cfg=make_cfg(
                profile_stages=True,
                profile_lazy_reid_candidates=True,
                profile_lazy_reid_embeddings=True,
            ),
            overall_profiled_frames=5,
            overall_stage_samples=samples,
            stage_summary_lines=[],
            overall_lazy_reid_frames=5,
            overall_lazy_reid_candidates=20,
            overall_lazy_reid_crops=40,
            overall_lazy_reid_self_pairs=10,
            overall_lazy_reid_self_pass=8,
            overall_lazy_reid_self_sim_sum=7.5,
            overall_lazy_reid_arbiter_checks=10,
            overall_lazy_reid_arbiter_approve=7,
        )
    )
    captured = capsys.readouterr()
    assert "lazy_reid_embeddings" in captured.out
    assert "lazy_reid_arbiter_dry_run" in captured.out


def test_lazy_reid_lines_appended(tmp_path):
    """Lazy reid stats appended to stage_summary_lines."""
    lines = []
    samples = {s: [10.0] for s in TOP_STAGES}
    print_overall_summary(
        **base_overall_kwargs(
            tmp_path,
            cfg=make_cfg(
                profile_stages=True,
                profile_lazy_reid_candidates=True,
                profile_lazy_reid_embeddings=True,
            ),
            overall_profiled_frames=5,
            overall_stage_samples=samples,
            stage_summary_lines=lines,
            overall_lazy_reid_frames=5,
            overall_lazy_reid_candidates=20,
            overall_lazy_reid_crops=40,
            overall_lazy_reid_self_pairs=10,
            overall_lazy_reid_self_pass=8,
            overall_lazy_reid_self_sim_sum=7.5,
            overall_lazy_reid_arbiter_checks=10,
            overall_lazy_reid_arbiter_approve=7,
        )
    )
    assert any("lazy_reid_candidates" in line for line in lines)
    assert any("lazy_reid_embeddings" in line for line in lines)
    assert any("lazy_reid_arbiter_dry_run" in line for line in lines)


def test_writes_birth_csv(tmp_path):
    """print_overall_summary writes birth CSV when rows present."""
    rows = [
        {
            "seq": "S1",
            "frame": 1,
            "policy": "boost",
            "det_idx": 0,
            "score_before": 0.3,
            "score_after": 0.6,
            "x1": 100.0,
            "y1": 200.0,
            "x2": 200.0,
            "y2": 400.0,
            "w": 100.0,
            "h": 200.0,
            "output_emitted": True,
            "output_local_track_id": 1,
            "output_track_id": 1,
            "output_score": 0.6,
            "output_x1": 100.0,
            "output_y1": 200.0,
            "output_x2": 200.0,
            "output_y2": 400.0,
        },
    ]
    csv_path = str(tmp_path / "birth.csv")
    print_overall_summary(
        **base_overall_kwargs(
            tmp_path,
            debug_birth_csv=csv_path,
            debug_birth_rows=rows,
        )
    )
    assert (tmp_path / "birth.csv").exists()
    with open(csv_path) as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 1
    assert reader[0]["policy"] == "boost"
    assert reader[0]["score_before"] == "0.3"


def test_no_birth_csv_when_no_path(tmp_path):
    """Does not write birth CSV when no path provided."""
    print_overall_summary(
        **base_overall_kwargs(
            tmp_path,
            debug_birth_csv="",
            debug_birth_rows=[{"seq": "S1"}],  # has rows but no path
        )
    )
    assert not (tmp_path / "birth.csv").exists()


def test_gmc_breakdown_in_overall_summary(tmp_path, capsys):
    """print_overall_summary shows GMC breakdown when samples present."""
    samples = {s: [2.0, 2.5, 3.0] for s in GMC_STAGES}
    print_overall_summary(
        **base_overall_kwargs(
            tmp_path,
            cfg=make_cfg(profile_stages=True),
            overall_profiled_frames=3,
            overall_stage_samples={s: [10.0] for s in TOP_STAGES},
            stage_summary_lines=[],
            overall_gmc_samples=samples,
        )
    )
    captured = capsys.readouterr()
    assert "GMC Breakdown" in captured.out
    assert "gmc_gray_downscale" in captured.out


def test_post_counts_in_overall_summary(tmp_path, capsys):
    """print_overall_summary shows post counts when present."""
    print_overall_summary(
        **base_overall_kwargs(
            tmp_path,
            cfg=make_cfg(profile_stages=True),
            overall_profiled_frames=3,
            overall_stage_samples={s: [10.0] for s in TOP_STAGES},
            stage_summary_lines=[],
            overall_post_counts={"post_filter": 9, "birth": 6},
        )
    )
    captured = capsys.readouterr()
    assert "Post Counts" in captured.out
    assert "post_filter" in captured.out


def test_breakdown_stage_in_overall_summary(tmp_path, capsys):
    """print_overall_summary shows breakdown stage when total > 0."""
    print_overall_summary(
        **base_overall_kwargs(
            tmp_path,
            cfg=make_cfg(profile_stages=True),
            overall_profiled_frames=3,
            overall_stage_samples={s: [10.0] for s in TOP_STAGES},
            stage_summary_lines=[],
            overall_stage_totals={"post_filter": 6.0},
            breakdown_stage_names=BREAKDOWN_STAGES,
        )
    )
    captured = capsys.readouterr()
    assert "Postprocess Breakdown" in captured.out
    assert "post_filter" in captured.out


def test_fps_summary_with_latency(tmp_path):
    """FPS summary includes OVERALL line when latency available."""
    print_overall_summary(
        **base_overall_kwargs(
            tmp_path,
            fps_summary_lines=["SEQ1\tfps=30"],
            overall_latency_ms=[20.0, 25.0, 30.0],
        )
    )
    content = (tmp_path / "_fps_summary.txt").read_text()
    assert "OVERALL" in content
    assert "fps=" in content
    assert "mean_ms=" in content


def test_fps_summary_no_latency(tmp_path):
    """FPS summary does not include OVERALL line when no latency."""
    print_overall_summary(
        **base_overall_kwargs(
            tmp_path,
            fps_summary_lines=["SEQ1\tfps=30"],
            overall_latency_ms=[],
        )
    )
    content = (tmp_path / "_fps_summary.txt").read_text()
    assert "OVERALL" not in content


# ─── print_sequence_summary: tile diag + lazy reid ───────────────────────────


def test_seq_tile_diagnostics_printed(capsys):
    """print_sequence_summary prints tile diagnostics when enabled."""
    lines = []
    print_sequence_summary(
        **base_seq_kwargs(
            cfg=make_cfg(tile_diagnostics=True),
            profile_stages=False,
            seq_tile_diag={
                "frames_tiled": 100,
                "pre_merge_seam_boxes": 50,
                "post_merge_seam_boxes": 20,
                "merged_clusters": 30,
                "merged_members": 60,
                "merged_outputs": 25,
            },
            stage_summary_lines=lines,
        )
    )
    captured = capsys.readouterr()
    assert "Tile diagnostics" in captured.out
    assert "compression=" in captured.out


def test_seq_no_tile_diag_when_disabled(capsys):
    """print_sequence_summary skips tile diagnostics when disabled."""
    lines = []
    print_sequence_summary(
        **base_seq_kwargs(
            cfg=make_cfg(tile_diagnostics=False),
            profile_stages=False,
            seq_tile_diag={
                "frames_tiled": 100,
                "pre_merge_seam_boxes": 50,
                "post_merge_seam_boxes": 20,
                "merged_clusters": 30,
                "merged_members": 60,
                "merged_outputs": 25,
            },
            stage_summary_lines=lines,
        )
    )
    captured = capsys.readouterr()
    assert "Tile diagnostics" not in captured.out


def test_seq_lazy_reid_appended(tmp_path):
    """print_sequence_summary appends lazy reid to stage_summary_lines."""
    lines = []
    samples = {s: [10.0] for s in TOP_STAGES}
    print_sequence_summary(
        **base_seq_kwargs(
            cfg=make_cfg(
                profile_stages=True,
                profile_lazy_reid_candidates=True,
                profile_lazy_reid_embeddings=True,
            ),
            profile_stages=True,
            seq_profiled_frames=5,
            seq_stage_samples=samples,
            seq_lazy_reid_frames=5,
            seq_lazy_reid_candidates=20,
            seq_lazy_reid_crops=40,
            seq_lazy_reid_self_pairs=10,
            seq_lazy_reid_self_pass=8,
            seq_lazy_reid_self_sim_sum=7.5,
            seq_lazy_reid_arbiter_checks=10,
            seq_lazy_reid_arbiter_approve=7,
            stage_summary_lines=lines,
        )
    )
    assert any("lazy_reid_candidates" in line for line in lines)
    assert any("lazy_reid_embeddings" in line for line in lines)
    assert any("lazy_reid_arbiter_dry_run" in line for line in lines)


def test_seq_native_reid_printed(capsys):
    """print_sequence_summary prints native ReID breakdown when samples present."""
    lines = []
    samples = {s: [5.0, 6.0] for s in TOP_STAGES}
    reid_samples = {s: [3.0, 3.5, 4.0] for s in NATIVE_REID_STAGES}
    print_sequence_summary(
        **base_seq_kwargs(
            cfg=make_cfg(),
            profile_stages=True,
            seq_profiled_frames=3,
            seq_stage_samples=samples,
            seq_native_reid_samples=reid_samples,
            stage_summary_lines=lines,
        )
    )
    captured = capsys.readouterr()
    assert "ReID Extract Breakdown" in captured.out
    assert "crop" in captured.out


def test_seq_post_counts_printed(capsys):
    """print_sequence_summary prints post counts when present."""
    lines = []
    samples = {s: [10.0] for s in TOP_STAGES}
    print_sequence_summary(
        **base_seq_kwargs(
            cfg=make_cfg(),
            profile_stages=True,
            seq_profiled_frames=3,
            seq_stage_samples=samples,
            seq_post_counts={"post_filter": 6, "birth": 3},
            overall_post_counts={"post_filter": 0, "birth": 0},
            stage_summary_lines=lines,
        )
    )
    captured = capsys.readouterr()
    assert "Post Counts" in captured.out


def test_seq_gmc_printed(capsys):
    """print_sequence_summary prints GMC breakdown when samples present."""
    lines = []
    samples = {s: [10.0] for s in TOP_STAGES}
    gmc_samples = {s: [1.0, 1.5] for s in GMC_STAGES}
    print_sequence_summary(
        **base_seq_kwargs(
            cfg=make_cfg(),
            profile_stages=True,
            seq_profiled_frames=3,
            seq_stage_samples=samples,
            seq_gmc_samples=gmc_samples,
            stage_summary_lines=lines,
        )
    )
    captured = capsys.readouterr()
    assert "GMC Breakdown" in captured.out


def test_seq_appends_blank_line_at_end(tmp_path):
    """print_sequence_summary appends blank line at end of stage_summary_lines."""
    lines = []
    samples = {s: [10.0] for s in TOP_STAGES}
    print_sequence_summary(
        **base_seq_kwargs(
            cfg=make_cfg(),
            profile_stages=True,
            seq_profiled_frames=3,
            seq_stage_samples=samples,
            stage_summary_lines=lines,
        )
    )
    assert lines[-1] == ""


def test_seq_accumulates_overall_stage_totals(capsys):
    """print_sequence_summary accumulates totals into overall_stage_totals."""
    totals = {s: 0.0 for s in TOP_STAGES + BREAKDOWN_STAGES}
    samples = {s: [10.0, 20.0] for s in TOP_STAGES}
    breakdown_totals = {s: 0.0 for s in BREAKDOWN_STAGES}
    print_sequence_summary(
        **base_seq_kwargs(
            cfg=make_cfg(),
            profile_stages=True,
            seq_profiled_frames=2,
            seq_stage_samples=samples,
            breakdown_stage_names=BREAKDOWN_STAGES,
            seq_stage_totals=breakdown_totals,
            overall_stage_totals=totals,
            stage_summary_lines=[],
        )
    )
    # detect total should be 10+20=30
    assert totals["detect"] == 30.0


def test_seq_accumulates_overall_post_counts(capsys):
    """print_sequence_summary accumulates post counts into overall_post_counts."""
    post_counts = {"post_filter": 0, "birth": 0}
    print_sequence_summary(
        **base_seq_kwargs(
            cfg=make_cfg(),
            profile_stages=True,
            seq_profiled_frames=3,
            seq_stage_samples={s: [10.0] for s in TOP_STAGES},
            seq_post_counts={"post_filter": 6, "birth": 3},
            overall_post_counts=post_counts,
            stage_summary_lines=[],
        )
    )
    assert post_counts["post_filter"] == 6
    assert post_counts["birth"] == 3


def test_seq_accumulates_overall_gmc_samples(capsys):
    """print_sequence_summary appends GMC samples to overall."""
    gmc_samples = {s: [] for s in GMC_STAGES}
    seq_gmc = {s: [1.0, 2.0] for s in GMC_STAGES}
    print_sequence_summary(
        **base_seq_kwargs(
            cfg=make_cfg(),
            profile_stages=True,
            seq_profiled_frames=2,
            seq_stage_samples={s: [10.0] for s in TOP_STAGES},
            seq_gmc_samples=seq_gmc,
            overall_gmc_samples=gmc_samples,
            stage_summary_lines=[],
        )
    )
    # Overall should have extended samples
    assert len(gmc_samples["gmc_gray_downscale"]) == 2
    assert gmc_samples["gmc_gray_downscale"] == [1.0, 2.0]
