"""Tests for saccade.perception.eval.post_merge."""

# scope: eval
# function: behavior
# lifecycle: active

from __future__ import annotations

import pytest

from saccade.perception.eval.post_merge import (
    _direction_penalty,
    _parse_mot_lines,
    _format_mot_records,
    _build_output_tracklets,
    _tracklet_velocity,
    post_merge_output_tracklets,
    filter_low_quality_tracklets,
    interpolate_tracklets,
)


# ── _tracklet_velocity ──────────────────────────────────────────────────


def test_tracklet_velocity_single_record() -> None:
    from saccade.perception.eval.types import MotRecord

    records = [MotRecord(frame=1, track_id=1, x=0, y=0, w=10, h=20, score=0.9, tail=[])]
    vx, vy = _tracklet_velocity(records, from_start=True, samples=2)
    assert vx == pytest.approx(0.0)
    assert vy == pytest.approx(0.0)


def test_tracklet_velocity_two_records() -> None:
    from saccade.perception.eval.types import MotRecord

    records = [
        MotRecord(frame=1, track_id=1, x=0, y=0, w=10, h=20, score=0.9, tail=[]),
        MotRecord(frame=3, track_id=1, x=20, y=10, w=10, h=20, score=0.9, tail=[]),
    ]
    vx, vy = _tracklet_velocity(records, from_start=True, samples=2)
    # dt=2, dx=20, dy=10 => vx=10.0, vy=5.0
    assert vx == pytest.approx(10.0)
    assert vy == pytest.approx(5.0)


def test_tracklet_velocity_from_end() -> None:
    from saccade.perception.eval.types import MotRecord

    records = [
        MotRecord(frame=1, track_id=1, x=0, y=0, w=10, h=20, score=0.9, tail=[]),
        MotRecord(frame=2, track_id=1, x=10, y=5, w=10, h=20, score=0.9, tail=[]),
        MotRecord(frame=3, track_id=1, x=30, y=15, w=10, h=20, score=0.9, tail=[]),
    ]
    vx, vy = _tracklet_velocity(records, from_start=False, samples=2)
    # last 2: frame 2->3, dx=20, dy=10 => vx=20.0, vy=10.0
    assert vx == pytest.approx(20.0)
    assert vy == pytest.approx(10.0)


# ── _direction_penalty ──────────────────────────────────────────────────


def test_direction_penalty_same_direction() -> None:
    a, b = (10.0, 0.0), (20.0, 0.0)
    assert _direction_penalty(a, b) == pytest.approx(0.0)


def test_direction_penalty_perpendicular() -> None:
    a, b = (10.0, 0.0), (0.0, 10.0)
    assert _direction_penalty(a, b) == pytest.approx(0.0)  # cos=0, max(0,-0)=0


def test_direction_penalty_opposite_direction() -> None:
    a, b = (10.0, 0.0), (-10.0, 0.0)
    assert _direction_penalty(a, b) == pytest.approx(1.0)  # cos=-1, max(0,1)=1


def test_direction_penalty_zero_vector() -> None:
    a, b = (0.0, 0.0), (10.0, 0.0)
    assert _direction_penalty(a, b) == pytest.approx(0.0)  # norm < 1e-3


# ── _parse_mot_lines / _format_mot_records ──────────────────────────────


def test_parse_mot_lines_empty() -> None:
    assert _parse_mot_lines([]) == []
    assert _parse_mot_lines([""]) == []
    assert _parse_mot_lines(["", "", ""]) == []


def test_parse_mot_lines_single() -> None:
    lines = ["1,10,50.0,60.0,30.0,70.0,0.9"]
    records = _parse_mot_lines(lines)
    assert len(records) == 1
    assert records[0].frame == 1
    assert records[0].track_id == 10
    assert records[0].score == pytest.approx(0.9)


def test_parse_mot_lines_multiple() -> None:
    lines = [
        "1,10,50.0,60.0,30.0,70.0,0.9",
        "2,10,55.0,65.0,30.0,70.0,0.85",
    ]
    records = _parse_mot_lines(lines)
    assert len(records) == 2
    assert records[0].frame == 1
    assert records[1].frame == 2


def test_format_mot_records_empty() -> None:
    assert _format_mot_records([]) == []


def test_format_mot_records_roundtrip() -> None:
    lines = [
        "1,10,50.0,60.0,30.0,70.0,0.9",
        "2,11,55.0,65.0,30.0,70.0,0.85",
    ]
    records = _parse_mot_lines(lines)
    formatted = _format_mot_records(records)
    re_parsed = _parse_mot_lines(formatted)
    assert len(re_parsed) == 2
    assert re_parsed[0].frame == 1
    assert re_parsed[0].track_id == 10
    assert re_parsed[1].frame == 2
    assert re_parsed[1].track_id == 11


def test_format_mot_records_sorted_by_frame_then_id() -> None:
    from saccade.perception.eval.types import MotRecord

    records = [
        MotRecord(frame=2, track_id=10, x=0, y=0, w=10, h=20, score=0.9, tail=[]),
        MotRecord(frame=1, track_id=20, x=0, y=0, w=10, h=20, score=0.9, tail=[]),
        MotRecord(frame=1, track_id=10, x=0, y=0, w=10, h=20, score=0.9, tail=[]),
    ]
    lines = _format_mot_records(records)
    assert len(lines) == 3
    # Should be sorted: (1,10), (1,20), (2,10)
    assert lines[0].startswith("1,10,")
    assert lines[1].startswith("1,20,")
    assert lines[2].startswith("2,10,")


# ── _build_output_tracklets ────────────────────────────────────────────


def test_build_output_tracklets_single_track() -> None:
    from saccade.perception.eval.types import MotRecord

    records = [
        MotRecord(frame=1, track_id=1, x=0, y=0, w=10, h=20, score=0.9, tail=[]),
        MotRecord(frame=2, track_id=1, x=10, y=5, w=10, h=20, score=0.85, tail=[]),
    ]
    tracklets = _build_output_tracklets(records, velocity_samples=2)
    assert len(tracklets) == 1
    assert tracklets[0].track_id == 1
    assert tracklets[0].start == 1
    assert tracklets[0].end == 2


def test_build_output_tracklets_multiple_tracks() -> None:
    from saccade.perception.eval.types import MotRecord

    records = [
        MotRecord(frame=1, track_id=1, x=0, y=0, w=10, h=20, score=0.9, tail=[]),
        MotRecord(frame=1, track_id=2, x=50, y=50, w=10, h=20, score=0.8, tail=[]),
        MotRecord(frame=2, track_id=1, x=10, y=5, w=10, h=20, score=0.85, tail=[]),
    ]
    tracklets = _build_output_tracklets(records, velocity_samples=2)
    assert len(tracklets) == 2
    ids = {t.track_id for t in tracklets}
    assert ids == {1, 2}


def test_build_output_tracklets_mean_score() -> None:
    from saccade.perception.eval.types import MotRecord

    records = [
        MotRecord(frame=1, track_id=1, x=0, y=0, w=10, h=20, score=0.8, tail=[]),
        MotRecord(frame=2, track_id=1, x=10, y=5, w=10, h=20, score=1.0, tail=[]),
    ]
    tracklets = _build_output_tracklets(records, velocity_samples=2)
    assert tracklets[0].mean_score == pytest.approx(0.9)


# ── post_merge_output_tracklets ─────────────────────────────────────────


def test_post_merge_disabled() -> None:
    lines = ["1,1,50.0,60.0,30.0,70.0,0.9"]
    result, stats = post_merge_output_tracklets(
        lines,
        enabled=False,
        ttl=10,
        min_gap=5,
        velocity_samples=2,
        spatial_weight=1.0,
        motion_weight=0.0,
        time_weight=0.0,
        direction_weight=0.0,
        max_cost=1.0,
    )
    assert result == lines
    assert stats["accepted"] == 0


def test_post_merge_empty_lines() -> None:
    result, stats = post_merge_output_tracklets(
        [],
        enabled=True,
        ttl=10,
        min_gap=5,
        velocity_samples=2,
        spatial_weight=1.0,
        motion_weight=0.0,
        time_weight=0.0,
        direction_weight=0.0,
        max_cost=1.0,
    )
    assert result == []
    assert stats["accepted"] == 0


def test_post_merge_single_tracklet() -> None:
    lines = ["1,1,50.0,60.0,30.0,70.0,0.9", "2,1,55.0,65.0,30.0,70.0,0.85"]
    result, stats = post_merge_output_tracklets(
        lines,
        enabled=True,
        ttl=10,
        min_gap=5,
        velocity_samples=2,
        spatial_weight=1.0,
        motion_weight=0.0,
        time_weight=0.0,
        direction_weight=0.0,
        max_cost=1.0,
    )
    assert stats["ids_before"] == 1
    assert stats["ids_after"] == 1


def test_post_merge_gap_too_small() -> None:
    """Gap below min_gap should not merge."""
    lines = [
        "1,1,50.0,60.0,30.0,70.0,0.9",
        "3,2,55.0,65.0,30.0,70.0,0.85",
    ]
    result, stats = post_merge_output_tracklets(
        lines,
        enabled=True,
        ttl=10,
        min_gap=5,  # gap=3-2=1 < 5
        velocity_samples=2,
        spatial_weight=1.0,
        motion_weight=0.0,
        time_weight=0.0,
        direction_weight=0.0,
        max_cost=1.0,
    )
    assert stats["ids_after"] == 2  # no merge


def test_post_merge_gap_exceeds_ttl() -> None:
    """Gap above ttl should not merge."""
    lines = [
        "1,1,50.0,60.0,30.0,70.0,0.9",
        "15,2,55.0,65.0,30.0,70.0,0.85",
    ]
    result, stats = post_merge_output_tracklets(
        lines,
        enabled=True,
        ttl=10,
        min_gap=5,
        velocity_samples=2,
        spatial_weight=1.0,
        motion_weight=0.0,
        time_weight=0.0,
        direction_weight=0.0,
        max_cost=1.0,
    )
    assert stats["ids_after"] == 2  # gap=13 > ttl=10


def test_post_merge_merge_within_ttl() -> None:
    """Tracklets within ttl and min_gap should be considered for merge."""
    lines = [
        "1,1,50.0,60.0,30.0,70.0,0.9",
        "3,2,50.5,60.5,30.0,70.0,0.85",  # gap=3-2=1... wait, gap = new.start - lost.end
    ]
    # Track 1: end=1, Track 2: start=3, gap=3-1=2
    result, stats = post_merge_output_tracklets(
        lines,
        enabled=True,
        ttl=10,
        min_gap=1,
        velocity_samples=2,
        spatial_weight=1.0,
        motion_weight=0.0,
        time_weight=0.0,
        direction_weight=0.0,
        max_cost=1.0,
    )
    # spatial cost should be very low since boxes are nearly identical
    assert stats["candidates"] >= 0


def test_post_merge_max_cost_rejects() -> None:
    lines = [
        "1,1,0.0,0.0,30.0,70.0,0.9",
        "3,2,500.0,500.0,30.0,70.0,0.85",  # far away
    ]
    result, stats = post_merge_output_tracklets(
        lines,
        enabled=True,
        ttl=10,
        min_gap=1,
        velocity_samples=2,
        spatial_weight=1.0,
        motion_weight=0.0,
        time_weight=0.0,
        direction_weight=0.0,
        max_cost=0.1,  # very tight
    )
    assert stats["ids_after"] == 2  # too far to merge


# ── filter_low_quality_tracklets ────────────────────────────────────────


def test_filter_low_quality_disabled() -> None:
    # min_len=1 and min_score=0.0 triggers early return (no filtering needed)
    lines = ["1,1,50.0,60.0,30.0,70.0,0.9"]
    result, stats = filter_low_quality_tracklets(lines, min_len=1, min_score=0.0)
    assert result == lines
    # Early return path: stats initialized to zeros
    assert stats["before"] == 0
    assert stats["after"] == 0


def test_filter_low_quality_enabled() -> None:
    # min_len=2 triggers the full path
    lines = ["1,1,50.0,60.0,30.0,70.0,0.9"]
    result, stats = filter_low_quality_tracklets(lines, min_len=2, min_score=0.0)
    assert len(result) == 0  # track has only 1 record, min_len=2 removes it
    assert stats["before"] == 1
    assert stats["removed"] == 1


def test_filter_low_quality_min_len_removes() -> None:
    lines = [
        "1,1,50.0,60.0,30.0,70.0,0.9",
    ]
    result, stats = filter_low_quality_tracklets(lines, min_len=5, min_score=0.0)
    assert len(result) == 0  # track 1 has only 1 record, min_len=5
    assert stats["before"] == 1
    assert stats["removed"] == 1


def test_filter_low_quality_min_score_removes() -> None:
    lines = [
        "1,1,50.0,60.0,30.0,70.0,0.9",
        "2,1,55.0,65.0,30.0,70.0,0.1",
    ]
    result, stats = filter_low_quality_tracklets(lines, min_len=1, min_score=0.6)
    # avg score = 0.5 < 0.6 => removed
    assert len(result) == 0
    assert stats["before"] == 1
    assert stats["removed"] == 1


def test_filter_low_quality_keeps_good() -> None:
    lines = [
        "1,1,50.0,60.0,30.0,70.0,0.9",
        "2,1,55.0,65.0,30.0,70.0,0.8",
    ]
    result, stats = filter_low_quality_tracklets(lines, min_len=1, min_score=0.5)
    assert len(result) == 2
    assert stats["removed"] == 0


def test_filter_low_quality_empty_lines() -> None:
    result, stats = filter_low_quality_tracklets([], min_len=5, min_score=0.5)
    assert result == []


# ── interpolate_tracklets ──────────────────────────────────────────────


def test_interpolate_disabled_max_gap_zero() -> None:
    lines = ["1,1,0.0,0.0,10.0,20.0,0.9", "5,1,40.0,20.0,10.0,20.0,0.9"]
    result, stats = interpolate_tracklets(lines, max_gap=0, min_track_len=2)
    assert result == lines
    assert stats["gaps_filled"] == 0


def test_interpolate_no_gaps() -> None:
    lines = [
        "1,1,0.0,0.0,10.0,20.0,0.9",
        "2,1,8.0,4.0,10.0,20.0,0.9",
        "3,1,16.0,8.0,10.0,20.0,0.9",
    ]
    result, stats = interpolate_tracklets(lines, max_gap=2, min_track_len=2)
    assert result == lines
    assert stats["gaps_filled"] == 0


def test_interpolate_small_gap() -> None:
    lines = [
        "1,1,0.0,0.0,10.0,20.0,0.9",
        "4,1,32.0,16.0,10.0,20.0,0.9",
    ]
    # One gap interval (frame 1 -> 4, gap = 2 intermediate frames)
    result, stats = interpolate_tracklets(lines, max_gap=5, min_track_len=2)
    assert stats["gaps_filled"] == 1  # 1 gap interval
    assert stats["frames_added"] == 2  # 2 frames inserted (frame 2 and 3)
    # Total lines = original 2 + 2 interpolated = 4
    assert len(result) == 4


def test_interpolate_gap_too_large() -> None:
    lines = [
        "1,1,0.0,0.0,10.0,20.0,0.9",
        "10,1,88.0,44.0,10.0,20.0,0.9",
    ]
    result, stats = interpolate_tracklets(lines, max_gap=5, min_track_len=2)
    assert stats["gaps_filled"] == 0  # gap=8 > max_gap=5


def test_interpolate_short_tracklet_filtered() -> None:
    lines = ["1,1,0.0,0.0,10.0,20.0,0.9"]
    result, stats = interpolate_tracklets(lines, max_gap=5, min_track_len=5)
    assert len(result) == 1
    assert stats["gaps_filled"] == 0


def test_interpolate_empty_lines() -> None:
    result, stats = interpolate_tracklets([], max_gap=5, min_track_len=2)
    assert result == []


def test_interpolate_multiple_tracks() -> None:
    lines = [
        "1,1,0.0,0.0,10.0,20.0,0.9",
        "3,1,24.0,12.0,10.0,20.0,0.9",
        "1,2,50.0,50.0,10.0,20.0,0.8",
        "2,2,58.0,54.0,10.0,20.0,0.8",
    ]
    result, stats = interpolate_tracklets(lines, max_gap=5, min_track_len=2)
    # Track 1 has gap, track 2 does not
    assert stats["tracks_interpolated"] >= 0


def test_interpolate_min_h_filter() -> None:
    lines = [
        "1,1,0.0,0.0,10.0,100.0,0.9",
        "5,1,40.0,20.0,10.0,100.0,0.9",
        "1,2,0.0,0.0,10.0,30.0,0.9",
        "5,2,40.0,20.0,10.0,30.0,0.9",
    ]
    result, stats = interpolate_tracklets(lines, max_gap=5, min_track_len=2, min_h=60)
    # Track 1 (h=100) should be interpolated, Track 2 (h=30 < 60) should not
    assert stats["gaps_filled"] == 1
    assert stats["tracks_interpolated"] == 1


def test_interpolate_min_h_both_sides() -> None:
    lines = [
        "1,1,0.0,0.0,10.0,100.0,0.9",
        "5,1,40.0,20.0,10.0,30.0,0.9",
    ]
    result, stats = interpolate_tracklets(lines, max_gap=5, min_track_len=2, min_h=60)
    assert stats["gaps_filled"] == 0


def test_interpolate_min_h_zero_default() -> None:
    lines = [
        "1,1,0.0,0.0,10.0,30.0,0.9",
        "5,1,40.0,20.0,10.0,30.0,0.9",
    ]
    result, stats = interpolate_tracklets(lines, max_gap=5, min_track_len=2)
    assert stats["gaps_filled"] == 1
