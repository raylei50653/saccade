"""Unit tests for the tracker reordering buffer (perception.tracking.reorder)."""

# scope: tracking
# function: behavior
# lifecycle: active

from unittest.mock import patch

from saccade.perception.tracking.reorder import ReorderingBuffer


def test_push_then_pop_with_full_buffer():
    buf = ReorderingBuffer(window_ms=150)
    for i in range(6):
        buf.push(i * 33, f"frame_{i}")
    result = buf.pop_ready()
    assert len(result) >= 1
    assert result[0] == "frame_0"


def test_pop_ready_empty_buffer():
    buf = ReorderingBuffer()
    assert buf.pop_ready() == []


def test_pop_ready_insufficient_frames_no_timeout():
    buf = ReorderingBuffer(window_ms=5000)
    buf.push(100, "A")
    buf.push(200, "B")
    with patch("saccade.perception.tracking.reorder.time") as mock_time:
        mock_time.time.return_value = 0.0
        result = buf.pop_ready()
    assert result == []


def test_pop_ready_returns_in_timestamp_order():
    # window_ms=50: after each pop, next item's ts diff (100ms) > 50ms → keeps draining
    buf = ReorderingBuffer(window_ms=50)
    buf.push(300, "C")
    buf.push(100, "A")
    buf.push(200, "B")
    for i in range(3):
        buf.push(400 + i * 33, f"extra_{i}")
    result = buf.pop_ready()
    assert result[0] == "A"
    assert result[1] == "B"
    assert result[2] == "C"


def test_reset_clears_state():
    buf = ReorderingBuffer()
    buf.push(100, "X")
    buf.push(200, "Y")
    buf.reset()
    assert buf.buffer == []
    assert buf.last_emitted_ts == 0
    assert buf.pop_ready() == []


def test_pop_ready_window_threshold_triggers():
    buf = ReorderingBuffer(window_ms=50)
    buf.last_emitted_ts = 0
    buf.push(100, "stale")
    result = buf.pop_ready()
    assert "stale" in result


def test_count_increments_on_each_push():
    buf = ReorderingBuffer()
    assert buf.count == 0
    buf.push(1, "a")
    buf.push(2, "b")
    assert buf.count == 2
