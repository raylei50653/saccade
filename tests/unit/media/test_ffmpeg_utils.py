"""Unit tests for the RTSP ffmpeg streamer (saccade.media.ffmpeg_utils)."""

# scope: media
# function: behavior
# lifecycle: active

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pytest

from saccade.media.ffmpeg_utils import RTSPStreamer


class _FakeProcess:
    def __init__(self, *, alive: bool = True, stdin: Any | None = None) -> None:
        self._alive = alive
        self.stdin = stdin if stdin is not None else BytesIO()
        self.terminated = False
        self.killed = False
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        return None if self._alive else 1

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def kill(self) -> None:
        self.killed = True
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        return 0


def test_start_builds_ffmpeg_command_and_normalizes_localhost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_popen(command: list[str], stdin: Any) -> _FakeProcess:
        calls.append(command)
        assert stdin is not None
        return _FakeProcess()

    monkeypatch.setattr("saccade.media.ffmpeg_utils.subprocess.Popen", fake_popen)
    monkeypatch.setattr("saccade.media.ffmpeg_utils.time.sleep", lambda _: None)
    streamer = RTSPStreamer(
        rtsp_url="rtsp://localhost:8554/out",
        fps=30,
        width=1280,
        height=720,
    )

    streamer.start()

    assert streamer.rtsp_url == "rtsp://127.0.0.1:8554/out"
    assert len(calls) == 1
    command = calls[0]
    assert command[0] == "ffmpeg"
    assert command[command.index("-s") + 1] == "1280x720"
    assert command[command.index("-r") + 1] == "30"
    assert command[-1] == "rtsp://127.0.0.1:8554/out"


def test_start_is_idempotent_when_process_is_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_popen(command: list[str], stdin: Any) -> _FakeProcess:
        nonlocal calls
        calls += 1
        return _FakeProcess()

    monkeypatch.setattr("saccade.media.ffmpeg_utils.subprocess.Popen", fake_popen)
    monkeypatch.setattr("saccade.media.ffmpeg_utils.time.sleep", lambda _: None)
    streamer = RTSPStreamer()

    streamer.start()
    streamer.start()

    assert calls == 1


def test_push_frame_resizes_and_writes_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = BytesIO()
    streamer = RTSPStreamer(width=4, height=2)
    streamer.process = _FakeProcess(stdin=stream)
    frame = np.ones((1, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "saccade.media.ffmpeg_utils.cv2.resize",
        lambda image, size: np.full((size[1], size[0], 3), 7, dtype=image.dtype),
    )

    streamer.push_frame(frame)

    written = stream.getvalue()
    assert len(written) == 2 * 4 * 3
    assert written == bytes([7]) * (2 * 4 * 3)


def test_push_frame_rejects_non_bgr_frames() -> None:
    streamer = RTSPStreamer()
    grayscale = np.zeros((480, 640), dtype=np.uint8)

    with pytest.raises(ValueError, match="shape"):
        streamer.push_frame(grayscale)


def test_stop_closes_stream_and_waits_for_process() -> None:
    stream = BytesIO()
    process = _FakeProcess(stdin=stream)
    streamer = RTSPStreamer()
    streamer.process = process

    streamer.stop()

    assert streamer.process is None
    assert stream.closed
    assert process.terminated
    assert process.wait_calls == [1.0]
