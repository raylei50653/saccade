"""Unit tests for the MediaMTX control client (saccade.media.mediamtx_client)."""

# scope: media
# function: behavior
# lifecycle: active

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from saccade.media import mediamtx_client
from saccade.media.mediamtx_client import MediaMTXClient


class _FakeSink:
    def __init__(self) -> None:
        self.connected: list[tuple[str, Any]] = []

    def connect(self, signal: str, callback: Any) -> None:
        self.connected.append((signal, callback))


class _FakeBus:
    def __init__(self) -> None:
        self.watch_added = False
        self.connected: list[tuple[str, Any]] = []

    def add_signal_watch(self) -> None:
        self.watch_added = True

    def connect(self, signal: str, callback: Any) -> None:
        self.connected.append((signal, callback))


class _FakePipeline:
    def __init__(self) -> None:
        self.sink = _FakeSink()
        self.bus = _FakeBus()
        self.states: list[Any] = []

    def get_by_name(self, name: str) -> _FakeSink:
        assert name == "sink"
        return self.sink

    def get_bus(self) -> _FakeBus:
        return self.bus

    def set_state(self, state: Any) -> None:
        self.states.append(state)


class _FakeMainLoop:
    def __init__(self) -> None:
        self.run_calls = 0
        self.quit_calls = 0

    def run(self) -> None:
        self.run_calls += 1

    def quit(self) -> None:
        self.quit_calls += 1


class _FakeThread:
    instances: list["_FakeThread"] = []

    def __init__(self, target: Any, daemon: bool) -> None:
        self.target = target
        self.daemon = daemon
        self.started = False
        self.join_calls: list[float | None] = []
        _FakeThread.instances.append(self)

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)
        self.started = False


@pytest.fixture(autouse=True)
def clear_fake_threads() -> None:
    _FakeThread.instances.clear()


def test_rtsp_pipeline_quotes_special_url_characters() -> None:
    url = "rtsp://reader:pa!ss word@127.0.0.1:8554/live"
    client = MediaMTXClient(rtsp_url=url)

    pipeline = client._get_pipeline_str()

    assert f"location='{url}'" in pipeline
    assert "protocols=tcp" in pipeline


def test_file_pipeline_quotes_paths_with_spaces(tmp_path: Path) -> None:
    video = tmp_path / "sample clip.mp4"
    video.write_bytes(b"not a real video")
    client = MediaMTXClient(dummy_video=str(video))

    pipeline = client._get_pipeline_str()

    assert f"filesrc location='{video}'" in pipeline
    assert "decodebin" in pipeline


def test_connect_releases_pipeline_when_first_frame_never_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pipeline = _FakePipeline()
    fake_loop = _FakeMainLoop()
    parsed: list[str] = []

    def fake_parse_launch(pipeline: str) -> _FakePipeline:
        parsed.append(pipeline)
        return fake_pipeline

    monkeypatch.setattr(mediamtx_client.Gst, "parse_launch", fake_parse_launch)
    monkeypatch.setattr(mediamtx_client.threading, "Thread", _FakeThread)
    monkeypatch.setattr(
        MediaMTXClient,
        "_await_first_frame",
        lambda self, timeout_sec=3.0: False,
    )
    client = MediaMTXClient(rtsp_url="rtsp://127.0.0.1:8554/live")
    client._mainloop = fake_loop

    connected = client.connect()

    assert connected is False
    assert parsed
    assert fake_pipeline.sink.connected == [("new-sample", client._on_new_sample)]
    assert fake_pipeline.bus.watch_added
    assert fake_pipeline.bus.connected == [("message", client._on_bus_message)]
    assert fake_pipeline.states == [
        mediamtx_client.Gst.State.PLAYING,
        mediamtx_client.Gst.State.NULL,
    ]
    assert fake_loop.quit_calls == 1
    assert _FakeThread.instances[0].join_calls == [1.0]
    assert client.pipeline is None
    assert client._loop_thread is None
    assert not client._running


def test_release_is_idempotent_and_clears_references() -> None:
    fake_pipeline = _FakePipeline()
    fake_loop = _FakeMainLoop()
    fake_thread = _FakeThread(target=fake_loop.run, daemon=True)
    fake_thread.start()
    client = MediaMTXClient(rtsp_url="rtsp://127.0.0.1:8554/live")
    client.pipeline = fake_pipeline
    client._mainloop = fake_loop
    client._loop_thread = fake_thread
    client._running = True

    client.release()
    client.release()

    assert fake_pipeline.states == [mediamtx_client.Gst.State.NULL]
    assert fake_loop.quit_calls == 2
    assert fake_thread.join_calls == [1.0]
    assert client.pipeline is None
    assert client._loop_thread is None
    assert not client._running
