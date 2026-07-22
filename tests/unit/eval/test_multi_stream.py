"""Tests for multi-stream evaluation orchestration (perception.eval.multi_stream)."""

# scope: eval
# function: behavior
# lifecycle: active

from __future__ import annotations

import threading
from contextlib import nullcontext

import pytest

from saccade.perception.eval import multi_stream


class _FakeProxy:
    def __init__(self, server, stream_id) -> None:
        self.server = server
        self.stream_id = stream_id
        self.stream = None


def test_run_sequence_worker_releases_init_lock_on_success(monkeypatch) -> None:
    lock = threading.Lock()
    eval_mod = multi_stream.run_eval_single.__module__
    evaluator_mod = __import__(eval_mod, fromlist=["dummy"])
    evaluator_mod._worker_init_lock = lock

    assert lock.acquire(blocking=False)
    lock.release()

    monkeypatch.setattr(
        "saccade.perception.multistream_mamba_server.MambaStreamProxy",
        _FakeProxy,
    )
    monkeypatch.setattr(multi_stream.torch.cuda, "Stream", lambda: object())
    monkeypatch.setattr(
        multi_stream.torch.cuda, "stream", lambda _stream: nullcontext()
    )
    monkeypatch.setattr(
        multi_stream,
        "run_eval_single",
        lambda **_kwargs: {"ok": True},
    )

    result = multi_stream._run_sequence_worker("seq", object(), {})

    assert result == {"ok": True}
    assert lock.acquire(blocking=False)
    lock.release()
    evaluator_mod._worker_init_lock = None


def test_run_sequence_worker_releases_init_lock_on_error(monkeypatch) -> None:
    lock = threading.Lock()
    eval_mod = multi_stream.run_eval_single.__module__
    evaluator_mod = __import__(eval_mod, fromlist=["dummy"])
    evaluator_mod._worker_init_lock = lock

    monkeypatch.setattr(
        "saccade.perception.multistream_mamba_server.MambaStreamProxy",
        _FakeProxy,
    )
    monkeypatch.setattr(multi_stream.torch.cuda, "Stream", lambda: object())
    monkeypatch.setattr(
        multi_stream.torch.cuda, "stream", lambda _stream: nullcontext()
    )

    def _raise(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(multi_stream, "run_eval_single", _raise)

    with pytest.raises(RuntimeError, match="boom"):
        multi_stream._run_sequence_worker("seq", object(), {})

    assert lock.acquire(blocking=False)
    lock.release()
    evaluator_mod._worker_init_lock = None
