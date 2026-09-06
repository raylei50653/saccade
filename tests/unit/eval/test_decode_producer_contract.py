"""Producer→consumer contract of TorchvisionGpuStreamer's decode worker.

Three invariants, none of which depends on any hypothesis about issue #340's
capture race:

  * a decode failure reaches the consumer as a failure, not as end-of-sequence;
  * a worker writes only into the queue it was started with, so one that outlives
    its join timeout cannot interleave frames into the next sequence;
  * the frame is released to the caching allocator against the *consumer's*
    stream, which is what ``multi_stream`` needs since it runs the consumer under
    a worker stream while the decode thread never sets one.

The stubbed streamer keeps these assertions device-free: what is under test is
the queue protocol and the consumer's control flow, not CUDA behaviour.
"""

# scope: eval
# function: contract
# lifecycle: active

import contextlib
import queue as queue_mod
import threading

import pytest

from saccade.perception.eval.streaming import TorchvisionGpuStreamer


class _Frame:
    """Stand-in for a decoded CUDA tensor; ``permute`` is a view, not a copy."""

    def __init__(self, name: str):
        self.name = name
        self.recorded_on: list[object] = []

    def permute(self, *_dims: int) -> "_Frame":
        return self

    def record_stream(self, stream: object) -> None:
        self.recorded_on.append(stream)


class _Stream:
    """Identity-comparable stand-in for a torch CUDA stream."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"<stream {self.name}>"


def _stub_streamer(files: list[str], consumer: "_Stream | None" = None):
    """A streamer with every torch/torchvision dependency stubbed out."""
    consumer_stream = consumer or _Stream("consumer")

    streamer = TorchvisionGpuStreamer.__new__(TorchvisionGpuStreamer)
    streamer.img_files = list(files)
    streamer._prefetch = 2
    streamer._idx = 0
    streamer._stop = threading.Event()
    streamer._worker = None
    streamer._queue = None  # type: ignore[assignment]
    streamer._rgb = None

    class _Cuda:
        stream = staticmethod(lambda _s: contextlib.nullcontext())
        current_stream = staticmethod(lambda: consumer_stream)

    class _Torch:
        cuda = _Cuda

    streamer._torch = _Torch  # type: ignore[assignment]
    streamer._read_file = lambda path: path  # type: ignore[assignment]
    streamer._decode = lambda data, device, mode: _Frame(data)  # type: ignore[assignment]
    return streamer, consumer_stream


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_decode_failure_is_not_end_of_sequence() -> None:
    """A failing decode must fail the run, not truncate the sequence quietly.

    The worker used to queue a bare ``None``, which ``__next__`` turned into
    ``StopIteration`` — the same event as a finished sequence. The protocol
    therefore permitted a broken run to write a short, plausible-looking result
    file. This asserts the failure now arrives as one.
    """
    streamer, _ = _stub_streamer(["a.jpg", "b.jpg", "c.jpg"])

    def boom(_path):  # type: ignore[no-untyped-def]
        raise OSError("truncated JPEG")

    streamer._read_file = boom  # type: ignore[assignment]

    iterator = iter(streamer)
    with pytest.raises(RuntimeError) as excinfo:
        next(iterator)

    # Not StopIteration, and not a bare re-raise: the message has to place the
    # failure in the decode worker at a named frame, and keep the original cause.
    assert "decode worker" in str(excinfo.value)
    assert "a.jpg" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, OSError)
    assert "truncated JPEG" in str(excinfo.value.__cause__)

    # Join the worker here so its re-raise is reported against this test.
    assert streamer._worker is not None
    streamer._worker.join(timeout=5)


def test_a_stale_worker_cannot_write_into_a_fresh_queue() -> None:
    """The worker owns its queue, so a join timeout cannot interleave sequences.

    ``_stop_worker`` joins with a 3 s timeout. Before the queue was passed in, a
    worker that outlived that timeout would resume pushing into whichever queue
    ``self._queue`` pointed at by then — the *next* sequence's. Asserted directly
    on the ownership invariant rather than by racing a real thread.
    """
    streamer, _ = _stub_streamer(["a.jpg"])
    stale_queue: queue_mod.Queue = queue_mod.Queue()
    fresh_queue: queue_mod.Queue = queue_mod.Queue()
    streamer._queue = fresh_queue

    # Run the stale worker inline against the queue it was handed, exactly as a
    # thread that outlived its join would.
    streamer._decode_worker(stale_queue)

    assert fresh_queue.empty(), "a stale worker reached the next sequence's queue"
    assert stale_queue.qsize() == 1


def test_frame_is_recorded_against_the_consumer_stream() -> None:
    """The allocator is told the consuming stream still holds the block.

    Producer and consumer streams already differ on the ``multi_stream`` path,
    where ``run_eval`` runs under a worker stream while the decode thread is a
    new thread that never set one. Without this the allocator may hand the block
    back to the next decode while the consumer still has reads queued.
    """
    consumer = _Stream("worker-stream")
    streamer, consumer_stream = _stub_streamer(["a.jpg", "b.jpg"], consumer=consumer)

    frames = list(streamer)

    assert len(frames) == 2
    for frame in frames:
        assert frame.recorded_on == [consumer_stream]


def test_normal_iteration_and_end_of_sequence_are_unchanged() -> None:
    """The success path still yields every frame, in order, then StopIteration."""
    files = ["a.jpg", "b.jpg", "c.jpg"]
    streamer, _ = _stub_streamer(files)

    iterator = iter(streamer)
    assert [next(iterator).name for _ in files] == files
    with pytest.raises(StopIteration):
        next(iterator)
