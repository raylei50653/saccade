import os
import queue
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator, List, cast

# DALI is an optional extra (`uv sync --extra dali`). Eval/unit tests must import
# this module without DALI (cloud CI / no-GPU hosts use TorchvisionGpuStreamer).
try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIGenericIterator

    HAS_DALI = True
except ImportError:  # pragma: no cover - exercised when extra not installed
    fn = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    Pipeline = object  # type: ignore[misc, assignment]
    DALIGenericIterator = None  # type: ignore[misc, assignment]
    HAS_DALI = False

_DALI_INSTALL_HINT = "nvidia-dali is not installed. Install with: uv sync --extra dali"


def _require_dali() -> None:
    if not HAS_DALI:
        raise ImportError(_DALI_INSTALL_HINT)


class JpgPipe(Pipeline):  # type: ignore[misc, valid-type]
    def __init__(
        self, batch_size: int, num_threads: int, device_id: int, files: List[str]
    ):
        _require_dali()
        cast(Any, super()).__init__(batch_size, num_threads, device_id)
        self.input = fn.readers.file(files=files, name="Reader")

    def define_graph(self) -> Any:
        jpegs, labels = self.input
        images = fn.decoders.image(
            jpegs, device="mixed", output_type=getattr(types, "RGB")
        )
        return images


class DALIStreamer:
    """High-speed JPEG sequence streamer using NVIDIA DALI."""

    def __init__(self, files: List[str], batch_size: int = 1):
        _require_dali()
        self.files = files
        self.batch_size = batch_size
        self.pipe = JpgPipe(
            batch_size=batch_size, num_threads=2, device_id=0, files=files
        )
        self.pipe.build()
        self.iterator = DALIGenericIterator([self.pipe], ["data"], size=len(files))

    def __iter__(self) -> Iterator[Any]:
        return iter(self.iterator)

    def __len__(self) -> int:
        return len(self.files)


def get_streamer(path_list: List[str], batch_size: int = 1) -> DALIStreamer:
    return DALIStreamer(path_list, batch_size)


class DALIStreamerStream:
    """Compatibility wrapper used by the eval runner for MOT image folders."""

    def __init__(self, img_dir: Path):
        _require_dali()
        self.img_files = sorted(str(path.absolute()) for path in img_dir.glob("*.jpg"))
        self.file_list_path: str | None = None
        self._setup()

    def _setup(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as handle:
            for img in self.img_files:
                handle.write(f"{img} 0\n")
            self.file_list_path = handle.name

        class _JpgPipe(Pipeline):  # type: ignore[misc, valid-type]
            def __init__(self, file_list: str):
                cast(Any, super()).__init__(1, 4, 0, prefetch_queue_depth=2)
                self.input = fn.readers.file(file_list=file_list, name="reader")

            def define_graph(self) -> Any:
                jpegs, _ = self.input
                return fn.decoders.image(
                    jpegs, device="cpu", output_type=getattr(types, "RGB")
                ).gpu()

        self.pipe = _JpgPipe(self.file_list_path)
        cast(Any, self.pipe).build()
        self.iterator = cast(Any, DALIGenericIterator)(
            [self.pipe], ["data"], auto_reset=True
        )

    def __iter__(self) -> "DALIStreamerStream":
        return self

    def __next__(self) -> Any:
        try:
            return next(self.iterator)[0]["data"][0]
        except StopIteration:
            if self.file_list_path and os.path.exists(self.file_list_path):
                os.remove(self.file_list_path)
            raise


class _DecodeFailure:
    """Queue item marking a failed decode, kept distinct from end-of-sequence.

    The producer used to queue a bare ``None`` on failure, which the consumer
    turned into ``StopIteration`` -- so a decode error and a finished sequence
    were the same event to the caller.  Carrying the exception instead lets
    :meth:`TorchvisionGpuStreamer.__next__` re-raise it on the consuming thread.
    """

    __slots__ = ("error",)

    def __init__(self, error: BaseException):
        self.error = error


class TorchvisionGpuStreamer:
    """Drop-in for DALIStreamerStream that decodes JPEGs on the GPU's dedicated
    NVJPG hardware engine via torchvision/nvJPEG.

    DALI's ``device="mixed"`` decoder can't build under WSL2 (its nvJPEG init
    queries NVML, which WSL2 stubs out), so the eval path falls back to CPU
    decode. torchvision's ``decode_jpeg(device="cuda")`` hits the same nvJPEG
    library directly and works — running on the NVJPG engine (~90% util) without
    touching the SMs, so it offloads decode off the CPU for free.

    A background daemon thread prefetches decoded frames so that the CPU-side
    Huffman decode (nvJPEG hybrid backend) overlaps with GPU computation on the
    main thread.

    Yields ``[H, W, C]`` uint8 CUDA tensors to match ``DALIStreamerStream``.

    Stream contract (issue #340 Phase 2B)
    -------------------------------------
    Three facts about ``decode_jpeg(device="cuda")``, measured with an LD_PRELOAD
    tally of the cudart entry points rather than read off the documentation:

    1. nvJPEG runs the decode on a private stream of its own, and torchvision
       calls ``cudaStreamSynchronize`` on that stream before returning.  The
       decode is therefore already complete when the tensor reaches the queue.
    2. torchvision then joins the result to the **caller's current torch stream**
       with ``cudaStreamWaitEvent``.  In the worker thread that used to be the
       legacy stream, which is the whole of this thread's legacy-stream
       footprint -- and the second precondition for Rule B (see
       :mod:`.cuda_capture`).
    3. Setting the worker's current stream redirects that join, so running the
       worker under its own stream takes the thread off the legacy stream
       entirely.  Verified by the same tally: 24 ``cudaStreamWaitEvent`` calls
       move from stream ``0x0`` to the dedicated stream, and the decode thread's
       legacy-stream call count goes to zero.

    Point 1 means the event handoff below is not what makes the pixels visible
    today.  It is here so that the ordering is *this module's* invariant rather
    than a private detail of torchvision's decoder, and because points 2 and 3
    make the consumer's stream no longer the producer's.  Two separate
    invariants follow, and neither substitutes for the other:

    * **Execution ordering** -- each frame is queued with an event recorded on
      the decode stream, and :meth:`__next__` makes the consuming stream wait on
      it.  The double-buffer path then records ``input_ready`` on that same
      consuming stream, so its side stream inherits the dependency transitively.
    * **Allocator lifetime** -- ``record_stream`` in :meth:`__next__`, which the
      producer/consumer stream split already required before this change and
      still does.  An event orders execution; only ``record_stream`` stops the
      caching allocator handing the block back while reads are still queued.

    The decode stream is owned by the *worker*, not by this object: see
    :meth:`_start_worker`.
    """

    def __init__(self, img_dir: Path, prefetch: int = 2):
        import torch
        from torchvision.io import ImageReadMode, decode_jpeg, read_file

        self._read_file = read_file
        self._decode = decode_jpeg
        self._rgb = ImageReadMode.RGB
        self._torch = torch
        self.img_files = sorted(str(path.absolute()) for path in img_dir.glob("*.jpg"))
        self._prefetch = max(1, prefetch)
        self._idx = 0
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        # The current worker generation's decode stream; ``None`` until the first
        # ``_start_worker``.  Owned by the worker, replaced with it.
        self._decode_stream: Any = None
        # Mode the worker thread held before its Rule A exemption; ``None`` until
        # a worker has successfully exchanged (see ``cuda_capture``).
        self.relaxed_capture_mode_from: str | None = None

    def _start_worker(self) -> None:
        self._stop.clear()
        # Bind the queue to the worker instead of letting it read ``self._queue``.
        # ``_stop_worker`` joins with a timeout, so a worker that outlives that
        # timeout would otherwise resume pushing into whichever queue
        # ``self._queue`` points at by then -- the next sequence's -- interleaving
        # two sequences' frames.
        work_queue: queue.Queue = queue.Queue(maxsize=self._prefetch + 1)
        self._queue = work_queue
        # The decode stream is bound to the worker for the same reason the queue
        # is.  A worker that outlived the join timeout is still issuing decodes;
        # if the stream belonged to this object, the next sequence's worker would
        # share a CUDA ordering domain with it, and the per-frame events of two
        # generations would be recorded on one stream.  Created here rather than
        # in ``__init__`` so that constructing a streamer never forces CUDA init
        # -- device-free tests build one.
        worker_stream = self._torch.cuda.Stream()
        self._decode_stream = worker_stream
        self._worker = threading.Thread(
            target=self._decode_worker, args=(work_queue, worker_stream), daemon=True
        )
        self._worker.start()

    def _decode_worker(self, out_queue: "queue.Queue", worker_stream: Any) -> None:
        from .cuda_capture import enter_relaxed_capture_mode

        torch = self._torch
        try:
            # Rule A exemption: this thread allocates through torch while the
            # main thread may hold a "global"-mode capture open.  Inside the try
            # deliberately -- the shared helper raises when the exchange does not
            # complete cleanly, and that has to reach the consumer through the
            # same failure path a decode error takes, or a loud failure here
            # would just hang ``__next__``.  Left unset on that path: the
            # post-call mode is unverified, so no prior mode is recorded.  Rule B
            # is untouched by any of this.
            self.relaxed_capture_mode_from = enter_relaxed_capture_mode()
            # Rule B's second precondition is this thread's legacy-stream
            # footprint, and the footprint is torchvision's join to the caller's
            # current stream.  Decoding under the worker's own stream removes it,
            # whoever else opens a blocking capture.
            with torch.cuda.stream(worker_stream):
                for f in self.img_files:
                    if self._stop.is_set():
                        return
                    data = self._read_file(f)
                    img_chw = self._decode(data, device="cuda", mode=self._rgb)
                    img_hwc = img_chw.permute(1, 2, 0)
                    ready = torch.cuda.Event()
                    ready.record(worker_stream)
                    out_queue.put((img_hwc, ready))
        except Exception as exc:
            # Rule B leaves an open question: `cudaErrorStreamCaptureImplicit`
            # needs a *blocking* capturing stream, and the stream responsible is
            # unidentified -- a stream can join a capture it never began, so our
            # own non-blocking capture origins do not rule one out (see
            # `.cuda_capture`). Printed unconditionally: rare and already fatal.
            try:
                from .cuda_capture import describe_capture_state

                print(describe_capture_state("decode_worker:error"), flush=True)
            except Exception:  # noqa: BLE001 - never mask the decode error
                pass
            # Hand the failure to the consumer as a failure.  A bare ``None``
            # sentinel used to be indistinguishable from end-of-sequence, so the
            # protocol permitted a decode error to truncate the output silently.
            out_queue.put(_DecodeFailure(exc))
            raise

    def _stop_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            return
        self._stop.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._worker.join(timeout=3)

    def __iter__(self) -> "TorchvisionGpuStreamer":
        self._stop_worker()
        self._idx = 0
        self._start_worker()
        return self

    def __next__(self) -> Any:
        if self._idx >= len(self.img_files):
            raise StopIteration
        result = self._queue.get()
        # Failure keeps its own type rather than a shape the success path could
        # be mistaken for: check before unpacking.
        if isinstance(result, _DecodeFailure):
            raise RuntimeError(
                f"GPU JPEG decode failed in the decode worker at frame index "
                f"{self._idx} ({self.img_files[self._idx]})"
            ) from result.error
        frame, ready = result
        consumer = self._torch.cuda.current_stream()
        # Execution ordering: the consuming stream may not run before the decode
        # stream reached this frame.
        consumer.wait_event(ready)
        # Allocator lifetime, which the event does not cover.  The frame is
        # allocated by the producer thread on the worker's decode stream, so once
        # the consumer drops its reference the caching allocator is free to hand
        # the block back to the next ``decode_jpeg`` while the consumer stream
        # still has reads queued against it.  ``record_stream`` is what tells the
        # allocator otherwise.  (``multi_stream`` makes the two streams differ on
        # the consumer side as well: it runs ``run_eval`` under a worker stream.)
        frame.record_stream(consumer)
        self._idx += 1
        return frame
