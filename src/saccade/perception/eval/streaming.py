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

    1. nvJPEG runs the decode on a private stream of its own, and torchvision calls
       ``cudaStreamSynchronize`` on that stream before returning.  The decode is
       therefore already complete when the tensor reaches the queue.
    2. torchvision then joins the result to the **caller's current torch stream**
       with ``cudaStreamWaitEvent``.  In the worker thread that used to be the
       legacy stream, which is the whole of this thread's legacy-stream
       footprint — and the second precondition for Rule B (see
       :mod:`.cuda_capture`).
    3. Setting the worker's current stream redirects that join, so running the
       worker under ``self._decode_stream`` takes the thread off the legacy
       stream entirely.  Verified by the same tally: 24 ``cudaStreamWaitEvent``
       calls move from stream ``0x0`` to the dedicated stream, and the decode
       thread's legacy-stream call count goes to zero.

    Point 1 means the handoff below is not what makes the pixels visible today.
    It is here so that the ordering is *this module's* invariant rather than a
    private detail of torchvision's decoder, and because points 2 and 3 make the
    consumer's stream no longer the producer's:

    * **Ordering.**  Each frame is queued with an event recorded on the decode
      stream, and :meth:`__next__` makes the consuming stream wait on it.  The
      double-buffer path then records ``input_ready`` on that same consuming
      stream, so its side stream inherits the dependency transitively.
    * **Buffer lifetime.**  The block is allocated on the decode stream, so once
      the consumer drops its reference the allocator may hand it straight back
      to the next ``decode_jpeg`` — which would overwrite pixels a consumer
      stream still has reads queued against.  :meth:`__next__` therefore calls
      ``record_stream`` on every frame with the stream that consumes it.  This is
      the invariant that same-stream production used to provide for free.
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
        # Created lazily on the first iteration: constructing a streamer must not
        # force CUDA init (unit tests build one on hosts without a device).
        self._decode_stream: Any = None
        # Rule A exemption status of the worker thread, for tests and diagnostics.
        self.relaxed_capture_mode_from: str | None = None

    def _start_worker(self) -> None:
        self._stop.clear()
        if self._decode_stream is None:
            self._decode_stream = self._torch.cuda.Stream()
        # Bind the queue to the worker instead of reading ``self._queue``: a
        # previous worker that outlived its join timeout would otherwise start
        # pushing frames into the fresh queue and interleave two sequences.
        work_queue: queue.Queue = queue.Queue(maxsize=self._prefetch + 1)
        self._queue = work_queue
        self._worker = threading.Thread(
            target=self._decode_worker, args=(work_queue,), daemon=True
        )
        self._worker.start()

    def _decode_worker(self, out_queue: "queue.Queue") -> None:
        torch = self._torch
        from .cuda_capture import describe_capture_state, enter_relaxed_capture_mode

        # Rule A: this thread allocates through torch while the main thread may
        # hold a "global"-mode capture open (make_graphed_callables takes no
        # capture_error_mode), so it exempts itself.  Rule B is handled by the
        # decode stream below, not by this call.
        try:
            self.relaxed_capture_mode_from = enter_relaxed_capture_mode()
            with torch.cuda.stream(self._decode_stream):
                for f in self.img_files:
                    if self._stop.is_set():
                        return
                    data = self._read_file(f)
                    img_chw = self._decode(data, device="cuda", mode=self._rgb)
                    img_hwc = img_chw.permute(1, 2, 0)
                    ready = torch.cuda.Event()
                    ready.record(self._decode_stream)
                    out_queue.put((img_hwc, ready))
        except BaseException as exc:  # noqa: BLE001 - re-raised after handoff
            # Rule B leaves an open question: `cudaErrorStreamCaptureImplicit`
            # needs a *blocking* capturing stream, and every stream we capture
            # on is non-blocking, so the capture responsible may not be ours.
            # `open_capture=None` in this dump proves exactly that. Printed
            # unconditionally: this path is rare and already fatal.
            try:
                print(describe_capture_state("decode_worker:error"), flush=True)
            except Exception:  # noqa: BLE001 - never mask the decode error
                pass
            # Hand the exception to the consumer.  Queueing a bare sentinel used
            # to make a decode failure indistinguishable from end-of-sequence,
            # which silently truncated the sequence instead of failing the run.
            out_queue.put((None, exc))
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
        frame, ready = self._queue.get()
        if frame is None:
            raise RuntimeError(
                f"GPU JPEG decode failed at frame index {self._idx} "
                f"({self.img_files[self._idx]})"
            ) from ready
        consumer = self._torch.cuda.current_stream()
        consumer.wait_event(ready)
        frame.record_stream(consumer)
        self._idx += 1
        return frame
