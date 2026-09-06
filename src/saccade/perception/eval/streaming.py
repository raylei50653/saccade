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
        self._worker = threading.Thread(
            target=self._decode_worker, args=(work_queue,), daemon=True
        )
        self._worker.start()

    def _decode_worker(self, out_queue: "queue.Queue") -> None:
        from .cuda_capture import enter_relaxed_capture_mode

        try:
            # Rule A exemption: this thread allocates through torch while the
            # main thread may hold a "global"-mode capture open.  Inside the try
            # deliberately -- the shared helper now raises on a failed exchange,
            # and that has to reach the consumer through the same failure path a
            # decode error takes, or a loud failure here would just hang
            # ``__next__``.  Rule B is untouched by any of this.
            self.relaxed_capture_mode_from = enter_relaxed_capture_mode()
            for f in self.img_files:
                if self._stop.is_set():
                    return
                data = self._read_file(f)
                img_chw = self._decode(data, device="cuda", mode=self._rgb)
                img_hwc = img_chw.permute(1, 2, 0)
                out_queue.put(img_hwc)
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
        if isinstance(result, _DecodeFailure):
            raise RuntimeError(
                f"GPU JPEG decode failed in the decode worker at frame index "
                f"{self._idx} ({self.img_files[self._idx]})"
            ) from result.error
        # The frame is allocated by the producer thread, on the producer thread's
        # current stream.  Where that differs from the consumer's -- ``multi_stream``
        # runs ``run_eval`` under a worker stream while the decode thread is a new
        # thread that never set one -- the caching allocator would be free to hand
        # the block back to the next ``decode_jpeg`` as soon as the consumer drops
        # its reference, while the consumer stream still has reads queued against
        # it.  ``record_stream`` is what tells the allocator otherwise.
        result.record_stream(self._torch.cuda.current_stream())
        self._idx += 1
        return result
