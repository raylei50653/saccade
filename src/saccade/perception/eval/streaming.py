import os
import queue
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator, List, cast

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator


class JpgPipe(Pipeline):
    def __init__(
        self, batch_size: int, num_threads: int, device_id: int, files: List[str]
    ):
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
        self.img_files = sorted(str(path.absolute()) for path in img_dir.glob("*.jpg"))
        self.file_list_path: str | None = None
        self._setup()

    def _setup(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as handle:
            for img in self.img_files:
                handle.write(f"{img} 0\n")
            self.file_list_path = handle.name

        class _JpgPipe(Pipeline):
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

    def _start_worker(self) -> None:
        self._stop.clear()
        self._queue = queue.Queue(maxsize=self._prefetch + 1)
        self._worker = threading.Thread(target=self._decode_worker, daemon=True)
        self._worker.start()

    @staticmethod
    def _enter_relaxed_capture_mode() -> None:
        """Scope this thread out of CUDA stream-capture safety checks.

        The worker issues nvJPEG/allocator calls concurrently with the main
        thread's per-sequence CUDA-graph captures (GMC/tracker/NMS/detect).
        Under the default cudaStreamCaptureModeGlobal, any such call
        invalidates an in-progress capture — intermittent
        cudaErrorStreamCaptureInvalidated at sequence start. Relaxed mode is
        the documented remedy for background threads.
        """
        import ctypes

        for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
            try:
                rt = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            return
        mode = ctypes.c_int(2)  # cudaStreamCaptureModeRelaxed
        rt.cudaThreadExchangeStreamCaptureMode(ctypes.byref(mode))

    def _decode_worker(self) -> None:
        self._enter_relaxed_capture_mode()
        try:
            for f in self.img_files:
                if self._stop.is_set():
                    return
                data = self._read_file(f)
                img_chw = self._decode(data, device="cuda", mode=self._rgb)
                img_hwc = img_chw.permute(1, 2, 0)
                self._queue.put(img_hwc)
        except Exception:
            self._queue.put(None)
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
        if result is None:
            raise StopIteration
        self._idx += 1
        return result
