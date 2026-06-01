import os
import tempfile
from pathlib import Path
from typing import Any, Iterator, List, cast

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator


class JpgPipe(Pipeline):  # type: ignore[misc]
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

        class _JpgPipe(Pipeline):  # type: ignore[misc]
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
