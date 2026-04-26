import os
import tempfile
from pathlib import Path

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator


class DALIStreamerStream:
    def __init__(self, img_dir: Path):
        self.img_files = sorted(list(img_dir.glob("*.jpg")))
        self._setup()

    def _setup(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            for img in self.img_files:
                f.write(f"{img.absolute()} 0\n")
            self.file_list_path = f.name

        class JpgPipe(Pipeline):
            def __init__(self, flist):
                super().__init__(1, 4, 0, prefetch_queue_depth=2)
                self.input = fn.readers.file(file_list=flist, name="reader")

            def define_graph(self):
                jpegs, _ = self.input
                return fn.decoders.image(jpegs, device="cpu", output_type=types.RGB).gpu()

        self.pipe = JpgPipe(self.file_list_path)
        self.pipe.build()
        self.iterator = DALIGenericIterator([self.pipe], ["data"], auto_reset=True)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)[0]["data"][0]
        except StopIteration:
            if os.path.exists(self.file_list_path):
                os.remove(self.file_list_path)
            raise
