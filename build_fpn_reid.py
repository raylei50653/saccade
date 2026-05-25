#!/usr/bin/env python3
"""Build the FPN ReID CUDA extension (JIT or pip)."""

from pathlib import Path
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import setuptools

project_root = Path(__file__).resolve().parent.parent

ext = CUDAExtension(
    name="saccade_fpn_reid",
    sources=[
        str(project_root / "src/tracking/fpn_reid_cuda.cu"),
    ],
    extra_compile_args={
        "cxx": ["-O3"],
        "nvcc": ["-O3", "-use_fast_math"],
    },
)

if __name__ == "__main__":
    setuptools.setup(
        name="saccade_fpn_reid",
        ext_modules=[ext],
        cmdclass={"build_ext": BuildExtension},
    )
