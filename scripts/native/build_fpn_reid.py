#!/usr/bin/env python3
"""Build the legacy FPN ReID CUDA extension with setuptools."""

from pathlib import Path

import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

project_root = Path(__file__).resolve().parents[2]

ext = CUDAExtension(
    name="saccade_fpn_reid_cuda",
    sources=[
        str(project_root / "src/tracking/fpn_reid_cuda.cu"),
        str(project_root / "src/tracking/fpn_reid_binding.cpp"),
    ],
    extra_compile_args={
        "cxx": ["-O3"],
        "nvcc": ["-O3", "-use_fast_math"],
    },
)

if __name__ == "__main__":
    setuptools.setup(
        name="saccade_fpn_reid_cuda",
        ext_modules=[ext],
        cmdclass={"build_ext": BuildExtension},
    )
