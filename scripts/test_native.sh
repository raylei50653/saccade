#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build-native-coverage"
cuda_root="${CUDA_HOME:-/opt/cuda}"

export CUDA_HOME="${cuda_root}"
export CUDAToolkit_ROOT="${cuda_root}"
export PATH="${cuda_root}/bin:${PATH}"

cmake -S "${repo_root}" -B "${build_dir}" \
  -DENABLE_NATIVE_COVERAGE=ON \
  -DCUDAToolkit_ROOT="${cuda_root}" \
  -DCUDA_TOOLKIT_ROOT_DIR="${cuda_root}" \
  -DCMAKE_CUDA_COMPILER="${cuda_root}/bin/nvcc"
cmake --build "${build_dir}" -j4
ctest --test-dir "${build_dir}" --output-on-failure
