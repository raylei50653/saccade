#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build-native-coverage"

cmake -S "${repo_root}" -B "${build_dir}" -DENABLE_NATIVE_COVERAGE=ON
cmake --build "${build_dir}" --target saccade_assignment_algorithms_test saccade_gpu_postprocess_test -j4
ctest --test-dir "${build_dir}" --output-on-failure
