#!/usr/bin/env bash
# status: stable
# Configure/build native targets with coverage instrumentation.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${repo_root}/build-native-coverage"
target="saccade_assignment_algorithms_test"
gcov_input="CMakeFiles/${target}.dir/tests/native/test_assignment_algorithms.cpp.gcno"

cmake -S "${repo_root}" -B "${build_dir}" -DENABLE_NATIVE_COVERAGE=ON
cmake --build "${build_dir}" --target "${target}" -j4
ctest --test-dir "${build_dir}" --output-on-failure

pushd "${build_dir}" >/dev/null
gcov -b -c -l -p "${gcov_input}" | awk '
    /^File '\''\/home\/ray\/developer\/ai\/saccade\// {
        keep = ($0 ~ /(include\/tracking|tests\/native)/)
        if (keep) {
            print
        }
        next
    }
    /^Lines executed:/ && keep {
        print
        keep = 0
    }
'
popd >/dev/null
