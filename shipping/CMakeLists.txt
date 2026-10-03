# Shipping runtime config loader (#465 Phase B PR-4a): dependency-free C++17,
# no CUDA/torch. Standalone project on purpose -- the root CMakeLists.txt is a
# runtime-identity environment input, and PR-4a does not owe a republication.
# PR-4b wires this into the root build together with that republication.
#
#   cmake -S shipping -B build-shipping && cmake --build build-shipping
#   ctest --test-dir build-shipping --output-on-failure
cmake_minimum_required(VERSION 3.18)
project(saccade_shipping LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(saccade_shipping_config STATIC
    src/strict_json.cpp
    src/resolved_config.cpp
)
target_include_directories(saccade_shipping_config PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(saccade_shipping_config PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
get_filename_component(SACCADE_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
add_executable(saccade_resolved_config_test ${SACCADE_REPO_ROOT}/tests/native/test_resolved_config.cpp)
target_link_libraries(saccade_resolved_config_test PRIVATE saccade_shipping_config)
target_compile_options(saccade_resolved_config_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME saccade_resolved_config_test
    COMMAND saccade_resolved_config_test
        ${SACCADE_REPO_ROOT}/configs/shipping/mamba_whole_graph.resolved.json)
