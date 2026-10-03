# Shipping runtime config (#465 Phase B).
#
# Two ways in:
#   * standalone (CI job `shipping-config-loader`, stock runner, no CUDA):
#       cmake -S shipping -B build-shipping && cmake --build build-shipping
#       ctest --test-dir build-shipping --output-on-failure
#     builds the strict loader (PR-4a) and the CUDA-free JSON -> native
#     parameter mapping + readback (PR-4b) with their tests;
#   * from the root CMakeLists.txt (add_subdirectory), where saccade_tracking
#     exists: additionally builds `saccade_shipping_native`, which constructs
#     the GPU tracker/GMC/pipeline from the resolved config and fails closed on
#     readback, and its GPU test (ENABLE_NATIVE_TESTS).
#
# Nothing here reads the process environment (tests/unit/
# test_shipping_config_loader_sources.py) or includes tracking/legacy_env.hpp.
cmake_minimum_required(VERSION 3.18)
project(saccade_shipping LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(SACCADE_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
set(SACCADE_RESOLVED_CONFIG ${SACCADE_REPO_ROOT}/configs/shipping/mamba_whole_graph.resolved.json)

add_library(saccade_shipping_config STATIC
    src/strict_json.cpp
    src/resolved_config.cpp
)
target_include_directories(saccade_shipping_config PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(saccade_shipping_config PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(saccade_shipping_config PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The tracker's CUDA-free parameter store. The root build defines this target
# before saccade_tracking; standalone, it is built here from the same source.
if(NOT TARGET saccade_tracker_params)
    add_library(saccade_tracker_params STATIC ${SACCADE_REPO_ROOT}/src/tracking/tracker_params.cpp)
    target_include_directories(saccade_tracker_params PUBLIC ${SACCADE_REPO_ROOT}/include)
    target_compile_options(saccade_tracker_params PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(saccade_tracker_params PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# JSON -> native parameter state and readback. Includes only the CUDA-free
# tracking headers (tracker_params.hpp, perception_params.hpp).
add_library(saccade_shipping_native_config STATIC src/native_config.cpp)
target_link_libraries(saccade_shipping_native_config PUBLIC saccade_shipping_config saccade_tracker_params)
target_compile_options(saccade_shipping_native_config PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(saccade_shipping_native_config PROPERTIES POSITION_INDEPENDENT_CODE ON)

enable_testing()
add_executable(saccade_resolved_config_test ${SACCADE_REPO_ROOT}/tests/native/test_resolved_config.cpp)
target_link_libraries(saccade_resolved_config_test PRIVATE saccade_shipping_config)
target_compile_options(saccade_resolved_config_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME saccade_resolved_config_test
    COMMAND saccade_resolved_config_test ${SACCADE_RESOLVED_CONFIG})

add_executable(saccade_shipping_native_config_test
    ${SACCADE_REPO_ROOT}/tests/native/test_shipping_native_config.cpp)
target_link_libraries(saccade_shipping_native_config_test PRIVATE saccade_shipping_native_config)
target_compile_options(saccade_shipping_native_config_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME saccade_shipping_native_config_test
    COMMAND saccade_shipping_native_config_test ${SACCADE_RESOLVED_CONFIG})

if(TARGET saccade_tracking)
    add_library(saccade_shipping_native STATIC src/native_build.cpp)
    target_link_libraries(saccade_shipping_native PUBLIC saccade_shipping_native_config saccade_tracking)
    target_include_directories(saccade_shipping_native PUBLIC ${SACCADE_REPO_ROOT}/include)
    set_target_properties(saccade_shipping_native PROPERTIES POSITION_INDEPENDENT_CODE ON)

    if(ENABLE_NATIVE_TESTS)
        # Needs a CUDA device: builds the real objects.
        add_executable(saccade_shipping_native_build_test
            ${SACCADE_REPO_ROOT}/tests/native/test_shipping_native_build.cpp)
        target_link_libraries(saccade_shipping_native_build_test PRIVATE
            saccade_shipping_native CUDA::cudart)
        add_test(NAME saccade_shipping_native_build_test
            COMMAND saccade_shipping_native_build_test ${SACCADE_RESOLVED_CONFIG})
    endif()
endif()
