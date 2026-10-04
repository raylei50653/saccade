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
#     readback, `saccade_shipping_ingest` (nvJPEG decode + normalize, PR-7),
#     the developer tools, and their GPU tests (ENABLE_NATIVE_TESTS).
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

# MOT ids, lines and sequence-tail interpolation (PR-6). No config, no CUDA.
# fp-contract off: interpolation rounds after the multiply and after the add,
# as numpy does (mot_output.cpp).
add_library(saccade_shipping_mot STATIC src/mot_output.cpp)
target_include_directories(saccade_shipping_mot PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(saccade_shipping_mot PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
set_target_properties(saccade_shipping_mot PROPERTIES POSITION_INDEPENDENT_CODE ON)

# JSON -> native parameter state and readback. Includes only the CUDA-free
# tracking headers (tracker_params.hpp, perception_params.hpp).
add_library(saccade_shipping_native_config STATIC
    src/native_config.cpp src/post_detector_plan.cpp src/sequence_output.cpp src/ingest_plan.cpp
    src/sha256.cpp src/detector_plan.cpp src/schedule_plan.cpp)
target_link_libraries(saccade_shipping_native_config PUBLIC
    saccade_shipping_config saccade_tracker_params saccade_shipping_mot)
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

# Post-detector host plan + CPU detection filters vs the Python oracle (PR-5).
add_executable(saccade_shipping_detection_filters_test
    ${SACCADE_REPO_ROOT}/tests/native/test_shipping_detection_filters.cpp)
target_link_libraries(saccade_shipping_detection_filters_test PRIVATE saccade_shipping_native_config)
target_compile_options(saccade_shipping_detection_filters_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME saccade_shipping_detection_filters_test
    COMMAND saccade_shipping_detection_filters_test ${SACCADE_RESOLVED_CONFIG}
            ${SACCADE_REPO_ROOT}/tests/native/fixtures/shipping_detection_filters.json)

# MOT ids / lines / interpolation and the sequence-output plan vs the Python
# oracle (PR-6).
add_executable(saccade_shipping_mot_output_test
    ${SACCADE_REPO_ROOT}/tests/native/test_shipping_mot_output.cpp)
target_link_libraries(saccade_shipping_mot_output_test PRIVATE saccade_shipping_native_config)
target_compile_options(saccade_shipping_mot_output_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME saccade_shipping_mot_output_test
    COMMAND saccade_shipping_mot_output_test ${SACCADE_RESOLVED_CONFIG}
            ${SACCADE_REPO_ROOT}/tests/native/fixtures/shipping_mot_output.json)

# Ingest plan, normalize host twin and sequence input vs the Python oracle (PR-7).
add_executable(saccade_shipping_ingest_plan_test
    ${SACCADE_REPO_ROOT}/tests/native/test_shipping_ingest_plan.cpp)
target_link_libraries(saccade_shipping_ingest_plan_test PRIVATE saccade_shipping_native_config)
target_compile_options(saccade_shipping_ingest_plan_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME saccade_shipping_ingest_plan_test
    COMMAND saccade_shipping_ingest_plan_test ${SACCADE_RESOLVED_CONFIG}
            ${SACCADE_REPO_ROOT}/tests/native/fixtures/shipping_ingest.json)

# Detector plan: resolved config + frozen PR-1L lineage + realization
# attestation, and SHA-256 (PR-8).
add_executable(saccade_shipping_detector_plan_test
    ${SACCADE_REPO_ROOT}/tests/native/test_shipping_detector_plan.cpp)
target_link_libraries(saccade_shipping_detector_plan_test PRIVATE saccade_shipping_native_config)
target_compile_options(saccade_shipping_detector_plan_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME saccade_shipping_detector_plan_test
    COMMAND saccade_shipping_detector_plan_test ${SACCADE_RESOLVED_CONFIG}
            ${SACCADE_REPO_ROOT}/tests/native/fixtures/shipping_head_lineage.json
            ${SACCADE_REPO_ROOT}/configs/shipping/mamba_head_realization.attestation.json)

if(TARGET saccade_tracking)
    add_library(saccade_shipping_native STATIC src/native_build.cpp src/post_detector_host.cpp)
    target_link_libraries(saccade_shipping_native PUBLIC saccade_shipping_native_config saccade_tracking)
    target_include_directories(saccade_shipping_native PUBLIC ${SACCADE_REPO_ROOT}/include)
    set_target_properties(saccade_shipping_native PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # Developer tool (#465 PR-5): replay a post-detector dump through the host.
    add_executable(saccade_replay tools/saccade_replay.cpp)
    target_link_libraries(saccade_replay PRIVATE saccade_shipping_native CUDA::cudart)

    # nvJPEG for the native ingest (PR-7): the oracle's own decoder build. The
    # oracle decodes with torchvision's bundled libnvjpeg, which is the
    # nvidia-nvjpeg 13.0.1.86 wheel's library byte for byte; the wheel also
    # carries the matching header. It is fetched pinned (like the TensorRT
    # headers above) rather than added to the `native-build` extra, which is
    # the build toolchain only (tests/contract/test_package_native_delivery.py);
    # whether shipping bundles nvJPEG is a Phase C decision. Fail closed unless
    # the library is still torchvision's: a different nvJPEG build is a
    # different decoder, and the ingest parity would have to be re-measured.
    FetchContent_Declare(
        saccade_nvjpeg_wheel
        URL https://files.pythonhosted.org/packages/b6/e6/9b96fcead7f78a5e2d5d9e726f0d0b1517faf2895146d571e006793180d7/nvidia_nvjpeg-13.0.1.86-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
        URL_HASH SHA256=f658a05c48702219b3db52e194c2ea1ff0ccd3fbf2b75127b4a573074cc22fd5
        DOWNLOAD_NAME nvidia_nvjpeg-13.0.1.86.zip
        DOWNLOAD_EXTRACT_TIMESTAMP NEW
    )
    FetchContent_Populate(saccade_nvjpeg_wheel)
    set(SACCADE_NVJPEG_LIBRARY "${saccade_nvjpeg_wheel_SOURCE_DIR}/nvidia/cu13/lib/libnvjpeg.so.13")
    file(GLOB _saccade_oracle_nvjpeg "${_saccade_purelib}/torchvision.libs/libnvjpeg*.so*")
    list(LENGTH _saccade_oracle_nvjpeg _saccade_oracle_nvjpeg_n)
    if(NOT _saccade_oracle_nvjpeg_n EQUAL 1)
        message(FATAL_ERROR "Expected exactly one bundled libnvjpeg under ${_saccade_purelib}/torchvision.libs (the oracle's decoder), found: ${_saccade_oracle_nvjpeg}")
    endif()
    file(SHA256 "${SACCADE_NVJPEG_LIBRARY}" _saccade_nvjpeg_sha)
    file(SHA256 "${_saccade_oracle_nvjpeg}" _saccade_oracle_nvjpeg_sha)
    if(NOT _saccade_nvjpeg_sha STREQUAL _saccade_oracle_nvjpeg_sha)
        message(FATAL_ERROR "The pinned nvJPEG (${_saccade_nvjpeg_sha}) is not the oracle's decoder ${_saccade_oracle_nvjpeg} (${_saccade_oracle_nvjpeg_sha}); re-pin it and re-run the ingest parity (#465 PR-7)")
    endif()
    add_library(saccade_nvjpeg SHARED IMPORTED)
    set_target_properties(saccade_nvjpeg PROPERTIES
        IMPORTED_LOCATION "${SACCADE_NVJPEG_LIBRARY}"
        IMPORTED_SONAME "libnvjpeg.so.13"
        INTERFACE_INCLUDE_DIRECTORIES "${saccade_nvjpeg_wheel_SOURCE_DIR}/nvidia/cu13/include")

    add_library(saccade_shipping_ingest STATIC src/ingest_host.cpp src/ingest_normalize.cu)
    target_link_libraries(saccade_shipping_ingest PUBLIC
        saccade_shipping_native_config CUDA::cudart saccade_nvjpeg)
    target_include_directories(saccade_shipping_ingest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(saccade_shipping_ingest PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # Developer tool (#465 PR-7): stream the native ingest of one sequence to
    # scripts/eval/diagnostics/native_ingest_parity.py.
    add_executable(saccade_ingest_probe tools/saccade_ingest_probe.cpp)
    target_link_libraries(saccade_ingest_probe PRIVATE saccade_shipping_ingest)

    # Native detector (PR-8): resize -> the existing TRTEngine backbone
    # (saccade_perception) -> the PR-1L LibTorch head -> S2. LibTorch only
    # (TORCH_LIBRARIES has no libtorch_python); the head's scan operator is
    # dlopen'ed at load time after its hash is checked.
    add_library(saccade_shipping_detector STATIC src/detector_host.cpp src/detector_s2.cu)
    target_link_libraries(saccade_shipping_detector PUBLIC
        saccade_shipping_native_config saccade_perception ${TORCH_LIBRARIES} CUDA::cudart
        ${CMAKE_DL_LIBS})
    target_include_directories(saccade_shipping_detector PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include ${SACCADE_REPO_ROOT}/include)
    set_target_properties(saccade_shipping_detector PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # Developer tool (#465 PR-8): stream the native ingest + detector of one
    # sequence to scripts/eval/diagnostics/native_detector_parity.py.
    add_executable(saccade_detector_probe tools/saccade_detector_probe.cpp)
    target_link_libraries(saccade_detector_probe PRIVATE saccade_shipping_detector saccade_shipping_ingest)

    # End-to-end runtimes: ingest -> detector -> post-detector host -> MOT emit
    # and tail, serial and eager (PR-9) and the oracle's double-buffer schedule
    # with CUDA graphs (PR-10); the shipping entrypoint saccade_track (boundary
    # §2) runs the schedule the resolved config names.
    add_library(saccade_shipping_runtime STATIC src/serial_runtime.cpp src/double_buffer_runtime.cpp)
    target_link_libraries(saccade_shipping_runtime PUBLIC
        saccade_shipping_detector saccade_shipping_ingest saccade_shipping_native)
    set_target_properties(saccade_shipping_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_executable(saccade_track tools/saccade_track.cpp)
    target_link_libraries(saccade_track PRIVATE saccade_shipping_runtime)
    # NEEDED = what the entrypoint uses. The toolchain default is
    # --no-as-needed, so otherwise any library that lands before torch's own
    # --as-needed switch on the link line (e.g. libnvinfer_plugin, which nothing
    # here calls) becomes a dependency by link order alone (#465 PR-11).
    target_link_options(saccade_track PRIVATE LINKER:--as-needed)
    # The entrypoint must not need OpenCV or Python (#465 PR-11): a build that
    # links either fails here.
    if(NOT CMAKE_READELF)
        find_program(CMAKE_READELF readelf REQUIRED)
    endif()
    add_custom_command(TARGET saccade_track POST_BUILD
        COMMAND ${CMAKE_COMMAND} -DREADELF=${CMAKE_READELF} -DBINARY=$<TARGET_FILE:saccade_track>
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_link_surface.cmake
        VERBATIM)

    # The shipping tree (#465 PR-12): `cmake --install <build> --component
    # shipping --prefix <tree>` gives
    #   bin/saccade_track         RUNPATH $ORIGIN/../lib only (no build or venv
    #                             path); lib/ is the slot a Phase C bundle
    #                             would fill, empty here
    #   share/saccade/...         the model root (install_model_root.cmake)
    # Third-party runtime libraries (CUDA runtime, nvJPEG, TensorRT, LibTorch)
    # are not in the tree: whether and how they ship is Phase C.
    # Device code: find_package(Torch) turns CMAKE_CUDA_ARCHITECTURES off and
    # compiles every CUDA target for TORCH_CUDA_ARCH_LIST (the local GPU when
    # unset). The shipping tree needs the explicit list below, LibTorch's own
    # (torch 2.11.0+cu130: sm_75/80/86/90/100/120) with PTX for the newest; the
    # install refuses a build configured otherwise. The list covers
    # saccade_track's device code only: the attested operator library stays
    # sm_120 SASS and the backbone engine is built for one SM and TensorRT
    # version (docs/reference/native_runtime_resolved_config.md §16).
    set(SACCADE_SHIPPING_TORCH_CUDA_ARCH_LIST "7.5;8.0;8.6;9.0;10.0;12.0+PTX")
    set_target_properties(saccade_track PROPERTIES
        INSTALL_RPATH "\$ORIGIN/../lib"
        INSTALL_RPATH_USE_LINK_PATH OFF)
    set(SACCADE_ATTESTED_OP_LIBRARY "${SACCADE_REPO_ROOT}/build/libsaccade_scan_torchop.so" CACHE FILEPATH
        "The attested operator library the shipping tree carries (its sha256 must be the realization attestation's)")
    install(CODE "
        if(NOT \"${TORCH_CUDA_ARCH_LIST}\" STREQUAL \"${SACCADE_SHIPPING_TORCH_CUDA_ARCH_LIST}\")
            message(FATAL_ERROR \"shipping tree: this build compiles CUDA for TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}'; configure with -DTORCH_CUDA_ARCH_LIST='${SACCADE_SHIPPING_TORCH_CUDA_ARCH_LIST}' (#465 PR-12)\")
        endif()" COMPONENT shipping)
    install(TARGETS saccade_track RUNTIME DESTINATION bin COMPONENT shipping)
    install(CODE "
        set(SACCADE_REPO_ROOT \"${SACCADE_REPO_ROOT}\")
        set(SACCADE_MODEL_ROOT_DEST \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/share/saccade\")
        set(SACCADE_SHIPPING_CONFIG configs/shipping/mamba_whole_graph.resolved.json)
        set(SACCADE_SHIPPING_LINEAGE models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json)
        set(SACCADE_SHIPPING_ATTESTATION configs/shipping/mamba_head_realization.attestation.json)
        set(SACCADE_ATTESTED_OP_LIBRARY \"${SACCADE_ATTESTED_OP_LIBRARY}\")
        include(\"${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_model_root.cmake\")" COMPONENT shipping)

    if(ENABLE_NATIVE_TESTS)
        # Needs a CUDA device: builds the real objects.
        add_executable(saccade_shipping_native_build_test
            ${SACCADE_REPO_ROOT}/tests/native/test_shipping_native_build.cpp)
        target_link_libraries(saccade_shipping_native_build_test PRIVATE
            saccade_shipping_native CUDA::cudart)
        add_test(NAME saccade_shipping_native_build_test
            COMMAND saccade_shipping_native_build_test ${SACCADE_RESOLVED_CONFIG})

        add_executable(saccade_shipping_post_detector_host_test
            ${SACCADE_REPO_ROOT}/tests/native/test_shipping_post_detector_host.cpp)
        target_link_libraries(saccade_shipping_post_detector_host_test PRIVATE
            saccade_shipping_native CUDA::cudart)
        add_test(NAME saccade_shipping_post_detector_host_test
            COMMAND saccade_shipping_post_detector_host_test ${SACCADE_RESOLVED_CONFIG})

        # Native S2 vs the oracle's compiled S2 (fixture), and the detector
        # loader's hash checks when the model files are present (PR-8).
        add_executable(saccade_shipping_detector_s2_test
            ${SACCADE_REPO_ROOT}/tests/native/test_shipping_detector_s2.cpp)
        target_link_libraries(saccade_shipping_detector_s2_test PRIVATE saccade_shipping_detector)
        add_test(NAME saccade_shipping_detector_s2_test
            COMMAND saccade_shipping_detector_s2_test
                    ${SACCADE_REPO_ROOT}/tests/native/fixtures/shipping_detector_s2.json
                    ${SACCADE_RESOLVED_CONFIG}
                    ${SACCADE_REPO_ROOT}/tests/native/fixtures/shipping_head_lineage.json
                    ${SACCADE_REPO_ROOT}/configs/shipping/mamba_head_realization.attestation.json
                    ${SACCADE_REPO_ROOT})

        # End-to-end serial runtime wiring (PR-9): per-sequence state and the
        # wiring mutations, on MOT17 frames when the models and data exist.
        add_executable(saccade_shipping_serial_runtime_test
            ${SACCADE_REPO_ROOT}/tests/native/test_shipping_serial_runtime.cpp)
        target_link_libraries(saccade_shipping_serial_runtime_test PRIVATE saccade_shipping_runtime)
        add_test(NAME saccade_shipping_serial_runtime_test
            COMMAND saccade_shipping_serial_runtime_test ${SACCADE_RESOLVED_CONFIG}
                    ${SACCADE_REPO_ROOT}/models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json
                    ${SACCADE_REPO_ROOT}/configs/shipping/mamba_head_realization.attestation.json
                    ${SACCADE_REPO_ROOT}
                    ${SACCADE_REPO_ROOT}/datasets/MOT17/train)

        # Double-buffer runtime with CUDA graphs (PR-10): serial-equal output,
        # the graph capture keys and counts, and the schedule negative controls.
        add_executable(saccade_shipping_double_buffer_runtime_test
            ${SACCADE_REPO_ROOT}/tests/native/test_shipping_double_buffer_runtime.cpp)
        target_link_libraries(saccade_shipping_double_buffer_runtime_test PRIVATE saccade_shipping_runtime)
        add_test(NAME saccade_shipping_double_buffer_runtime_test
            COMMAND saccade_shipping_double_buffer_runtime_test ${SACCADE_RESOLVED_CONFIG}
                    ${SACCADE_REPO_ROOT}/models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json
                    ${SACCADE_REPO_ROOT}/configs/shipping/mamba_head_realization.attestation.json
                    ${SACCADE_REPO_ROOT}
                    ${SACCADE_REPO_ROOT}/datasets/MOT17/train)

        add_executable(saccade_shipping_ingest_host_test
            ${SACCADE_REPO_ROOT}/tests/native/test_shipping_ingest_host.cpp)
        target_link_libraries(saccade_shipping_ingest_host_test PRIVATE saccade_shipping_ingest)
        add_test(NAME saccade_shipping_ingest_host_test
            COMMAND saccade_shipping_ingest_host_test ${SACCADE_RESOLVED_CONFIG}
                    ${SACCADE_REPO_ROOT}/tests/native/fixtures/shipping_ingest.json
                    ${SACCADE_REPO_ROOT}/tests/native/fixtures/shipping_ingest)
    endif()
endif()
