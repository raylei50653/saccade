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
    # Measurement boundary (#465 Phase C PR-C2). Each runtime library below is
    # built twice from the same sources: <name>, the shipping objects, and
    # <name>_measurement, compiled with SACCADE_SHIPPING_MEASUREMENT_HOOKS
    # (PUBLIC, so every consumer sees the same declarations), which adds the
    # negative-control mutations and the other measurement setters
    # (*_for_measurement). The shipping objects do not contain them: a shipping
    # caller cannot name one (it is not declared) and the shipping entrypoint
    # carries no mutation code (check_shipping_bundle.py static,
    # entrypoint_no_measurement_surface). Only developer tools and tests link a
    # _measurement library; none of them is installed, and no target links
    # both variants.
    function(saccade_shipping_variants name)
        add_library(${name} STATIC ${ARGN})
        add_library(${name}_measurement STATIC ${ARGN})
        target_compile_definitions(${name}_measurement PUBLIC SACCADE_SHIPPING_MEASUREMENT_HOOKS=1)
        set_target_properties(${name} ${name}_measurement PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endfunction()

    saccade_shipping_variants(saccade_shipping_native src/native_build.cpp src/post_detector_host.cpp)
    foreach(_v "" "_measurement")
        target_link_libraries(saccade_shipping_native${_v} PUBLIC saccade_shipping_native_config saccade_tracking)
        target_include_directories(saccade_shipping_native${_v} PUBLIC ${SACCADE_REPO_ROOT}/include)
    endforeach()

    # Developer tool (#465 PR-5): replay a post-detector dump through the host.
    add_executable(saccade_replay tools/saccade_replay.cpp)
    target_link_libraries(saccade_replay PRIVATE saccade_shipping_native_measurement CUDA::cudart)

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

    saccade_shipping_variants(saccade_shipping_ingest src/ingest_host.cpp src/ingest_normalize.cu)
    foreach(_v "" "_measurement")
        target_link_libraries(saccade_shipping_ingest${_v} PUBLIC
            saccade_shipping_native_config CUDA::cudart saccade_nvjpeg)
        target_include_directories(saccade_shipping_ingest${_v} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    endforeach()

    # Developer tool (#465 PR-7): stream the native ingest of one sequence to
    # scripts/eval/diagnostics/native_ingest_parity.py.
    add_executable(saccade_ingest_probe tools/saccade_ingest_probe.cpp)
    target_link_libraries(saccade_ingest_probe PRIVATE saccade_shipping_ingest_measurement)

    # Native detector (PR-8): resize -> the existing TRTEngine backbone
    # (saccade_perception) -> the PR-1L LibTorch head -> S2. LibTorch only
    # (TORCH_LIBRARIES has no libtorch_python); the head's scan operator is
    # dlopen'ed at load time after its hash is checked.
    saccade_shipping_variants(saccade_shipping_detector src/detector_host.cpp src/detector_s2.cu)
    foreach(_v "" "_measurement")
        target_link_libraries(saccade_shipping_detector${_v} PUBLIC
            saccade_shipping_native_config saccade_perception ${TORCH_LIBRARIES} CUDA::cudart
            ${CMAKE_DL_LIBS})
        target_include_directories(saccade_shipping_detector${_v} PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include ${SACCADE_REPO_ROOT}/include)
    endforeach()

    # Developer tool (#465 PR-8): stream the native ingest + detector of one
    # sequence to scripts/eval/diagnostics/native_detector_parity.py.
    add_executable(saccade_detector_probe tools/saccade_detector_probe.cpp)
    target_link_libraries(saccade_detector_probe PRIVATE
        saccade_shipping_detector_measurement saccade_shipping_ingest_measurement)

    # End-to-end runtimes: ingest -> detector -> post-detector host -> MOT emit
    # and tail, serial and eager (PR-9) and the oracle's double-buffer schedule
    # with CUDA graphs (PR-10); the shipping entrypoint saccade_track (boundary
    # §2) runs the schedule the resolved config names.
    saccade_shipping_variants(saccade_shipping_runtime src/serial_runtime.cpp src/double_buffer_runtime.cpp)
    foreach(_v "" "_measurement")
        target_link_libraries(saccade_shipping_runtime${_v} PUBLIC
            saccade_shipping_detector${_v} saccade_shipping_ingest${_v} saccade_shipping_native${_v})
    endforeach()

    # Developer build of the entrypoint (PR-C2): saccade_track's interface plus
    # the negative controls, the serial override and --max-frames, on the
    # measurement variant (tools/saccade_track_measurement.cpp). Not installed.
    add_executable(saccade_track_measurement tools/saccade_track_measurement.cpp)
    target_link_libraries(saccade_track_measurement PRIVATE saccade_shipping_runtime_measurement)
    target_link_options(saccade_track_measurement PRIVATE LINKER:--as-needed)

    add_executable(saccade_track tools/saccade_track.cpp)
    target_link_libraries(saccade_track PRIVATE saccade_shipping_runtime)
    # NEEDED = what the entrypoint uses. The toolchain default is
    # --no-as-needed, so otherwise any library that lands before torch's own
    # --as-needed switch on the link line (e.g. libnvinfer_plugin, which nothing
    # here calls) becomes a dependency by link order alone (#465 PR-11).
    target_link_options(saccade_track PRIVATE LINKER:--as-needed)
    # The entrypoint must not need OpenCV or Python (#465 PR-11) and must carry
    # no measurement hook or developer option (PR-C2): a build that links
    # either, or a _measurement library, fails here.
    if(NOT CMAKE_READELF)
        find_program(CMAKE_READELF readelf REQUIRED)
    endif()
    add_custom_command(TARGET saccade_track POST_BUILD
        COMMAND ${CMAKE_COMMAND} -DREADELF=${CMAKE_READELF} -DBINARY=$<TARGET_FILE:saccade_track>
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_link_surface.cmake
        COMMAND ${CMAKE_COMMAND} -DBINARY=$<TARGET_FILE:saccade_track>
                -DSURFACE=${CMAKE_CURRENT_SOURCE_DIR}/measurement_surface.json
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_no_measurement_surface.cmake
        VERBATIM)

    # The shipping tree (#465 PR-12, bundled in Phase C PR-C1): `cmake --install
    # <build> --component shipping --prefix <tree>` gives
    #   bin/saccade_track         the launcher (launcher/saccade_track.sh): runs
    #                             the entrypoint through the system loader with
    #                             --library-path lib/vendor and --audit
    #   libexec/saccade_track     the entrypoint, RUNPATH $ORIGIN/../lib only (no
    #                             build or venv path); the pinned bytes
    #                             (entrypoint_pin.json; PR-C2: no measurement
    #                             hook or developer option)
    #   lib/saccade_loader_audit.so  the loader provenance check (src/loader_audit.c)
    #   lib/vendor/<SONAME>       the bundled third-party set, pinned by sha256
    #                             in third_party_set.json (owner decision C-D1)
    #   licenses/                 each wheel's license files, THIRD_PARTY.md,
    #                             Saccade's LICENSE / NOTICE
    #   share/saccade/...         the model root (install_model_root.cmake)
    # docs/reference/native_runtime_resolved_config.md §16 (tree), §17 (bundle),
    # §18 (CLI surface).
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
    # The entrypoint bytes (PR-C1): a rebuild of the same sources is not
    # byte-identical (entrypoint_pin.json), so the package carries the pinned
    # binary (PR-12's until PR-C2, then PR-C2's) named by
    # SACCADE_SHIPPING_ENTRYPOINT, hash-checked at
    # install. Unset, the install takes this build's saccade_track (a
    # development tree; the bundle check reports that it is not the pin).
    set(SACCADE_SHIPPING_ENTRYPOINT "" CACHE FILEPATH
        "The pinned saccade_track the shipping tree carries (sha256 = shipping/entrypoint_pin.json); empty = this build's")
    if(SACCADE_SHIPPING_ENTRYPOINT)
        install(CODE "
            file(READ \"${CMAKE_CURRENT_SOURCE_DIR}/entrypoint_pin.json\" _pin)
            string(JSON _want GET \"\${_pin}\" sha256)
            if(NOT EXISTS \"${SACCADE_SHIPPING_ENTRYPOINT}\")
                message(FATAL_ERROR \"shipping tree: ${SACCADE_SHIPPING_ENTRYPOINT} does not exist\")
            endif()
            file(SHA256 \"${SACCADE_SHIPPING_ENTRYPOINT}\" _got)
            if(NOT _got STREQUAL _want)
                message(FATAL_ERROR \"shipping tree: ${SACCADE_SHIPPING_ENTRYPOINT} sha256 \${_got} != \${_want} (shipping/entrypoint_pin.json)\")
            endif()
            set(_dst \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/libexec/saccade_track\")
            file(MAKE_DIRECTORY \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/libexec\")
            message(STATUS \"Installing: \${_dst} (pinned, sha256 \${_want})\")
            file(COPY_FILE \"${SACCADE_SHIPPING_ENTRYPOINT}\" \"\${_dst}\")
            file(CHMOD \"\${_dst}\" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)" COMPONENT shipping)
    else()
        install(TARGETS saccade_track RUNTIME DESTINATION libexec COMPONENT shipping)
    endif()
    install(PROGRAMS launcher/saccade_track.sh DESTINATION bin RENAME saccade_track COMPONENT shipping)

    # The loader provenance check (PR-C1): C and libc only (it runs in the
    # loader's audit namespace). Its bundled-name list comes from the pinned set.
    enable_language(C)
    set(SACCADE_THIRD_PARTY_SET ${CMAKE_CURRENT_SOURCE_DIR}/third_party_set.json)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${SACCADE_THIRD_PARTY_SET})
    file(READ ${SACCADE_THIRD_PARTY_SET} _saccade_tp)
    string(JSON _saccade_tp_n LENGTH "${_saccade_tp}" entries)
    math(EXPR _saccade_tp_last "${_saccade_tp_n} - 1")
    set(_saccade_tp_names "")
    foreach(_i RANGE ${_saccade_tp_last})
        string(JSON _saccade_tp_soname GET "${_saccade_tp}" entries ${_i} soname)
        string(APPEND _saccade_tp_names "    \"${_saccade_tp_soname}\",\n")
    endforeach()
    file(CONFIGURE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated/loader_audit_names.inc
        CONTENT "/* Generated from shipping/third_party_set.json; do not edit. */\nstatic const char *const kBundled[] = {\n${_saccade_tp_names}    NULL};\n")
    add_library(saccade_loader_audit SHARED src/loader_audit.c)
    target_include_directories(saccade_loader_audit PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
    target_compile_options(saccade_loader_audit PRIVATE -Wall -Wextra -Werror)
    set_target_properties(saccade_loader_audit PROPERTIES
        PREFIX "" C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
        INSTALL_RPATH "" INSTALL_RPATH_USE_LINK_PATH OFF)
    add_dependencies(saccade_track saccade_loader_audit)
    install(TARGETS saccade_loader_audit LIBRARY DESTINATION lib COMPONENT shipping)
    install(CODE "
        set(SACCADE_REPO_ROOT \"${SACCADE_REPO_ROOT}\")
        set(SACCADE_PREFIX_DEST \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}\")
        set(SACCADE_THIRD_PARTY_SET \"${SACCADE_THIRD_PARTY_SET}\")
        set(SACCADE_ROOT_purelib \"${_saccade_purelib}\")
        set(SACCADE_ROOT_nvjpeg_wheel \"${saccade_nvjpeg_wheel_SOURCE_DIR}\")
        include(\"${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_third_party.cmake\")" COMPONENT shipping)
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
            saccade_shipping_native_measurement CUDA::cudart)
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
        target_link_libraries(saccade_shipping_serial_runtime_test PRIVATE saccade_shipping_runtime_measurement)
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
        target_link_libraries(saccade_shipping_double_buffer_runtime_test PRIVATE saccade_shipping_runtime_measurement)
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
