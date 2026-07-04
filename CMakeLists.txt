cmake_minimum_required(VERSION 3.18)
project(Saccade CXX CUDA)

# Suppress deprecation warning for FetchContent_Populate (we only need headers, not build TensorRT targets)
cmake_policy(SET CMP0169 OLD)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

option(ENABLE_NATIVE_TESTS "Build native C++ tests" ON)
option(ENABLE_NATIVE_COVERAGE "Enable coverage instrumentation for native C++ test targets" OFF)

# 1. 全局編譯配置
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fvisibility=hidden -O3 -Wall -fno-lto")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fPIC,-fno-lto")

# 設置 RPATH 以支持動態連結
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

function(enable_native_coverage target_name)
    if(ENABLE_NATIVE_COVERAGE)
        target_compile_options(${target_name} PRIVATE -O0 -g --coverage)
        target_link_options(${target_name} PRIVATE --coverage)
    endif()
endfunction()

# 2. 依賴搜尋
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)

# 3. Python 基礎環境搜尋 (用於獲取 Pybind11 與 TensorRT 庫路徑)
if(DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH "Python 3 executable" FORCE)
endif()

if(NOT DEFINED Python3_EXECUTABLE)
    set(_repo_venv_python "${PROJECT_SOURCE_DIR}/.venv/bin/python3")
    if(EXISTS "${_repo_venv_python}")
        set(Python3_EXECUTABLE "${_repo_venv_python}" CACHE FILEPATH "Python 3 executable" FORCE)
    endif()
endif()

if(NOT DEFINED Python3_EXECUTABLE AND DEFINED ENV{VIRTUAL_ENV})
    set(_venv_python "$ENV{VIRTUAL_ENV}/bin/python")
    if(EXISTS "${_venv_python}")
        set(Python3_EXECUTABLE "${_venv_python}" CACHE FILEPATH "Python 3 executable" FORCE)
    endif()
endif()

find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Interpreter Development.Module)

# ─── Find LibTorch ───
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_DIR}")
find_package(Torch REQUIRED)

# Find PyTorch Python bindings library (torch_python) for pybind11 at::Tensor type-casting
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_library(TORCH_PYTHON_LIBRARY
    NAMES torch_python libtorch_python.so
    PATHS ${TORCH_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED
)

# LibTorch/Caffe2 injects -flto=auto into CMAKE_CXX_FLAGS which causes LTO
# version mismatches when linking pybind11 modules. Strip it out immediately.
string(REPLACE "-flto=auto" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-fno-fat-lto-objects" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "CXX compiler flags" FORCE)


# 4. TensorRT 依賴處理 (方案 B: 自動下載頭文件 + 使用 Python 庫)
include(FetchContent)
FetchContent_Declare(
    tensorrt_oss
    URL https://github.com/NVIDIA/TensorRT/archive/refs/tags/v10.16.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP NEW
)
FetchContent_Populate(tensorrt_oss)
set(TENSORRT_INCLUDE_DIR "${tensorrt_oss_SOURCE_DIR}/include")

# Eigen 3.4 (header-only linear algebra)
FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP NEW
)
FetchContent_Populate(eigen)
set(EIGEN_INCLUDE_DIR "${eigen_SOURCE_DIR}")

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import tensorrt_libs; print(tensorrt_libs.__path__[0])"
    OUTPUT_VARIABLE TENSORRT_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_library(TENSORRT_NVINFER_LIBRARY
    NAMES nvinfer libnvinfer.so.10
    PATHS ${TENSORRT_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED
)
find_library(TENSORRT_NVINFER_PLUGIN_LIBRARY
    NAMES nvinfer_plugin libnvinfer_plugin.so.10
    PATHS ${TENSORRT_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED
)

# 5. 全局包含路徑
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
# TensorRT headers as SYSTEM to suppress third-party deprecation warnings
include_directories(SYSTEM ${TENSORRT_INCLUDE_DIR})
# Eigen headers as SYSTEM
include_directories(SYSTEM ${EIGEN_INCLUDE_DIR})

# 6. 定義 Perception 模組 (Dynamic Library)
add_library(saccade_perception STATIC
    src/perception/trt_engine.cpp
    src/perception/preprocessor.cpp
    src/perception/preprocessor_gpu.cu
    src/perception/letterbox_kernel.cu
    src/perception/feature_extractor.cpp
    src/perception/mamba_gated_detector.cpp
    src/perception/batched_mamba_detector.cpp
)
# 設定導出宏
target_compile_definitions(saccade_perception PRIVATE SACCADE_EXPORT_PERCEPTION)

# 7. 定義 Tracking 模組 (Dynamic Library)
add_library(saccade_tracking STATIC
    src/tracking/tracker_gpu.cu
    src/tracking/relink_gate.cu
    src/tracking/gmc.cpp
    src/tracking/gmc_kernel.cu
    src/tracking/pipeline.cpp
    src/tracking/crop_pool.cpp
    src/tracking/frame_buffer.cpp
    src/tracking/reid_queue.cpp
    src/tracking/workbench.cpp
    src/tracking/quality_filter.cu
    src/tracking/copy_pad.cu
    src/tracking/mamba_scan.cu
    src/tracking/temporal_birth_pool.cu
    src/tracking/multi_signal_birth_pool.cu
    src/tracking/seq_runner.cpp
    src/tracking/eval_pool.cpp
    src/tracking/cheb_gr_online.cpp
    src/tracking/cheb_gr_kreciprocal.cpp
    src/tracking/dynamic_reid_controller.cpp
)
set_target_properties(saccade_tracking PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_link_libraries(saccade_tracking
    CUDA::cudart
    CUDA::cufft
    ${OpenCV_LIBS}
    saccade_perception
)
target_compile_definitions(saccade_tracking PRIVATE SACCADE_EXPORT_TRACKING)

# 8. 定義 Media 模組 (Dynamic Library)
add_library(saccade_media STATIC
    src/media/gst_client.cpp
    src/media/buffer_pool.cpp
)
target_link_libraries(saccade_media
    ${GSTREAMER_LIBRARIES}
    CUDA::cudart
    saccade_perception
)
target_compile_definitions(saccade_media PRIVATE SACCADE_EXPORT_MEDIA)

# 9. Python Bindings
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
set(Python_EXECUTABLE "${Python3_EXECUTABLE}")
set(Python_ROOT_DIR "${PROJECT_SOURCE_DIR}/.venv")
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})

set(SACCADE_BUILD_RPATH
    "${PROJECT_BINARY_DIR}"
    "${TENSORRT_LIB_DIR}"
)
set(SACCADE_INSTALL_RPATH
    "${CMAKE_INSTALL_PREFIX}/lib"
    "${TENSORRT_LIB_DIR}"
)

target_link_libraries(saccade_perception
    ${TENSORRT_NVINFER_LIBRARY}
    ${TENSORRT_NVINFER_PLUGIN_LIBRARY}
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cufft
    ${OpenCV_LIBS}
)
set_target_properties(
    saccade_perception
    saccade_tracking
    saccade_media
    PROPERTIES
        BUILD_RPATH "${SACCADE_BUILD_RPATH}"
        INSTALL_RPATH "${SACCADE_INSTALL_RPATH}"
)

# 7b. SelectiveScan TensorRT plugin — shared library loaded at runtime via
#     IPluginRegistry::loadLibrary() (see scripts/model/build_mamba_head_trt.py).
#     Built as a MODULE so it exports initLibNvInferPlugins/getCreators.
#     mamba_scan.cu is compiled in directly (hidden visibility) so the .so is
#     self-contained and does not interpose symbols on saccade_tracking_ext.
add_library(saccade_scan_plugin MODULE
    src/tracking/mamba_scan_plugin.cpp
    src/tracking/mamba_scan.cu
)
target_include_directories(saccade_scan_plugin PRIVATE ${TENSORRT_INCLUDE_DIR})
# TENSORRT_BUILD_LIB makes NvInfer's TENSORRTAPI macro expand to
# visibility("default") so initLibNvInferPlugins / getCreators /
# setLoggerFinder are exported from the .so (otherwise the global
# -fvisibility=hidden hides them and TRT loadLibrary() cannot find them).
target_compile_definitions(saccade_scan_plugin PRIVATE TENSORRT_BUILD_LIB)
target_link_libraries(saccade_scan_plugin PRIVATE
    ${TENSORRT_NVINFER_LIBRARY}
    CUDA::cudart
)
set_target_properties(saccade_scan_plugin PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "saccade_scan_plugin"
    CUDA_SEPARABLE_COMPILATION OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    BUILD_RPATH "${SACCADE_BUILD_RPATH}"
    INSTALL_RPATH "${SACCADE_INSTALL_RPATH}"
)

pybind11_add_module(saccade_media_ext NO_EXTRAS src/media/gst_client_python.cpp)
target_link_libraries(saccade_media_ext PRIVATE 
    saccade_media
    ${GSTREAMER_LIBRARIES}
    CUDA::cudart
)
target_compile_options(saccade_media_ext PRIVATE -fno-lto)


pybind11_add_module(saccade_tracking_ext NO_EXTRAS src/tracking/tracker_gpu_python.cpp)
target_link_libraries(saccade_tracking_ext PRIVATE
    saccade_tracking
    CUDA::cudart
)
target_compile_options(saccade_tracking_ext PRIVATE -fno-lto)

pybind11_add_module(saccade_eval_ext NO_EXTRAS src/tracking/eval_python.cpp)
target_link_libraries(saccade_eval_ext PRIVATE
    saccade_tracking
    saccade_perception
    CUDA::cudart
)
target_compile_options(saccade_eval_ext PRIVATE -fno-lto)

pybind11_add_module(saccade_cheb_gr_online_ext NO_EXTRAS src/tracking/cheb_gr_online_python.cpp)
target_link_libraries(saccade_cheb_gr_online_ext PRIVATE
    saccade_tracking
    CUDA::cudart
)
target_compile_options(saccade_cheb_gr_online_ext PRIVATE -fno-lto)

pybind11_add_module(saccade_perception_ext NO_EXTRAS src/perception/perception_python.cpp)
target_link_libraries(saccade_perception_ext PRIVATE
    saccade_perception
    CUDA::cudart
    ${TORCH_PYTHON_LIBRARY}
)
target_compile_options(saccade_perception_ext PRIVATE -fno-lto)
set_target_properties(
    saccade_media_ext
    saccade_tracking_ext
    saccade_eval_ext
    saccade_perception_ext
    saccade_cheb_gr_online_ext
    PROPERTIES
        BUILD_RPATH "${SACCADE_BUILD_RPATH}"
        INSTALL_RPATH "${SACCADE_INSTALL_RPATH}"
        CXX_VISIBILITY_PRESET "default"
        VISIBILITY_INLINES_HIDDEN OFF
)

# 7. 主程式
add_executable(saccade_node src/main.cpp)
target_link_libraries(saccade_node 
    saccade_perception 
    saccade_tracking
    saccade_media
    ${GSTREAMER_LIBRARIES}
)
set_target_properties(
    saccade_node
    PROPERTIES
        BUILD_RPATH "${SACCADE_BUILD_RPATH}"
        INSTALL_RPATH "${SACCADE_INSTALL_RPATH}"
)

if(ENABLE_NATIVE_TESTS)
    enable_testing()

    add_executable(saccade_assignment_algorithms_test
        tests/native/test_assignment_algorithms.cpp
    )
    target_include_directories(saccade_assignment_algorithms_test PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )
    enable_native_coverage(saccade_assignment_algorithms_test)
    add_test(NAME saccade_assignment_algorithms_test COMMAND saccade_assignment_algorithms_test)

    add_executable(saccade_gpu_postprocess_test
        tests/native/test_gpu_postprocess.cpp
        src/tracking/tracker_gpu.cu
    )
    target_include_directories(saccade_gpu_postprocess_test PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )
    target_link_libraries(saccade_gpu_postprocess_test PRIVATE
        CUDA::cudart
    )
    set_target_properties(saccade_gpu_postprocess_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME saccade_gpu_postprocess_test COMMAND saccade_gpu_postprocess_test)

    # ── tracking unit tests (link against saccade_tracking static lib) ──────
    # These cover the OOB clamp, mamba_scan fwd + broadcast_C, fpn_reid
    # launchers, and the GMC Hanning divide-by-zero guard.
    function(add_tracking_native_test name source)
        add_executable(${name} ${source})
        target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
        target_link_libraries(${name} PRIVATE saccade_tracking CUDA::cudart)
        enable_native_coverage(${name})
        set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    add_tracking_native_test(saccade_temporal_birth_pool_test
        tests/native/test_temporal_birth_pool.cpp)
    add_tracking_native_test(saccade_mamba_scan_test
        tests/native/test_mamba_scan.cpp)
    add_tracking_native_test(saccade_gmc_hanning_guard_test
        tests/native/test_gmc_hanning_guard.cpp)
    add_tracking_native_test(saccade_crop_pool_test
        tests/native/test_crop_pool.cpp)

    # fpn_reid launchers live in fpn_reid_cuda.cu, which is NOT part of
    # saccade_tracking (it ships as a standalone torch extension). Compile
    # the .cu directly into the test, mirroring saccade_gpu_postprocess_test.
    add_executable(saccade_fpn_reid_launchers_test
        tests/native/test_fpn_reid_launchers.cpp
        src/tracking/fpn_reid_cuda.cu
    )
    target_include_directories(saccade_fpn_reid_launchers_test PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )
    target_link_libraries(saccade_fpn_reid_launchers_test PRIVATE CUDA::cudart)
    set_target_properties(saccade_fpn_reid_launchers_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    enable_native_coverage(saccade_fpn_reid_launchers_test)
    add_test(NAME saccade_fpn_reid_launchers_test COMMAND saccade_fpn_reid_launchers_test)

    add_tracking_native_test(saccade_cheb_gr_online_test
        tests/native/test_cheb_gr_online.cpp)

    # ── media buffer-pool race tests (S1-S9 regression) ─────────────────────
    # BufferPool is GStreamer-independent; the test links saccade_media
    # (which contains buffer_pool.cpp) + Threads for the concurrent-acquire
    # scenarios. No GStreamer runtime needed.
    add_executable(saccade_gst_buffer_pool_test
        tests/native/test_gst_buffer_pool.cpp
    )
    target_include_directories(saccade_gst_buffer_pool_test PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )
    target_link_libraries(saccade_gst_buffer_pool_test PRIVATE
        saccade_media
        CUDA::cudart
        Threads::Threads
    )
    enable_native_coverage(saccade_gst_buffer_pool_test)
    add_test(NAME saccade_gst_buffer_pool_test COMMAND saccade_gst_buffer_pool_test)
endif()
