cmake_minimum_required(VERSION 3.18)
project(Saccade CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(ENABLE_NATIVE_TESTS "Build native C++ tests" ON)
option(ENABLE_NATIVE_COVERAGE "Enable coverage instrumentation for native C++ test targets" OFF)

# 1. 全局編譯配置
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fvisibility=hidden -O3 -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fPIC")

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
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# 3. 定義 Perception 模組 (Dynamic Library)
add_library(saccade_perception SHARED
    src/perception/trt_engine.cpp
    src/perception/preprocessor.cpp
    src/perception/preprocessor_gpu.cu
    src/perception/feature_extractor.cpp
)
# 設定導出宏
target_compile_definitions(saccade_perception PRIVATE SACCADE_EXPORT_PERCEPTION)

# 4. 定義 Tracking 模組 (Dynamic Library)
add_library(saccade_tracking SHARED
    src/tracking/tracker_gpu.cu
    src/tracking/gmc.cpp
    src/tracking/gmc_kernel.cu
    src/tracking/pipeline.cpp
)
set_target_properties(saccade_tracking PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(saccade_tracking
    CUDA::cudart
    ${OpenCV_LIBS}
    saccade_perception
)
target_compile_definitions(saccade_tracking PRIVATE SACCADE_EXPORT_TRACKING)

# 5. 定義 Media 模組 (Dynamic Library)
add_library(saccade_media SHARED
    src/media/gst_client.cpp
)
target_link_libraries(saccade_media
    ${GSTREAMER_LIBRARIES}
    CUDA::cudart
    saccade_perception
)
target_compile_definitions(saccade_media PRIVATE SACCADE_EXPORT_MEDIA)

# 6. Python Bindings
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
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import tensorrt_libs; print(tensorrt_libs.__path__[0])"
    OUTPUT_VARIABLE TENSORRT_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})

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
    CUDA::cudart
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

pybind11_add_module(saccade_media_ext src/media/gst_client_python.cpp)
target_link_libraries(saccade_media_ext PRIVATE 
    saccade_media
    ${GSTREAMER_LIBRARIES}
    CUDA::cudart
)

pybind11_add_module(saccade_tracking_ext src/tracking/tracker_gpu_python.cpp)
target_link_libraries(saccade_tracking_ext PRIVATE 
    saccade_tracking
    CUDA::cudart
)

pybind11_add_module(saccade_perception_ext src/perception/perception_python.cpp)
target_link_libraries(saccade_perception_ext PRIVATE 
    saccade_perception
    CUDA::cudart
)
set_target_properties(
    saccade_media_ext
    saccade_tracking_ext
    saccade_perception_ext
    PROPERTIES
        BUILD_RPATH "${SACCADE_BUILD_RPATH}"
        INSTALL_RPATH "${SACCADE_INSTALL_RPATH}"
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
endif()
