cmake_minimum_required(VERSION 3.18)
project(Saccade CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 1. 全局編譯配置
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fvisibility=hidden -O3 -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fPIC")

# 設置 RPATH 以支持動態連結
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# 2. 依賴搜尋
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# 3. 定義 Perception 模組 (Dynamic Library)
add_library(saccade_perception SHARED
    src/perception/trt_engine.cpp
    src/perception/preprocessor.cpp
    src/perception/preprocessor_gpu.cu
)
target_link_libraries(saccade_perception
    nvinfer
    nvinfer_plugin
    nppc nppig nppidei nppist nppicc
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
)
# 設定導出宏
target_compile_definitions(saccade_perception PRIVATE SACCADE_EXPORT_PERCEPTION)

# 4. 定義 Tracking 模組 (Dynamic Library)
add_library(saccade_tracking SHARED
    src/tracking/tracker_gpu.cu
    src/tracking/smart_tracker.cu
)
set_target_properties(saccade_tracking PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(saccade_tracking
    ${CUDA_LIBRARIES}
    saccade_perception
)
target_compile_definitions(saccade_tracking PRIVATE SACCADE_EXPORT_TRACKING)

# 5. 定義 Media 模組 (Dynamic Library)
add_library(saccade_media SHARED
    src/media/gst_client.cpp
)
target_link_libraries(saccade_media
    ${GSTREAMER_LIBRARIES}
    ${CUDA_LIBRARIES}
    saccade_perception
)
target_compile_definitions(saccade_media PRIVATE SACCADE_EXPORT_MEDIA)

# 6. Python Bindings
find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})

pybind11_add_module(saccade_media_ext src/media/gst_client_python.cpp)
target_link_libraries(saccade_media_ext PRIVATE 
    saccade_media
    ${GSTREAMER_LIBRARIES}
    ${CUDA_LIBRARIES}
)

pybind11_add_module(saccade_tracking_ext src/tracking/tracker_gpu_python.cpp)
target_link_libraries(saccade_tracking_ext PRIVATE 
    saccade_tracking
    ${CUDA_LIBRARIES}
)

pybind11_add_module(saccade_perception_ext src/perception/perception_python.cpp)
target_link_libraries(saccade_perception_ext PRIVATE 
    saccade_perception
    ${CUDA_LIBRARIES}
)

# 7. 主程式
add_executable(saccade_node src/main.cpp)
target_link_libraries(saccade_node 
    saccade_perception 
    saccade_tracking
    saccade_media
    ${GSTREAMER_LIBRARIES}
)
