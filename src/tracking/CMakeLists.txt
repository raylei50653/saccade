cmake_minimum_required(VERSION 3.18)
project(saccade_fpn_reid LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

add_library(fpn_reid_cuda SHARED
    fpn_reid_cuda.cu
)

target_compile_features(fpn_reid_cuda PUBLIC cxx_std_17)
set_target_properties(fpn_reid_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(fpn_reid_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>)

target_include_directories(fpn_reid_cuda PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

target_link_libraries(fpn_reid_cuda PRIVATE
    ${TORCH_LIBRARIES}
    Python::Python
)

# PyTorch C++ extensions that use PYBIND11_MODULE need libtorch_python
find_library(TORCH_PYTHON_LIB torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)
if(TORCH_PYTHON_LIB)
    target_link_libraries(fpn_reid_cuda PRIVATE "${TORCH_PYTHON_LIB}")
endif()
