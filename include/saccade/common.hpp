#pragma once

// 導出宏定義 (Visibility Control)
#if defined(_WIN32)
    #define SACCADE_EXPORT __declspec(dllexport)
    #define SACCADE_IMPORT __declspec(dllimport)
#else
    #define SACCADE_EXPORT __attribute__((visibility("default")))
    #define SACCADE_IMPORT __attribute__((visibility("default")))
#endif

#ifdef SACCADE_EXPORT_PERCEPTION
    #define SACCADE_PERCEPTION_API SACCADE_EXPORT
#else
    #define SACCADE_PERCEPTION_API SACCADE_IMPORT
#endif

#ifdef SACCADE_EXPORT_TRACKING
    #define SACCADE_TRACKING_API SACCADE_EXPORT
#else
    #define SACCADE_TRACKING_API SACCADE_IMPORT
#endif

#ifdef SACCADE_EXPORT_MEDIA
    #define SACCADE_MEDIA_API SACCADE_EXPORT
#else
    #define SACCADE_MEDIA_API SACCADE_IMPORT
#endif

#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>

namespace saccade {

// 全局 CUDA 錯誤檢查
inline void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        throw std::runtime_error("CUDA Error");
    }
}

} // namespace saccade
