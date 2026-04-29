#pragma once

#include "saccade/common.hpp"
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

namespace saccade {

/**
 * @brief Camera Motion Compensation via Sparse Optical Flow (BoT-SORT style).
 * 
 * Estimates an affine warp between consecutive frames using goodFeaturesToTrack 
 * + Lucas-Kanade + RANSAC.
 */
class SACCADE_TRACKING_API GMC {
public:
    GMC(int downscale = 8,
        int max_corners = 100,
        float quality_level = 0.01,
        float min_distance = 10.0,
        int min_inliers = 8,
        float ransac_threshold = 3.0);
    ~GMC() = default;

    /**
     * @brief Estimate affine camera warp between previous and current frame.
     * @param frame_gpu_ptr GPU pointer (RGB interleaved, HWC, float32, [0, 1])
     * @param width Original width
     * @param height Original height
     * @param stream CUDA stream
     * @return 6-float vector [H00, H01, H02, H10, H11, H12], or empty if failed.
     */
    std::vector<float> estimate(const float* frame_gpu_ptr, int width, int height, cudaStream_t stream);

    /**
     * @brief Estimate affine camera warp using CPU Mat (for Python compatibility).
     * @param frame BGR or RGB Mat
     * @param downscale Internal downscale factor (overrides ctor if > 0)
     * @return 6-float vector
     */
    std::vector<float> estimate_mat(const cv::Mat& frame, int downscale = -1);

    void reset();

private:
    int downscale_;
    int max_corners_;
    float quality_level_;
    float min_distance_;
    int min_inliers_;
    float ransac_threshold_;

    cv::Mat prev_gray_;
    std::vector<cv::Point2f> prev_pts_;

    // Buffer for GPU -> CPU transfer
    void* d_gray_small_ = nullptr;
    size_t gray_small_size_ = 0;
};

} // namespace saccade
