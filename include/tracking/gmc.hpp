#pragma once

#include "saccade/common.hpp"
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cufft.h>

namespace saccade {

/**
 * @brief Camera Motion Compensation.
 * 
 * Supports both CPU-based (BoT-SORT style LK + RANSAC) and 
 * pure GPU-based (UCMCTrack style Phase Correlation) modes.
 */
class SACCADE_TRACKING_API GMC {
public:
    GMC(int downscale = 8,
        int max_corners = 100,
        float quality_level = 0.01,
        float min_distance = 10.0,
        int min_inliers = 8,
        float ransac_threshold = 3.0);
    ~GMC();

    /**
     * @brief Estimate affine camera warp between previous and current frame.
     * @param frame_gpu_ptr GPU pointer (CHW float32 [0,1])
     * @param width Original width
     * @param height Original height
     * @param stream CUDA stream
     * @param use_gpu_phase_corr If true, uses pure GPU phase correlation (translation only, high performance).
     * @return 6-float vector [H00, H01, H02, H10, H11, H12], or empty if failed.
     */
    std::vector<float> estimate(const float* frame_gpu_ptr, int width, int height, cudaStream_t stream, bool use_gpu_phase_corr = true);

    /**
     * @brief Set foreground bounding boxes to be zeroed before phase correlation.
     * Boxes are in original-frame pixel coordinates: [x1,y1,x2,y2, ...] flat.
     * Call before estimate() to suppress foreground bias.
     */
    void set_fg_mask_boxes(const std::vector<float>& boxes_xyxy);

    /** @brief PCR (peak-to-RMS ratio) from the most recent GPU phase correlation. */
    float pcr_score() const { return last_pcr_score_; }

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

    // Buffer for GPU -> CPU transfer (Legacy/Fallback)
    void* d_gray_small_ = nullptr;
    size_t gray_small_size_ = 0;

    // Pure GPU Phase Correlation members
    float* d_prev_gray_ = nullptr;
    void* d_tmp_complex_a_ = nullptr;
    void* d_tmp_complex_b_ = nullptr;
    float* d_tmp_float_ = nullptr;
    float* d_peak_x_ = nullptr;
    float* d_peak_y_ = nullptr;
    float* d_peak_val_ = nullptr;
    float* d_pcr_score_ = nullptr;
    float last_pcr_score_ = 0.0f;
    cufftHandle plan_r2c_ = 0;
    cufftHandle plan_c2r_ = 0;
    bool plans_created_ = false;
    int last_w_ = 0, last_h_ = 0;

    // Foreground mask boxes (optional, set each frame before estimate())
    float* d_fg_boxes_ = nullptr;
    int n_fg_boxes_ = 0;
    size_t fg_boxes_cap_ = 0;
    int orig_w_ = 0, orig_h_ = 0;

    void ensure_gpu_resources(int w, int h);

    cudaStream_t gmc_stream_ = nullptr;
    cudaEvent_t prep_event_ = nullptr;
};


} // namespace saccade
