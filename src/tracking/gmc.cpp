#include "tracking/gmc.hpp"
#include <cuda_runtime.h>
#include <iostream>

namespace saccade {

// Externs from gmc_kernel.cu
void launch_grayscale_downscale(
    const float* src, float* dst, 
    int src_w, int src_h, int dst_w, int dst_h, 
    cudaStream_t stream);

extern "C" void launch_phase_correlation(
    const float* prev_gray, const float* curr_gray,
    int w, int h, float* dx, float* dy,
    void* d_tmp_complex_a, void* d_tmp_complex_b, void* d_tmp_float,
    float* d_peak_x, float* d_peak_y, float* d_peak_val, float* d_pcr_score,
    cufftHandle plan_r2c, cufftHandle plan_c2r, cudaStream_t stream);

void launch_zero_fg_rects(
    float* gray, int w, int h,
    const float* d_boxes, int n_boxes,
    float orig_w, float orig_h,
    cudaStream_t stream);

GMC::GMC(int downscale,
         int max_corners,
         float quality_level,
         float min_distance,
         int min_inliers,
         float ransac_threshold)
    : downscale_(downscale),
      max_corners_(max_corners),
      quality_level_(quality_level),
      min_distance_(min_distance),
      min_inliers_(min_inliers),
      ransac_threshold_(ransac_threshold) {
    cudaStreamCreate(&gmc_stream_);
    cudaEventCreate(&prep_event_);
}

GMC::~GMC() {
    if (gmc_stream_) cudaStreamDestroy(gmc_stream_);
    if (prep_event_) cudaEventDestroy(prep_event_);
    if (d_gray_small_) cudaFree(d_gray_small_);
    
    // PC cleanup
    if (d_prev_gray_) cudaFree(d_prev_gray_);
    if (d_tmp_complex_a_) cudaFree(d_tmp_complex_a_);
    if (d_tmp_complex_b_) cudaFree(d_tmp_complex_b_);
    if (d_tmp_float_) cudaFree(d_tmp_float_);
    if (d_peak_x_) cudaFree(d_peak_x_);
    if (d_peak_y_) cudaFree(d_peak_y_);
    if (d_peak_val_) cudaFree(d_peak_val_);
    if (d_pcr_score_) cudaFree(d_pcr_score_);
    if (d_fg_boxes_) cudaFree(d_fg_boxes_);
    if (plans_created_) {
        cufftDestroy(plan_r2c_);
        cufftDestroy(plan_c2r_);
    }
}

void GMC::reset() {
    prev_gray_.release();
    prev_pts_.clear();
    if (d_prev_gray_) cudaMemset(d_prev_gray_, 0, last_w_ * last_h_ * sizeof(float));
}

void GMC::ensure_gpu_resources(int w, int h) {
    if (plans_created_ && last_w_ == w && last_h_ == h) return;

    if (plans_created_) {
        cufftDestroy(plan_r2c_);
        cufftDestroy(plan_c2r_);
    }
    
    cufftPlan2d(&plan_r2c_, h, w, CUFFT_R2C);
    cufftPlan2d(&plan_c2r_, h, w, CUFFT_C2R);
    cufftSetStream(plan_r2c_, gmc_stream_);
    cufftSetStream(plan_c2r_, gmc_stream_);
    
    size_t size_gray = w * h * sizeof(float);
    // cuFFT R2C output for h×w input: h rows × (w/2+1) complex columns
    size_t size_complex = (size_t)h * (w / 2 + 1) * sizeof(cuComplex);
    size_t size_float = w * h * sizeof(float);

    if (d_prev_gray_) cudaFree(d_prev_gray_);
    if (d_tmp_complex_a_) cudaFree(d_tmp_complex_a_);
    if (d_tmp_complex_b_) cudaFree(d_tmp_complex_b_);
    if (d_tmp_float_) cudaFree(d_tmp_float_);
    if (d_peak_x_) cudaFree(d_peak_x_);
    if (d_peak_y_) cudaFree(d_peak_y_);
    if (d_peak_val_) cudaFree(d_peak_val_);
    if (d_pcr_score_) cudaFree(d_pcr_score_);

    cudaMalloc(&d_prev_gray_, size_gray);
    cudaMalloc(&d_tmp_complex_a_, size_complex);
    cudaMalloc(&d_tmp_complex_b_, size_complex);
    cudaMalloc(&d_tmp_float_, size_float);
    cudaMalloc(&d_peak_x_, sizeof(float));
    cudaMalloc(&d_peak_y_, sizeof(float));
    cudaMalloc(&d_peak_val_, sizeof(float));
    cudaMalloc(&d_pcr_score_, sizeof(float));

    cudaMemset(d_prev_gray_, 0, size_gray);
    
    last_w_ = w; last_h_ = h;
    plans_created_ = true;
}

std::vector<float> GMC::estimate(const float* frame_gpu_ptr, int width, int height, cudaStream_t stream, bool use_gpu_phase_corr) {
    int dst_w = width / downscale_;
    int dst_h = height / downscale_;
    size_t needed = dst_w * dst_h * sizeof(float);
    
    if (d_gray_small_ == nullptr || gray_small_size_ < needed) {
        if (d_gray_small_) cudaFree(d_gray_small_);
        cudaMalloc(&d_gray_small_, needed);
        gray_small_size_ = needed;
    }

    cudaEventRecord(prep_event_, stream);
    cudaStreamWaitEvent(gmc_stream_, prep_event_, 0);

    // Optimized: HWC -> Float Gray with Hanning Window
    launch_grayscale_downscale(frame_gpu_ptr, (float*)d_gray_small_, width, height, dst_w, dst_h, gmc_stream_);
    
    if (use_gpu_phase_corr) {
        ensure_gpu_resources(dst_w, dst_h);
        orig_w_ = width; orig_h_ = height;

        // Apply foreground mask before FFT if boxes were provided this frame.
        if (n_fg_boxes_ > 0) {
            launch_zero_fg_rects(
                (float*)d_gray_small_, dst_w, dst_h,
                d_fg_boxes_, n_fg_boxes_,
                (float)width, (float)height,
                gmc_stream_);
            n_fg_boxes_ = 0;  // consume once per frame
        }

        float dx = 0, dy = 0;
        launch_phase_correlation(
            (const float*)d_prev_gray_, (const float*)d_gray_small_,
            dst_w, dst_h, &dx, &dy,
            d_tmp_complex_a_, d_tmp_complex_b_, d_tmp_float_,
            d_peak_x_, d_peak_y_, d_peak_val_, d_pcr_score_,
            plan_r2c_, plan_c2r_, gmc_stream_
        );

        // Update previous frame regardless of PCR outcome
        cudaMemcpyAsync(d_prev_gray_, d_gray_small_, needed, cudaMemcpyDeviceToDevice, gmc_stream_);

        // launch_phase_correlation already synced the stream; read PCR synchronously.
        float h_pcr = 0.0f;
        cudaMemcpy(&h_pcr, d_pcr_score_, sizeof(float), cudaMemcpyDeviceToHost);
        last_pcr_score_ = h_pcr;

        // NaN means PCR check failed (static camera, low texture) — skip correction
        if (std::isnan(dx) || std::isnan(dy)) return {};

        if (std::abs(dx) > dst_w * 0.25f || std::abs(dy) > dst_h * 0.25f) return {};

        std::vector<float> warp(6);
        warp[0] = 1.0f; warp[1] = 0.0f; warp[2] = dx * downscale_;
        warp[3] = 0.0f; warp[4] = 1.0f; warp[5] = dy * downscale_;
        return warp;
    } else {
        return {}; // Optimized GPU path only for this ADR
    }
}

void GMC::set_fg_mask_boxes(const std::vector<float>& boxes_xyxy) {
    if (boxes_xyxy.empty()) {
        n_fg_boxes_ = 0;
        return;
    }
    size_t needed = boxes_xyxy.size() * sizeof(float);
    if (d_fg_boxes_ == nullptr || fg_boxes_cap_ < needed) {
        if (d_fg_boxes_) cudaFree(d_fg_boxes_);
        cudaMalloc(&d_fg_boxes_, needed);
        fg_boxes_cap_ = needed;
    }
    cudaMemcpy(d_fg_boxes_, boxes_xyxy.data(), needed, cudaMemcpyHostToDevice);
    n_fg_boxes_ = static_cast<int>(boxes_xyxy.size() / 4);
}

std::vector<float> GMC::estimate_mat(const cv::Mat& frame, int downscale_override) {
    int ds = (downscale_override > 0) ? downscale_override : downscale_;
    cv::Mat curr_gray;
    if (frame.channels() == 3) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        if (ds > 1) {
            cv::resize(gray, curr_gray, cv::Size(frame.cols / ds, frame.rows / ds), 0, 0, cv::INTER_AREA);
        } else {
            curr_gray = gray;
        }
    } else {
        if (ds > 1) {
            cv::resize(frame, curr_gray, cv::Size(frame.cols / ds, frame.rows / ds), 0, 0, cv::INTER_AREA);
        } else {
            curr_gray = frame;
        }
    }

    std::vector<float> warp;

    if (!prev_gray_.empty()) {
        try {
            if (prev_pts_.size() < 20) {
                cv::goodFeaturesToTrack(prev_gray_, prev_pts_, max_corners_, quality_level_, min_distance_);
            }

            if (prev_pts_.size() >= (size_t)min_inliers_) {
                std::vector<cv::Point2f> curr_pts;
                std::vector<uchar> status;
                std::vector<float> err;
                cv::calcOpticalFlowPyrLK(prev_gray_, curr_gray, prev_pts_, curr_pts, status, err);

                std::vector<cv::Point2f> good_prev, good_curr;
                for (size_t i = 0; i < status.size(); i++) {
                    if (status[i]) {
                        good_prev.push_back(prev_pts_[i]);
                        good_curr.push_back(curr_pts[i]);
                    }
                }

                if (good_prev.size() >= (size_t)min_inliers_) {
                    cv::Mat inliers;
                    cv::Mat M = cv::estimateAffinePartial2D(good_prev, good_curr, inliers, cv::RANSAC, ransac_threshold_);
                    
                    if (!M.empty() && cv::countNonZero(inliers) >= min_inliers_) {
                        // Rescale translation if downscaled
                        // Note: estimate_mat expects original size if downscale_override is -1
                        // But if called from estimate(float*), ds=1 and scaling is already handled in kernel
                        float scale_w = (float)frame.cols / curr_gray.cols;
                        float scale_h = (float)frame.rows / curr_gray.rows;
                        
                        warp.resize(6);
                        warp[0] = M.at<double>(0, 0);
                        warp[1] = M.at<double>(0, 1);
                        warp[2] = M.at<double>(0, 2) * scale_w;
                        warp[3] = M.at<double>(1, 0);
                        warp[4] = M.at<double>(1, 1);
                        warp[5] = M.at<double>(1, 2) * scale_h;
                    }
                    prev_pts_ = good_curr;
                }
            }
        } catch (const std::exception& e) {
            prev_pts_.clear();
        }
    }

    prev_gray_ = curr_gray.clone();
    return warp;
}

} // namespace saccade
