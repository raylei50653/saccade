#include "tracking/gmc.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

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

extern "C" void launch_phase_correlation_into_warp(
    const float* prev_gray, const float* curr_gray,
    int w, int h, float* out_warp,
    void* d_tmp_complex_a, void* d_tmp_complex_b, void* d_tmp_float,
    float* d_peak_x, float* d_peak_y, float* d_peak_val, float* d_pcr_score,
    cufftHandle plan_r2c, cufftHandle plan_c2r, cudaStream_t stream,
    float downscale);

// Sub-step launchers for per-stage profiling
extern "C" void launch_fft_pair(
    const float* prev_gray, const float* curr_gray,
    void* d_complex_a, void* d_complex_b,
    cufftHandle plan_r2c, cudaStream_t stream);

extern "C" void launch_cross_power_spectrum_step(
    void* d_complex_a, const void* d_complex_b,
    int n_complex, cudaStream_t stream);

extern "C" void launch_ifft_step(
    void* d_complex_a, float* d_float_out,
    cufftHandle plan_c2r, cudaStream_t stream);

extern "C" void launch_peak_and_warp(
    const float* d_float, int w, int h,
    float* d_peak_x, float* d_peak_y, float* d_peak_val, float* d_pcr_score,
    float downscale, float* out_warp, cudaStream_t stream);

void launch_identity_warp(float* out_warp, cudaStream_t stream);

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
    cudaEventCreateWithFlags(&gmc_done_event_, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&pcr_ready_event_, cudaEventDisableTiming);
    cudaHostAlloc(&h_pcr_score_async_, sizeof(float), cudaHostAllocPortable);
    if (h_pcr_score_async_) *h_pcr_score_async_ = 0.0f;
}

GMC::~GMC() {
    if (gmc_stream_) cudaStreamDestroy(gmc_stream_);
    if (prep_event_) cudaEventDestroy(prep_event_);
    if (gmc_done_event_) cudaEventDestroy(gmc_done_event_);
    if (pcr_ready_event_) cudaEventDestroy(pcr_ready_event_);
    if (h_pcr_score_async_) cudaFreeHost(h_pcr_score_async_);
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
    last_pcr_score_ = 0.0f;
    pcr_pending_ = false;
    if (h_pcr_score_async_) *h_pcr_score_async_ = 0.0f;
    if (d_prev_gray_) cudaMemset(d_prev_gray_, 0, last_w_ * last_h_ * sizeof(float));
}

void GMC::set_profiling_enabled(bool enabled) {
    profiling_enabled_ = enabled;
}

void GMC::reset_profile_stats() {
    last_profile_stats_ = ProfileStats{};
}

GMC::ProfileStats GMC::get_profile_stats() const {
    return last_profile_stats_;
}

float GMC::pcr_score() {
    refresh_pcr_score();
    return last_pcr_score_;
}

void GMC::refresh_pcr_score() {
    if (!pcr_pending_ || pcr_ready_event_ == nullptr || h_pcr_score_async_ == nullptr) return;
    const auto status = cudaEventQuery(pcr_ready_event_);
    if (status == cudaSuccess) {
        last_pcr_score_ = *h_pcr_score_async_;
        pcr_pending_ = false;
    }
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

void GMC::estimate_into(
    const float* frame_gpu_ptr,
    int width,
    int height,
    cudaStream_t stream,
    float* d_out_warp,
    bool use_gpu_phase_corr) {
    refresh_pcr_score();
    if (profiling_enabled_) {
        reset_profile_stats();
        last_profile_stats_.frames = 1;
    }

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

    auto measure_gpu_stage = [&](double& out_ms, auto&& fn) {
        if (!profiling_enabled_) {
            fn();
            return;
        }
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, gmc_stream_);
        fn();
        cudaEventRecord(stop, gmc_stream_);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        out_ms += ms;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    measure_gpu_stage(last_profile_stats_.gray_downscale_ms, [&] {
        launch_grayscale_downscale(
            frame_gpu_ptr, (float*)d_gray_small_, width, height, dst_w, dst_h, gmc_stream_);
    });

    if (!use_gpu_phase_corr) {
        measure_gpu_stage(last_profile_stats_.phase_corr_ms, [&] {
            launch_identity_warp(d_out_warp, gmc_stream_);
        });
        last_pcr_score_ = 0.0f;
        pcr_pending_ = false;
        if (h_pcr_score_async_) *h_pcr_score_async_ = 0.0f;
        cudaEventRecord(gmc_done_event_, gmc_stream_);
        cudaStreamWaitEvent(stream, gmc_done_event_, 0);
        if (profiling_enabled_) {
            last_profile_stats_.total_ms =
                last_profile_stats_.gray_downscale_ms
                + last_profile_stats_.fg_mask_ms
                + last_profile_stats_.phase_corr_ms
                + last_profile_stats_.handoff_ms;
        }
        return;
    }

    ensure_gpu_resources(dst_w, dst_h);
    orig_w_ = width;
    orig_h_ = height;

    if (n_fg_boxes_ > 0) {
        measure_gpu_stage(last_profile_stats_.fg_mask_ms, [&] {
            launch_zero_fg_rects(
                (float*)d_gray_small_, dst_w, dst_h,
                d_fg_boxes_, n_fg_boxes_,
                (float)width, (float)height,
                gmc_stream_);
        });
        n_fg_boxes_ = 0;
    }

    if (profiling_enabled_) {
        int n_complex = dst_h * (dst_w / 2 + 1);
        measure_gpu_stage(last_profile_stats_.fft_ms, [&] {
            launch_fft_pair(
                (const float*)d_prev_gray_, (const float*)d_gray_small_,
                d_tmp_complex_a_, d_tmp_complex_b_,
                plan_r2c_, gmc_stream_);
        });
        measure_gpu_stage(last_profile_stats_.cross_power_ms, [&] {
            launch_cross_power_spectrum_step(
                d_tmp_complex_a_, d_tmp_complex_b_, n_complex, gmc_stream_);
        });
        measure_gpu_stage(last_profile_stats_.ifft_ms, [&] {
            launch_ifft_step(d_tmp_complex_a_, (float*)d_tmp_float_, plan_c2r_, gmc_stream_);
        });
        measure_gpu_stage(last_profile_stats_.peak_find_ms, [&] {
            launch_peak_and_warp(
                (const float*)d_tmp_float_, dst_w, dst_h,
                d_peak_x_, d_peak_y_, d_peak_val_, d_pcr_score_,
                static_cast<float>(downscale_), d_out_warp, gmc_stream_);
        });
        last_profile_stats_.phase_corr_ms =
            last_profile_stats_.fft_ms
            + last_profile_stats_.cross_power_ms
            + last_profile_stats_.ifft_ms
            + last_profile_stats_.peak_find_ms;
    } else {
        launch_phase_correlation_into_warp(
            (const float*)d_prev_gray_, (const float*)d_gray_small_,
            dst_w, dst_h, d_out_warp,
            d_tmp_complex_a_, d_tmp_complex_b_, d_tmp_float_,
            d_peak_x_, d_peak_y_, d_peak_val_, d_pcr_score_,
            plan_r2c_, plan_c2r_, gmc_stream_, static_cast<float>(downscale_));
    }

    measure_gpu_stage(last_profile_stats_.handoff_ms, [&] {
        cudaMemcpyAsync(
            d_prev_gray_, d_gray_small_, needed, cudaMemcpyDeviceToDevice, gmc_stream_);
        cudaMemcpyAsync(
            h_pcr_score_async_, d_pcr_score_, sizeof(float), cudaMemcpyDeviceToHost, gmc_stream_);
        cudaEventRecord(pcr_ready_event_, gmc_stream_);
        pcr_pending_ = true;
        cudaEventRecord(gmc_done_event_, gmc_stream_);
    });
    cudaStreamWaitEvent(stream, gmc_done_event_, 0);
    if (profiling_enabled_) {
        last_profile_stats_.total_ms =
            last_profile_stats_.gray_downscale_ms
            + last_profile_stats_.fg_mask_ms
            + last_profile_stats_.phase_corr_ms
            + last_profile_stats_.handoff_ms;
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
