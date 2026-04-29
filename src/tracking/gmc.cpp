#include "tracking/gmc.hpp"
#include <cuda_runtime.h>
#include <iostream>

namespace saccade {

// Extern from gmc_kernel.cu
void launch_grayscale_downscale(
    const float* src, uint8_t* dst, 
    int src_w, int src_h, int dst_w, int dst_h, 
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
      ransac_threshold_(ransac_threshold) {}

void GMC::reset() {
    prev_gray_.release();
    prev_pts_.clear();
}

std::vector<float> GMC::estimate(const float* frame_gpu_ptr, int width, int height, cudaStream_t stream) {
    int dst_w = width / downscale_;
    int dst_h = height / downscale_;
    size_t needed = dst_w * dst_h;

    if (d_gray_small_ == nullptr || gray_small_size_ < needed) {
        if (d_gray_small_) cudaFree(d_gray_small_);
        cudaMalloc(&d_gray_small_, needed);
        gray_small_size_ = needed;
    }

    launch_grayscale_downscale(frame_gpu_ptr, (uint8_t*)d_gray_small_, width, height, dst_w, dst_h, stream);
    
    cv::Mat curr_gray(dst_h, dst_w, CV_8UC1);
    cudaMemcpyAsync(curr_gray.data, d_gray_small_, needed, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return estimate_mat(curr_gray, 1); // downscale already done
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
