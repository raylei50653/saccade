// GMC's CPU mode (#465 PR-11: moved verbatim out of gmc.cpp, which no longer
// needs OpenCV). The previous gray frame and the tracked points used to be GMC
// members; they now live in a state object the GMC holds type-erased, so
// GMC::reset() still clears them.
#include "tracking/gmc_cpu.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <memory>

namespace saccade {

namespace {

struct GmcCpuFlowState {
    cv::Mat prev_gray;
    std::vector<cv::Point2f> prev_pts;
};

} // namespace

struct GmcCpuFlow {
    static std::vector<float> estimate(GMC& gmc, const cv::Mat& frame, int downscale_override);
};

std::vector<float> GmcCpuFlow::estimate(GMC& gmc, const cv::Mat& frame, int downscale_override) {
    if (!gmc.cpu_flow_state_) gmc.cpu_flow_state_ = std::make_shared<GmcCpuFlowState>();
    auto& state = *static_cast<GmcCpuFlowState*>(gmc.cpu_flow_state_.get());
    // Local names match the former members, so the body below is unchanged.
    cv::Mat& prev_gray_ = state.prev_gray;
    std::vector<cv::Point2f>& prev_pts_ = state.prev_pts;
    const int downscale_ = gmc.downscale_;
    const int max_corners_ = gmc.max_corners_;
    const float quality_level_ = gmc.quality_level_;
    const float min_distance_ = gmc.min_distance_;
    const int min_inliers_ = gmc.min_inliers_;
    const float ransac_threshold_ = gmc.ransac_threshold_;

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

std::vector<float> gmc_estimate_mat(GMC& gmc, const cv::Mat& frame, int downscale) {
    return GmcCpuFlow::estimate(gmc, frame, downscale);
}

} // namespace saccade
