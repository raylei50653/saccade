#pragma once

#include "tracking/gmc.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace saccade {

/**
 * @brief GMC's CPU mode (BoT-SORT style LK + RANSAC) on a host Mat -- the
 * Python `GMC.estimate_mat`. Kept out of gmc.cpp so that the GPU GMC, which
 * the shipping runtime links, needs no OpenCV (#465 PR-11).
 *
 * @param gmc       supplies the LK / RANSAC parameters and keeps the previous
 *                  frame between calls (GMC::reset() clears it)
 * @param frame     BGR or gray uint8 Mat
 * @param downscale internal downscale factor (overrides the GMC's if > 0)
 * @return 6-float vector [H00, H01, H02, H10, H11, H12], or empty if failed
 */
std::vector<float> gmc_estimate_mat(GMC& gmc, const cv::Mat& frame, int downscale = -1);

} // namespace saccade
