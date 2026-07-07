#pragma once

#include <Eigen/Dense>
#include <vector>

namespace saccade {

Eigen::MatrixXf cheb_gr_kreciprocal(
    const Eigen::MatrixXf& query_feats,
    const Eigen::MatrixXf& gallery_feats,
    float cheb_lambda = 2.0f,
    int k2 = 6,
    int max_fwd = 50,
    float fuse_lambda = 0.3f);

Eigen::MatrixXf cheb_gr_kreciprocal_self(
    const Eigen::MatrixXf& feats,
    float cheb_lambda = 2.0f,
    int k2 = 6,
    int max_fwd = 50,
    float fuse_lambda = 0.3f);

}  // namespace saccade
