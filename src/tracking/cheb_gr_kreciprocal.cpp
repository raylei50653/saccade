#include "tracking/cheb_gr_kreciprocal.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace saccade {

namespace {

void topk_indices(const Eigen::Ref<const Eigen::VectorXf>& row,
                  int k,
                  std::vector<int>& out) {
    int n = static_cast<int>(row.size());
    out.resize(n);
    std::iota(out.begin(), out.end(), 0);
    if (k <= 0 || n == 0) {
        out.clear();
        return;
    }
    if (k >= n) return;
    std::sort(out.begin(), out.end(),
              [&row](int a, int b) {
                  if (row(a) != row(b)) return row(a) < row(b);
                  return a < b;
              });
    out.resize(k);
}

}  // namespace

Eigen::MatrixXf cheb_gr_kreciprocal(
    const Eigen::MatrixXf& query_feats,
    const Eigen::MatrixXf& gallery_feats,
    float cheb_lambda,
    int k2,
    int max_fwd,
    float fuse_lambda) {

    int nq = static_cast<int>(query_feats.rows());
    int ng = static_cast<int>(gallery_feats.rows());
    int n = nq + ng;

    Eigen::MatrixXf combined(n, query_feats.cols());
    combined.topRows(nq) = query_feats;
    combined.bottomRows(ng) = gallery_feats;

    Eigen::MatrixXf dist = 2.0f * Eigen::MatrixXf::Ones(n, n) -
                           2.0f * combined * combined.transpose();
    dist = dist.cwiseMax(0.0f);

    Eigen::VectorXf mu = dist.rowwise().mean();
    Eigen::VectorXf sigma(n);
    for (int i = 0; i < n; ++i) {
        float m = mu(i);
        float var = (dist.row(i).array() - m).square().mean();
        sigma(i) = std::sqrt(var);
    }

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> fwd(n, n);
    for (int i = 0; i < n; ++i) {
        float thresh = mu(i) - cheb_lambda * sigma(i);
        for (int j = 0; j < n; ++j) {
            fwd(i, j) = dist(i, j) <= thresh;
        }
        fwd(i, i) = true;
    }

    if (max_fwd > 0 && max_fwd < n) {
        for (int i = 0; i < n; ++i) {
            std::vector<int> idx;
            topk_indices(dist.row(i), max_fwd, idx);
            std::vector<bool> cap_row(n, false);
            for (int j : idx) cap_row[j] = true;
            for (int j = 0; j < n; ++j) fwd(i, j) = fwd(i, j) && cap_row[j];
        }
    }

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            fwd(i, j) = fwd(i, j) && fwd(j, i);

    int k2_eff = std::min(std::max(k2, 1), n);
    std::vector<std::vector<int>> knn(n);
    for (int i = 0; i < n; ++i) {
        topk_indices(dist.row(i), k2_eff, knn[i]);
    }

    Eigen::MatrixXf v(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            v(i, j) = std::exp(-dist(i, j)) * (fwd(i, j) ? 1.0f : 0.0f);
        }
        float row_sum = v.row(i).sum();
        if (row_sum > 1e-12f) v.row(i) /= row_sum;
    }

    if (k2_eff > 1) {
        Eigen::MatrixXf a = Eigen::MatrixXf::Zero(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j : knn[i]) {
                a(i, j) += 1.0f / static_cast<float>(k2_eff);
            }
        }
        v = a * v;
    }

    Eigen::MatrixXf d_jaccard(nq, ng);
    for (int i = 0; i < nq; ++i) {
        std::vector<int> support;
        for (int j = 0; j < n; ++j) {
            if (v(i, j) > 0.0f) support.push_back(j);
        }
        for (int j = 0; j < ng; ++j) {
            if (support.empty()) {
                d_jaccard(i, j) = 1.0f;
                continue;
            }
            float s = 0.0f;
            for (int col : support) {
                s += std::min(v(i, col), v(nq + j, col));
            }
            d_jaccard(i, j) = 1.0f - s / std::max(2.0f - s, 1e-12f);
        }
    }

    if (fuse_lambda >= 1.0f) return d_jaccard;

    Eigen::MatrixXf cos = query_feats * gallery_feats.transpose();
    Eigen::MatrixXf d_orig = 0.5f * (1.0f - cos.array());
    return fuse_lambda * d_jaccard + (1.0f - fuse_lambda) * d_orig;
}

Eigen::MatrixXf cheb_gr_kreciprocal_self(
    const Eigen::MatrixXf& feats,
    float cheb_lambda,
    int k2,
    int max_fwd,
    float fuse_lambda) {
    return cheb_gr_kreciprocal(feats, feats, cheb_lambda, k2, max_fwd,
                               fuse_lambda);
}

}  // namespace saccade
