#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace saccade {

class SinkhornAlgorithm {
public:
    // 解決熵正規化最佳傳輸問題 (Sinkhorn-Knopp Algorithm)
    // cost_matrix: [N x M] 成本矩陣 (在追蹤中為 1.0 - IoU)
    // assignment: 輸出陣列，大小為 N。
    // lambda: 熵正規化權重 (數值越大越接近硬分配，但容易數值不穩定；數值越小分配越平滑)
    static void Solve(const std::vector<std::vector<float>>& cost_matrix, std::vector<int>& assignment, float lambda = 30.0f, int max_iters = 50) {
        int n_bidders = cost_matrix.size();
        if (n_bidders == 0) return;
        int n_items = cost_matrix[0].size();
        if (n_items == 0) {
            assignment.assign(n_bidders, -1);
            return;
        }

        // Sinkhorn 通常需要方陣且滿足邊際機率 (Marginal Probabilities)
        int n = std::max(n_bidders, n_items);
        std::vector<std::vector<float>> C(n, std::vector<float>(n, 1.0f)); // 預設最高成本 (IoU = 0)
        for (int i = 0; i < n_bidders; ++i) {
            for (int j = 0; j < n_items; ++j) {
                C[i][j] = cost_matrix[i][j];
            }
        }

        // K = exp(-lambda * C) -> 建立親和力矩陣 (Affinity Matrix)
        std::vector<std::vector<float>> K(n, std::vector<float>(n, 0.0f));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                K[i][j] = std::exp(-lambda * C[i][j]);
            }
        }

        // 初始化縮放向量
        std::vector<float> u(n, 1.0f / n);
        std::vector<float> v(n, 1.0f / n);

        // Sinkhorn 迭代 (純矩陣操作)
        for (int iter = 0; iter < max_iters; ++iter) {
            // u = 1 / (K * v)
            for (int i = 0; i < n; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < n; ++j) {
                    sum += K[i][j] * v[j];
                }
                u[i] = 1.0f / (sum + 1e-9f);
            }
            // v = 1 / (K^T * u)
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int i = 0; i < n; ++i) {
                    sum += K[i][j] * u[i];
                }
                v[j] = 1.0f / (sum + 1e-9f);
            }
        }

        // 計算最終的軟分配矩陣 P = diag(u) * K * diag(v)
        std::vector<std::vector<float>> P(n, std::vector<float>(n, 0.0f));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                P[i][j] = u[i] * K[i][j] * v[j];
            }
        }

        // 貪婪解碼：從連續的機率矩陣中提取硬分配 (Hard Assignment)
        assignment.assign(n_bidders, -1);
        std::vector<bool> item_used(n_items, false);
        
        struct Match {
            int i, j;
            float prob;
            bool operator<(const Match& o) const { return prob > o.prob; } // 機率由大到小排序
        };
        
        std::vector<Match> matches;
        matches.reserve(n_bidders * n_items);
        for (int i = 0; i < n_bidders; ++i) {
            for (int j = 0; j < n_items; ++j) {
                matches.push_back({i, j, P[i][j]});
            }
        }
        std::sort(matches.begin(), matches.end());

        std::vector<bool> bidder_used(n_bidders, false);
        for (const auto& m : matches) {
            if (!bidder_used[m.i] && !item_used[m.j]) {
                assignment[m.i] = m.j;
                bidder_used[m.i] = true;
                item_used[m.j] = true;
            }
        }
    }
};

} // namespace saccade