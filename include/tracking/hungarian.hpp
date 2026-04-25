#pragma once
#include <vector>
#include <limits>
#include <algorithm>

namespace saccade {

class HungarianAlgorithm {
public:
    // 解決線性分配問題 (Linear Assignment Problem)
    // cost_matrix: [N x M] 成本矩陣 (例如 1.0 - IoU)
    // assignment: 輸出陣列，大小為 N。assignment[i] 表示第 i 個偵測框配對到的軌跡索引。未配對為 -1。
    void Solve(const std::vector<std::vector<float>>& cost_matrix, std::vector<int>& assignment) {
        int n = cost_matrix.size();
        if (n == 0) return;
        int m = cost_matrix[0].size();
        if (m == 0) {
            assignment.assign(n, -1);
            return;
        }
        
        // 匈牙利演算法要求 Row 數量 <= Col 數量 (n <= m)
        // 如果 n > m，我們轉置矩陣求解，最後再映射回來。
        bool transposed = false;
        std::vector<std::vector<float>> cost;
        if (n > m) {
            transposed = true;
            cost.assign(m, std::vector<float>(n));
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < m; ++j)
                    cost[j][i] = cost_matrix[i][j];
            std::swap(n, m);
        } else {
            cost = cost_matrix;
        }

        std::vector<float> u(n + 1, 0.0f), v(m + 1, 0.0f);
        std::vector<int> p(m + 1, 0), way(m + 1, 0);

        for (int i = 1; i <= n; ++i) {
            p[0] = i;
            int j0 = 0;
            std::vector<float> minv(m + 1, std::numeric_limits<float>::max());
            std::vector<bool> used(m + 1, false);

            do {
                used[j0] = true;
                int i0 = p[j0], j1 = 0;
                float delta = std::numeric_limits<float>::max();

                for (int j = 1; j <= m; ++j) {
                    if (!used[j]) {
                        float cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                        if (cur < minv[j]) {
                            minv[j] = cur;
                            way[j] = j0;
                        }
                        if (minv[j] < delta) {
                            delta = minv[j];
                            j1 = j;
                        }
                    }
                }

                for (int j = 0; j <= m; ++j) {
                    if (used[j]) {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }
                j0 = j1;
            } while (p[j0] != 0);

            do {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0 != 0);
        }

        // 映射結果
        assignment.assign(transposed ? m : n, -1);
        if (transposed) {
            for (int j = 1; j <= m; ++j) {
                if (p[j] != 0) assignment[j - 1] = p[j] - 1;
            }
        } else {
            for (int j = 1; j <= m; ++j) {
                if (p[j] != 0) assignment[p[j] - 1] = j - 1;
            }
        }
    }
};

} // namespace saccade
