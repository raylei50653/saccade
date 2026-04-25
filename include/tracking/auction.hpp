#pragma once
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

namespace saccade {

class AuctionAlgorithm {
public:
    // 解決線性分配問題 (Maximization Problem)
    // profit_matrix: [N x M] 收益矩陣 (在追蹤中為 IoU)
    // assignment: 輸出陣列，大小為 N。assignment[i] 表示第 i 個偵測框配對到的軌跡索引。未配對為 -1。
    // epsilon: 最低競標加價，影響收斂速度與精度。
    static void Solve(const std::vector<std::vector<float>>& profit_matrix, std::vector<int>& assignment, float epsilon = 0.01f) {
        int n_bidders = profit_matrix.size();
        if (n_bidders == 0) return;
        int n_items = profit_matrix[0].size();
        if (n_items == 0) {
            assignment.assign(n_bidders, -1);
            return;
        }
        
        // 拍賣演算法需要人數等於商品數，我們以 0 收益 (0 IoU) 補齊空缺變成方陣
        int n = std::max(n_bidders, n_items);
        std::vector<std::vector<float>> benefit(n, std::vector<float>(n, 0.0f));
        for (int i = 0; i < n_bidders; ++i) {
            for (int j = 0; j < n_items; ++j) {
                benefit[i][j] = profit_matrix[i][j];
            }
        }
        
        std::vector<int> owner(n, -1);      // owner[j] 表示第 j 個商品目前的擁有者
        std::vector<int> bid_owner(n, -1);  // bid_owner[i] 表示第 i 個競標者目前擁有的商品
        std::vector<float> prices(n, 0.0f); // 每個商品的目前價格
        
        std::vector<int> unassigned;
        for (int i = 0; i < n; ++i) unassigned.push_back(i);
        
        int iter = 0;
        const int max_iters = 10000; // 防止陷入死迴圈
        
        while (!unassigned.empty() && iter < max_iters) {
            iter++;
            
            std::vector<int> bids(n, -1);
            std::vector<float> bid_increments(n, 0.0f);
            
            // 1. 競標階段 (Bidding Phase)
            // 每個尚未分配的競標者，找出對自己最有利的商品
            for (int i : unassigned) {
                float best_val = -1e9f;
                float second_best_val = -1e9f;
                int best_item = -1;
                
                for (int j = 0; j < n; ++j) {
                    float val = benefit[i][j] - prices[j];
                    if (val > best_val) {
                        second_best_val = best_val;
                        best_val = val;
                        best_item = j;
                    } else if (val > second_best_val) {
                        second_best_val = val;
                    }
                }
                
                if (best_item != -1) {
                    bids[i] = best_item;
                    // 加價幅度為 (最佳利潤 - 次佳利潤 + epsilon)
                    float inc = best_val - second_best_val + epsilon;
                    if (second_best_val <= -1e8f) inc = epsilon; // 只有一個有效商品時的保護機制
                    bid_increments[i] = inc;
                }
            }
            
            // 2. 分配階段 (Assignment Phase)
            // 處理多個買家同時競標同一個商品的情況
            std::vector<int> item_highest_bidder(n, -1);
            std::vector<float> item_max_bid(n, -1e9f);
            
            for (int i : unassigned) {
                int j = bids[i];
                if (j != -1) {
                    float bid = prices[j] + bid_increments[i];
                    if (bid > item_max_bid[j]) {
                        item_max_bid[j] = bid;
                        item_highest_bidder[j] = i;
                    }
                }
            }
            
            std::vector<int> newly_unassigned;
            for (int j = 0; j < n; ++j) {
                int highest_bidder = item_highest_bidder[j];
                if (highest_bidder != -1) {
                    int current_owner = owner[j];
                    if (current_owner != -1) {
                        // 舊買家被踢出，重新回到未分配名單
                        newly_unassigned.push_back(current_owner);
                        bid_owner[current_owner] = -1;
                    }
                    owner[j] = highest_bidder;
                    bid_owner[highest_bidder] = j;
                    prices[j] = item_max_bid[j]; // 價格推高
                }
            }
            
            // 防死鎖：如果一整輪沒有任何有效的競標變更，提早結束
            if (newly_unassigned.size() == unassigned.size()) {
                bool stuck = true;
                for (int i : unassigned) {
                    if (bids[i] != -1 && item_highest_bidder[bids[i]] == i) {
                        stuck = false; break;
                    }
                }
                if (stuck) break;
            }
            
            unassigned = newly_unassigned;
        }
        
        // 映射並輸出結果
        assignment.assign(n_bidders, -1);
        for (int i = 0; i < n_bidders; ++i) {
            int j = bid_owner[i];
            if (j != -1 && j < n_items) {
                assignment[i] = j;
            }
        }
    }
};

} // namespace saccade