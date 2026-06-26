#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <vector>

#define WARP_SIZE 32

namespace saccade {
namespace kernel {

/**
 * @brief Warp-level Top-K (K=3) 插入歸併
 * 
 * 每個 Warp 負責處理代價矩陣的一行。
 * 使用暫存器存儲 local Top-3，並透過 Shuffle 指令進行 Warp 內歸併。
 */
__device__ __forceinline__ void merge_top3(float* v, int* idx, float new_v, int new_idx) {
    if (new_v > v[0]) {
        v[2] = v[1]; idx[2] = idx[1];
        v[1] = v[0]; idx[1] = idx[0];
        v[0] = new_v; idx[0] = new_idx;
    } else if (new_v > v[1]) {
        v[2] = v[1]; idx[2] = idx[1];
        v[1] = new_v; idx[1] = new_idx;
    } else if (new_v > v[2]) {
        v[2] = new_v; idx[2] = new_idx;
    }
}

__device__ __forceinline__ void warp_merge_top3(float* v, int* idx) {
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        for (int i = 0; i < 3; ++i) {
            float other_v = __shfl_xor_sync(0xffffffff, v[i], mask);
            int other_idx = __shfl_xor_sync(0xffffffff, idx[i], mask);
            merge_top3(v, idx, other_v, other_idx);
        }
    }
}

/**
 * @brief 稀疏化 Kernel 原型
 * row_ptr: 代價矩陣行指針
 * n_cols: 矩陣列數
 * out_indices: 輸出的 Top-3 索引 [N x 3]
 * out_probs: 輸出的 Top-3 機率 [N x 3]
 */
__global__ void sparsify_top3_kernel(const float* __restrict__ matrix, int n_rows, int n_cols,
                                     int* out_indices, float* out_probs) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    int tid = threadIdx.x;
    const float* row_data = matrix + row * n_cols;

    // 1. 初始化 local Top-3
    float local_v[3] = {-1e9f, -1e9f, -1e9f};
    int local_idx[3] = {-1, -1, -1};

    // 2. 線性掃描 (Warp 協作)
    for (int col = tid; col < n_cols; col += WARP_SIZE) {
        merge_top3(local_v, local_idx, row_data[col], col);
    }

    // 3. Warp 內歸併
    warp_merge_top3(local_v, local_idx);

    // 4. 由 Warp Leader (Lane 0) 寫回結果
    if (tid == 0) {
        for (int i = 0; i < 3; ++i) {
            out_indices[row * 3 + i] = local_idx[i];
            out_probs[row * 3 + i] = local_v[i];
        }
    }
}

} // namespace kernel
} // namespace saccade

// --- 簡單的測試主程式 ---
int main() {
    const int N = 1, M = 100; // 測試單行 100 個元素
    std::vector<float> h_matrix(N * M);
    for (int i = 0; i < M; ++i) h_matrix[i] = (float)(i % 13) / 13.0f; // 構造一些變化
    h_matrix[42] = 0.99f; h_matrix[7] = 0.88f; h_matrix[99] = 0.95f; // 設定明確的 Top-3

    float *d_matrix, *d_probs;
    int *d_indices;
    cudaMalloc(&d_matrix, N * M * sizeof(float));
    cudaMalloc(&d_probs, N * 3 * sizeof(float));
    cudaMalloc(&d_indices, N * 3 * sizeof(int));

    cudaMemcpy(d_matrix, h_matrix.data(), N * M * sizeof(float), cudaMemcpyHostToDevice);

    saccade::kernel::sparsify_top3_kernel<<<N, 32>>>(d_matrix, N, M, d_indices, d_probs);

    std::vector<float> h_probs(N * 3);
    std::vector<int> h_indices(N * 3);
    cudaMemcpy(h_probs.data(), d_probs, N * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices.data(), d_indices, N * 3 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Top-3 results:\n");
    for (int i = 0; i < 3; ++i) {
        printf("Rank %d: Index %d, Value %.4f\n", i+1, h_indices[i], h_probs[i]);
    }

    cudaFree(d_matrix); cudaFree(d_probs); cudaFree(d_indices);
    return 0;
}
