#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <vector>

#define WARP_SIZE 32

namespace saccade {
namespace kernel {

/**
 * @brief 修正後的 6-way merge to Top-3 (Branchless 傾向)
 * 
 * 傳入兩個已排序的 Top-3 數組 (a, b)，合併出最終的 Top-3 並存回 a。
 * 這是 warp_merge_top3 的核心，解決了重複寫入與晉升失敗的問題。
 */
__device__ __forceinline__ void merge_top3_fixed(float* a_v, int* a_idx, const float* b_v, const int* b_idx) {
    float res_v[3];
    int res_idx[3];
    
    int i = 0, j = 0;
    // 嚴格的 6-way 選取前 3 個，保證 100% 正確性與唯一性 (假設輸入互斥)
    for (int k = 0; k < 3; ++k) {
        if (a_v[i] >= b_v[j]) {
            res_v[k] = a_v[i];
            res_idx[k] = a_idx[i];
            i++;
        } else {
            res_v[k] = b_v[j];
            res_idx[k] = b_idx[j];
            j++;
        }
    }

    // 寫回 a
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        a_v[k] = res_v[k];
        a_idx[k] = res_idx[k];
    }
}

__device__ __forceinline__ void warp_merge_top3_fixed(float* v, int* idx) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        float other_v[3];
        int other_idx[3];
        
        // 一次性交換整個 Top-3 窗口
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            other_v[i] = __shfl_xor_sync(0xffffffff, v[i], mask);
            other_idx[i] = __shfl_xor_sync(0xffffffff, idx[i], mask);
        }
        
        merge_top3_fixed(v, idx, other_v, other_idx);
    }
}

/**
 * @brief 單一元素插入 (用於線性掃描階段)
 */
__device__ __forceinline__ void insert_single_top3(float* v, int* idx, float new_v, int new_idx) {
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

__global__ void sparsify_top3_kernel_fixed(const float* __restrict__ matrix, int n_rows, int n_cols,
                                           int* out_indices, float* out_probs) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    int tid = threadIdx.x;
    const float* row_data = matrix + row * n_cols;

    float local_v[3] = {-1e9f, -1e9f, -1e9f};
    int local_idx[3] = {-1, -1, -1};

    for (int col = tid; col < n_cols; col += WARP_SIZE) {
        insert_single_top3(local_v, local_idx, row_data[col], col);
    }

    warp_merge_top3_fixed(local_v, local_idx);

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            out_indices[row * 3 + i] = local_idx[i];
            out_probs[row * 3 + i] = local_v[i];
        }
    }
}

} // namespace kernel
} // namespace saccade

int main() {
    const int N = 1, M = 100;
    std::vector<float> h_matrix(N * M);
    for (int i = 0; i < M; ++i) h_matrix[i] = (float)(i % 13) / 13.0f;
    h_matrix[42] = 0.99f; h_matrix[7] = 0.88f; h_matrix[99] = 0.95f;

    float *d_matrix, *d_probs;
    int *d_indices;
    cudaMalloc(&d_matrix, N * M * sizeof(float));
    cudaMalloc(&d_probs, N * 3 * sizeof(float));
    cudaMalloc(&d_indices, N * 3 * sizeof(int));

    cudaMemcpy(d_matrix, h_matrix.data(), N * M * sizeof(float), cudaMemcpyHostToDevice);

    saccade::kernel::sparsify_top3_kernel_fixed<<<N, 32>>>(d_matrix, N, M, d_indices, d_probs);

    std::vector<float> h_probs(N * 3);
    std::vector<int> h_indices(N * 3);
    cudaMemcpy(h_probs.data(), d_probs, N * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices.data(), d_indices, N * 3 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Fixed Top-3 results:\n");
    for (int i = 0; i < 3; ++i) {
        printf("Rank %d: Index %d, Value %.4f\n", i+1, h_indices[i], h_probs[i]);
    }

    cudaFree(d_matrix); cudaFree(d_probs); cudaFree(d_indices);
    return 0;
}
