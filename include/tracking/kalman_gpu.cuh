#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace saccade {
namespace kf_gpu {

// 4x4 矩陣求逆 (使用 Cramer's Rule / 伴隨矩陣，專為 GPU Kernel 優化)
__device__ __forceinline__ void invert4x4(const float m[16], float inv[16]) {
    float inv0  = m[5]  * m[10] * m[15] - 
                  m[5]  * m[11] * m[14] - 
                  m[9]  * m[6]  * m[15] + 
                  m[9]  * m[7]  * m[14] +
                  m[13] * m[6]  * m[11] - 
                  m[13] * m[7]  * m[10];

    float inv4  = -m[4]  * m[10] * m[15] + 
                   m[4]  * m[11] * m[14] + 
                   m[8]  * m[6]  * m[15] - 
                   m[8]  * m[7]  * m[14] - 
                   m[12] * m[6]  * m[11] + 
                   m[12] * m[7]  * m[10];

    float inv8  = m[4]  * m[9] * m[15] - 
                  m[4]  * m[11] * m[13] - 
                  m[8]  * m[5] * m[15] + 
                  m[8]  * m[7] * m[13] + 
                  m[12] * m[5] * m[11] - 
                  m[12] * m[7] * m[9];

    float inv12 = -m[4]  * m[9] * m[14] + 
                   m[4]  * m[10] * m[13] +
                   m[8]  * m[5] * m[14] - 
                   m[8]  * m[6] * m[13] - 
                   m[12] * m[5] * m[10] + 
                   m[12] * m[6] * m[9];

    float inv1  = -m[1]  * m[10] * m[15] + 
                   m[1]  * m[11] * m[14] + 
                   m[9]  * m[2] * m[15] - 
                   m[9]  * m[3] * m[14] - 
                   m[13] * m[2] * m[11] + 
                   m[13] * m[3] * m[10];

    float inv5  = m[0]  * m[10] * m[15] - 
                  m[0]  * m[11] * m[14] - 
                  m[8]  * m[2] * m[15] + 
                  m[8]  * m[3] * m[14] + 
                  m[12] * m[2] * m[11] - 
                  m[12] * m[3] * m[10];

    float inv9  = -m[0]  * m[9] * m[15] + 
                   m[0]  * m[11] * m[13] + 
                   m[8]  * m[1] * m[15] - 
                   m[8]  * m[3] * m[13] - 
                   m[12] * m[1] * m[11] + 
                   m[12] * m[3] * m[9];

    float inv13 = m[0]  * m[9] * m[14] - 
                  m[0]  * m[10] * m[13] - 
                  m[8]  * m[1] * m[14] + 
                  m[8]  * m[2] * m[13] + 
                  m[12] * m[1] * m[10] - 
                  m[12] * m[2] * m[9];

    float inv2  = m[1]  * m[6] * m[15] - 
                  m[1]  * m[7] * m[14] - 
                  m[5]  * m[2] * m[15] + 
                  m[5]  * m[3] * m[14] + 
                  m[13] * m[2] * m[7] - 
                  m[13] * m[3] * m[6];

    float inv6  = -m[0]  * m[6] * m[15] + 
                   m[0]  * m[7] * m[14] + 
                   m[4]  * m[2] * m[15] - 
                   m[4]  * m[3] * m[14] - 
                   m[12] * m[2] * m[7] + 
                   m[12] * m[3] * m[6];

    float inv10 = m[0]  * m[5] * m[15] - 
                  m[0]  * m[7] * m[13] - 
                  m[4]  * m[1] * m[15] + 
                  m[4]  * m[3] * m[13] + 
                  m[12] * m[1] * m[7] - 
                  m[12] * m[3] * m[5];

    float inv14 = -m[0]  * m[5] * m[14] + 
                   m[0]  * m[6] * m[13] + 
                   m[4]  * m[1] * m[14] - 
                   m[4]  * m[2] * m[13] - 
                   m[12] * m[1] * m[6] + 
                   m[12] * m[2] * m[5];

    float inv3  = -m[1] * m[6] * m[11] + 
                   m[1] * m[7] * m[10] + 
                   m[5] * m[2] * m[11] - 
                   m[5] * m[3] * m[10] - 
                   m[9] * m[2] * m[7] + 
                   m[9] * m[3] * m[6];

    float inv7  = m[0] * m[6] * m[11] - 
                  m[0] * m[7] * m[10] - 
                  m[4] * m[2] * m[11] + 
                  m[4] * m[3] * m[10] + 
                  m[8] * m[2] * m[7] - 
                  m[8] * m[3] * m[6];

    float inv11 = -m[0] * m[5] * m[11] + 
                   m[0] * m[7] * m[9] + 
                   m[4] * m[1] * m[11] - 
                   m[4] * m[3] * m[9] - 
                   m[8] * m[1] * m[7] + 
                   m[8] * m[3] * m[5];

    float inv15 = m[0] * m[5] * m[10] - 
                  m[0] * m[6] * m[9] - 
                  m[4] * m[1] * m[10] + 
                  m[4] * m[2] * m[9] + 
                  m[8] * m[1] * m[6] - 
                  m[8] * m[2] * m[5];

    float det = m[0] * inv0 + m[1] * inv4 + m[2] * inv8 + m[3] * inv12;
    if (fabs(det) < 1e-12f) det = 1e-12f; // 防止除以零
    float inv_det = 1.0f / det;

    inv[0] = inv0 * inv_det;
    inv[1] = inv1 * inv_det;
    inv[2] = inv2 * inv_det;
    inv[3] = inv3 * inv_det;
    inv[4] = inv4 * inv_det;
    inv[5] = inv5 * inv_det;
    inv[6] = inv6 * inv_det;
    inv[7] = inv7 * inv_det;
    inv[8] = inv8 * inv_det;
    inv[9] = inv9 * inv_det;
    inv[10] = inv10 * inv_det;
    inv[11] = inv11 * inv_det;
    inv[12] = inv12 * inv_det;
    inv[13] = inv13 * inv_det;
    inv[14] = inv14 * inv_det;
    inv[15] = inv15 * inv_det;
}

// 初始化協方差矩陣 P (8x8)
__device__ __forceinline__ void init_covariance(float P[64]) {
    for (int i = 0; i < 64; ++i) P[i] = 0.0f;
    // 較大的初始不確定性
    P[0] = 10.0f;  P[9] = 10.0f;  P[18] = 10.0f; P[27] = 10.0f;
    P[36] = 10000.0f; P[45] = 10000.0f; P[54] = 10000.0f; P[63] = 10000.0f;
}

// 取得過程噪聲矩陣 Q
__device__ __forceinline__ void get_Q(float h, float Q[64]) {
    for (int i = 0; i < 64; ++i) Q[i] = 0.0f;
    float std_weight_position = 1.0f / 20.0f;
    float std_weight_velocity = 1.0f / 160.0f;
    float pos_std = std_weight_position * h;
    float vel_std = std_weight_velocity * h;
    
    Q[0] = pos_std * pos_std; 
    Q[9] = pos_std * pos_std; 
    Q[18] = 1e-4f; 
    Q[27] = pos_std * pos_std;
    Q[36] = vel_std * vel_std; 
    Q[45] = vel_std * vel_std; 
    Q[54] = 1e-10f; 
    Q[63] = vel_std * vel_std;
}

// 取得測量噪聲矩陣 R
__device__ __forceinline__ void get_R(float h, float R[16], float light_factor = 0.0f) {
    for (int i = 0; i < 16; ++i) R[i] = 0.0f;
    float std_weight_position = 1.0f / 20.0f;
    float pos_std = std_weight_position * h;
    
    float multiplier = 1.0f + 2.0f * light_factor;
    R[0] = pos_std * pos_std * multiplier; 
    R[5] = pos_std * pos_std * multiplier; 
    R[10] = 1e-2f * multiplier; 
    R[15] = pos_std * pos_std * multiplier;
}

// 卡爾曼預測步
__device__ __forceinline__ void predict(float x[8], float P[64]) {
    // 1. x = F * x (因 F 特殊結構，等同於 x[0:4] += x[4:8])
    x[0] += x[4];
    x[1] += x[5];
    x[2] += x[6];
    x[3] += x[7];

    // 2. P = F * P * F^T + Q
    // 由於 F = [I, I; 0, I]
    // P_new_top_left = P_00 + P_01 + P_10 + P_11 + Q_00
    // P_new_top_right = P_01 + P_11
    // P_new_bottom_left = P_10 + P_11
    // P_new_bottom_right = P_11 + Q_11
    float P_new[64];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            P_new[i*8+j] = P[i*8+j] + P[i*8+(j+4)] + P[(i+4)*8+j] + P[(i+4)*8+(j+4)];
            P_new[i*8+(j+4)] = P[i*8+(j+4)] + P[(i+4)*8+(j+4)];
            P_new[(i+4)*8+j] = P[(i+4)*8+j] + P[(i+4)*8+(j+4)];
            P_new[(i+4)*8+(j+4)] = P[(i+4)*8+(j+4)];
        }
    }
    
    float Q[64];
    get_Q(x[3], Q);
    for (int i = 0; i < 64; ++i) {
        P[i] = P_new[i] + Q[i];
    }
}

// 卡爾曼更新步
__device__ __forceinline__ void update(float x[8], float P[64], const float z[4], float light_factor = 0.0f) {
    // 1. S = H * P * H^T + R
    // 由於 H = [I, 0]，H*P*H^T 就是 P 的左上 4x4
    float R[16];
    get_R(x[3], R, light_factor);
    
    float S[16];
    for(int i = 0; i < 4; ++i) {
        for(int j = 0; j < 4; ++j) {
            S[i*4+j] = P[i*8+j] + R[i*4+j];
        }
    }
    
    // 2. S_inv = inv(S)
    float S_inv[16];
    invert4x4(S, S_inv);
    
    // 3. K = P * H^T * S_inv
    // P * H^T 就是 P 的左邊 8x4
    float K[32]; // 8x4
    for(int i = 0; i < 8; ++i) {
        for(int j = 0; j < 4; ++j) {
            float sum = 0.0f;
            for(int k = 0; k < 4; ++k) {
                sum += P[i*8+k] * S_inv[k*4+j];
            }
            K[i*4+j] = sum;
        }
    }
    
    // 4. y = z - H * x
    float y[4];
    y[0] = z[0] - x[0];
    y[1] = z[1] - x[1];
    y[2] = z[2] - x[2];
    y[3] = z[3] - x[3];
    
    // 5. x = x + K * y
    for(int i = 0; i < 8; ++i) {
        float sum = 0.0f;
        for(int j = 0; j < 4; ++j) {
            sum += K[i*4+j] * y[j];
        }
        x[i] += sum;
    }
    
    // 6. P = (I - K * H) * P = P - K * (H * P)
    // H * P 就是 P 的上半 4x8
    float P_new[64];
    for(int i = 0; i < 8; ++i) {
        for(int j = 0; j < 8; ++j) {
            float sum = 0.0f;
            for(int k = 0; k < 4; ++k) {
                sum += K[i*4+k] * P[k*8+j];
            }
            P_new[i*8+j] = P[i*8+j] - sum;
        }
    }
    
    for(int i = 0; i < 64; ++i) P[i] = P_new[i];
}

} // namespace kf_gpu
} // namespace saccade