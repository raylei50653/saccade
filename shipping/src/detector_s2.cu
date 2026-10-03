// Native S2 kernels (#465 Phase B PR-8). See saccade_shipping/detector_s2.hpp.
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "saccade_shipping/detector_s2.hpp"

namespace saccade::shipping {
namespace {

void launch_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("shipping detector S2: ") + what + ": " +
                                 cudaGetErrorString(e));
    }
}

// tl.sigmoid as Inductor's kernel runs it (see the header): the same PTX
// instructions, one asm statement each.
__device__ __forceinline__ float inductor_sigmoid(float x) {
    const float zero = 0.0f;
    const float one = 1.0f;
    float t, e, d, s;
    asm("sub.f32 %0, %1, %2;" : "=f"(t) : "f"(zero), "f"(x));
    asm("mul.f32 %0, %1, 0f3FB8AA3B;" : "=f"(t) : "f"(t));
    asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
    asm("add.f32 %0, %1, 0f3F800000;" : "=f"(d) : "f"(e));
    asm("div.full.f32 %0, %1, %2;" : "=f"(s) : "f"(one), "f"(d));
    return s;
}

// triton_helpers.maximum(a, b): a if (a > b or a is NaN) else b.
__device__ __forceinline__ float triton_maximum(float a, float b) {
    return (a > b || a != a) ? a : b;
}

// triton_helpers.maximum_with_index((a, ai), (b, bi)): NaN is the largest
// value, NaNs are equal to each other, equal values keep the lower index.
__device__ __forceinline__ void triton_maximum_with_index(float& a, int& ai, float b, int bi) {
    const bool a_nan = a != a;
    const bool b_nan = b != b;
    bool mask = a > b || (a_nan && !b_nan);
    const bool equal = a == b || (a_nan && b_nan);
    mask = mask || (equal && ai < bi);
    if (!mask) {
        a = b;
        ai = bi;
    }
}

struct Levels {
    S2Level l[3];
    int base1, base2, total;  // first anchor of levels 1 and 2; anchor count
};

__device__ __forceinline__ int level_of(const Levels& L, int a, int& local) {
    if (a < L.base1) {
        local = a;
        return 0;
    }
    if (a < L.base2) {
        local = a - L.base1;
        return 1;
    }
    local = a - L.base2;
    return 2;
}

__global__ void score_max_kernel(Levels L, int nc, float* __restrict__ scores_max,
                                 std::int64_t* __restrict__ class_idx) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= L.total) return;
    int local = 0;
    const S2Level& lv = L.l[level_of(L, a, local)];
    const int plane = lv.side * lv.side;
    float vmax = -INFINITY;  // `maximum` accumulator (initial -inf, as Inductor's)
    float wv = -INFINITY;    // `maximum_with_index` accumulator
    int wi = 2147483647;
    for (int r = 0; r < nc; ++r) {
        const float s = inductor_sigmoid(lv.cls[static_cast<std::size_t>(r) * plane + local]);
        vmax = triton_maximum(vmax, s);
        triton_maximum_with_index(wv, wi, s, r);
    }
    scores_max[a] = vmax;
    class_idx[a] = wi;
}

__global__ void decode_gather_kernel(Levels L, const std::int64_t* __restrict__ topk_idx,
                                     const float* __restrict__ topk_score,
                                     const std::int64_t* __restrict__ class_idx, int k, float sx,
                                     float sy, float* __restrict__ raw, float* __restrict__ scaled) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    const int a = static_cast<int>(topk_idx[j]);
    int local = 0;
    const S2Level& lv = L.l[level_of(L, a, local)];
    const int plane = lv.side * lv.side;
    // _precompute_anchor_grid: arange(side) + 0.5 (exact in float32).
    const float ax = __fadd_rn(static_cast<float>(local % lv.side), 0.5f);
    const float ay = __fadd_rn(static_cast<float>(local / lv.side), 0.5f);
    // reg_max 1: the distances are the head's reg channels (l, t, r, b).
    const float l = lv.reg[local];
    const float t = lv.reg[plane + local];
    const float r = lv.reg[2 * plane + local];
    const float b = lv.reg[3 * plane + local];
    // _dist2bbox_xywh (ultralytics 8.4.37 order).
    const float x1a = __fsub_rn(ax, l), y1a = __fsub_rn(ay, t);
    const float x2a = __fadd_rn(ax, r), y2a = __fadd_rn(ay, b);
    const float cx = __fmul_rn(__fadd_rn(x1a, x2a), 0.5f);
    const float cy = __fmul_rn(__fadd_rn(y1a, y2a), 0.5f);
    const float w = __fsub_rn(x2a, x1a), h = __fsub_rn(y2a, y1a);
    // bboxes * strides_t, then xy -/+ wh / 2.
    const float cxs = __fmul_rn(cx, lv.stride), cys = __fmul_rn(cy, lv.stride);
    const float hw = __fmul_rn(__fmul_rn(w, lv.stride), 0.5f);
    const float hh = __fmul_rn(__fmul_rn(h, lv.stride), 0.5f);
    const float x1 = __fsub_rn(cxs, hw), y1 = __fsub_rn(cys, hh);
    const float x2 = __fadd_rn(cxs, hw), y2 = __fadd_rn(cys, hh);
    // results[b, :, 5] = class_ids[b][topk_idx].float()
    const float cls = static_cast<float>(class_idx[a]);
    float* o = raw + static_cast<std::size_t>(j) * 6;
    o[0] = x1;
    o[1] = y1;
    o[2] = x2;
    o[3] = y2;
    o[4] = topk_score[j];
    o[5] = cls;
    // _whole_graph_fn: detections[:, :, [0, 2]] *= sx; [:, :, [1, 3]] *= sy.
    float* q = scaled + static_cast<std::size_t>(j) * 6;
    q[0] = __fmul_rn(x1, sx);
    q[1] = __fmul_rn(y1, sy);
    q[2] = __fmul_rn(x2, sx);
    q[3] = __fmul_rn(y2, sy);
    q[4] = topk_score[j];
    q[5] = cls;
}

Levels make_levels(const S2Level levels[3]) {
    Levels L{};
    for (int i = 0; i < 3; ++i) {
        if (levels[i].side <= 0 || levels[i].cls == nullptr || levels[i].reg == nullptr) {
            throw std::invalid_argument("shipping detector S2: bad level");
        }
        L.l[i] = levels[i];
    }
    L.base1 = levels[0].side * levels[0].side;
    L.base2 = L.base1 + levels[1].side * levels[1].side;
    L.total = L.base2 + levels[2].side * levels[2].side;
    return L;
}

}  // namespace

void s2_score_max(const S2Level levels[3], int num_classes, float* scores_max,
                  std::int64_t* class_idx, cudaStream_t stream) {
    if (num_classes <= 0) throw std::invalid_argument("shipping detector S2: num_classes <= 0");
    const Levels L = make_levels(levels);
    constexpr int kThreads = 128;
    score_max_kernel<<<(L.total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(L, num_classes,
                                                                                   scores_max, class_idx);
    launch_check("score_max");
}

void s2_decode_gather(const S2Level levels[3], const std::int64_t* topk_idx, const float* topk_score,
                      const std::int64_t* class_idx, int k, float sx, float sy, float* raw,
                      float* scaled, cudaStream_t stream) {
    if (k <= 0) return;
    const Levels L = make_levels(levels);
    constexpr int kThreads = 128;
    decode_gather_kernel<<<(k + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        L, topk_idx, topk_score, class_idx, k, sx, sy, raw, scaled);
    launch_check("decode_gather");
}

}  // namespace saccade::shipping
