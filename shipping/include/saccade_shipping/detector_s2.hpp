// Native S2 kernels (#465 Phase B PR-8, U3b-2). CUDA runtime only, no torch.
//
// The oracle's S2 is `_postprocess_mamba_fixed`, torch.compile'd (mode
// "default"). With the headline shapes Inductor lowers it (torch 2.11.0+cu130,
// triton 3.6.0) to:
//
//   1. one reduction kernel over the 80 classes of every anchor: the class
//      logit x (level-concatenated) -> tl.sigmoid(x), compiled to the PTX
//        sub.f32 t, 0, x;  mul.f32 t, t, 0f3FB8AA3B;  ex2.approx.f32 e, t;
//        add.f32 d, e, 1.0;  div.full.f32 s, 1.0, d
//      and two reductions of s: `maximum` (value; NaN propagates) and
//      `maximum_with_index` (index; NaN counts as largest, ties take the lowest
//      index) -- both order-independent, so the result does not depend on the
//      autotuned block shape;
//   2. aten.topk(scores_max, max_det) (an ATen fallback, not a Triton kernel);
//   3. pointwise box kernels: x1y1 = a - lt, x2y2 = a + rb, c = (x1y1 + x2y2)
//      * 0.5, wh = x2y2 - x1y1, then (c * s) -/+ (wh * s) * 0.5. Inductor's
//      SASS contracts c * s into an FFMA, but with power-of-two strides every
//      product here is exact, so the value is that of the separate float32
//      ops below (which is also what eager computes);
//   4. a gather into (1, max_det, 6): box, top-k score, float(class index).
//
// `s2_score_max` is step 1 with the same PTX instructions (inline asm, so
// neither nvcc nor a fast-math flag can change them). `s2_decode_gather` is
// steps 3 + 4 for the selected anchors only, followed by the oracle's
// coordinate scaling (`_whole_graph_fn`: x *= sx, y *= sy), all with explicit
// round-to-nearest float32 operations (no contraction). Step 2 is the caller's
// (ATen, the oracle's own topk kernel).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace saccade::shipping {

// One pyramid level of the S2 input: the head's cls [nc, side, side] and
// reg [4, side, side] (batch 1, contiguous).
struct S2Level {
    const float* cls;
    const float* reg;
    int side;
    float stride;
};

// scores_max[i], class_idx[i] for i < anchors (level-concatenated, row-major
// within a level).
void s2_score_max(const S2Level levels[3], int num_classes, float* scores_max,
                  std::int64_t* class_idx, cudaStream_t stream);

// For j < k: anchor a = topk_idx[j]; raw[j] = (x1, y1, x2, y2, topk_score[j],
// float(class_idx[a])) in img_size space, scaled[j] = raw[j] with x *= sx,
// y *= sy (the S2 output at the detector boundary).
void s2_decode_gather(const S2Level levels[3], const std::int64_t* topk_idx, const float* topk_score,
                      const std::int64_t* class_idx, int k, float sx, float sy, float* raw,
                      float* scaled, cudaStream_t stream);

// The whole native S2 on device buffers: s2_score_max, ATen topk (top-k,
// largest, sorted -- the oracle's aten.topk), s2_decode_gather. `raw` and
// `scaled` hold k rows of 6. Defined in detector_host.cpp (it needs ATen); the
// detector host and its GPU test both call it.
void s2_run(const S2Level levels[3], int num_classes, int k, float sx, float sy, float* raw,
            float* scaled, cudaStream_t stream);

}  // namespace saccade::shipping
