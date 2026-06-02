#pragma once

#include "saccade/common.hpp"
#include "perception/base_detector.hpp"
#include "perception/trt_engine.hpp"
#include <torch/script.h>
#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>

namespace saccade {

/**
 * @brief High-performance MambaGatedDetector utilizing TensorRT for YOLO backbone 
 * and LibTorch for Mamba SSM detection head and anchor box post-processing.
 */
class SACCADE_PERCEPTION_API MambaGatedDetector : public BaseDetector {
public:
    MambaGatedDetector(
        const std::string& trt_backbone_path,
        const std::string& mamba_head_script_path,
        int img_size = 640,
        float conf_thr = 0.05f
    );
    
    ~MambaGatedDetector() override;

    /**
     * @brief Execute end-to-end GPU inference pipeline
     * @param d_input_img  GPU Float Tensor of shape [1, 3, 640, 640] normalized to [0, 1]
     * @return GPU Float Tensor of shape [num_valid_dets, 6] (xyxy, score, class)
     */
    torch::Tensor forward(torch::Tensor d_input_img) override;
    torch::Tensor extract_fpn_embeddings(torch::Tensor d_boxes_xyxy) override;

    /**
     * @brief Run only the Mamba SSM head + decode/NMS on pre-computed FPN feats.
     *
     * Skips the TRT backbone so callers that already have feats (e.g. a shared
     * batched backbone serving multiple streams) can run the head GIL-free.
     * @param p3 [1,128,80,80] / p4 [1,256,40,40] / p5 [1,512,20,20] CUDA float
     * @return GPU Float Tensor [num_valid_dets, 6] (xyxy, score, class)
     */
    torch::Tensor forward_from_feats(torch::Tensor p3, torch::Tensor p4, torch::Tensor p5);

    /**
     * @brief Head + decode on pre-computed feats, returning the *padded* serving
     * format: [max_det, 6] (xyxy, score, class), conf-thresholded + topk, NO NMS.
     *
     * Mirrors Python ``_postprocess_mamba`` exactly so it is a drop-in for the
     * multistream server's per-stream head (the downstream run_eval postprocess
     * applies the real confidence gate + NMS).
     */
    torch::Tensor forward_feats_padded(torch::Tensor p3, torch::Tensor p4, torch::Tensor p5,
                                       float conf_thr, int max_det);

    /**
     * @brief Raw pointer bindings for Python pybind11 integration
     */
    int forward_ptr(uintptr_t d_input_img, uintptr_t d_out_dets) override;
    int forward_from_feats_ptr(uintptr_t p3, uintptr_t p4, uintptr_t p5, uintptr_t d_out_dets);
    void forward_feats_padded_ptr(uintptr_t p3, uintptr_t p4, uintptr_t p5,
                                  float conf_thr, int max_det, uintptr_t d_out_dets);
    void extract_fpn_embeddings_ptr(uintptr_t d_boxes_xyxy, int num_dets, uintptr_t d_out_embs) override;

    int get_img_size() const override { return img_size_; }
    int get_fpn_dim() const override { return 128; } // Traced JDE reduces dimension to 128

private:
    // Run TorchScript head + anchor decode on feats; returns
    // (boxes_xyxy [N,4], scores_max [N], class_ids [N]). Shared by
    // forward_from_feats (NMS) and forward_feats_padded (serving format).
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    decode_feats(torch::Tensor p3, torch::Tensor p4, torch::Tensor p5);

    std::unique_ptr<TRTEngine> backbone_;
    torch::jit::script::Module mamba_head_;
    std::string mamba_head_script_path_;
    int img_size_;
    float conf_thr_;

    // Dimensions of FPN features
    const int p3_c_ = 128, p3_h_ = 80, p3_w_ = 80;
    const int p4_c_ = 256, p4_h_ = 40, p4_w_ = 40;
    const int p5_c_ = 512, p5_h_ = 20, p5_w_ = 20;
};

} // namespace saccade
