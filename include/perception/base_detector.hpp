#pragma once

#include "saccade/common.hpp"
#include <torch/torch.h>

namespace saccade {

/**
 * @brief Abstract Base Class for polymorphic detectors (TRT Only vs Hybrid PyTorch/Mamba).
 * Allows the multi-threaded EvaluatorPool to call either detector dynamically.
 */
class SACCADE_PERCEPTION_API BaseDetector {
public:
    virtual ~BaseDetector() = default;

    /**
     * @brief Execute end-to-end GPU inference pipeline
     * @param d_input_img  GPU Float Tensor of shape [1, 3, H, W] normalized to [0, 1]
     * @return GPU Float Tensor of shape [num_valid_dets, 6] (xyxy, score, class)
     */
    virtual torch::Tensor forward(torch::Tensor d_input_img) = 0;

    /**
     * @brief Extract appearance FPN embeddings for specified boxes
     * @param d_boxes_xyxy GPU Float Tensor of shape [N, 4] containing bounding box coordinates
     * @return GPU Float Tensor of shape [N, fpn_dim] containing L2-normalized embeddings
     */
    virtual torch::Tensor extract_fpn_embeddings(torch::Tensor d_boxes_xyxy) = 0;

    /**
     * @brief Raw pointer bindings for C++ multi-threaded/Python integration
     * @param d_input_img GPU Float buffer pointer [1, 3, H, W]
     * @param d_out_dets  GPU Float output buffer pointer [max_dets, 6]
     * @return Number of valid detections written
     */
    virtual int forward_ptr(uintptr_t d_input_img, uintptr_t d_out_dets) = 0;

    /**
     * @brief Raw pointer bindings for ReID embedding extraction
     */
    virtual void extract_fpn_embeddings_ptr(uintptr_t d_boxes_xyxy, int num_dets, uintptr_t d_out_embs) = 0;

    virtual int get_img_size() const = 0;
    virtual int get_fpn_dim() const = 0;
};

} // namespace saccade
