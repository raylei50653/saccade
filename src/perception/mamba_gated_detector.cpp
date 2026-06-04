#include "perception/mamba_gated_detector.hpp"
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <stdexcept>
#include <mutex>

namespace saccade {

namespace {

std::mutex g_mamba_head_mutex;

torch::Tensor nms_gpu(torch::Tensor bboxes, torch::Tensor scores, float iou_threshold) {
    if (bboxes.size(0) == 0) {
        return torch::empty({0}, torch::TensorOptions().dtype(torch::kLong).device(bboxes.device()));
    }

    auto x1 = bboxes.select(1, 0);
    auto y1 = bboxes.select(1, 1);
    auto x2 = bboxes.select(1, 2);
    auto y2 = bboxes.select(1, 3);
    auto areas = (x2 - x1) * (y2 - y1);

    auto order = scores.argsort(-1, true); // descending sort order

    std::vector<int64_t> keep;
    while (order.numel() > 0) {
        int64_t i = order[0].item<int64_t>();
        keep.push_back(i);

        if (order.numel() == 1) {
            break;
        }

        auto idx = order.slice(0, 1, order.numel()); // remaining indices
        
        auto xx1 = torch::clamp_min(x1.index({idx}), x1[i]);
        auto yy1 = torch::clamp_min(y1.index({idx}), y1[i]);
        auto xx2 = torch::clamp_max(x2.index({idx}), x2[i]);
        auto yy2 = torch::clamp_max(y2.index({idx}), y2[i]);

        auto w = torch::clamp_min(xx2 - xx1, 0.0f);
        auto h = torch::clamp_min(yy2 - yy1, 0.0f);
        auto inter = w * h;

        auto ovr = inter / (areas[i] + areas.index({idx}) - inter);
        auto mask = ovr <= iou_threshold;
        
        if (!mask.any().item<bool>()) {
            break;
        }
        order = idx.index({mask});
    }

    return torch::tensor(keep, torch::TensorOptions().dtype(torch::kLong).device(bboxes.device()));
}

struct ThreadState {
    nvinfer1::IExecutionContext* ctx = nullptr;
    cudaStream_t stream = nullptr;
    void* d_p3 = nullptr;
    void* d_p4 = nullptr;
    void* d_p5 = nullptr;
    torch::jit::script::Module mamba_head;
    bool has_mamba_head = false;

    ~ThreadState() {
        if (d_p3) cudaFree(d_p3);
        if (d_p4) cudaFree(d_p4);
        if (d_p5) cudaFree(d_p5);
        if (stream) cudaStreamDestroy(stream);
        if (ctx) TRTEngine::delete_context(ctx);
    }
};

ThreadState& get_thread_state(TRTEngine* backbone, const std::string& mamba_head_path, int p3_bytes, int p4_bytes, int p5_bytes) {
    static thread_local ThreadState state;
    if (!state.ctx) {
        state.ctx = backbone->create_context();
        if (!state.ctx) {
            throw std::runtime_error("MambaGatedDetector: Failed to create TRT execution context for thread");
        }
        cudaStreamCreate(&state.stream);
        cudaMalloc(&state.d_p3, p3_bytes);
        cudaMalloc(&state.d_p4, p4_bytes);
        cudaMalloc(&state.d_p5, p5_bytes);
    }
    if (!state.has_mamba_head) {
        try {
            state.mamba_head = torch::jit::load(mamba_head_path);
            state.mamba_head.to(torch::kCUDA);
            state.mamba_head.eval();
            state.has_mamba_head = true;
        } catch (const c10::Error& e) {
            throw std::runtime_error("MambaGatedDetector: Failed to load TorchScript module on thread: " + std::string(e.what()));
        }
    }
    return state;
}

} // namespace

MambaGatedDetector::MambaGatedDetector(
    const std::string& trt_backbone_path,
    const std::string& mamba_head_script_path,
    int img_size,
    float conf_thr
) : mamba_head_script_path_(mamba_head_script_path), img_size_(img_size), conf_thr_(conf_thr) {
    
    // Force LibTorch to initialize CUDA, cuBLAS, and its internal handlers
    auto dummy_x = torch::zeros({1, 1}, torch::TensorOptions().device(torch::kCUDA));
    auto dummy_w = torch::zeros({1, 1}, torch::TensorOptions().device(torch::kCUDA));
    auto dummy_y = torch::matmul(dummy_x, dummy_w);

    // 1. Create TRT engine instance
    backbone_ = std::make_unique<TRTEngine>(trt_backbone_path);

    // 2. Load JIT TorchScript model for Mamba SSM head (on main thread to catch early errors)
    try {
        mamba_head_ = torch::jit::load(mamba_head_script_path);
        mamba_head_.to(torch::kCUDA);
        mamba_head_.eval();
    } catch (const c10::Error& e) {
        throw std::runtime_error("MambaGatedDetector: Failed to load TorchScript module: " + std::string(e.what()));
    }
}

MambaGatedDetector::~MambaGatedDetector() = default;

torch::Tensor MambaGatedDetector::forward(torch::Tensor d_input_img) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto current_stream = at::cuda::getCurrentCUDAStream();

    // Get thread-local state
    auto& state = get_thread_state(
        backbone_.get(),
        mamba_head_script_path_,
        1 * p3_c_ * p3_h_ * p3_w_ * sizeof(float),
        1 * p4_c_ * p4_h_ * p4_w_ * sizeof(float),
        1 * p5_c_ * p5_h_ * p5_w_ * sizeof(float)
    );

    // Ensure the input tensor is contiguous
    auto img_contiguous = d_input_img.contiguous();

    // 1. Execute TensorRT backbone directly on the current PyTorch stream
    backbone_->infer_with_context(state.ctx, {img_contiguous.data_ptr<float>(), state.d_p3, state.d_p4, state.d_p5}, current_stream.stream());

    // 4. Wrap pre-allocated device memory as LibTorch Tensors and run head/decode.
    return forward_from_feats(
        torch::from_blob(state.d_p3, {1, p3_c_, p3_h_, p3_w_}, options),
        torch::from_blob(state.d_p4, {1, p4_c_, p4_h_, p4_w_}, options),
        torch::from_blob(state.d_p5, {1, p5_c_, p5_h_, p5_w_}, options)
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
MambaGatedDetector::decode_feats(torch::Tensor p3, torch::Tensor p4, torch::Tensor p5) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    // Thread-local state owns this thread's TorchScript head instance (avoids
    // cross-thread contention on a shared jit::Module during concurrent calls).
    auto& state = get_thread_state(
        backbone_.get(),
        mamba_head_script_path_,
        1 * p3_c_ * p3_h_ * p3_w_ * sizeof(float),
        1 * p4_c_ * p4_h_ * p4_w_ * sizeof(float),
        1 * p5_c_ * p5_h_ * p5_w_ * sizeof(float)
    );

    std::vector<torch::Tensor> feats = {p3, p4, p5};

    // 5. Run TorchScript Mamba SSM Head Wrapper
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(feats);

    auto outputs = state.mamba_head.forward(inputs).toTuple();
    auto cls_preds = outputs->elements()[0].toTensorList().vec();
    auto reg_preds = outputs->elements()[1].toTensorList().vec();

    // 6. Anchor generation & box decoding in LibTorch C++
    std::vector<float> strides_vec = {8.0f, 16.0f, 32.0f};
    std::vector<torch::Tensor> anchor_points;
    std::vector<torch::Tensor> stride_tensor;

    std::vector<torch::Tensor> cls_flat, reg_flat;
    for (size_t i = 0; i < cls_preds.size(); ++i) {
        cls_flat.push_back(cls_preds[i].flatten(2));
        reg_flat.push_back(reg_preds[i].flatten(2));

        int h = cls_preds[i].size(2);
        int w = cls_preds[i].size(3);
        float stride = strides_vec[i];

        auto sx = torch::arange(w, options) + 0.5f;
        auto sy = torch::arange(h, options) + 0.5f;
        auto mesh = torch::meshgrid({sy, sx}, "ij");
        auto anchor = torch::stack({mesh[1], mesh[0]}, -1).view({-1, 2});

        anchor_points.push_back(anchor);
        stride_tensor.push_back(torch::full({h * w, 1}, stride, options));
    }

    auto cls_all = torch::cat(cls_flat, 2); // [1, num_classes, N]
    auto reg_all = torch::cat(reg_flat, 2); // [1, 4, N]

    auto anchors = torch::cat(anchor_points, 0); // [N, 2]
    auto anchor_strides = torch::cat(stride_tensor, 0); // [N, 1]

    // 7. dist2bbox & scaling
    auto lt = reg_all.slice(1, 0, 2); // [1, 2, N]
    auto rb = reg_all.slice(1, 2, 4); // [1, 2, N]
    auto anchor_pts = anchors.transpose(0, 1).unsqueeze(0); // [1, 2, N]

    auto c_xy = anchor_pts + (rb - lt) / 2.0f;
    auto h_w = lt + rb;
    auto bboxes = torch::cat({c_xy, h_w}, 1); // [1, 4, N] (xywh)

    auto strides_t = anchor_strides.squeeze(-1).unsqueeze(0).unsqueeze(1); // [1, 1, N]
    bboxes = bboxes * strides_t;

    auto xywh = bboxes.permute({0, 2, 1}); // [1, N, 4]
    auto c_xy_final = xywh.slice(2, 0, 2);
    auto wh_final = xywh.slice(2, 2, 4);
    auto x1y1 = c_xy_final - wh_final / 2.0f;
    auto x2y2 = c_xy_final + wh_final / 2.0f;
    auto boxes_xyxy = torch::cat({x1y1, x2y2}, -1).squeeze(0); // [N, 4]

    // 8. Per-anchor max class score + id
    auto scores = cls_all.sigmoid(); // [1, num_classes, N]
    auto max_scores_and_ids = scores.max(1);
    auto scores_max = std::get<0>(max_scores_and_ids).squeeze(0); // [N]
    auto class_ids = std::get<1>(max_scores_and_ids).squeeze(0).to(torch::kFloat32); // [N]

    return {boxes_xyxy, scores_max, class_ids};
}

torch::Tensor MambaGatedDetector::forward_from_feats(torch::Tensor p3, torch::Tensor p4, torch::Tensor p5) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto decoded = decode_feats(p3, p4, p5);
    auto boxes_xyxy = std::get<0>(decoded); // [N, 4]
    auto scores_max = std::get<1>(decoded); // [N]
    auto class_ids = std::get<2>(decoded);  // [N]

    auto mask = (scores_max >= conf_thr_) &
                ((boxes_xyxy.select(1, 2) - boxes_xyxy.select(1, 0)) > 0.0f) & 
                ((boxes_xyxy.select(1, 3) - boxes_xyxy.select(1, 1)) > 0.0f);
    
    // Guard against empty detections
    if (!mask.any().item<bool>()) {
        return torch::zeros({0, 6}, options);
    }

    auto final_scores = scores_max.index({mask});
    auto final_classes = class_ids.index({mask});
    auto final_boxes = boxes_xyxy.index({mask, torch::indexing::Slice()});

    // 9. GPU-based NMS (IoU threshold = 0.5f)
    auto keep = nms_gpu(final_boxes, final_scores, 0.5f);
    
    if (keep.size(0) == 0) {
        return torch::zeros({0, 6}, options);
    }
    
    auto nms_boxes = final_boxes.index({keep, torch::indexing::Slice()});
    auto nms_scores = final_scores.index({keep});
    auto nms_classes = final_classes.index({keep});

    return torch::cat({nms_boxes, nms_scores.unsqueeze(-1), nms_classes.unsqueeze(-1)}, 1);
}

torch::Tensor MambaGatedDetector::forward_feats_padded(
    torch::Tensor p3, torch::Tensor p4, torch::Tensor p5, float conf_thr, int max_det) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto decoded = decode_feats(p3, p4, p5);
    auto boxes_xyxy = std::get<0>(decoded); // [N, 4]
    auto scores_max = std::get<1>(decoded); // [N]
    auto class_ids = std::get<2>(decoded);  // [N]

    // Mirror Python _postprocess_mamba EXACTLY: conf-threshold + topk(max_det),
    // padded [max_det, 6], NO NMS and NO w/h validity filter. The downstream
    // run_eval postprocess applies the real confidence gate + NMS, so this entry
    // must stay a faithful drop-in for the Python serving head.
    auto results = torch::zeros({max_det, 6}, options);

    auto mask = scores_max >= conf_thr;
    auto s = scores_max.index({mask});
    auto c = class_ids.index({mask});
    auto bx = boxes_xyxy.index({mask, torch::indexing::Slice()});

    int64_t total = s.size(0);
    int64_t n = std::min<int64_t>(total, max_det);
    if (n > 0) {
        if (total > max_det) {
            auto topk = std::get<1>(s.topk(max_det));
            s = s.index({topk});
            c = c.index({topk});
            bx = bx.index({topk, torch::indexing::Slice()});
        }
        using torch::indexing::Slice;
        results.index_put_({Slice(0, n), Slice(0, 4)}, bx.slice(0, 0, n));
        results.index_put_({Slice(0, n), 4}, s.slice(0, 0, n));
        results.index_put_({Slice(0, n), 5}, c.slice(0, 0, n));
    }

    return results;
}

torch::Tensor MambaGatedDetector::extract_fpn_embeddings(torch::Tensor d_boxes_xyxy) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    
    if (d_boxes_xyxy.size(0) == 0) {
        return torch::zeros({0, get_fpn_dim()}, options);
    }

    // Get thread-local state
    auto& state = get_thread_state(
        backbone_.get(),
        mamba_head_script_path_,
        1 * p3_c_ * p3_h_ * p3_w_ * sizeof(float),
        1 * p4_c_ * p4_h_ * p4_w_ * sizeof(float),
        1 * p5_c_ * p5_h_ * p5_w_ * sizeof(float)
    );

    std::vector<torch::Tensor> feats = {
        torch::from_blob(state.d_p3, {1, p3_c_, p3_h_, p3_w_}, options),
        torch::from_blob(state.d_p4, {1, p4_c_, p4_h_, p4_w_}, options),
        torch::from_blob(state.d_p5, {1, p5_c_, p5_h_, p5_w_}, options)
    };

    std::vector<torch::Tensor> parts;
    auto cx = (d_boxes_xyxy.select(1, 0) + d_boxes_xyxy.select(1, 2)) * 0.5f;
    auto cy = (d_boxes_xyxy.select(1, 1) + d_boxes_xyxy.select(1, 3)) * 0.5f;

    for (const auto& f : feats) {
        int f_h = f.size(2);
        int f_w = f.size(3);

        auto cx_norm = cx / static_cast<float>(img_size_);
        auto cy_norm = cy / static_cast<float>(img_size_);

        auto cx_idx = (cx_norm * f_w).to(torch::kLong).clamp(0, f_w - 1);
        auto cy_idx = (cy_norm * f_h).to(torch::kLong).clamp(0, f_h - 1);

        auto center_feat = f.index({0, torch::indexing::Slice(), cy_idx, cx_idx}).transpose(0, 1);
        parts.push_back(center_feat);
    }

    auto concat_feats = torch::cat(parts, 1);
    return concat_feats / (concat_feats.norm(2, 1, true) + 1e-12f);
}

int MambaGatedDetector::forward_ptr(uintptr_t d_input_img, uintptr_t d_out_dets) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto input_tensor = torch::from_blob(reinterpret_cast<void*>(d_input_img), {1, 3, img_size_, img_size_}, options);
    
    auto detections = forward(input_tensor);
    int num_dets = detections.size(0);
    
    if (num_dets > 0) {
        auto current_stream = at::cuda::getCurrentCUDAStream();
        cudaMemcpyAsync(reinterpret_cast<void*>(d_out_dets), detections.data_ptr<float>(),
                        num_dets * 6 * sizeof(float), cudaMemcpyDeviceToDevice, current_stream.stream());
    }
    return num_dets;
}

int MambaGatedDetector::forward_from_feats_ptr(uintptr_t p3, uintptr_t p4, uintptr_t p5, uintptr_t d_out_dets) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto p3_t = torch::from_blob(reinterpret_cast<void*>(p3), {1, p3_c_, p3_h_, p3_w_}, options);
    auto p4_t = torch::from_blob(reinterpret_cast<void*>(p4), {1, p4_c_, p4_h_, p4_w_}, options);
    auto p5_t = torch::from_blob(reinterpret_cast<void*>(p5), {1, p5_c_, p5_h_, p5_w_}, options);

    auto detections = forward_from_feats(p3_t, p4_t, p5_t);
    int num_dets = detections.size(0);

    if (num_dets > 0) {
        auto current_stream = at::cuda::getCurrentCUDAStream();
        cudaMemcpyAsync(reinterpret_cast<void*>(d_out_dets), detections.data_ptr<float>(),
                        num_dets * 6 * sizeof(float), cudaMemcpyDeviceToDevice, current_stream.stream());
    }
    return num_dets;
}

void MambaGatedDetector::forward_feats_padded_ptr(
    uintptr_t p3, uintptr_t p4, uintptr_t p5, float conf_thr, int max_det, uintptr_t d_out_dets) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto p3_t = torch::from_blob(reinterpret_cast<void*>(p3), {1, p3_c_, p3_h_, p3_w_}, options);
    auto p4_t = torch::from_blob(reinterpret_cast<void*>(p4), {1, p4_c_, p4_h_, p4_w_}, options);
    auto p5_t = torch::from_blob(reinterpret_cast<void*>(p5), {1, p5_c_, p5_h_, p5_w_}, options);

    // [max_det, 6] padded (zeros for empty slots), matching Python serving head.
    auto results = forward_feats_padded(p3_t, p4_t, p5_t, conf_thr, max_det);

    auto current_stream = at::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(reinterpret_cast<void*>(d_out_dets), results.data_ptr<float>(),
                    static_cast<size_t>(max_det) * 6 * sizeof(float),
                    cudaMemcpyDeviceToDevice, current_stream.stream());
}

void MambaGatedDetector::extract_fpn_embeddings_ptr(uintptr_t d_boxes_xyxy, int num_dets, uintptr_t d_out_embs) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    
    if (num_dets == 0) return;
    
    auto boxes_tensor = torch::from_blob(reinterpret_cast<void*>(d_boxes_xyxy), {num_dets, 4}, options);
    auto embeddings = extract_fpn_embeddings(boxes_tensor);
    
    auto current_stream = at::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(reinterpret_cast<void*>(d_out_embs), embeddings.data_ptr<float>(),
                    num_dets * get_fpn_dim() * sizeof(float), cudaMemcpyDeviceToDevice, current_stream.stream());
}

} // namespace saccade
