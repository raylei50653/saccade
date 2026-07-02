#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "perception/batched_mamba_detector.hpp"
#include "perception/trt_engine.hpp"
#include "perception/preprocessor.hpp"
#include "perception/feature_extractor.hpp"
#include "perception/mamba_gated_detector.hpp"
#include <cuda_runtime_api.h>

namespace py = pybind11;

namespace saccade {

/**
 * @brief 為感知模組提供 Python 綁定
 */
void init_perception_ext(py::module &m) {
    py::class_<IPerceptionEngine, std::shared_ptr<IPerceptionEngine>>(m, "IPerceptionEngine");

    py::class_<BaseDetector, std::shared_ptr<BaseDetector>>(m, "BaseDetector")
        .def("forward", &BaseDetector::forward, py::arg("d_input_img"))
        .def("extract_fpn_embeddings", &BaseDetector::extract_fpn_embeddings, py::arg("d_boxes_xyxy"))
        .def("forward_ptr", &BaseDetector::forward_ptr, py::arg("d_input_img"), py::arg("d_out_dets"))
        .def("extract_fpn_embeddings_ptr", &BaseDetector::extract_fpn_embeddings_ptr, py::arg("d_boxes_xyxy"), py::arg("num_dets"), py::arg("d_out_embs"))
        .def_property_readonly("img_size", &BaseDetector::get_img_size)
        .def_property_readonly("fpn_dim", &BaseDetector::get_fpn_dim)
        .def_property_readonly("cpp_ptr", [](BaseDetector& self) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(&self);
        });

    py::class_<TRTEngine, IPerceptionEngine, BaseDetector, std::shared_ptr<TRTEngine>>(m, "TRTEngine")
        .def(py::init<const std::string &>(), py::arg("model_path"))
        .def("infer", [](TRTEngine &self, const std::vector<size_t> &binding_ptrs, size_t stream_ptr) {
            std::vector<void*> bindings;
            for (auto ptr : binding_ptrs) {
                bindings.push_back(reinterpret_cast<void*>(ptr));
            }
            return self.infer(bindings, reinterpret_cast<cudaStream_t>(stream_ptr));
        }, py::arg("bindings"), py::arg("stream"))
        .def("set_tensor_address", [](TRTEngine &self, const std::string &name, uintptr_t ptr) {
            return self.set_tensor_address(name.c_str(), reinterpret_cast<void*>(ptr));
        }, py::arg("name"), py::arg("ptr"))
        .def("enqueue_v3", [](TRTEngine &self, size_t stream_ptr) {
            return self.enqueue_v3(reinterpret_cast<cudaStream_t>(stream_ptr));
        }, py::arg("stream"))
        .def("get_tensor_shape", [](TRTEngine &self, const std::string &name) {
            auto dims = self.getTensorDims(name.c_str());
            std::vector<int64_t> shape;
            for (int i = 0; i < dims.nbDims; ++i) {
                shape.push_back(dims.d[i]);
            }
            return shape;
        }, py::arg("name"))
        .def("set_input_shape", &TRTEngine::set_input_shape, py::arg("name"), py::arg("shape"))
        .def("create_context", [](TRTEngine &self) -> uintptr_t {
            auto* ctx = self.create_context();
            return reinterpret_cast<uintptr_t>(ctx);
        }, py::return_value_policy::reference_internal,
           "Create a new execution context for concurrent inference.")
        .def("infer_with_context", [](TRTEngine &self, uintptr_t ctx_ptr,
                                      const std::vector<size_t> &binding_ptrs,
                                      size_t stream_ptr) {
            auto* ctx = reinterpret_cast<nvinfer1::IExecutionContext*>(ctx_ptr);
            std::vector<void*> bindings;
            bindings.reserve(binding_ptrs.size());
            for (auto ptr : binding_ptrs) {
                bindings.push_back(reinterpret_cast<void*>(ptr));
            }
            return self.infer_with_context(ctx, bindings,
                                           reinterpret_cast<cudaStream_t>(stream_ptr));
        }, py::arg("ctx_ptr"), py::arg("bindings"), py::arg("stream"),
           "Run inference using a specific context (thread-safe).")
        .def_static("delete_context", [](uintptr_t ctx_ptr) {
            TRTEngine::delete_context(
                reinterpret_cast<nvinfer1::IExecutionContext*>(ctx_ptr));
        }, py::arg("ctx_ptr"),
           "Destroy a context created by create_context().")
        .def_property_readonly("cpp_ptr", [](TRTEngine& self) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(&self);
        }, "Raw pointer to the underlying C++ TRTEngine (for eval_pool etc.).");

    py::class_<Preprocessor>(m, "Preprocessor")
        .def(py::init<int, int>(), py::arg("target_width"), py::arg("target_height"))
        .def("process_gpu", [](Preprocessor &self, uintptr_t input_ptr, int src_w, int src_h, uintptr_t output_ptr, uintptr_t stream_ptr) {
            self.process_gpu(reinterpret_cast<void*>(input_ptr), src_w, src_h, reinterpret_cast<void*>(output_ptr), reinterpret_cast<cudaStream_t>(stream_ptr));
        }, py::arg("input_ptr"), py::arg("src_w"), py::arg("src_h"), py::arg("output_ptr"), py::arg("stream_ptr"));

    py::class_<Cropper>(m, "Cropper")
        .def(py::init<int, int>(), py::arg("crop_width"), py::arg("crop_height"))
        .def("process_gpu", [](Cropper &self, uintptr_t input_ptr, int src_w, int src_h, uintptr_t boxes_ptr, int num_boxes, uintptr_t output_ptr, uintptr_t stream_ptr) {
            self.process_gpu(reinterpret_cast<void*>(input_ptr), src_w, src_h, reinterpret_cast<float*>(boxes_ptr), num_boxes, reinterpret_cast<void*>(output_ptr), reinterpret_cast<cudaStream_t>(stream_ptr));
        }, py::arg("input_ptr"), py::arg("src_w"), py::arg("src_h"), py::arg("boxes_ptr"), py::arg("num_boxes"), py::arg("output_ptr"), py::arg("stream_ptr"))
        .def_property_readonly("cpp_ptr", [](Cropper& self) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(&self);
        });

    py::enum_<ModelType>(m, "ModelType")
        .value("SIGLIP2", ModelType::SIGLIP2)
        .value("DINOV2", ModelType::DINOV2)
        .value("TRANSREID", ModelType::TRANSREID)
        .value("OSNET", ModelType::OSNET)
        .value("FASTREID", ModelType::FASTREID)
        .value("MOBILENETV4_REID", ModelType::MOBILENETV4_REID)
        .export_values();

    py::class_<FeatureExtractor>(m, "FeatureExtractor")
        .def(py::init<const std::string&, ModelType, int>(), py::arg("model_path"), py::arg("type"), py::arg("max_batch") = 32)
        .def("extract", [](FeatureExtractor &self, uintptr_t input_ptr, int num, uintptr_t output_ptr, uintptr_t stream_ptr) {
            self.extract(reinterpret_cast<void*>(input_ptr), num, reinterpret_cast<void*>(output_ptr), reinterpret_cast<cudaStream_t>(stream_ptr));
        }, py::arg("input_ptr"), py::arg("num"), py::arg("output_ptr"), py::arg("stream_ptr"))
        .def("extract_parts_fused", [](FeatureExtractor &self, uintptr_t input_ptr, int num_dets, uintptr_t output_ptr, uintptr_t stream_ptr) {
            self.extract_parts_fused(reinterpret_cast<void*>(input_ptr), num_dets, reinterpret_cast<void*>(output_ptr), reinterpret_cast<cudaStream_t>(stream_ptr));
        }, py::arg("input_ptr"), py::arg("num_dets"), py::arg("output_ptr"), py::arg("stream_ptr"),
           "Extract 3-part crops, apply weighted fusion [0.5,0.3,0.2], and L2-normalize. "
           "Input: [3*num_dets, 3, H, W]. Output: [num_dets, feat_dim].")
        .def("extract_with_stability", [](
                FeatureExtractor &self,
                uintptr_t input_ptr, int num_images,
                uintptr_t embedding_ptr, uintptr_t stability_ptr,
                uintptr_t stream_ptr,
                float sigma_embed, float sigma_gate, float top_k_ratio) {
            self.extract_with_stability(
                reinterpret_cast<void*>(input_ptr), num_images,
                reinterpret_cast<void*>(embedding_ptr),
                reinterpret_cast<void*>(stability_ptr),
                reinterpret_cast<cudaStream_t>(stream_ptr),
                sigma_embed, sigma_gate, top_k_ratio);
        }, py::arg("input_ptr"), py::arg("num_images"),
           py::arg("embedding_ptr"), py::arg("stability_ptr"), py::arg("stream_ptr"),
           py::arg("sigma_embed") = 0.015f, py::arg("sigma_gate") = 0.040f,
           py::arg("top_k_ratio") = 0.5f,
           "LaSt-ViT refined extraction (SigLIP2 only). "
           "Returns L2-normalized embedding [N, feat_dim] and stability scores [N].")
        .def_property_readonly("feature_dim", &FeatureExtractor::get_feature_dim)
        .def_property_readonly("max_batch", &FeatureExtractor::get_max_batch)
        .def_property_readonly("input_hw", &FeatureExtractor::get_input_hw)
        .def_property_readonly("cpp_ptr", [](FeatureExtractor& self) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(&self);
        });

    py::class_<MambaGatedDetector, BaseDetector, std::shared_ptr<MambaGatedDetector>>(m, "MambaGatedDetector")
        .def(py::init<const std::string &, const std::string &, int, float>(),
             py::arg("trt_backbone_path"),
             py::arg("mamba_head_script_path"),
             py::arg("img_size") = 640,
             py::arg("conf_thr") = 0.05f)
        .def("forward_ptr", [](MambaGatedDetector &self, uintptr_t d_input_img, uintptr_t d_out_dets) {
            return self.forward_ptr(d_input_img, d_out_dets);
        }, py::arg("d_input_img"), py::arg("d_out_dets"),
           py::call_guard<py::gil_scoped_release>())
        .def("forward_from_feats_ptr", [](MambaGatedDetector &self, uintptr_t p3, uintptr_t p4, uintptr_t p5, uintptr_t d_out_dets) {
            return self.forward_from_feats_ptr(p3, p4, p5, d_out_dets);
        }, py::arg("p3"), py::arg("p4"), py::arg("p5"), py::arg("d_out_dets"),
           py::call_guard<py::gil_scoped_release>())
        .def("forward_feats_padded_ptr", [](MambaGatedDetector &self, uintptr_t p3, uintptr_t p4, uintptr_t p5, float conf_thr, int max_det, uintptr_t d_out_dets) {
            self.forward_feats_padded_ptr(p3, p4, p5, conf_thr, max_det, d_out_dets);
        }, py::arg("p3"), py::arg("p4"), py::arg("p5"), py::arg("conf_thr"), py::arg("max_det"), py::arg("d_out_dets"),
           py::call_guard<py::gil_scoped_release>())
        .def("extract_fpn_embeddings_ptr", [](MambaGatedDetector &self, uintptr_t d_boxes_xyxy, int num_dets, uintptr_t d_out_embs) {
            self.extract_fpn_embeddings_ptr(d_boxes_xyxy, num_dets, d_out_embs);
        }, py::arg("d_boxes_xyxy"), py::arg("num_dets"), py::arg("d_out_embs"),
           py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("img_size", &MambaGatedDetector::get_img_size)
        .def_property_readonly("fpn_dim", &MambaGatedDetector::get_fpn_dim)
        .def_property_readonly("cpp_ptr", [](MambaGatedDetector& self) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(&self);
        });

    // BatchedBackbone: pure TRT batch-N backbone, Mamba head stays in Python
    py::class_<BatchedBackbone>(m, "BatchedBackbone")
        .def(py::init<const std::string&, int, int>(),
             py::arg("trt_engine_path"),
             py::arg("max_batch") = 4,
             py::arg("img_size")  = 640)
        .def("batch_infer",
             &BatchedBackbone::batch_infer,
             py::arg("batch_frames"),
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("img_size",  &BatchedBackbone::get_img_size)
        .def_property_readonly("max_batch", &BatchedBackbone::get_max_batch);
}

PYBIND11_MODULE(saccade_perception_ext, m) {
    m.doc() = "Saccade Perception C++ Extension";
    init_perception_ext(m);
}

} // namespace saccade
