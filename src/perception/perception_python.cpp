#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "perception/trt_engine.hpp"
#include "perception/preprocessor.hpp"
#include "perception/feature_extractor.hpp"
#include <cuda_runtime_api.h>

namespace py = pybind11;

namespace saccade {

/**
 * @brief 為感知模組提供 Python 綁定
 */
void init_perception_ext(py::module &m) {
    py::class_<IPerceptionEngine>(m, "IPerceptionEngine");

    py::class_<TRTEngine, IPerceptionEngine>(m, "TRTEngine")
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
        .def("set_input_shape", &TRTEngine::set_input_shape, py::arg("name"), py::arg("shape"));

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
        .def_property_readonly("feature_dim", &FeatureExtractor::get_feature_dim)
        .def_property_readonly("max_batch", &FeatureExtractor::get_max_batch)
        .def_property_readonly("input_hw", &FeatureExtractor::get_input_hw)
        .def_property_readonly("cpp_ptr", [](FeatureExtractor& self) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(&self);
        });
}

PYBIND11_MODULE(saccade_perception_ext, m) {
    m.doc() = "Saccade Perception C++ Extension";
    init_perception_ext(m);
}

} // namespace saccade
