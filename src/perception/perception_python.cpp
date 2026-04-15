#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "perception/trt_engine.hpp"
#include <cuda_runtime_api.h>

namespace py = pybind11;

namespace saccade {

/**
 * @brief 為 TRTEngine 提供 Python 綁定
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
        .def("get_tensor_shape", [](TRTEngine &self, const std::string &name) {
            auto dims = self.getTensorDims(name.c_str());
            std::vector<int64_t> shape;
            for (int i = 0; i < dims.nbDims; ++i) {
                shape.push_back(dims.d[i]);
            }
            return shape;
        }, py::arg("name"));
}

PYBIND11_MODULE(saccade_perception_ext, m) {
    m.doc() = "Saccade Perception C++ Extension";
    init_perception_ext(m);
}

} // namespace saccade
