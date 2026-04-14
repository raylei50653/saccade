#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tracking/tracker_gpu.hpp"
#include "tracking/smart_tracker.hpp"

namespace py = pybind11;
using namespace saccade;

PYBIND11_MODULE(saccade_tracking_ext, m) {
    m.doc() = "Saccade GPU Tracker (Python Bindings)";

    py::class_<TrackResult>(m, "TrackResult")
        .def_readonly("x1", &TrackResult::x1)
        .def_readonly("y1", &TrackResult::y1)
        .def_readonly("x2", &TrackResult::x2)
        .def_readonly("y2", &TrackResult::y2)
        .def_readonly("obj_id", &TrackResult::obj_id)
        .def_readonly("score", &TrackResult::score)
        .def_readonly("class_id", &TrackResult::class_id);

    py::class_<GPUByteTracker>(m, "GPUByteTracker")
        .def(py::init<int>(), py::arg("max_objects") = 2048)
        .def("update", [](GPUByteTracker& self, uintptr_t boxes_ptr, uintptr_t scores_ptr, uintptr_t classes_ptr, int num_dets, uintptr_t stream_ptr) {
            return self.update(
                reinterpret_cast<float*>(boxes_ptr),
                reinterpret_cast<float*>(scores_ptr),
                reinterpret_cast<int*>(classes_ptr),
                num_dets,
                reinterpret_cast<cudaStream_t>(stream_ptr)
            );
        }, 
        py::arg("boxes_ptr"), py::arg("scores_ptr"), py::arg("classes_ptr"), py::arg("num_dets"), py::arg("stream_ptr"),
        "Update tracker with raw GPU pointers and stream");

    py::class_<SmartTracker>(m, "SmartTracker")
        .def(py::init<float, float, int>(), py::arg("iou_threshold") = 0.7f, py::arg("velocity_angle_threshold") = 45.0f, py::arg("max_objects") = 2048)
        .def("update_and_filter", [](SmartTracker& self, uintptr_t ids_ptr, uintptr_t boxes_ptr, uintptr_t mask_ptr, int num_objs, uintptr_t stream_ptr) {
            self.update_and_filter(
                reinterpret_cast<const int*>(ids_ptr),
                reinterpret_cast<const float*>(boxes_ptr),
                reinterpret_cast<bool*>(mask_ptr),
                num_objs,
                reinterpret_cast<cudaStream_t>(stream_ptr)
            );
        },
        py::arg("ids_ptr"), py::arg("boxes_ptr"), py::arg("mask_ptr"), py::arg("num_objs"), py::arg("stream_ptr"),
        "Filter boxes for feature extraction using GPU");
}

