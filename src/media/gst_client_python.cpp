#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "media/gst_client.hpp"

namespace py = pybind11;
using namespace saccade;

PYBIND11_MODULE(saccade_media_ext, m) {
    m.doc() = "Saccade GStreamer Media Client with GPU Buffer Pool (Python Bindings)";

    py::class_<FrameData>(m, "FrameData")
        .def_property_readonly("cuda_ptr", [](const FrameData& self) {
            return reinterpret_cast<uintptr_t>(self.cuda_ptr);
        }, "Pointer to GPU memory as an integer address")
        .def_readonly("width", &FrameData::width)
        .def_readonly("height", &FrameData::height)
        .def_readonly("channels", &FrameData::channels)
        .def_readonly("timestamp", &FrameData::timestamp);

    py::class_<GstClient>(m, "GstClient")
        .def(py::init<const std::string&>())
        .def("connect", &GstClient::connect)
        .def("release", &GstClient::release)
        .def("set_frame_callback", &GstClient::setFrameCallback, 
             "Set a Python function to be called on every frame arrival.");
}
