#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "media/gst_client.hpp"
#include <iostream>

namespace py = pybind11;
using namespace saccade;

PYBIND11_MODULE(saccade_media_ext, m) {
    m.doc() = "Saccade Industrial GStreamer Client with Resource-Aware Buffer Management";

    // 封裝 FrameData 為 Python Class
    py::class_<FrameData>(m, "FrameData")
        .def_property_readonly("cuda_ptr", [](const FrameData& self) {
            return reinterpret_cast<uintptr_t>(self.cuda_ptr);
        })
        .def_property_readonly("stream_ptr", [](const FrameData& self) {
            return reinterpret_cast<uintptr_t>(self.stream_ptr);
        })
        .def_readonly("buffer_index", &FrameData::buffer_index)
        .def_readonly("width", &FrameData::width)
        .def_readonly("height", &FrameData::height)
        .def_readonly("timestamp", &FrameData::timestamp)
        .def("mark_processing", [](FrameData& self) {
            auto* client_impl = static_cast<IMediaClient*>(self.owner_ptr);
            if (client_impl) client_impl->markProcessing(self.buffer_index);
        })
        .def("release", [](FrameData& self) {
            auto* client_impl = static_cast<IMediaClient*>(self.owner_ptr);
            if (client_impl) client_impl->releaseBuffer(self.buffer_index);
        })
        // 支援 Python with 語句 (RAII)
        .def("__enter__", [](FrameData& self) {
            auto* client_impl = static_cast<IMediaClient*>(self.owner_ptr);
            if (client_impl) client_impl->markProcessing(self.buffer_index);
            return self;
        })
        .def("__exit__", [](FrameData& self, py::object exc_type, py::object exc_value, py::object traceback) {
            auto* client_impl = static_cast<IMediaClient*>(self.owner_ptr);
            if (client_impl) client_impl->releaseBuffer(self.buffer_index);
        });

    py::class_<GstClient>(m, "GstClient")
        .def(py::init<const std::string&>())
        .def("connect", &GstClient::connect, py::call_guard<py::gil_scoped_release>())
        .def("release", &GstClient::release, py::call_guard<py::gil_scoped_release>())
        .def("sync_buffer", &GstClient::syncBuffer, py::call_guard<py::gil_scoped_release>())
        .def("set_frame_callback", [](GstClient& self, py::object cb) {
            // 封裝 C++ Callback，加入 Exception Handling
            self.setFrameCallback([cb](const FrameData& data) {
                py::gil_scoped_acquire acquire;
                try {
                    cb(data);
                } catch (py::error_already_set& e) {
                    std::cerr << "❌ [Python Callback Error] " << e.what() << std::endl;
                    // 如果 Python 報錯，確保 Buffer 狀態被重置回 EMPTY，否則會發生 Buffer Leak
                    auto* client_impl = static_cast<IMediaClient*>(data.owner_ptr);
                    if (client_impl) client_impl->releaseBuffer(data.buffer_index);
                }
            });
        }, py::arg("callback"));
}
