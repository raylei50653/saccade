#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tracking/seq_runner.hpp"
#include "tracking/eval_pool.hpp"
#include "tracking/pipeline.hpp"
#include "perception/trt_engine.hpp"

namespace py = pybind11;
using namespace saccade;

// Helper: convert a list of FrameResult to a dict of numpy arrays
static py::dict frame_results_to_py(const std::vector<FrameResult>& results) {
    // Collect all tracks as flat arrays: frame_id, track_id, x1, y1, x2, y2, score
    size_t total = 0;
    for (const auto& r : results) total += r.track_ids.size();

    py::array_t<float> out_boxes({(ssize_t)total, (ssize_t)4});
    py::array_t<int>   out_ids({(ssize_t)total});
    py::array_t<int>   out_frames({(ssize_t)total});
    py::array_t<float> out_scores({(ssize_t)total});

    auto b = out_boxes.mutable_unchecked<2>();
    auto id = out_ids.mutable_unchecked<1>();
    auto fr = out_frames.mutable_unchecked<1>();
    auto sc = out_scores.mutable_unchecked<1>();

    size_t k = 0;
    for (const auto& r : results) {
        for (size_t j = 0; j < r.track_ids.size(); ++j, ++k) {
            fr(k)    = r.frame_id;
            id(k)    = r.track_ids[j];
            b(k, 0)  = r.boxes[j][0];
            b(k, 1)  = r.boxes[j][1];
            b(k, 2)  = r.boxes[j][2];
            b(k, 3)  = r.boxes[j][3];
            sc(k)    = r.scores[j];
        }
    }

    return py::dict(
        py::arg("frame_ids") = out_frames,
        py::arg("track_ids") = out_ids,
        py::arg("boxes")     = out_boxes,
        py::arg("scores")    = out_scores
    );
}

PYBIND11_MODULE(saccade_eval_ext, m) {
    m.doc() = "C++ multi-threaded evaluation pool (saccade_eval_ext)";

    // ── SequenceConfig ────────────────────────────────────────────────────────
    py::class_<SequenceConfig>(m, "CppSequenceConfig")
        .def(py::init<>())
        .def_readwrite("name",               &SequenceConfig::name)
        .def_readwrite("frame_paths",        &SequenceConfig::frame_paths)
        .def_readwrite("width",              &SequenceConfig::width)
        .def_readwrite("height",             &SequenceConfig::height)
        .def_readwrite("w_aspect",           &SequenceConfig::w_aspect)
        .def_readwrite("w_center",           &SequenceConfig::w_center)
        .def_readwrite("w_area",             &SequenceConfig::w_area)
        .def_readwrite("fp_hard_filter",     &SequenceConfig::fp_hard_filter)
        .def_readwrite("fp_min_score",       &SequenceConfig::fp_min_score)
        .def_readwrite("fp_max_area",        &SequenceConfig::fp_max_area)
        .def_readwrite("fp_max_susp_score",  &SequenceConfig::fp_max_susp_score)
        .def_readwrite("narrow_bonus",       &SequenceConfig::narrow_bonus)
        .def_readwrite("narrow_aspect_thresh", &SequenceConfig::narrow_aspect_thresh)
        .def_readwrite("narrow_height_thresh", &SequenceConfig::narrow_height_thresh)
        .def_readwrite("person_class",       &SequenceConfig::person_class)
        .def_readwrite("trt_input_size",     &SequenceConfig::trt_input_size)
        .def_readwrite("max_raw_dets",       &SequenceConfig::max_raw_dets)
        // Tracker params
        .def_readwrite("track_thresh",           &SequenceConfig::track_thresh)
        .def_readwrite("high_thresh",            &SequenceConfig::high_thresh)
        .def_readwrite("match_thresh",           &SequenceConfig::match_thresh)
        .def_readwrite("new_track_thresh",       &SequenceConfig::new_track_thresh)
        .def_readwrite("mid_thresh",             &SequenceConfig::mid_thresh)
        .def_readwrite("confirm_streak",         &SequenceConfig::confirm_streak)
        .def_readwrite("confirm_score_thresh",   &SequenceConfig::confirm_score_thresh)
        .def_readwrite("fuse_score_weight",      &SequenceConfig::fuse_score_weight)
        .def_readwrite("vel_dir_weight",         &SequenceConfig::vel_dir_weight)
        .def_readwrite("stage2_match_thresh",    &SequenceConfig::stage2_match_thresh)
        .def_readwrite("birth_low_score_thresh", &SequenceConfig::birth_low_score_thresh)
        .def_readwrite("track_buffer",           &SequenceConfig::track_buffer)
        // GMC
        .def_readwrite("gmc_enabled",    &SequenceConfig::gmc_enabled)
        .def_readwrite("gmc_downscale",  &SequenceConfig::gmc_downscale)
        .def_readwrite("gmc_phase_corr", &SequenceConfig::gmc_phase_corr)
        // ReID
        .def_readwrite("reid_engine_path", &SequenceConfig::reid_engine_path)
        .def_readwrite("reid_model_type",  &SequenceConfig::reid_model_type)
        .def_readwrite("reid_budget",      &SequenceConfig::reid_budget)
        .def_readwrite("reid_interval",    &SequenceConfig::reid_interval)
        .def_readwrite("reid_crop_h",      &SequenceConfig::reid_crop_h)
        .def_readwrite("reid_crop_w",      &SequenceConfig::reid_crop_w);

    // ── SequenceRunner ────────────────────────────────────────────────────────
    py::class_<SequenceRunner>(m, "CppSequenceRunner")
        .def(py::init([](uintptr_t detect_engine_ptr,
                         const PerceptionPipeline::Config& pipe_cfg,
                         int max_dets, int max_tracks, int device_id) {
                 return new SequenceRunner(
                     reinterpret_cast<TRTEngine*>(detect_engine_ptr),
                     pipe_cfg, max_dets, max_tracks, device_id);
             }),
             py::arg("detect_engine_ptr"),
             py::arg("pipe_cfg"),
             py::arg("max_dets")   = 2048,
             py::arg("max_tracks") = 256,
             py::arg("device_id")  = 0)
        .def("run",
             [](SequenceRunner& self, const SequenceConfig& cfg) -> py::dict {
                 py::gil_scoped_release release;
                 auto frame_results = self.run(cfg);
                 py::gil_scoped_acquire acquire;
                 return frame_results_to_py(frame_results);
             },
             py::arg("cfg"),
             "Run one sequence. Returns dict with frame_ids, track_ids, boxes, scores.");

    // ── EvaluatorPool ─────────────────────────────────────────────────────────
    py::class_<EvaluatorPool>(m, "CppEvaluatorPool")
        .def(py::init([](uintptr_t detect_engine_ptr,
                         const PerceptionPipeline::Config& pipe_cfg,
                         int n_threads, int max_dets, int max_tracks, int device_id) {
                 return new EvaluatorPool(
                     reinterpret_cast<TRTEngine*>(detect_engine_ptr),
                     pipe_cfg, n_threads, max_dets, max_tracks, device_id);
             }),
             py::arg("detect_engine_ptr"),
             py::arg("pipe_cfg"),
             py::arg("n_threads")  = 4,
             py::arg("max_dets")   = 2048,
             py::arg("max_tracks") = 256,
             py::arg("device_id")  = 0)
        .def("run_sequences",
             [](EvaluatorPool& self,
                const std::vector<SequenceConfig>& seqs) -> py::dict {
                 // Release GIL for the entire multi-sequence evaluation
                 py::gil_scoped_release release;
                 auto cpp_results = self.run_sequences(seqs);
                 py::gil_scoped_acquire acquire;

                 py::dict out;
                 for (auto& [name, frame_results] : cpp_results) {
                     out[py::str(name)] = frame_results_to_py(frame_results);
                 }
                 return out;
             },
             py::arg("seqs"),
             "Run all sequences in parallel (GIL released for entire call). "
             "Returns dict[seq_name, dict{frame_ids, track_ids, boxes, scores}].");
}
