#include "tracking/cheb_gr_online.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <unordered_map>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(saccade_cheb_gr_online_ext, m) {
    m.doc() = "C++ causal Cheb-GR online handover";

    py::class_<saccade::MotRecord>(m, "MotRecord")
        .def(py::init<>())
        .def_readwrite("frame", &saccade::MotRecord::frame)
        .def_readwrite("track_id", &saccade::MotRecord::track_id)
        .def_readwrite("x", &saccade::MotRecord::x)
        .def_readwrite("y", &saccade::MotRecord::y)
        .def_readwrite("w", &saccade::MotRecord::w)
        .def_readwrite("h", &saccade::MotRecord::h)
        .def_readwrite("score", &saccade::MotRecord::score)
        .def_readwrite("tail", &saccade::MotRecord::tail);

    py::class_<saccade::OutputTracklet>(m, "OutputTracklet")
        .def(py::init<>())
        .def_readwrite("track_id", &saccade::OutputTracklet::track_id)
        .def_readwrite("start", &saccade::OutputTracklet::start)
        .def_readwrite("end", &saccade::OutputTracklet::end)
        .def_readwrite("mean_score", &saccade::OutputTracklet::mean_score)
        .def_property(
            "start_box",
            [](const saccade::OutputTracklet& t) {
                return std::vector<float>(t.start_box, t.start_box + 4);
            },
            [](saccade::OutputTracklet& t, const std::vector<float>& v) {
                for (int i = 0; i < 4; ++i) t.start_box[i] = v[i];
            })
        .def_property(
            "end_box",
            [](const saccade::OutputTracklet& t) {
                return std::vector<float>(t.end_box, t.end_box + 4);
            },
            [](saccade::OutputTracklet& t, const std::vector<float>& v) {
                for (int i = 0; i < 4; ++i) t.end_box[i] = v[i];
            })
        .def_property(
            "start_velocity",
            [](const saccade::OutputTracklet& t) {
                return std::vector<float>(t.start_velocity,
                                          t.start_velocity + 2);
            },
            [](saccade::OutputTracklet& t, const std::vector<float>& v) {
                for (int i = 0; i < 2; ++i) t.start_velocity[i] = v[i];
            })
        .def_property(
            "end_velocity",
            [](const saccade::OutputTracklet& t) {
                return std::vector<float>(t.end_velocity,
                                          t.end_velocity + 2);
            },
            [](saccade::OutputTracklet& t, const std::vector<float>& v) {
                for (int i = 0; i < 2; ++i) t.end_velocity[i] = v[i];
            });

    py::class_<saccade::HandoverStats>(m, "HandoverStats")
        .def(py::init<>())
        .def_readwrite("events", &saccade::HandoverStats::events)
        .def_readwrite("events_with_candidates",
                       &saccade::HandoverStats::events_with_candidates)
        .def_readwrite("handovers", &saccade::HandoverStats::handovers)
        .def_readwrite("reject_no_head",
                       &saccade::HandoverStats::reject_no_head)
        .def_readwrite("reject_min_head",
                       &saccade::HandoverStats::reject_min_head)
        .def_readwrite("reject_cost", &saccade::HandoverStats::reject_cost)
        .def_readwrite("reject_margin",
                       &saccade::HandoverStats::reject_margin)
        .def_readwrite("decisions_logged",
                       &saccade::HandoverStats::decisions_logged)
        .def_readwrite("ids_before", &saccade::HandoverStats::ids_before)
        .def_readwrite("ids_after", &saccade::HandoverStats::ids_after);

    py::class_<saccade::HandoverParams>(m, "HandoverParams")
        .def(py::init<>())
        .def_readwrite("enabled", &saccade::HandoverParams::enabled)
        .def_readwrite("max_cost", &saccade::HandoverParams::max_cost)
        .def_readwrite("max_gap", &saccade::HandoverParams::max_gap)
        .def_readwrite("decide_n", &saccade::HandoverParams::decide_n)
        .def_readwrite("min_head_samples",
                       &saccade::HandoverParams::min_head_samples)
        .def_readwrite("margin", &saccade::HandoverParams::margin)
        .def_readwrite("pool_frac", &saccade::HandoverParams::pool_frac)
        .def_readwrite("cheb_lambda", &saccade::HandoverParams::cheb_lambda)
        .def_readwrite("k2", &saccade::HandoverParams::k2)
        .def_readwrite("max_fwd", &saccade::HandoverParams::max_fwd)
        .def_readwrite("fuse_lambda", &saccade::HandoverParams::fuse_lambda);

    py::class_<saccade::HandoverDecision>(m, "HandoverDecision")
        .def(py::init<>())
        .def_readwrite("newborn_id", &saccade::HandoverDecision::newborn_id)
        .def_readwrite("newborn_start",
                       &saccade::HandoverDecision::newborn_start)
        .def_readwrite("newborn_end",
                       &saccade::HandoverDecision::newborn_end)
        .def_readwrite("candidate_id",
                       &saccade::HandoverDecision::candidate_id)
        .def_readwrite("candidate_label",
                       &saccade::HandoverDecision::candidate_label)
        .def_readwrite("candidate_start",
                       &saccade::HandoverDecision::candidate_start)
        .def_readwrite("candidate_end",
                       &saccade::HandoverDecision::candidate_end)
        .def_readwrite("gap", &saccade::HandoverDecision::gap)
        .def_readwrite("head_n", &saccade::HandoverDecision::head_n)
        .def_readwrite("bank_n", &saccade::HandoverDecision::bank_n)
        .def_readwrite("candidate_count",
                       &saccade::HandoverDecision::candidate_count)
        .def_readwrite("best_cost", &saccade::HandoverDecision::best_cost)
        .def_readwrite("second_cost",
                       &saccade::HandoverDecision::second_cost)
        .def_readwrite("margin", &saccade::HandoverDecision::margin)
        .def_readwrite("required_margin",
                       &saccade::HandoverDecision::required_margin)
        .def_readwrite("max_cost", &saccade::HandoverDecision::max_cost)
        .def_readwrite("accepted", &saccade::HandoverDecision::accepted)
        .def_readwrite("reason", &saccade::HandoverDecision::reason);

    m.def("parse_mot_lines", &saccade::parse_mot_lines,
          py::arg("lines"));
    m.def("format_mot_records", &saccade::format_mot_records,
          py::arg("records"));
    m.def("build_output_tracklets", &saccade::build_output_tracklets,
          py::arg("records"), py::arg("velocity_samples"));

    m.def(
        "causal_handover_lines",
        [](const std::vector<std::string>& results_lines,
           const std::unordered_map<int, std::vector<float>>& head_embs,
           const std::unordered_map<int, std::vector<float>>& bank_embs,
           int embedding_dim,
           const saccade::HandoverParams& params,
           bool log_decisions) {
            std::vector<saccade::HandoverDecision> decisions;
            auto [lines, stats] = saccade::causal_handover_lines(
                results_lines, head_embs, bank_embs, embedding_dim, params,
                log_decisions ? &decisions : nullptr);
            py::dict stats_dict;
            stats_dict["events"] = stats.events;
            stats_dict["events_with_candidates"] =
                stats.events_with_candidates;
            stats_dict["handovers"] = stats.handovers;
            stats_dict["reject_no_head"] = stats.reject_no_head;
            stats_dict["reject_min_head"] = stats.reject_min_head;
            stats_dict["reject_cost"] = stats.reject_cost;
            stats_dict["reject_margin"] = stats.reject_margin;
            stats_dict["decisions_logged"] = stats.decisions_logged;
            stats_dict["ids_before"] = stats.ids_before;
            stats_dict["ids_after"] = stats.ids_after;
            return py::make_tuple(lines, stats_dict,
                                  log_decisions
                                      ? py::cast(decisions)
                                      : py::none());
        },
        py::arg("results_lines"), py::arg("head_embs"),
        py::arg("bank_embs"), py::arg("embedding_dim"),
        py::arg("params"), py::arg("log_decisions") = false);
}
