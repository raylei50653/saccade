#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "tracking/tracker_gpu.hpp"

namespace py = pybind11;
using namespace saccade;

namespace {

float compute_iou(const float* a, const float* b) {
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);
    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float inter = w * h;
    const float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    const float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    return inter / (area_a + area_b - inter + 1e-6f);
}

py::tuple merge_cross_tile_duplicates_cpu(
    py::array_t<float, py::array::c_style | py::array::forcecast> boxes,
    py::array_t<float, py::array::c_style | py::array::forcecast> scores,
    py::array_t<int, py::array::c_style | py::array::forcecast> classes,
    float iou_threshold,
    float center_threshold,
    float area_ratio_threshold
) {
    if (boxes.ndim() != 2 || boxes.shape(1) != 4) {
        throw std::invalid_argument("boxes must have shape [N, 4]");
    }
    if (scores.ndim() != 1 || classes.ndim() != 1) {
        throw std::invalid_argument("scores and classes must have shape [N]");
    }
    const ssize_t num = boxes.shape(0);
    if (scores.shape(0) != num || classes.shape(0) != num) {
        throw std::invalid_argument("boxes, scores, and classes must have the same length");
    }

    auto boxes_in = boxes.unchecked<2>();
    auto scores_in = scores.unchecked<1>();
    auto classes_in = classes.unchecked<1>();

    if (num <= 1) {
        return py::make_tuple(boxes, scores, classes);
    }

    std::vector<int> order(static_cast<size_t>(num));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return scores_in(lhs) > scores_in(rhs);
    });

    std::vector<unsigned char> consumed(static_cast<size_t>(num), 0);
    std::vector<float> out_boxes;
    std::vector<float> out_scores;
    std::vector<int> out_classes;
    out_boxes.reserve(static_cast<size_t>(num) * 4);
    out_scores.reserve(static_cast<size_t>(num));
    out_classes.reserve(static_cast<size_t>(num));

    for (int anchor_idx : order) {
        if (consumed[static_cast<size_t>(anchor_idx)]) {
            continue;
        }

        const int anchor_class = classes_in(anchor_idx);
        const float anchor_box[4] = {
            boxes_in(anchor_idx, 0),
            boxes_in(anchor_idx, 1),
            boxes_in(anchor_idx, 2),
            boxes_in(anchor_idx, 3),
        };
        const float anchor_w = std::max(1e-6f, anchor_box[2] - anchor_box[0]);
        const float anchor_h = std::max(1e-6f, anchor_box[3] - anchor_box[1]);
        const float anchor_area = anchor_w * anchor_h;
        const float anchor_cx = 0.5f * (anchor_box[0] + anchor_box[2]);
        const float anchor_cy = 0.5f * (anchor_box[1] + anchor_box[3]);

        std::vector<int> cluster_indices;
        cluster_indices.reserve(4);
        for (int candidate_idx : order) {
            if (consumed[static_cast<size_t>(candidate_idx)]) {
                continue;
            }
            if (classes_in(candidate_idx) != anchor_class) {
                continue;
            }

            const float candidate_box[4] = {
                boxes_in(candidate_idx, 0),
                boxes_in(candidate_idx, 1),
                boxes_in(candidate_idx, 2),
                boxes_in(candidate_idx, 3),
            };
            const float iou = compute_iou(anchor_box, candidate_box);
            const float candidate_w = std::max(1e-6f, candidate_box[2] - candidate_box[0]);
            const float candidate_h = std::max(1e-6f, candidate_box[3] - candidate_box[1]);
            const float candidate_area = candidate_w * candidate_h;
            const float min_w = std::min(anchor_w, candidate_w);
            const float min_h = std::min(anchor_h, candidate_h);
            const float center_gate = std::sqrt(min_w * min_w + min_h * min_h) * center_threshold;
            const float candidate_cx = 0.5f * (candidate_box[0] + candidate_box[2]);
            const float candidate_cy = 0.5f * (candidate_box[1] + candidate_box[3]);
            const float center_dx = candidate_cx - anchor_cx;
            const float center_dy = candidate_cy - anchor_cy;
            const float center_dist = std::sqrt(center_dx * center_dx + center_dy * center_dy);
            const float area_ratio = std::min(
                candidate_area / std::max(anchor_area, 1e-6f),
                anchor_area / std::max(candidate_area, 1e-6f)
            );

            if (iou >= iou_threshold || (center_dist <= center_gate && area_ratio >= area_ratio_threshold)) {
                cluster_indices.push_back(candidate_idx);
            }
        }

        float weight_sum = 0.0f;
        float fused_box[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float fused_score = 0.0f;
        for (int idx : cluster_indices) {
            const float score = scores_in(idx);
            weight_sum += score;
            fused_score = std::max(fused_score, score);
            for (int k = 0; k < 4; ++k) {
                fused_box[k] += boxes_in(idx, k) * score;
            }
            consumed[static_cast<size_t>(idx)] = 1;
        }

        const float inv_weight_sum = 1.0f / std::max(weight_sum, 1e-6f);
        for (float& coord : fused_box) {
            coord *= inv_weight_sum;
            out_boxes.push_back(coord);
        }
        out_scores.push_back(fused_score);
        out_classes.push_back(anchor_class);
    }

    const ssize_t out_num = static_cast<ssize_t>(out_scores.size());
    py::array_t<float> out_boxes_arr({out_num, static_cast<ssize_t>(4)});
    py::array_t<float> out_scores_arr({out_num});
    py::array_t<int> out_classes_arr({out_num});

    auto out_boxes_mut = out_boxes_arr.mutable_unchecked<2>();
    auto out_scores_mut = out_scores_arr.mutable_unchecked<1>();
    auto out_classes_mut = out_classes_arr.mutable_unchecked<1>();
    for (ssize_t i = 0; i < out_num; ++i) {
        for (ssize_t k = 0; k < 4; ++k) {
            out_boxes_mut(i, k) = out_boxes[static_cast<size_t>(i) * 4 + static_cast<size_t>(k)];
        }
        out_scores_mut(i) = out_scores[static_cast<size_t>(i)];
        out_classes_mut(i) = out_classes[static_cast<size_t>(i)];
    }

    return py::make_tuple(out_boxes_arr, out_scores_arr, out_classes_arr);
}

} // namespace

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

    py::class_<TrackStateSnapshot>(m, "TrackStateSnapshot")
        .def_readonly("obj_id", &TrackStateSnapshot::obj_id)
        .def_readonly("class_id", &TrackStateSnapshot::class_id)
        .def_readonly("age", &TrackStateSnapshot::age)
        .def_readonly("score", &TrackStateSnapshot::score)
        .def_readonly("state", &TrackStateSnapshot::state)
        .def_readonly("covariance", &TrackStateSnapshot::covariance);

    py::class_<GPUByteTracker>(m, "GPUByteTracker")
        .def(py::init<int, int>(), py::arg("max_objects") = 2048, py::arg("embedding_dim") = 768)
        .def("set_params", &GPUByteTracker::set_params, 
             py::arg("track_thresh"), py::arg("high_thresh"), py::arg("match_thresh"), py::arg("track_buffer"))
        .def("update_reference_features", [](GPUByteTracker& self, uintptr_t ids_ptr, uintptr_t features_ptr, int num, uintptr_t stream_ptr) {
            self.update_reference_features(
                reinterpret_cast<int*>(ids_ptr),
                reinterpret_cast<float*>(features_ptr),
                num,
                reinterpret_cast<cudaStream_t>(stream_ptr)
            );
        }, py::arg("ids_ptr"), py::arg("features_ptr"), py::arg("num"), py::arg("stream_ptr"))
        .def("update", [](GPUByteTracker& self, uintptr_t boxes_ptr, uintptr_t scores_ptr, uintptr_t classes_ptr, int num_dets, uintptr_t stream_ptr, 
                          std::optional<uintptr_t> embeddings_ptr, std::optional<uintptr_t> gmc_ptr, float light_factor) {
            return self.update(
                reinterpret_cast<float*>(boxes_ptr),
                reinterpret_cast<float*>(scores_ptr),
                reinterpret_cast<int*>(classes_ptr),
                num_dets,
                reinterpret_cast<cudaStream_t>(stream_ptr),
                embeddings_ptr ? reinterpret_cast<float*>(*embeddings_ptr) : nullptr,
                gmc_ptr ? reinterpret_cast<float*>(*gmc_ptr) : nullptr,
                light_factor
            );
        }, 
        py::arg("boxes_ptr"), py::arg("scores_ptr"), py::arg("classes_ptr"), py::arg("num_dets"), py::arg("stream_ptr"),
        py::arg("embeddings_ptr") = std::nullopt, py::arg("gmc_ptr") = std::nullopt, py::arg("light_factor") = 0.0f,
        "Update tracker with raw GPU pointers and stream")
        .def("get_state_snapshots", [](GPUByteTracker& self, uintptr_t stream_ptr) {
            return self.get_state_snapshots(reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("stream_ptr"),
        "Return active Kalman state and covariance snapshots");

    m.def(
        "merge_cross_tile_duplicates",
        &merge_cross_tile_duplicates_cpu,
        py::arg("boxes"),
        py::arg("scores"),
        py::arg("classes"),
        py::arg("iou_threshold") = 0.45f,
        py::arg("center_threshold") = 0.18f,
        py::arg("area_ratio_threshold") = 0.6f,
        "Merge duplicate detections across overlapping tiles on CPU."
    );

    m.def(
        "merge_cross_tile_duplicates_cuda",
        [](
            uintptr_t boxes_ptr,
            uintptr_t scores_ptr,
            uintptr_t classes_ptr,
            int num_dets,
            uintptr_t anchor_indices_ptr,
            uintptr_t box_sums_ptr,
            uintptr_t score_sums_ptr,
            uintptr_t score_bits_max_ptr,
            uintptr_t cluster_counts_ptr,
            uintptr_t out_boxes_ptr,
            uintptr_t out_scores_ptr,
            uintptr_t out_classes_ptr,
            uintptr_t out_count_ptr,
            float iou_threshold,
            float center_threshold,
            float area_ratio_threshold,
            uintptr_t stream_ptr
        ) {
            merge_cross_tile_duplicates_cuda(
                reinterpret_cast<const float*>(boxes_ptr),
                reinterpret_cast<const float*>(scores_ptr),
                reinterpret_cast<const int*>(classes_ptr),
                num_dets,
                reinterpret_cast<int*>(anchor_indices_ptr),
                reinterpret_cast<float*>(box_sums_ptr),
                reinterpret_cast<float*>(score_sums_ptr),
                reinterpret_cast<int*>(score_bits_max_ptr),
                reinterpret_cast<int*>(cluster_counts_ptr),
                reinterpret_cast<float*>(out_boxes_ptr),
                reinterpret_cast<float*>(out_scores_ptr),
                reinterpret_cast<int*>(out_classes_ptr),
                reinterpret_cast<int*>(out_count_ptr),
                iou_threshold,
                center_threshold,
                area_ratio_threshold,
                reinterpret_cast<cudaStream_t>(stream_ptr)
            );
        },
        py::arg("boxes_ptr"),
        py::arg("scores_ptr"),
        py::arg("classes_ptr"),
        py::arg("num_dets"),
        py::arg("anchor_indices_ptr"),
        py::arg("box_sums_ptr"),
        py::arg("score_sums_ptr"),
        py::arg("score_bits_max_ptr"),
        py::arg("cluster_counts_ptr"),
        py::arg("out_boxes_ptr"),
        py::arg("out_scores_ptr"),
        py::arg("out_classes_ptr"),
        py::arg("out_count_ptr"),
        py::arg("iou_threshold") = 0.45f,
        py::arg("center_threshold") = 0.18f,
        py::arg("area_ratio_threshold") = 0.6f,
        py::arg("stream_ptr"),
        "Merge duplicate detections across overlapping tiles on the caller's CUDA stream."
    );
}
