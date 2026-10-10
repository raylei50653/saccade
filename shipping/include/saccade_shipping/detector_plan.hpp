// Native detector plan (#465 Phase B PR-8, U3b-2). CUDA-free.
//
// The oracle's detector in the headline configuration, with the head replaced
// by the owner-accepted U1 form (PR-2L, `A_L`): per frame, from the ingest's
// float32 frame buffer [3, H, W] (PR-7),
//
//   resize    F.interpolate(frame[None], (img_size, img_size), "bilinear",
//             align_corners=False)                  (_whole_graph_fn)
//   backbone  the TRT backbone engine -> p3 / p4 / p5 (TRTYoloBackbone)
//   head      the PR-1L TorchScript artifact (LibTorch, graph executor
//             optimize off, `saccade_native::selective_scan_fwd`)
//             -> cls_p3..p5 [1, nc, h, w], reg_p3..p5 [1, 4, h, w]
//   S2        _postprocess_mamba_fixed, torch.compile'd (the oracle's
//             default): sigmoid / class max, top-max_det over all anchors,
//             LTRB -> xyxy on the stride-8/16/32 anchor grid (offset 0.5);
//             then `detections[..., x] *= sx; [..., y] *= sy` with
//             sx = float32(w / img_size), sy = float32(h / img_size)
//   rows      detect_single_patch_640 / _run_native_tensor_prep: boxes
//             float32 [n, 4], scores float32 [n], classes int32 [n],
//             n = max(max_det, nms pad 0) = max_det; is_tiled false; no
//             keypoints -- the PR-5 `detector.bin` boundary.
//
// `plan_detector` reads every value of that path from the resolved config
// (B2) and the frozen PR-1L lineage (B5), and fails closed (ConfigError) on
// any oracle gate it does not implement or any lineage disagreement. The
// operator library named by the frozen lineage is a per-machine build; a
// realization attestation (committed, `configs/shipping/`) may name the build
// that realizes it on this machine, and binds itself to the frozen lineage by
// hash. Code literals of the oracle (the strides, the anchor offset, the nms
// pad 0, the S2 op sequence) are pinned to the oracle source by
// tests/unit/test_native_detector_oracle_pins.py. Nothing here reads the
// process environment.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "saccade_shipping/resolved_config.hpp"
#include "saccade_shipping/strict_json.hpp"

namespace saccade::shipping {

inline constexpr const char* kHeadLineageSchema = "saccade.head_artifact_lineage_torchscript/v1";
inline constexpr const char* kHeadRealizationSchema = "saccade.head_realization_attestation/v1";
inline constexpr const char* kNativeScanOp = "saccade_native::selective_scan_fwd";
inline constexpr const char* kPythonScanOp = "saccade::selective_scan_fwd";

// MambaGatedDetector: `self.stride = torch.tensor([8.0, 16.0, 32.0])`;
// _precompute_anchor_grid: cell centres at +0.5.
inline constexpr std::array<int, 3> kDetectorStrides = {8, 16, 32};
inline constexpr float kAnchorOffset = 0.5f;

// A file the detector loads, with the sha256 it must have.
struct FileBinding {
    std::string path;  // as recorded (relative to the model root)
    std::string sha256;
};

// The four runtime requirements of the PR-1L artifact (lineage).
struct HeadRuntimeRequirements {
    bool graph_executor_optimize = false;
    bool cudnn_benchmark = false;
    bool cudnn_allow_tf32 = true;
    bool matmul_allow_tf32 = false;
};

struct DetectorPlan {
    int img_size = 0;  // stretch-resize target and backbone input side
    int max_det = 0;   // S2 top-k = rows per frame (nms pad 0)
    std::array<int, 3> in_channels{};  // p3 / p4 / p5 channels
    int num_classes = 0;
    int reg_channels = 4;  // reg_max 1: LTRB distances, no DFL
    std::array<std::array<int, 4>, 3> feature_shapes{};  // [1, c, img/s, img/s]
    int anchors = 0;  // sum of (img/s)^2
    FileBinding backbone_engine;
    FileBinding head_artifact;
    FileBinding op_library;  // the build that is loaded (lineage or attestation)
    std::string op_library_lineage_sha256;  // what the frozen lineage names
    bool op_library_from_attestation = false;
    int native_scan_calls = 0;
    HeadRuntimeRequirements runtime;
    double conf_thr_unused = 0.0;  // build.conf_thr: the fixed S2 never reads it
};

// sha256 of the lineage file is the caller's (it read the bytes).
DetectorPlan plan_detector(const ResolvedShippingConfig& cfg, const JsonValue& lineage,
                           const std::string& lineage_sha256,
                           const JsonValue* realization_attestation);

// Reads the three files and calls plan_detector (attestation optional: empty
// path = none).
struct DetectorInputs {
    std::string lineage_path;
    std::string attestation_path;  // may be empty
};
DetectorPlan plan_detector_files(const ResolvedShippingConfig& cfg, const DetectorInputs& in);

// Where a plan path is read from: relative paths against the model root,
// absolute ones as they are. Gate A (preflight.hpp) and the detector load
// (detector_host.hpp) both read the files here.
std::string resolve_model_path(const std::string& model_root, const std::string& path);

// set_whole_graph_img_dims: `self._whole_graph_sx.fill_(w_orig / self.img_size)`
// -- a Python float (double) division, stored in a float32 tensor.
inline float coordinate_scale(int orig, int img_size) {
    return static_cast<float>(static_cast<double>(orig) / static_cast<double>(img_size));
}

}  // namespace saccade::shipping
