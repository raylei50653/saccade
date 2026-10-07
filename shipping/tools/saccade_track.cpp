// saccade_track: the native shipping entrypoint (#465 boundary §2; Phase B
// PR-9 serial, PR-10 double buffer + CUDA graphs; Phase C PR-C2 interface).
//
// For each sequence directory (seqinfo.ini + img1/*.jpg), in the order given:
// native ingest -> detector -> post-detector host -> MOT emit and tail, and
// `<out>/<sequence>.txt` with the oracle's line format. The schedule is the
// one the resolved config names (schedule_plan.hpp): the oracle's double
// buffer with its CUDA graphs (saccade_shipping/double_buffer_runtime.hpp) for
// the headline config. One process runs every sequence, as one oracle run
// does: the decoder, detector and PerceptionPipeline are per run; the frame
// pools, tracker, GMC and track ids are per sequence. The only inputs are the
// resolved config, the frozen head lineage, the operator library's
// realization attestation and the files they bind; no environment variable is
// read.
//
// Exit 0: every sequence written; 2: any error (message on stderr; a refused
// argument, config or input stops before or at that sequence; an argument
// that is not listed below is refused before any file is read).
//
// Usage:
//   saccade_track --config configs/shipping/mamba_whole_graph.resolved.json
//       --lineage models/yolo/<stem>.lineage.json
//       [--attestation configs/shipping/mamba_head_realization.attestation.json]
//       [--model-root DIR] --out DIR [--report JSON] [--trace DIR] SEQUENCE_DIR...
// (--report / --trace: track_driver.hpp). This is the whole interface: the
// shipping build has no developer option and no measurement hook (PR-C2); the
// negative controls, the serial override and --max-frames are
// saccade_track_measurement, a developer build that is not installed.
#include <cstdio>
#include <exception>
#include <string>

#include "saccade_shipping/double_buffer_runtime.hpp"
#include "saccade_shipping/serial_runtime.hpp"
#include "track_driver.hpp"

#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
#error "saccade_track is the shipping entrypoint: it links the shipping runtime, not the measurement variant"
#endif

namespace sh = saccade::shipping;
namespace track = saccade::shipping::track;

namespace {

constexpr const char* kUsage =
    "usage: saccade_track --config JSON --lineage JSON [--attestation JSON] "
    "[--model-root DIR] --out DIR [--report JSON] [--trace DIR] SEQUENCE_DIR...";

track::Options parse_args(int argc, char** argv) {
    track::Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) track::fail(a + " needs a value");
            return argv[++i];
        };
        if (!track::parse_interface_arg(a, next, o)) track::fail("unknown argument " + a);
    }
    if (!track::interface_complete(o)) track::fail(kUsage);
    return o;
}

int run(const track::Options& opt) {
    track::require_distinct_sequences(opt);
    const sh::ResolvedShippingConfig cfg = sh::load_resolved_shipping_config_file(opt.config);
    const sh::DetectorInputs inputs{opt.lineage, opt.attestation};
    if (sh::select_schedule(cfg, /*serial_requested=*/false) == sh::Schedule::Serial) {
        sh::SerialRuntime rt(cfg, inputs, opt.model_root);
        return track::run_sequences(rt, opt, "saccade_track", "serial", 0, nullptr);
    }
    sh::DoubleBufferRuntime rt(cfg, inputs, opt.model_root);
    return track::run_sequences(rt, opt, "saccade_track", "double_buffer", 0, nullptr);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "saccade_track: %s\n", e.what());
        return 2;
    }
}
