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
// Exit 0: every sequence written and the journal's state=complete written;
// 2: any error (message on stderr; an argument that is not listed below is
// refused before any file is read; a refused config, lineage, attestation,
// model file, sequence input or output directory stops in Gate A, before any
// CUDA call; a frame the decoder refuses stops at that sequence). Once the
// arguments parse, stderr's first line is "saccade_track: run_id <id>"; the
// run id names this run in <out>/saccade_track.journal.json and the report;
// "saccade_track: preflight passed" follows when Gate A passes. --out is exclusive: a
// second run on the same <out> exits 2 and changes nothing. A run removes its
// own sequences' earlier <out>/<sequence>.txt, trace files and the --report
// file before it starts, so a failed rerun leaves none of them; keep earlier
// outputs with another --out. Only a journal with this run_id and
// state=complete marks a complete run; a txt counts only when the journal has
// it `written` with its sha256 (track_driver.hpp, run_completion.hpp).
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
#include <optional>
#include <string>
#include <utility>

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

int run(const track::Options& opt, sh::RunCompletion& completion) {
    // Gate A (track_driver.hpp): no CUDA call happens before it returns.
    sh::PreflightResult pre =
        track::preflight(opt, "saccade_track", completion, 0, /*serial_requested=*/false);
    if (pre.schedule == sh::Schedule::Serial) {
        sh::SerialRuntime rt(pre.config, std::move(pre.detector), opt.model_root);
        return track::run_sequences(rt, opt, completion, "saccade_track", "serial", 0, nullptr);
    }
    sh::DoubleBufferRuntime rt(pre.config, std::move(pre.detector), opt.model_root);
    return track::run_sequences(rt, opt, completion, "saccade_track", "double_buffer", 0, nullptr);
}

}  // namespace

int main(int argc, char** argv) {
    std::optional<sh::RunCompletion> completion;  // holds <out>'s lock until exit
    try {
        const track::Options opt = parse_args(argc, argv);
        track::begin_run(opt, "saccade_track", completion);
        return run(opt, *completion);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "saccade_track: %s\n", e.what());
        if (completion) completion->fail(e.what());
        return 2;
    }
}
