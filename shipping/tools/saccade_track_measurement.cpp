// saccade_track_measurement: the developer build of the shipping entrypoint
// (#465 Phase C PR-C2; developer_build_debug, never installed).
//
// The same sequence loop, trace and report as saccade_track
// (track_driver.hpp), linked against the measurement variant of the runtime
// libraries (SACCADE_SHIPPING_MEASUREMENT_HOOKS), plus the options the
// parity harness (scripts/eval/diagnostics/native_track_parity.py) needs and
// a shipping binary must not have:
//   --measurement-mutation M
//                   break one wiring rule on purpose (negative controls):
//                   serial: shared_post_host|stale_image_dims|gmc_previous_frame;
//                   double buffer: stale_detector_input|stale_gmc_input|
//                   swapped_detection_parity
//   --schedule serial
//                   the PR-9 serial, eager runtime (serial_runtime.hpp) instead
//                   of the config's schedule (the PR-9 reference)
//   --max-frames N  frames 1..min(N, seqLength), the oracle's --max-frames
// The report (format saccade.native_track_report/v5) names this entrypoint and
// records the three under "measurement". A mutation name is checked after Gate
// A and before any model is loaded. Run id, <out> lock and journal: as saccade_track
// (track_driver.hpp).
//
// Usage: saccade_track_measurement <saccade_track's arguments>
//            [--measurement-mutation M] [--schedule serial] [--max-frames N]
#include <cstdio>
#include <exception>
#include <optional>
#include <string>
#include <utility>

#include "saccade_shipping/double_buffer_runtime.hpp"
#include "saccade_shipping/serial_runtime.hpp"
#include "track_driver.hpp"

#ifndef SACCADE_SHIPPING_MEASUREMENT_HOOKS
#error "saccade_track_measurement links the measurement variant of the runtime libraries"
#endif

namespace sh = saccade::shipping;
namespace track = saccade::shipping::track;
using sh::JsonValue;

namespace {

constexpr const char* kEntrypoint = "saccade_track_measurement";

struct Options {
    track::Options track;
    std::string mutation = "none", schedule;
    int max_frames = 0;
};

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) track::fail(a + " needs a value");
            return argv[++i];
        };
        if (a == "--measurement-mutation") o.mutation = next();
        else if (a == "--schedule") o.schedule = next();
        else if (a == "--max-frames") o.max_frames = std::stoi(next());
        else if (!track::parse_interface_arg(a, next, o.track)) track::fail("unknown argument " + a);
    }
    if (!track::interface_complete(o.track) || o.max_frames < 0 ||
        !(o.schedule.empty() || o.schedule == "serial")) {
        track::fail("usage: saccade_track_measurement (--config JSON --lineage JSON "
                    "[--attestation JSON] [--model-root DIR] | --model-bundle DIR) "
                    "[--require-identity none|checksum_matched|expected_source_verified] --out DIR "
                    "[--report JSON] [--trace DIR] SEQUENCE_DIR... [--measurement-mutation M] "
                    "[--schedule serial] [--max-frames N]");
    }
    return o;
}

JsonValue measurement_json(const Options& o, const char* mutation) {
    JsonValue m = JsonValue::make_object();
    m.set("mutation", JsonValue::make_string(mutation));
    m.set("schedule_override", o.schedule.empty() ? JsonValue::make_null() : JsonValue::make_string(o.schedule));
    m.set("max_frames", JsonValue::make_int(o.max_frames));
    return m;
}

int run(const Options& opt, sh::RunCompletion& completion) {
    // Gate A, as saccade_track's: it validates the config's schedule plan
    // before the override is looked at (--schedule serial must not bypass the
    // fail-closed checks) and reads only the frames --max-frames will run.
    sh::PreflightResult pre =
        track::preflight(opt.track, kEntrypoint, completion, opt.max_frames, opt.schedule == "serial");
    if (pre.schedule == sh::Schedule::Serial) {
        const sh::RuntimeMutation m = sh::parse_runtime_mutation(opt.mutation);
        const JsonValue meas = measurement_json(opt, sh::runtime_mutation_name(m));
        auto rt = track::load_runtime<sh::SerialRuntime>(opt.track, pre, completion);
        rt->set_mutation_for_measurement(m);
        return track::run_sequences(*rt, opt.track, completion, kEntrypoint, "serial", opt.max_frames, &meas);
    }
    const sh::DoubleBufferMutation m = sh::parse_double_buffer_mutation(opt.mutation);
    const JsonValue meas = measurement_json(opt, sh::double_buffer_mutation_name(m));
    auto rt = track::load_runtime<sh::DoubleBufferRuntime>(opt.track, pre, completion);
    rt->set_mutation_for_measurement(m);
    return track::run_sequences(*rt, opt.track, completion, kEntrypoint, "double_buffer", opt.max_frames,
                                &meas);
}

}  // namespace

int main(int argc, char** argv) {
    std::optional<sh::RunCompletion> completion;  // holds <out>'s lock until exit
    try {
        const Options opt = parse_args(argc, argv);
        track::begin_run(opt.track, kEntrypoint, completion);
        return run(opt, *completion);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "saccade_track_measurement: %s\n", e.what());
        if (completion) completion->fail(e.what());
        return 2;
    }
}
