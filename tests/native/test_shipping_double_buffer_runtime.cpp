// End-to-end double-buffer runtime with CUDA graphs (#465 Phase B PR-10, U5).
// Needs a CUDA device, the model files and three MOT17 sequences; SKIPs when
// they are absent.
//
// Usage: saccade_shipping_double_buffer_runtime_test <resolved config> <lineage>
//            <attestation> <model root> <MOT17 train dir>
//
// The parity against the oracle is the harness's (native_track_parity.py);
// this checks the schedule and graph rules double_buffer_runtime.hpp states,
// on the first kFrames frames of MOT17-10 (X, 1920x1080), MOT17-05 (Y,
// 640x480) and MOT17-11 (Z, 1920x1080):
//   * the double buffer gives the serial runtime's lines and detector rows on
//     X, Y, X, Z in one runtime (serial is the PR-9-measured reference);
//   * the whole-detect graph is captured per key and kept across sequences of
//     the same dims: X 1 capture (4 warm-up runs), Y 1 (new dims), X again 1,
//     Z 0 (same dims as X); per sequence one main NMS, GMC and tracker
//     capture; replays: detector / NMS / tracker every frame, GMC every frame
//     but the first (the capture branch does not replay);
//   * the double-buffer negative controls change the output: a stale detector
//     input changes detector rows and lines, a stale GMC input changes lines
//     but not detector rows, a swapped detection parity changes both;
//   * a config whose schedule is not the double buffer is refused.
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "saccade_shipping/double_buffer_runtime.hpp"
#include "saccade_shipping/serial_runtime.hpp"

namespace sh = saccade::shipping;

namespace {

constexpr int kFrames = 40;
int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

struct Paths {
    std::string config, lineage, attestation, model_root;
    std::filesystem::path x, y, z;
};

// Every frame's detector rows, concatenated.
class Rows : public sh::FrameObserver {
public:
    void on_frame(const sh::FrameTrace& t) override {
        const sh::DetectionRows& r = *t.detections;
        frames.push_back(t.frame);
        boxes.insert(boxes.end(), r.boxes.begin(), r.boxes.end());
        scores.insert(scores.end(), r.scores.begin(), r.scores.end());
        classes.insert(classes.end(), r.classes.begin(), r.classes.end());
    }
    bool operator==(const Rows& o) const {
        return frames == o.frames && boxes == o.boxes && scores == o.scores && classes == o.classes;
    }
    std::vector<int> frames;
    std::vector<float> boxes, scores;
    std::vector<std::int32_t> classes;
};

struct Run {
    sh::SequenceRunResult result;
    Rows rows;
};

template <class Runtime>
Run run(Runtime& rt, const std::filesystem::path& seq) {
    Run r;
    r.result = rt.run_sequence(seq, kFrames, &r.rows);
    return r;
}

std::unique_ptr<sh::DoubleBufferRuntime> make_db(const Paths& p, sh::DoubleBufferMutation m) {
    auto rt = std::make_unique<sh::DoubleBufferRuntime>(
        sh::load_resolved_shipping_config_file(p.config), sh::DetectorInputs{p.lineage, p.attestation},
        p.model_root);
    rt->set_mutation_for_measurement(m);
    return rt;
}

void check_graphs(const sh::SequenceRunResult& r, int detector_captures, const std::string& tag) {
    const sh::ScheduleStats& g = r.stats.schedule;
    check(g.detector_captures == detector_captures &&
              g.detector_warmup_runs == 4 * detector_captures,
          tag + ": " + std::to_string(detector_captures) + " detector capture(s), 4 warm-up runs each");
    check(g.detector_replays == kFrames && g.post.nms_replays == kFrames &&
              g.post.tracker_replays == kFrames && g.post.gmc_replays == kFrames - 1,
          tag + ": replays (detector / NMS / tracker every frame, GMC all but the first)");
    check(g.post.nms_captures == 1 && g.post.gmc_captures == 1 && g.post.tracker_captures == 1,
          tag + ": one main NMS, GMC and tracker capture");
    check(r.stats.pre_roll_updates == 4 && r.stats.frames == kFrames && r.stats.tracker_updates == kFrames,
          tag + ": pre-roll 4, every frame tracked");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s <config> <lineage> <attestation> <model root> <MOT17 train dir>\n",
                     argv[0]);
        return 2;
    }
    const std::filesystem::path train(argv[5]);
    const Paths p{argv[1], argv[2], argv[3], argv[4], train / "MOT17-10-SDP", train / "MOT17-05-SDP",
                  train / "MOT17-11-SDP"};
    for (const std::filesystem::path& need : {std::filesystem::path(p.lineage), p.x / "seqinfo.ini",
                                              p.y / "seqinfo.ini", p.z / "seqinfo.ini"}) {
        if (!std::filesystem::exists(need)) {
            std::printf("SKIP: %s not present\n", need.string().c_str());
            return 0;
        }
    }
    try {
        using M = sh::DoubleBufferMutation;
        const sh::ResolvedShippingConfig cfg = sh::load_resolved_shipping_config_file(p.config);
        const sh::DetectorInputs inputs{p.lineage, p.attestation};

        std::vector<Run> serial;
        {
            sh::SerialRuntime rt(cfg, inputs, p.model_root);
            for (const auto& s : {p.x, p.y, p.x, p.z}) serial.push_back(run(rt, s));
        }

        auto db = make_db(p, M::None);
        check(db->schedule_plan().double_buffer, "the config's schedule is the double buffer");
        std::vector<Run> dbl;
        for (const auto& s : {p.x, p.y, p.x, p.z}) dbl.push_back(run(*db, s));
        const char* tags[] = {"X", "Y", "X again", "Z"};
        for (std::size_t i = 0; i < dbl.size(); ++i) {
            const std::string t = tags[i];
            check(dbl[i].result.lines == serial[i].result.lines, t + ": double-buffer lines == serial");
            check(dbl[i].rows == serial[i].rows, t + ": double-buffer detector rows == serial");
        }
        check_graphs(dbl[0].result, 1, "X");
        check_graphs(dbl[1].result, 1, "Y (new dims)");
        check_graphs(dbl[2].result, 1, "X again (dims changed back)");
        check_graphs(dbl[3].result, 0, "Z (same dims as X: graph kept)");
        check(db->detector_graph_stats().cache_clears == 2, "two cache clears (X -> Y -> X)");
        check(sh::mapped_python_libraries().empty(), "no Python library mapped");
        db.reset();

        auto stale_det = make_db(p, M::StaleDetectorInput);
        const Run sd = run(*stale_det, p.x);
        check(!(sd.rows == dbl[0].rows) && sd.result.lines != dbl[0].result.lines,
              "stale_detector_input: rows and lines change");
        stale_det.reset();

        auto stale_gmc = make_db(p, M::StaleGmcInput);
        const Run sg = run(*stale_gmc, p.x);
        check(sg.rows == dbl[0].rows && sg.result.lines != dbl[0].result.lines,
              "stale_gmc_input: rows unchanged, lines change");
        stale_gmc.reset();

        auto swapped = make_db(p, M::SwappedDetectionParity);
        const Run sw = run(*swapped, p.x);
        check(!(sw.rows == dbl[0].rows) && sw.result.lines != dbl[0].result.lines,
              "swapped_detection_parity: rows and lines change");
        swapped.reset();

        sh::ResolvedShippingConfig serial_cfg = cfg;
        serial_cfg.host_params.steps.schedule_double_buffer = false;
        serial_cfg.host_params.env.double_buffer.reset();
        bool refused = false;
        try {
            sh::DoubleBufferRuntime rt(serial_cfg, inputs, p.model_root);
        } catch (const sh::ConfigError&) {
            refused = true;
        }
        check(refused, "a serial-schedule config is refused by the double-buffer runtime");
        check(!sh::plan_schedule(serial_cfg).double_buffer, "the serial config plans the serial schedule");

        sh::ResolvedShippingConfig torn = cfg;
        torn.host_params.env.detect_barrier = std::string("full");
        refused = false;
        try {
            sh::plan_schedule(torn);
        } catch (const sh::ConfigError&) {
            refused = true;
        }
        check(refused, "double buffer without the event barrier is refused");

        refused = false;
        try {
            sh::parse_double_buffer_mutation("no_such_mutation");
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        check(refused, "unknown mutation name refused");
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf("%s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
