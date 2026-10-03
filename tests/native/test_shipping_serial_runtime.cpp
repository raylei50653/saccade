// End-to-end serial runtime wiring (#465 Phase B PR-9, U3b-3). Needs a CUDA
// device, the model files and two MOT17 sequences; SKIPs when they are absent.
//
// Usage: saccade_shipping_serial_runtime_test <resolved config> <lineage>
//            <attestation> <model root> <MOT17 train dir>
//
// Each stage is measured elsewhere (PR-5..PR-8); this checks the ownership
// rules serial_runtime.hpp states, on the first kFrames frames of MOT17-10
// (1920x1080) and MOT17-05 (640x480):
//   * per-sequence state: in one runtime, X, Y, X gives the same lines for
//     both X runs; a fresh runtime that runs Y first gives Y's lines;
//   * nothing Python is mapped; every frame ingested, pre-roll 4, tracker
//     updates on every frame;
//   * the wiring mutations are visible: a PostDetectorHost shared across
//     sequences changes the second X; image dims set only for the first
//     sequence change Y; GMC fed the previous frame changes X.
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

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
    std::filesystem::path x, y;
};

std::unique_ptr<sh::SerialRuntime> make(const Paths& p, sh::RuntimeMutation m) {
    auto rt = std::make_unique<sh::SerialRuntime>(sh::load_resolved_shipping_config_file(p.config),
                                                  sh::DetectorInputs{p.lineage, p.attestation},
                                                  p.model_root);
    rt->set_mutation_for_measurement(m);
    return rt;
}

void check_stats(const sh::SequenceRunResult& r, const std::string& tag) {
    const auto& s = r.stats;
    check(s.frames == kFrames && s.tracker_updates == kFrames && s.skipped_empty == 0,
          tag + ": every frame ingested and tracked");
    check(s.pre_roll_updates == 4, tag + ": graphed-update pre-roll 4");
    check(!r.lines.empty() && s.lines == r.lines.size() && s.track_ids > 0, tag + ": lines and ids");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s <config> <lineage> <attestation> <model root> <MOT17 train dir>\n",
                     argv[0]);
        return 2;
    }
    const Paths p{argv[1], argv[2], argv[3], argv[4],
                  std::filesystem::path(argv[5]) / "MOT17-10-SDP",
                  std::filesystem::path(argv[5]) / "MOT17-05-SDP"};
    for (const std::filesystem::path& need :
         {std::filesystem::path(p.lineage), p.x / "seqinfo.ini", p.y / "seqinfo.ini"}) {
        if (!std::filesystem::exists(need)) {
            std::printf("SKIP: %s not present\n", need.string().c_str());
            return 0;
        }
    }
    try {
        using M = sh::RuntimeMutation;
        auto rt = make(p, M::None);
        const auto x1 = rt->run_sequence(p.x, kFrames);
        const auto y1 = rt->run_sequence(p.y, kFrames);
        const auto x2 = rt->run_sequence(p.x, kFrames);
        check_stats(x1, "X");
        check_stats(y1, "Y");
        check(x1.stats.im_width == 1920 && y1.stats.im_width == 640, "the two geometries differ");
        check(x2.lines == x1.lines, "X, Y, X: the second X equals the first");
        check(sh::mapped_python_libraries().empty(), "no Python library mapped");
        check(rt->sequences_run() == 3, "three sequences run");
        rt.reset();

        auto fresh = make(p, M::None);
        check(fresh->run_sequence(p.y, kFrames).lines == y1.lines, "Y first in a fresh runtime equals Y after X");
        fresh.reset();

        auto shared = make(p, M::SharedPostHost);
        check(shared->run_sequence(p.x, kFrames).lines == x1.lines, "shared_post_host: first X unchanged");
        check(shared->run_sequence(p.x, kFrames).lines != x1.lines, "shared_post_host: second X changes");
        shared.reset();

        auto stale = make(p, M::StaleImageDims);
        check(stale->run_sequence(p.x, kFrames).lines == x1.lines, "stale_image_dims: X unchanged");
        check(stale->run_sequence(p.y, kFrames).lines != y1.lines, "stale_image_dims: Y changes");
        stale.reset();

        auto prev = make(p, M::GmcPreviousFrame);
        check(prev->run_sequence(p.x, kFrames).lines != x1.lines, "gmc_previous_frame: X changes");
        prev.reset();

        bool refused = false;
        try {
            sh::parse_runtime_mutation("no_such_mutation");
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
