// Native ingest plan, normalize twin and sequence input vs the Python oracle
// (#465 Phase B PR-7, U3b-1). CPU only; CI job `shipping-config-loader`.
//
// Usage: saccade_shipping_ingest_plan_test <resolved config> <shipping_ingest.json>
//
// The fixture is written by scripts/model/render_shipping_ingest_fixture.py
// from the oracle's own code. This test checks:
//   * plan_ingest accepts the headline config and refuses every ingest gate
//     it does not implement (GPU decode off, NV12 buffer, preprocess modes,
//     workbench);
//   * ingest_normalize(x) is bit-identical to torch's CUDA ingest op for all
//     256 values (and the table is not x / 255.0f, so the check discriminates);
//   * read_sequence_input gives the oracle's geometry, listing and consumed
//     frames on every materialized case, refuses where the oracle fails, and
//     refuses the cases the fixture marks native-stricter;
//   * read_frame_file refuses what decode_jpeg would (non-regular, empty).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "saccade_shipping/ingest_plan.hpp"
#include "saccade_shipping/strict_json.hpp"

namespace fs = std::filesystem;
namespace sh = saccade::shipping;
using sh::JsonValue;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        ++g_checks;                                                                       \
        if (!(cond)) {                                                                    \
            ++g_failures;                                                                 \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                                 \
    } while (0)

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void write_file(const fs::path& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary);
    out << bytes;
    if (!out) throw std::runtime_error("cannot write " + path.string());
}

const JsonValue& at(const JsonValue& v, const char* key) {
    const JsonValue* p = v.find(key);
    if (p == nullptr) throw std::runtime_error(std::string("fixture: missing ") + key);
    return *p;
}

std::uint32_t hex_le_u32(const std::string& h) {
    if (h.size() != 8) throw std::runtime_error("fixture: bad float hex " + h);
    std::uint32_t v = 0;
    for (int k = 3; k >= 0; --k) {
        v = (v << 8) | static_cast<std::uint32_t>(std::stoul(h.substr(static_cast<std::size_t>(k) * 2, 2), nullptr, 16));
    }
    return v;
}

std::uint32_t bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

bool throws_input(const std::function<void()>& f) {
    try {
        f();
    } catch (const sh::InputError&) {
        return true;
    }
    return false;
}

bool refused(const std::string& config_text, const char* section, const char* key, JsonValue value) {
    JsonValue doc = sh::parse_strict_json(config_text);
    JsonValue* slot = doc.find("host_params")->find(section)->find(key);
    if (slot == nullptr) throw std::runtime_error(std::string("no host_params.") + section + "." + key);
    *slot = std::move(value);
    const sh::ResolvedShippingConfig cfg = sh::load_resolved_shipping_config(doc);
    try {
        sh::plan_ingest(cfg);
    } catch (const sh::ConfigError&) {
        return true;
    }
    std::fprintf(stderr, "not refused: %s.%s\n", section, key);
    return false;
}

void check_plan(const std::string& config_text) {
    const sh::IngestPlan p = sh::plan_ingest(sh::parse_resolved_shipping_config(config_text));
    CHECK(bits(p.normalize_scale) == bits(1.0f / 255.0f));
    CHECK(refused(config_text, "steps", "ingest.gpu_decode", JsonValue::make_bool(false)));
    CHECK(refused(config_text, "steps", "ingest.nv12_buffer", JsonValue::make_bool(true)));
    CHECK(refused(config_text, "steps", "track.workbench", JsonValue::make_bool(true)));
    for (const char* mode : {"gamma", "contrast", "letterbox"}) {
        CHECK(refused(config_text, "cfg", "preprocess_modes",
                      JsonValue::make_array({JsonValue::make_string(mode)})));
    }
}

void check_normalize(const JsonValue& table) {
    CHECK(table.array.size() == 256);
    int differs_from_division = 0;
    for (int i = 0; i < 256 && i < static_cast<int>(table.array.size()); ++i) {
        const std::uint32_t want = hex_le_u32(table.array[static_cast<std::size_t>(i)].string);
        const float got = sh::ingest_normalize(static_cast<std::uint8_t>(i));
        if (bits(got) != want) std::fprintf(stderr, "normalize(%d) differs from torch\n", i);
        CHECK(bits(got) == want);
        if (bits(static_cast<float>(i) / 255.0f) != want) ++differs_from_division;
    }
    CHECK(differs_from_division > 0);  // the table tells the two forms apart
}

std::vector<std::string> strings(const JsonValue& v) {
    std::vector<std::string> out;
    for (const JsonValue& s : v.array) out.push_back(s.string);
    return out;
}

fs::path materialize(const fs::path& root, const JsonValue& c) {
    const fs::path seq = root / at(c, "name").string;
    fs::create_directories(seq);
    const JsonValue& ini = at(c, "seqinfo");
    if (ini.kind != JsonValue::Kind::Null) write_file(seq / "seqinfo.ini", ini.string);
    const JsonValue& img = at(c, "img1");
    if (img.kind != JsonValue::Kind::Null) {
        fs::create_directories(seq / "img1");
        for (const JsonValue& e : img.array) {
            const fs::path p = seq / "img1" / at(e, "name").string;
            const std::string& kind = at(e, "kind").string;
            if (kind == "file") write_file(p, "\xff\xd8\xff");
            else if (kind == "dir") fs::create_directories(p);
            else if (kind == "symlink") fs::create_symlink(at(e, "target").string, p);
            else throw std::runtime_error("fixture: entry kind " + kind);
        }
    }
    return seq;
}

void check_sequence_cases(const JsonValue& cases, const fs::path& root) {
    CHECK(cases.array.size() >= 15);
    for (const JsonValue& c : cases.array) {
        const std::string& name = at(c, "name").string;
        const JsonValue& oracle = at(c, "oracle");
        const bool native_same = at(c, "native").string == "same";
        const fs::path seq = materialize(root, c);
        const int max_frames = static_cast<int>(at(c, "max_frames").integer);
        if (!at(oracle, "ok").boolean || !native_same) {
            const bool r = throws_input([&] { sh::read_sequence_input(seq, max_frames); });
            if (!r) std::fprintf(stderr, "sequence case %s: not refused\n", name.c_str());
            CHECK(r);
            continue;
        }
        try {
            const sh::SequenceInput in = sh::read_sequence_input(seq, max_frames);
            const bool eq = in.im_width == at(oracle, "im_width").integer &&
                            in.im_height == at(oracle, "im_height").integer &&
                            in.seq_length == at(oracle, "seq_length").integer &&
                            in.listed == strings(at(oracle, "listed")) &&
                            in.frames == strings(at(oracle, "frames")) && in.img_dir == seq / "img1";
            if (!eq) std::fprintf(stderr, "sequence case %s: differs from the oracle\n", name.c_str());
            CHECK(eq);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "sequence case %s: %s\n", name.c_str(), e.what());
            CHECK(false);
        }
    }
}

void check_seqinfo_and_files(const fs::path& root) {
    const sh::SeqInfo s = sh::parse_seqinfo("[Sequence]\nimWidth=+640\nimHeight=480\nseqLength=7\n");
    CHECK(s.im_width == 640 && s.im_height == 480 && s.seq_length == 7);
    for (const char* bad : {"[Sequence]\nimWidth=-\nimHeight=1\nseqLength=1\n",
                            "[Sequence]\nimWidth=99999999999\nimHeight=1\nseqLength=1\n",
                            "[Sequence]\nimWidth=1\nimHeight=1\nseqLength=-1\n",
                            "[Sequence]\nimWidth=%(x)s\nimHeight=1\nseqLength=1\n",
                            "[Sequence]\nimWidth\nimHeight=1\nseqLength=1\n",
                            "[Sequence]\n[Sequence]\nimWidth=1\nimHeight=1\nseqLength=1\n"}) {
        CHECK(throws_input([&] { sh::parse_seqinfo(bad); }));
    }
    const fs::path dir = root / "files";
    fs::create_directories(dir / "sub.jpg");
    write_file(dir / "a.jpg", "\xff\xd8");
    write_file(dir / "empty.jpg", "");
    fs::create_symlink("a.jpg", dir / "link.jpg");
    fs::create_symlink("missing.jpg", dir / "dangling.jpg");
    CHECK(sh::read_frame_file(dir / "a.jpg").size() == 2);
    CHECK(sh::read_frame_file(dir / "link.jpg").size() == 2);
    CHECK(throws_input([&] { sh::read_frame_file(dir / "sub.jpg"); }));
    CHECK(throws_input([&] { sh::read_frame_file(dir / "empty.jpg"); }));
    CHECK(throws_input([&] { sh::read_frame_file(dir / "dangling.jpg"); }));
    CHECK(throws_input([&] { sh::read_frame_file(dir / "absent.jpg"); }));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <resolved config> <shipping_ingest.json>\n", argv[0]);
        return 2;
    }
    fs::path root;
    try {
        const std::string config_text = read_file(argv[1]);
        const JsonValue fixture = sh::parse_strict_json(read_file(argv[2]));
        CHECK(at(fixture, "format").string == "saccade.shipping_ingest_fixture/v1");
        root = fs::temp_directory_path() /
               ("saccade_ingest_plan_test_" + std::to_string(static_cast<long long>(::getpid())));
        fs::remove_all(root);
        fs::create_directories(root);
        check_plan(config_text);
        check_normalize(at(fixture, "normalize_f32_hex"));
        check_sequence_cases(at(fixture, "sequence_cases"), root / "cases");
        check_seqinfo_and_files(root);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        if (!root.empty()) fs::remove_all(root);
        return 2;
    }
    fs::remove_all(root);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
