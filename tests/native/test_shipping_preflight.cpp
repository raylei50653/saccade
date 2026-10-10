// Gate A of saccade_track (#536 CC-536-01-02 N-T2,
// docs/architecture/ship_export_contracts_536.md). CPU only; CI job
// `shipping-config-loader`. Links saccade_shipping_preflight and no CUDA
// library: everything checked here runs without a device by construction.
//
// Usage: saccade_shipping_preflight_test <resolved config> <lineage> <attestation> [--keep DIR]
//
// The lineage is tests/native/fixtures/shipping_head_lineage.json (a byte copy
// of the frozen PR-1L lineage). Each case builds, in its own directory, a
// self-consistent bundle: a model root whose operator library, head artifact
// and backbone engine are small stand-in files, and a copy of the lineage
// whose three sha256 are those files' (no attestation: the committed one is
// bound to the frozen lineage's bytes). Gate A passes on it -- it checks
// bytes, not where they came from (the CC-536-01-02 known limit) -- and each
// case breaks one thing. The entrypoints' error path is reproduced: the run
// takes <out> (RunCompletion), Gate A runs, a failure is recorded with
// RunCompletion::fail. Cases:
//   * pass:              the bundle passes; the result is the headline
//                        schedule (double buffer) and the stand-ins' hashes;
//                        the trace directories exist; no probe file is left;
//                        the journal is still running with identity null;
//   * pass_max_frames:   a sequence listing fewer frames than seqLength
//                        passes with --max-frames within the listing;
//   * config_*:          a missing config, a torn schedule
//                        (steps.schedule.double_buffer flipped);
//   * lineage_*:         a missing lineage; a lineage whose preset sha256
//                        disagrees with the config;
//   * attestation_other_lineage: the committed attestation with the
//                        stand-in lineage (bound to other bytes);
//   * sequence_*:        no seqinfo.ini; fewer img1 entries than seqLength;
//   * report_dir_missing, trace_unwritable: outputs that cannot be written;
//   * artifact_missing_*, artifact_replaced_*: each of the three files absent,
//                        or present with other bytes (checked last: every
//                        earlier step passed).
// The same refusals at the built entrypoint, with no CUDA call, and the
// missing-attestation case on the real model files:
// tests/unit/test_saccade_track_preflight_cli.py.
// Every refusal: PreflightError whose message starts "preflight: " and names
// the failed check; journal state failed, failure.sequence null, the message
// recorded, identity level null, every sequence pending; the earlier outputs
// were removed (RunCompletion), nothing else written.

#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "saccade_shipping/preflight.hpp"
#include "saccade_shipping/run_completion.hpp"
#include "saccade_shipping/sha256.hpp"
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

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + p.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void write_file(const fs::path& p, const std::string& s) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f << s;
    if (!f) throw std::runtime_error("cannot write " + p.string());
}

std::string sha(const std::string& s) { return sh::sha256_hex(s.data(), s.size()); }

JsonValue& at(JsonValue& o, const std::string& key) {
    JsonValue* v = o.find(key);
    if (v == nullptr) throw std::runtime_error("fixture has no " + key);
    return *v;
}

const std::string kOpLibrary = "operator library stand-in";
const std::string kHead = "head artifact stand-in";
const std::string kEngine = "backbone engine stand-in";
const std::vector<std::string> kSeqs = {"SEQ-A", "SEQ-B"};

struct Fixture {
    std::string config_text, lineage_text, attestation_path;
};

// One case's files: <dir>/root (model root, lineage, sequences), <dir>/out,
// <dir>/report.json, <dir>/trace.
struct Bundle {
    fs::path dir, root, lineage, config, out, report, trace;
    std::string attestation;
    std::vector<std::string> sequences;
    int max_frames = 0;
    JsonValue lineage_json;
};

void write_sequence(const fs::path& seq, int seq_length, int listed) {
    write_file(seq / "seqinfo.ini", "[Sequence]\nname=" + seq.filename().string() +
                                        "\nimWidth=8\nimHeight=6\nseqLength=" + std::to_string(seq_length) + "\n");
    for (int k = 1; k <= listed; ++k) {
        char name[16];
        std::snprintf(name, sizeof name, "%06d.jpg", k);
        write_file(seq / "img1" / name, "not decoded by Gate A");
    }
}

Bundle make_bundle(const Fixture& fx, const fs::path& dir) {
    Bundle b;
    b.dir = dir;
    b.root = dir / "root";
    b.out = dir / "out";
    b.report = dir / "report.json";
    b.trace = dir / "trace";
    b.config = dir / "config.json";
    write_file(b.config, fx.config_text);
    b.lineage_json = sh::parse_strict_json(fx.lineage_text);
    JsonValue& ts = at(b.lineage_json, "torchscript");
    JsonValue& op = at(b.lineage_json, "op_library");
    JsonValue& eng = at(at(b.lineage_json, "companions"), "backbone_engine");
    write_file(b.root / at(op, "path").string, kOpLibrary);
    write_file(b.root / at(ts, "path").string, kHead);
    write_file(b.root / at(eng, "path").string, kEngine);
    at(op, "sha256").string = sha(kOpLibrary);
    at(ts, "sha256").string = sha(kHead);
    at(eng, "sha256").string = sha(kEngine);
    b.lineage = b.root / "lineage.json";
    write_file(b.lineage, sh::dump_python_json(b.lineage_json));
    for (const std::string& s : kSeqs) {
        write_sequence(b.root / "seqs" / s, 3, 3);
        b.sequences.push_back((b.root / "seqs" / s).string());
    }
    return b;
}

struct Outcome {
    std::optional<sh::PreflightResult> result;
    std::string error;  // PreflightError's message; empty when it passed
    std::string other;  // any other exception
    JsonValue journal;
    std::string run_id;
};

// The entrypoints' sequence: take <out>, run Gate A, record a failure.
Outcome run_gate(const Bundle& b) {
    Outcome o;
    o.run_id = sh::new_run_id();
    sh::RunOutputs outputs{b.out, b.report, b.trace, kSeqs};
    std::optional<sh::RunCompletion> c;
    try {
        c.emplace(o.run_id, "saccade_shipping_preflight_test", outputs);
        o.result = sh::run_preflight(sh::PreflightInputs{b.config.string(), b.lineage.string(), b.attestation,
                                                         b.root.string(), b.sequences, b.max_frames, false},
                                     *c);
    } catch (const sh::PreflightError& e) {
        o.error = e.what();
        if (c) c->fail(e.what());
    } catch (const std::exception& e) {
        o.other = e.what();
        if (c) c->fail(e.what());
    }
    o.journal = sh::parse_strict_json(read_file(b.out / sh::kRunJournalName));
    return o;
}

const std::string& str(const JsonValue& o, const char* key) {
    const JsonValue* v = o.find(key);
    if (v == nullptr || v->kind != JsonValue::Kind::String) throw std::runtime_error(std::string("no string ") + key);
    return v->string;
}

bool is_null(const JsonValue& o, const char* key) {
    const JsonValue* v = o.find(key);
    return v != nullptr && v->kind == JsonValue::Kind::Null;
}

std::vector<std::string> probe_files(const fs::path& dir) {
    std::vector<std::string> out;
    for (const auto& e : fs::recursive_directory_iterator(dir)) {
        const std::string n = e.path().filename().string();
        if (n.rfind(".saccade_track.preflight.", 0) == 0) out.push_back(e.path().string());
    }
    return out;
}

void check_untouched_journal(const Outcome& o) {
    CHECK(str(o.journal, "run_id") == o.run_id);
    CHECK(is_null(*o.journal.find("identity"), "level"));
    for (const JsonValue& s : o.journal.find("sequences")->array) CHECK(str(s, "state") == "pending");
}

void expect_pass(const char* name, const Bundle& b, const Outcome& o) {
    if (!o.error.empty() || !o.other.empty()) {
        std::fprintf(stderr, "%s: expected to pass: %s%s\n", name, o.error.c_str(), o.other.c_str());
    }
    CHECK(o.result.has_value());
    CHECK(str(o.journal, "state") == "running");
    CHECK(is_null(o.journal, "failure"));
    check_untouched_journal(o);
    CHECK(probe_files(b.dir).empty());
}

void expect_refused(const char* name, const Bundle& b, const Outcome& o, const std::string& needle) {
    const bool ok = !o.result && o.other.empty() && o.error.rfind("preflight: ", 0) == 0 &&
                    o.error.find(needle) != std::string::npos;
    if (!ok) {
        std::fprintf(stderr, "%s: expected a preflight refusal naming '%s', got error='%s' other='%s'%s\n", name,
                     needle.c_str(), o.error.c_str(), o.other.c_str(), o.result ? " (passed)" : "");
    }
    CHECK(ok);
    CHECK(str(o.journal, "state") == "failed");
    const JsonValue* f = o.journal.find("failure");
    CHECK(f != nullptr && f->kind == JsonValue::Kind::Object);
    if (f != nullptr && f->kind == JsonValue::Kind::Object) {
        CHECK(is_null(*f, "sequence"));
        CHECK(str(*f, "message") == o.error);
    }
    check_untouched_journal(o);
    CHECK(!fs::exists(b.report));
    for (const std::string& s : kSeqs) CHECK(!fs::exists(b.out / (s + ".txt")));
    CHECK(probe_files(b.dir).empty());
}

struct Case {
    const char* name;
    std::function<void(Bundle&)> setup;
    const char* needle;  // nullptr: Gate A passes
    std::function<void(const Bundle&, const Outcome&)> extra;
};

void rewrite_lineage(Bundle& b, const std::function<void(JsonValue&)>& edit) {
    edit(b.lineage_json);
    write_file(b.lineage, sh::dump_python_json(b.lineage_json));
}

void rewrite_config(Bundle& b, const std::function<void(JsonValue&)>& edit) {
    JsonValue c = sh::parse_strict_json(read_file(b.config));
    edit(c);
    write_file(b.config, sh::dump_python_json(c));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4 && !(argc == 6 && std::string(argv[4]) == "--keep")) {
        std::fprintf(stderr, "usage: %s <resolved config> <lineage> <attestation> [--keep DIR]\n", argv[0]);
        return 2;
    }
    const Fixture fx{read_file(argv[1]), read_file(argv[2]), argv[3]};
    char tmpl[] = "/tmp/saccade_preflight_test.XXXXXX";
    const fs::path base = argc == 6 ? fs::path(argv[5]) : fs::path(::mkdtemp(tmpl));
    fs::create_directories(base);

    const std::string op_path = sh::parse_strict_json(fx.lineage_text).find("op_library")->find("path")->string;
    const std::string head_path = sh::parse_strict_json(fx.lineage_text).find("torchscript")->find("path")->string;
    const std::string engine_path = sh::parse_strict_json(fx.lineage_text)
                                        .find("companions")->find("backbone_engine")->find("path")->string;

    const std::vector<Case> cases = {
        {"pass", [](Bundle&) {}, nullptr,
         [](const Bundle& b, const Outcome& o) {
             if (!o.result) return;
             CHECK(o.result->schedule == sh::Schedule::DoubleBuffer);
             CHECK(o.result->detector.op_library.sha256 == sha(kOpLibrary));
             CHECK(o.result->detector.head_artifact.sha256 == sha(kHead));
             CHECK(o.result->detector.backbone_engine.sha256 == sha(kEngine));
             CHECK(!o.result->detector.op_library_from_attestation);
             for (const std::string& s : kSeqs) CHECK(fs::is_directory(b.trace / s));
         }},
        {"pass_max_frames",
         [](Bundle& b) {
             write_sequence(b.root / "seqs" / kSeqs[1], 3, 2);
             b.max_frames = 2;
         },
         nullptr, nullptr},
        {"config_missing", [](Bundle& b) { b.config = b.dir / "no_such.resolved.json"; }, "no_such.resolved.json",
         nullptr},
        {"config_torn_schedule",
         [](Bundle& b) {
             rewrite_config(b, [](JsonValue& c) {
                 at(at(at(c, "host_params"), "steps"), "schedule.double_buffer").boolean = false;
             });
         },
         "shipping schedule:", nullptr},
        {"lineage_missing", [](Bundle& b) { b.lineage = b.dir / "no_such.lineage.json"; }, "no_such.lineage.json",
         nullptr},
        {"lineage_disagrees",
         [](Bundle& b) {
             rewrite_lineage(b, [](JsonValue& l) { at(at(l, "preset"), "sha256").string = std::string(64, '0'); });
         },
         "preset path/sha256 differ", nullptr},
        {"attestation_other_lineage", [&](Bundle& b) { b.attestation = fx.attestation_path; },
         "bound to a different lineage", nullptr},
        {"sequence_no_seqinfo", [](Bundle& b) { fs::remove(b.root / "seqs" / kSeqs[1] / "seqinfo.ini"); },
         "seqinfo.ini", nullptr},
        {"sequence_short_listing", [](Bundle& b) { write_sequence(b.root / "seqs" / kSeqs[1], 4, 3); },
         "fewer than the 4 frames", nullptr},
        {"report_dir_missing", [](Bundle& b) { b.report = b.dir / "no_such_dir" / "report.json"; },
         "the directory of --report", nullptr},
        {"trace_unwritable",
         [](Bundle& b) {
             fs::create_directories(b.trace / kSeqs[0]);
             ::chmod((b.trace / kSeqs[0]).c_str(), 0500);
         },
         "cannot write in", nullptr},
        {"artifact_missing_op_library", [&](Bundle& b) { fs::remove(b.root / op_path); },
         "operator library", nullptr},
        {"artifact_missing_head", [&](Bundle& b) { fs::remove(b.root / head_path); }, "head artifact", nullptr},
        {"artifact_missing_engine", [&](Bundle& b) { fs::remove(b.root / engine_path); }, "backbone engine",
         nullptr},
        {"artifact_replaced_op_library", [&](Bundle& b) { write_file(b.root / op_path, "other bytes"); },
         "operator library", nullptr},
        {"artifact_replaced_head", [&](Bundle& b) { write_file(b.root / head_path, "other bytes"); },
         "head artifact", nullptr},
        {"artifact_replaced_engine", [&](Bundle& b) { write_file(b.root / engine_path, "other bytes"); },
         "backbone engine", nullptr},
    };

    // The trace case needs a directory this process cannot write in.
    const bool can_refuse_writes = ::geteuid() != 0;
    for (const Case& c : cases) {
        if (!can_refuse_writes && std::string(c.name) == "trace_unwritable") {
            std::fprintf(stderr, "%s: skipped (running as root)\n", c.name);
            continue;
        }
        const fs::path dir = base / c.name;
        fs::remove_all(dir);
        Bundle b = make_bundle(fx, dir);
        // Earlier outputs of this run's sequences: RunCompletion removes them.
        write_file(b.report, "earlier report");
        for (const std::string& s : kSeqs) write_file(b.out / (s + ".txt"), "earlier");
        c.setup(b);
        const Outcome o = run_gate(b);
        if (c.needle == nullptr) {
            expect_pass(c.name, b, o);
        } else {
            expect_refused(c.name, b, o, c.needle);
            if (std::string(c.name).rfind("artifact_", 0) == 0) {
                // The artifact checks come last: the earlier steps all passed.
                CHECK(o.error.find("sequence ") == std::string::npos);
            }
        }
        if (c.extra) c.extra(b, o);
        ::chmod((b.trace / kSeqs[0]).c_str(), 0700);
        std::fprintf(stderr, "%s: %s\n", c.name, o.error.empty() ? "passed" : o.error.c_str());
    }
    if (argc != 6) fs::remove_all(base);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
