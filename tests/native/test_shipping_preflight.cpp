// Gate A of saccade_track (#536 CC-536-01-02 N-T2,
// docs/architecture/ship_export_contracts_536.md). CPU only; CI job
// `shipping-config-loader`. Links saccade_shipping_preflight and no CUDA
// library: everything checked here runs without a device by construction.
//
// Usage: saccade_shipping_preflight_test <resolved config> <lineage> <attestation>
//        [--keep DIR] [--frozen-model-root ROOT]
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
//                        the journal is still running with checksum identity;
//   * pass_max_frames:   a sequence listing fewer frames than seqLength
//                        passes with --max-frames within the listing, and is
//                        refused with --max-frames beyond it;
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
// Journal v3 (#549 S2-1): the legacy identity says mode legacy, the requested
// policy, no allowlist and no bundle manifest, six bindings. Policy
// (--require-identity): checksum_matched passes; expected_source_verified is
// refused after Gate A with the checksum level kept (MB-57: legacy mode
// cannot exceed checksum_matched). Manifest mode:
// test_shipping_preflight_manifest.cpp.

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

const JsonValue& at(const JsonValue& o, const std::string& key) {
    const JsonValue* v = o.find(key);
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
    sh::IdentityLevel required = sh::IdentityLevel::None;
    JsonValue lineage_json;
};

void write_sequence(const fs::path& seq, int seq_length, int listed) {
    fs::remove_all(seq);
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
    JsonValue initial_identity;
    std::string run_id;
};

// The entrypoints' sequence: take <out>, run Gate A, record a failure.
Outcome run_gate(const Bundle& b) {
    Outcome o;
    o.run_id = sh::new_run_id();
    sh::RunOutputs outputs{b.out, b.report, b.trace, kSeqs};
    sh::PreflightInputs inputs{b.config.string(), b.lineage.string(), b.attestation,
                              b.root.string(), b.sequences, b.max_frames, false};
    inputs.required = b.required;
    std::optional<sh::RunCompletion> c;
    try {
        c.emplace(o.run_id, "saccade_shipping_preflight_test", outputs, false, b.required);
        o.initial_identity = *sh::parse_strict_json(read_file(b.out / sh::kRunJournalName)).find("identity");
        o.result = sh::run_preflight(inputs, *c);
    } catch (const sh::PreflightError& e) {
        o.error = e.what();
        if (c) {
            const JsonValue rejected = c->identity();
            const std::string journal_before = read_file(b.out / sh::kRunJournalName);
            std::string repeated_error;
            try {
                sh::run_preflight(inputs, *c);
            } catch (const sh::PreflightError& repeated) {
                repeated_error = repeated.what();
            }
            CHECK(repeated_error == "preflight: identity already finalized");
            CHECK(c->identity() == rejected);
            CHECK(read_file(b.out / sh::kRunJournalName) == journal_before);
            c->fail(e.what());
        }
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

void check_pending_sequences(const Outcome& o) {
    CHECK(str(o.journal, "run_id") == o.run_id);
    for (const JsonValue& s : o.journal.find("sequences")->array) CHECK(str(s, "state") == "pending");
}

const JsonValue& binding(const JsonValue& identity, const char* name) {
    const JsonValue* bindings = identity.find("bindings");
    if (bindings == nullptr || bindings->kind != JsonValue::Kind::Object || bindings->find(name) == nullptr) {
        throw std::runtime_error(std::string("no identity binding ") + name);
    }
    return *bindings->find(name);
}

void check_nullable_string(const JsonValue& object, const char* key, const std::optional<std::string>& want) {
    if (want) CHECK(str(object, key) == *want);
    else CHECK(is_null(object, key));
}

void check_binding(const JsonValue& identity, const char* name, const std::optional<std::string>& path,
                   const std::optional<std::string>& expected, const std::optional<std::string>& observed,
                   const char* status, const std::optional<std::string>& source_path = std::nullopt,
                   const std::optional<std::string>& pointer = std::nullopt) {
    const JsonValue& v = binding(identity, name);
    CHECK(v.kind == JsonValue::Kind::Object && v.object.size() == 5);
    check_nullable_string(v, "path", path);
    check_nullable_string(v, "expected_sha256", expected);
    check_nullable_string(v, "observed_sha256", observed);
    CHECK(str(v, "status") == status);
    if (source_path) {
        const JsonValue* src = v.find("expected_source");
        CHECK(src != nullptr && src->kind == JsonValue::Kind::Object && src->object.size() == 2);
        if (src != nullptr && src->kind == JsonValue::Kind::Object) {
            CHECK(str(*src, "path") == *source_path);
            CHECK(pointer.has_value() && str(*src, "json_pointer") == *pointer);
        }
    } else {
        CHECK(is_null(v, "expected_source"));
    }
}

void check_identity_shape(const JsonValue& identity) {
    CHECK(identity.kind == JsonValue::Kind::Object);
    CHECK(identity.object.size() == 9);
    CHECK(is_null(identity, "expected_source"));
    CHECK(str(identity, "publisher_authentication") == "not_checked_by_runtime");
    CHECK(str(identity, "mode") == "legacy");
    const std::string& required = str(identity, "required");
    CHECK(required == "none" || required == "checksum_matched" || required == "expected_source_verified");
    CHECK(is_null(identity, "allowlist_sha256"));
    CHECK(is_null(identity, "allowlist_entry"));
    CHECK(is_null(identity, "bundle_manifest_sha256"));
    CHECK(is_null(identity, "level") || str(identity, "level") == "checksum_matched");
    const JsonValue* bindings = identity.find("bindings");
    CHECK(bindings != nullptr && bindings->kind == JsonValue::Kind::Object && bindings->object.size() == 6);
    for (const char* key : {"config", "lineage", "attestation", "op_library", "head", "engine"}) {
        CHECK(binding(identity, key).kind == JsonValue::Kind::Object);
    }
}

void check_initial_identity(const JsonValue& identity) {
    check_identity_shape(identity);
    CHECK(is_null(identity, "level"));
    for (const char* key : {"config", "lineage", "attestation", "op_library", "head", "engine"}) {
        check_binding(identity, key, std::nullopt, std::nullopt, std::nullopt, "unchecked");
    }
}

void check_success_identity(const Bundle& b, const Outcome& o) {
    const JsonValue& identity = *o.journal.find("identity");
    check_identity_shape(identity);
    CHECK(str(identity, "level") == "checksum_matched");
    check_binding(identity, "config", b.config.string(), std::nullopt, sha(read_file(b.config)), "unchecked");
    const std::string lineage_sha = sha(read_file(b.lineage));
    if (b.attestation.empty()) {
        check_binding(identity, "lineage", b.lineage.string(), std::nullopt, lineage_sha, "unchecked");
        check_binding(identity, "attestation", std::nullopt, std::nullopt, std::nullopt, "unchecked");
    } else {
        check_binding(identity, "lineage", b.lineage.string(), lineage_sha, lineage_sha, "matched",
                      b.attestation, "/frozen_lineage/sha256");
        check_binding(identity, "attestation", b.attestation, std::nullopt, sha(read_file(b.attestation)), "unchecked");
    }
    if (!o.result) return;
    const auto& d = o.result->detector;
    check_binding(identity, "op_library", sh::resolve_model_path(b.root.string(), d.op_library.path),
                  d.op_library.sha256, d.op_library.sha256, "matched",
                  d.op_library_from_attestation ? b.attestation : b.lineage.string(), "/op_library/sha256");
    check_binding(identity, "head", sh::resolve_model_path(b.root.string(), d.head_artifact.path),
                  d.head_artifact.sha256, d.head_artifact.sha256, "matched", b.lineage.string(), "/torchscript/sha256");
    check_binding(identity, "engine", sh::resolve_model_path(b.root.string(), d.backbone_engine.path),
                  d.backbone_engine.sha256, d.backbone_engine.sha256, "matched", b.lineage.string(),
                  "/companions/backbone_engine/sha256");
}

void expect_pass(const char* name, const Bundle& b, const Outcome& o) {
    if (!o.error.empty() || !o.other.empty()) {
        std::fprintf(stderr, "%s: expected to pass: %s%s\n", name, o.error.c_str(), o.other.c_str());
    }
    CHECK(o.result.has_value());
    CHECK(str(o.journal, "state") == "running");
    CHECK(is_null(o.journal, "failure"));
    check_pending_sequences(o);
    check_initial_identity(o.initial_identity);
    check_success_identity(b, o);
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
    check_pending_sequences(o);
    check_initial_identity(o.initial_identity);
    check_identity_shape(*o.journal.find("identity"));
    CHECK(is_null(*o.journal.find("identity"), "level"));
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

void make_attestation(const Fixture& fx, Bundle& b) {
    JsonValue a = sh::parse_strict_json(read_file(fx.attestation_path));
    JsonValue& frozen = at(a, "frozen_lineage");
    at(frozen, "sha256").string = sha(read_file(b.lineage));
    at(frozen, "torchscript_sha256").string = at(at(b.lineage_json, "torchscript"), "sha256").string;
    at(frozen, "torchscript_content_sha256").string = at(at(b.lineage_json, "torchscript"), "content_sha256").string;
    at(frozen, "op_library_sha256").string = at(at(b.lineage_json, "op_library"), "sha256").string;
    at(at(a, "op_library"), "path").string = at(at(b.lineage_json, "op_library"), "path").string;
    at(at(a, "op_library"), "sha256").string = sha(kOpLibrary);
    b.attestation = (b.dir / "attestation.json").string();
    write_file(b.attestation, sh::dump_python_json(a));
}

void rewrite_attestation(Bundle& b, const std::function<void(JsonValue&)>& edit) {
    JsonValue a = sh::parse_strict_json(read_file(b.attestation));
    edit(a);
    write_file(b.attestation, sh::dump_python_json(a));
}

void check_gate_a_once(const Fixture& fx, const fs::path& dir) {
    const Bundle b = make_bundle(fx, dir);
    sh::RunCompletion completion(sh::new_run_id(), "saccade_shipping_preflight_test",
                                 sh::RunOutputs{b.out, b.report, b.trace, kSeqs});
    const sh::PreflightInputs inputs{b.config.string(), b.lineage.string(), "", b.root.string(), b.sequences};
    sh::run_preflight(inputs, completion);
    const JsonValue sealed = completion.identity();
    const std::string journal_before = read_file(b.out / sh::kRunJournalName);
    // A repeated Gate A must refuse before observing these changed bytes.
    write_file(b.config, "corrupt config after Gate A");
    std::string refusal;
    try {
        sh::run_preflight(inputs, completion);
    } catch (const sh::PreflightError& e) {
        refusal = e.what();
    }
    CHECK(refusal == "preflight: identity already finalized");
    CHECK(completion.identity() == sealed);
    CHECK(read_file(b.out / sh::kRunJournalName) == journal_before);
    completion.fail("later load failed");
    const JsonValue failed = sh::parse_strict_json(read_file(b.out / sh::kRunJournalName));
    CHECK(*failed.find("identity") == sealed);
    CHECK(str(failed, "state") == "failed");
    CHECK(str(*failed.find("failure"), "message") == "later load failed");
}

// --require-identity in legacy mode: the policy is compared after every Gate A
// check, against the sealed level.
void check_legacy_policy(const Fixture& fx, const fs::path& dir) {
    {
        Bundle b = make_bundle(fx, dir / "require_checksum");
        b.required = sh::IdentityLevel::ChecksumMatched;
        const Outcome o = run_gate(b);
        expect_pass("legacy_require_checksum", b, o);
        CHECK(str(*o.journal.find("identity"), "required") == "checksum_matched");
    }
    {
        // MB-57: legacy mode cannot exceed checksum_matched.
        Bundle b = make_bundle(fx, dir / "require_expected_source");
        b.required = sh::IdentityLevel::ExpectedSourceVerified;
        const Outcome o = run_gate(b);
        CHECK(!o.result);
        CHECK(o.error == "preflight: identity level checksum_matched is below --require-identity "
                         "expected_source_verified (legacy mode cannot exceed checksum_matched)");
        CHECK(str(o.journal, "state") == "failed");
        CHECK(str(*o.journal.find("failure"), "message") == o.error);
        const JsonValue& id = *o.journal.find("identity");
        check_identity_shape(id);
        CHECK(str(id, "level") == "checksum_matched");  // the proven level is kept
        CHECK(str(id, "required") == "expected_source_verified");
        CHECK(is_null(o.journal, "load_verification"));
        CHECK(str(binding(id, "engine"), "status") == "matched");
        check_pending_sequences(o);
        std::fprintf(stderr, "legacy_require_expected_source: %s\n", o.error.c_str());
    }
}

void check_frozen_model(const fs::path& dir, const fs::path& root, const fs::path& config,
                        const fs::path& lineage, const std::string& attestation) {
    Bundle b;
    b.dir = dir;
    b.root = root;
    b.config = config;
    b.lineage = lineage;
    b.attestation = attestation;
    b.out = dir / "out";
    b.report = dir / "report.json";
    b.trace = dir / "trace";
    fs::create_directories(dir);
    for (const std::string& name : kSeqs) {
        const fs::path sequence = dir / "seqs" / name;
        write_sequence(sequence, 3, 3);
        b.sequences.push_back(sequence.string());
    }
    // Read actual frozen inputs directly. There is no model copy, decode,
    // dlopen, engine deserialization or CUDA link in this executable.
    const Outcome outcome = run_gate(b);
    expect_pass("frozen_model", b, outcome);
    CHECK(outcome.result && outcome.result->detector.op_library_from_attestation);
    std::fprintf(stderr, "frozen_model: %s\n", outcome.error.empty() ? "passed" : outcome.error.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <resolved config> <lineage> <attestation> [--keep DIR] [--frozen-model-root ROOT]\n", argv[0]);
        return 2;
    }
    fs::path keep, frozen_model_root;
    for (int i = 4; i < argc; i += 2) {
        if (i + 1 >= argc) return 2;
        const std::string option(argv[i]);
        if (option == "--keep") keep = argv[i + 1];
        else if (option == "--frozen-model-root") frozen_model_root = argv[i + 1];
        else return 2;
    }
    const Fixture fx{read_file(argv[1]), read_file(argv[2]), argv[3]};
    char tmpl[] = "/tmp/saccade_preflight_test.XXXXXX";
    const fs::path base = keep.empty() ? fs::path(::mkdtemp(tmpl)) : keep;
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
        {"pass_attestation", [&](Bundle& b) { make_attestation(fx, b); }, nullptr,
         [](const Bundle&, const Outcome& o) { CHECK(o.result && o.result->detector.op_library_from_attestation); }},
        {"pass_attestation_operator_override",
         [&](Bundle& b) {
             make_attestation(fx, b);
             write_file(b.root / op_path, "realized operator bytes");
             rewrite_attestation(b, [](JsonValue& a) {
                 at(at(a, "op_library"), "sha256").string = sha("realized operator bytes");
             });
         },
         nullptr, [](const Bundle&, const Outcome& o) {
             CHECK(o.result && o.result->detector.op_library.sha256 == sha("realized operator bytes"));
         }},
        {"max_frames_beyond_listing",
         [](Bundle& b) {
             write_sequence(b.root / "seqs" / kSeqs[1], 3, 2);
             b.max_frames = 3;
         },
         "fewer than the 3 frames", nullptr},
        {"config_missing", [](Bundle& b) { b.config = b.dir / "no_such.resolved.json"; }, "no_such.resolved.json",
         [](const Bundle& b, const Outcome& o) {
             check_binding(*o.journal.find("identity"), "config", b.config.string(), std::nullopt,
                           std::nullopt, "missing");
         }},
        {"config_not_regular", [](Bundle& b) { fs::remove(b.config); fs::create_directory(b.config); },
         "not a regular file", [](const Bundle& b, const Outcome& o) {
             check_binding(*o.journal.find("identity"), "config", b.config.string(), std::nullopt,
                           std::nullopt, "missing");
         }},
        {"config_malformed", [](Bundle& b) { write_file(b.config, "{bad config"); }, "json syntax at byte 1: expected an object key string",
         [](const Bundle& b, const Outcome& o) {
             check_binding(*o.journal.find("identity"), "config", b.config.string(), std::nullopt,
                           sha(read_file(b.config)), "unchecked");
         }},
        {"config_torn_schedule",
         [](Bundle& b) {
             rewrite_config(b, [](JsonValue& c) {
                 at(at(at(c, "host_params"), "steps"), "schedule.double_buffer").boolean = false;
             });
         },
         "shipping schedule:", nullptr},
        {"lineage_missing", [](Bundle& b) { b.lineage = b.dir / "no_such.lineage.json"; }, "no_such.lineage.json",
         [](const Bundle& b, const Outcome& o) {
             check_binding(*o.journal.find("identity"), "lineage", b.lineage.string(), std::nullopt,
                           std::nullopt, "missing");
         }},
        {"lineage_not_regular", [](Bundle& b) { fs::remove(b.lineage); fs::create_directory(b.lineage); },
         "not a regular file", [](const Bundle& b, const Outcome& o) {
             check_binding(*o.journal.find("identity"), "lineage", b.lineage.string(), std::nullopt,
                           std::nullopt, "missing");
         }},
        {"lineage_malformed", [](Bundle& b) { write_file(b.lineage, "{bad lineage"); }, "json syntax at byte 1: expected an object key string",
         [](const Bundle& b, const Outcome& o) {
             check_binding(*o.journal.find("identity"), "lineage", b.lineage.string(), std::nullopt,
                           sha(read_file(b.lineage)), "unchecked");
         }},
        {"lineage_disagrees",
         [](Bundle& b) {
             rewrite_lineage(b, [](JsonValue& l) { at(at(l, "preset"), "sha256").string = std::string(64, '0'); });
         },
         "preset path/sha256 differ", nullptr},
        {"attestation_other_lineage", [&](Bundle& b) { b.attestation = fx.attestation_path; },
         "bound to a different lineage", [](const Bundle& b, const Outcome& o) {
             const JsonValue a = sh::parse_strict_json(read_file(b.attestation));
             check_binding(*o.journal.find("identity"), "lineage", b.lineage.string(),
                           str(*a.find("frozen_lineage"), "sha256"), sha(read_file(b.lineage)), "mismatch",
                           b.attestation, "/frozen_lineage/sha256");
         }},
        {"attestation_missing", [](Bundle& b) { b.attestation = (b.dir / "missing.attestation.json").string(); },
         "missing.attestation.json", [](const Bundle& b, const Outcome& o) {
             check_binding(*o.journal.find("identity"), "attestation", b.attestation, std::nullopt,
                           std::nullopt, "missing");
         }},
        {"attestation_not_regular", [](Bundle& b) {
             b.attestation = (b.dir / "attestation.directory").string();
             fs::create_directory(b.attestation);
         }, "not a regular file", [](const Bundle& b, const Outcome& o) {
             check_binding(*o.journal.find("identity"), "attestation", b.attestation, std::nullopt,
                           std::nullopt, "missing");
         }},
        {"attestation_malformed", [](Bundle& b) {
             b.attestation = (b.dir / "bad.attestation.json").string();
             write_file(b.attestation, "{bad attestation");
         }, "json syntax at byte 1: expected an object key string", [](const Bundle& b, const Outcome& o) {
             check_binding(*o.journal.find("identity"), "attestation", b.attestation, std::nullopt,
                           sha(read_file(b.attestation)), "unchecked");
         }},
        {"attestation_bad_schema", [&](Bundle& b) {
             make_attestation(fx, b);
             rewrite_attestation(b, [](JsonValue& a) { at(a, "schema").string = "unsupported"; });
         }, "realization attestation: schema is not", [](const Bundle& b, const Outcome& o) {
             check_binding(*o.journal.find("identity"), "lineage", b.lineage.string(), std::nullopt,
                           sha(read_file(b.lineage)), "unchecked");
         }},
        {"attestation_bad_expected_hash", [&](Bundle& b) {
             make_attestation(fx, b);
             rewrite_attestation(b, [](JsonValue& a) { at(at(a, "frozen_lineage"), "sha256").string = "bad"; });
         }, "frozen_lineage.sha256 is not a sha256 hex digest", [](const Bundle& b, const Outcome& o) {
             check_binding(*o.journal.find("identity"), "lineage", b.lineage.string(), std::nullopt,
                           sha(read_file(b.lineage)), "unchecked");
         }},
        {"attestation_head_metadata", [&](Bundle& b) {
             make_attestation(fx, b);
             rewrite_attestation(b, [](JsonValue& a) {
                 at(at(a, "frozen_lineage"), "torchscript_sha256").string = std::string(64, '0');
             });
         }, "frozen_lineage disagrees with the lineage", nullptr},
        {"attestation_content_metadata", [&](Bundle& b) {
             make_attestation(fx, b);
             rewrite_attestation(b, [](JsonValue& a) {
                 at(at(a, "frozen_lineage"), "torchscript_content_sha256").string = std::string(64, '0');
             });
         }, "frozen_lineage disagrees with the lineage", nullptr},
        {"attestation_operator_metadata", [&](Bundle& b) {
             make_attestation(fx, b);
             rewrite_attestation(b, [](JsonValue& a) {
                 at(at(a, "frozen_lineage"), "op_library_sha256").string = std::string(64, '0');
             });
         }, "frozen_lineage disagrees with the lineage", nullptr},
        {"attestation_operator_path", [&](Bundle& b) {
             make_attestation(fx, b);
             rewrite_attestation(b, [](JsonValue& a) { at(at(a, "op_library"), "path").string = "another.so"; });
         }, "op_library.path differs from the lineage", nullptr},
        {"attestation_operator_python_dependency", [&](Bundle& b) {
             make_attestation(fx, b);
             rewrite_attestation(b, [](JsonValue& a) {
                 at(at(a, "op_library"), "needed").array.push_back(JsonValue::make_string("libpython3.so"));
             });
         }, "realized op library links libpython3.so", nullptr},
        {"attestation_reproduction_false", [&](Bundle& b) {
             make_attestation(fx, b);
             rewrite_attestation(b, [](JsonValue& a) { at(at(a, "a_l_reproduction"), "identical").boolean = false; });
         }, "A_L reproduction is not byte-identical", nullptr},
        {"attestation_operator_hash_mismatch", [&](Bundle& b) {
             make_attestation(fx, b);
             rewrite_attestation(b, [](JsonValue& a) { at(at(a, "op_library"), "sha256").string = std::string(64, '0'); });
         }, "operator library", [](const Bundle& b, const Outcome& o) {
             check_binding(*o.journal.find("identity"), "op_library",
                           (b.root / at(at(b.lineage_json, "op_library"), "path").string).string(),
                           std::string(64, '0'), sha(kOpLibrary), "mismatch", b.attestation, "/op_library/sha256");
         }},
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
        {"artifact_not_regular_op_library", [&](Bundle& b) {
             fs::remove(b.root / op_path); fs::create_directory(b.root / op_path);
         }, "operator library", nullptr},
        {"artifact_not_regular_head", [&](Bundle& b) {
             fs::remove(b.root / head_path); fs::create_directory(b.root / head_path);
         }, "head artifact", nullptr},
        {"artifact_not_regular_engine", [&](Bundle& b) {
             fs::remove(b.root / engine_path); fs::create_directory(b.root / engine_path);
         }, "backbone engine", nullptr},
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
        if (std::string(c.name).rfind("artifact_", 0) == 0) {
            const bool replaced = std::string(c.name).rfind("artifact_replaced_", 0) == 0;
            const bool missing = !replaced;
            const std::string key = std::string(c.name).substr(replaced ? 18 :
                                    std::string(c.name).rfind("artifact_missing_", 0) == 0 ? 17 : 21);
            const JsonValue& id = *o.journal.find("identity");
            const std::string path = key == "op_library" ? op_path : key == "head" ? head_path : engine_path;
            const std::string expected = sha(key == "op_library" ? kOpLibrary : key == "head" ? kHead : kEngine);
            const std::string pointer = key == "op_library" ? "/op_library/sha256" :
                                        key == "head" ? "/torchscript/sha256" : "/companions/backbone_engine/sha256";
            check_binding(id, key.c_str(), (b.root / path).string(), expected,
                          missing ? std::nullopt : std::optional<std::string>{sha("other bytes")},
                          missing ? "missing" : "mismatch", b.lineage.string(), pointer);
            CHECK(o.error.find(missing ? "is missing (not a regular file)" :
                              "sha256 " + sha("other bytes") + " != " + expected) != std::string::npos);
            if (key != "op_library") CHECK(str(binding(id, "op_library"), "status") == "matched");
            if (key == "engine") CHECK(str(binding(id, "head"), "status") == "matched");
            if (key == "op_library") CHECK(str(binding(id, "head"), "status") == "unchecked");
            if (key != "engine") CHECK(str(binding(id, "engine"), "status") == "unchecked");
        }
        const std::string case_name(c.name);
        if (case_name == "attestation_head_metadata" || case_name == "attestation_content_metadata" ||
            case_name == "attestation_operator_metadata" || case_name == "attestation_operator_path" ||
            case_name == "attestation_operator_python_dependency" || case_name == "attestation_reproduction_false") {
            // A matching SHA records only a comparison. Later metadata-field
            // rejection cannot erase it or elevate the whole identity.
            const JsonValue& id = *o.journal.find("identity");
            check_binding(id, "lineage", b.lineage.string(), sha(read_file(b.lineage)), sha(read_file(b.lineage)),
                          "matched", b.attestation, "/frozen_lineage/sha256");
            check_binding(id, "op_library", (b.root / op_path).string(), sha(kOpLibrary), std::nullopt,
                          "unchecked", b.lineage.string(), "/op_library/sha256");
        }
        if (fs::is_regular_file(b.config)) {
            CHECK(str(binding(*o.journal.find("identity"), "config"), "observed_sha256") == sha(read_file(b.config)));
        }
        ::chmod((b.trace / kSeqs[0]).c_str(), 0700);
        std::fprintf(stderr, "%s: %s\n", c.name, o.error.empty() ? "passed" : o.error.c_str());
    }
    check_gate_a_once(fx, base / "gate_a_once");
    check_legacy_policy(fx, base / "legacy_policy");
    if (!frozen_model_root.empty()) check_frozen_model(base / "frozen_model", frozen_model_root, argv[1], argv[2], argv[3]);
    if (keep.empty()) fs::remove_all(base);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
