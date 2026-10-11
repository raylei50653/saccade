// Gate A of saccade_track in manifest mode (#549 S2-1, --model-bundle;
// saccade_shipping/preflight.hpp, model_bundle.hpp). CPU only; CI job
// `shipping-config-loader`. Links saccade_shipping_preflight and no CUDA
// library: nothing here can initialize a device.
//
// Usage: saccade_shipping_preflight_manifest_test <resolved config> <lineage>
//        <attestation> <example manifest> [--keep DIR] [--frozen-model-root ROOT]
//
// Each case builds, in its own directory, two roots:
//   runtime/  the runtime package's share/saccade/: a stand-in operator
//             library at the lineage's op_library path and an allowlist;
//   bundle/   the model bundle: the config, a copy of the lineage whose three
//             sha256 are those of small stand-in files, an attestation bound
//             to that lineage and the stand-in operator, the stand-in head and
//             engine, and model_bundle.json -- the published example manifest
//             with every member's bytes / sha256 set to these files.
// Gate A gets the allowlist's sha256 as the entrypoint would (the build pin),
// and each case changes one thing. The verification-matrix rows each case
// exercises (docs/architecture/model_bundle_549/verification_matrix.json):
//   MB-30 approved entry -> expected_source_verified (also under the policy);
//   MB-31 self-consistent substitution (new engine, lineage, attestation and
//         manifest) -> checksum_matched; refused under expected_source_verified;
//   MB-32 allowlist bytes other than the pin (and a missing allowlist);
//   MB-33 example / revoked entry -> checksum_matched; refused under the policy;
//   MB-35 config bytes edited, still field-consistent;
//   MB-36 a member whose size differs (refused before it is hashed);
//   MB-37 a member missing; MB-38 a member that is a symlink (inside / outside);
//   MB-39 a directory below the bundle root that is a symlink out of it;
//   MB-40 the whole bundle relocated (and reached through a symlinked root);
//   MB-14 an unknown manifest schema; R-03 a pairing to the wrong role;
//   MB-57's manifest counterpart: required levels at and above the proven one.
// Every refusal: PreflightError naming the check, journal failed, the
// bindings' statuses, level null -- except a policy refusal, which keeps the
// proven level and records `required`. Every pass: the resolved bindings are
// the absolute paths beneath each root with their bytes and sha256.
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "saccade_shipping/model_bundle.hpp"
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

// string_view keys: no temporary the result could seem to refer to.
JsonValue& at(JsonValue& o, std::string_view key) {
    JsonValue* v = o.find(key);
    if (v == nullptr) throw std::runtime_error("fixture has no " + std::string(key));
    return *v;
}

const JsonValue& at(const JsonValue& o, std::string_view key) {
    const JsonValue* v = o.find(key);
    if (v == nullptr) throw std::runtime_error("fixture has no " + std::string(key));
    return *v;
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

const std::string kOpLibrary = "operator library stand-in";
const std::string kHead = "head artifact stand-in";
const std::string kEngine = "backbone engine stand-in";
const std::vector<std::string> kSeqs = {"SEQ-A", "SEQ-B"};
const char* kApproval = R"({"decision_ref": "https://github.com/raylei50653/saccade/issues/549#issuecomment-1", "date": "2026-10-11"})";

struct Fixture {
    std::string config_text, lineage_text, attestation_text, manifest_text, allowlist_text;
};

// Member ids of the example manifest, by slot.
struct Ids {
    std::size_t backbone = 0, head = 1, op = 2, config = 3, lineage = 4, attestation = 5;
};

struct Bundle {
    fs::path dir, bundle, runtime, out, report, trace;
    JsonValue lineage_json, attestation_json, manifest_json, allowlist_json;
    std::string pin;  // the allowlist sha256 "built into the entrypoint"
    sh::IdentityLevel required = sh::IdentityLevel::None;
    std::vector<std::string> sequences;
    std::string model_bundle;  // what --model-bundle names (default: bundle)
    std::string legacy_config;  // a legacy option given too (refused)

    const JsonValue& member(std::size_t i) const { return at(manifest_json, "members").array.at(i); }
    fs::path member_path(std::size_t i) const {
        const JsonValue& m = member(i);
        return (str(m, "carried_by") == "model_bundle" ? bundle : runtime) / str(m, "path");
    }
    std::string manifest_sha() const { return sha(read_file(bundle / sh::kModelBundleManifestName)); }
};

void write_sequence(const fs::path& seq, int seq_length) {
    fs::remove_all(seq);
    write_file(seq / "seqinfo.ini", "[Sequence]\nname=" + seq.filename().string() +
                                        "\nimWidth=8\nimHeight=6\nseqLength=" + std::to_string(seq_length) + "\n");
    for (int k = 1; k <= seq_length; ++k) {
        char name[16];
        std::snprintf(name, sizeof name, "%06d.jpg", k);
        write_file(seq / "img1" / name, "not decoded by Gate A");
    }
}

// Writes the manifest (members' bytes / sha256 = the files now on disk).
void write_manifest(Bundle& b) {
    JsonValue& members = at(b.manifest_json, "members");
    for (std::size_t i = 0; i < members.array.size(); ++i) {
        const std::string bytes = read_file(b.member_path(i));
        at(members.array[i], "bytes") = JsonValue::make_int(static_cast<std::int64_t>(bytes.size()));
        at(members.array[i], "sha256").string = sha(bytes);
    }
    write_file(b.bundle / sh::kModelBundleManifestName, sh::dump_python_json(b.manifest_json) + "\n");
}

void write_allowlist(Bundle& b) {
    const std::string text = sh::dump_python_json(b.allowlist_json) + "\n";
    write_file(b.runtime / sh::kTrustedModelBundlesName, text);
    b.pin = sha(text);
}

// An allowlist entry for the current manifest, in `state`.
void allow(Bundle& b, const char* state) {
    JsonValue e = JsonValue::make_object();
    e.set("manifest_sha256", JsonValue::make_string(b.manifest_sha()));
    e.set("bundle_name", JsonValue::make_string("saccade-headline-s-mamba"));
    e.set("bundle_version", JsonValue::make_string("0.1.0-test"));
    e.set("state", JsonValue::make_string(state));
    const std::string s = state;
    e.set("approval", s == "example" ? JsonValue::make_null() : sh::parse_strict_json(kApproval));
    if (s == "revoked") {
        JsonValue r = sh::parse_strict_json(kApproval);
        r.set("reason", JsonValue::make_string("test revocation"));
        e.set("revocation", r);
    } else {
        e.set("revocation", JsonValue::make_null());
    }
    at(b.allowlist_json, "entries").array.push_back(e);
    write_allowlist(b);
}

void write_lineage_and_attestation(Bundle& b, const Ids& ids) {
    write_file(b.member_path(ids.lineage), sh::dump_python_json(b.lineage_json));
    JsonValue& frozen = at(b.attestation_json, "frozen_lineage");
    at(frozen, "sha256").string = sha(read_file(b.member_path(ids.lineage)));
    at(frozen, "torchscript_sha256").string = str(at(b.lineage_json, "torchscript"), "sha256");
    at(frozen, "torchscript_content_sha256").string = str(at(b.lineage_json, "torchscript"), "content_sha256");
    at(frozen, "op_library_sha256").string = str(at(b.lineage_json, "op_library"), "sha256");
    at(at(b.attestation_json, "op_library"), "path").string = str(at(b.lineage_json, "op_library"), "path");
    at(at(b.attestation_json, "op_library"), "sha256").string = sha(read_file(b.member_path(ids.op)));
    write_file(b.member_path(ids.attestation), sh::dump_python_json(b.attestation_json));
}

Bundle make_bundle(const Fixture& fx, const fs::path& dir) {
    const Ids ids;
    Bundle b;
    b.dir = dir;
    b.bundle = dir / "bundle";
    b.runtime = dir / "runtime";
    b.out = dir / "out";
    b.report = dir / "report.json";
    b.trace = dir / "trace";
    b.model_bundle = b.bundle.string();
    b.manifest_json = sh::parse_strict_json(fx.manifest_text);
    b.allowlist_json = sh::parse_strict_json(fx.allowlist_text);
    at(b.allowlist_json, "entries").array.clear();  // as the production allowlist
    b.lineage_json = sh::parse_strict_json(fx.lineage_text);
    b.attestation_json = sh::parse_strict_json(fx.attestation_text);
    write_file(b.member_path(ids.op), kOpLibrary);
    write_file(b.member_path(ids.head), kHead);
    write_file(b.member_path(ids.backbone), kEngine);
    write_file(b.member_path(ids.config), fx.config_text);
    at(at(b.lineage_json, "op_library"), "sha256").string = sha(kOpLibrary);
    at(at(b.lineage_json, "torchscript"), "sha256").string = sha(kHead);
    at(at(at(b.lineage_json, "companions"), "backbone_engine"), "sha256").string = sha(kEngine);
    write_lineage_and_attestation(b, ids);
    write_manifest(b);
    write_allowlist(b);
    for (const std::string& s : kSeqs) {
        write_sequence(dir / "seqs" / s, 3);
        b.sequences.push_back((dir / "seqs" / s).string());
    }
    return b;
}

struct Outcome {
    std::optional<sh::PreflightResult> result;
    std::string error, other;
    JsonValue journal, initial_identity;
};

sh::PreflightInputs inputs_of(const Bundle& b) {
    sh::PreflightInputs in;
    in.config = b.legacy_config;
    in.model_root.clear();
    in.sequences = b.sequences;
    in.required = b.required;
    in.model_bundle = b.model_bundle;
    in.runtime_root = b.runtime.string();
    in.allowlist_sha256 = b.pin;
    return in;
}

Outcome run_gate(const Bundle& b) {
    Outcome o;
    std::optional<sh::RunCompletion> c;
    const sh::PreflightInputs in = inputs_of(b);
    try {
        c.emplace(sh::new_run_id(), "saccade_shipping_preflight_manifest_test",
                  sh::RunOutputs{b.out, b.report, b.trace, kSeqs}, true, b.required);
        o.initial_identity = *sh::parse_strict_json(read_file(b.out / sh::kRunJournalName)).find("identity");
        o.result = sh::run_preflight(in, *c);
    } catch (const sh::PreflightError& e) {
        o.error = e.what();
        if (c) {
            const JsonValue sealed = c->identity();
            std::string again;
            try {
                sh::run_preflight(in, *c);
            } catch (const sh::PreflightError& r) {
                again = r.what();
            }
            CHECK(again == "preflight: identity already finalized");
            CHECK(c->identity() == sealed);
            c->fail(e.what());
        }
    } catch (const std::exception& e) {
        o.other = e.what();
        if (c) c->fail(e.what());
    }
    o.journal = sh::parse_strict_json(read_file(b.out / sh::kRunJournalName));
    return o;
}

const JsonValue& identity(const Outcome& o) { return at(o.journal, "identity"); }
const JsonValue& binding(const JsonValue& id, const char* name) { return at(at(id, "bindings"), name); }

void check_identity_shape(const JsonValue& id, const sh::IdentityLevel required) {
    CHECK(str(id, "publisher_authentication") == "not_checked_by_runtime");
    CHECK(str(id, "mode") == "model_bundle");
    CHECK(str(id, "required") == sh::identity_level_name(required));
    const JsonValue& bindings = at(id, "bindings");
    CHECK(bindings.object.size() == 7);
    const char* names[] = {"bundle_manifest", "config", "lineage", "attestation", "op_library", "head", "engine"};
    for (std::size_t i = 0; i < 7 && i < bindings.object.size(); ++i) CHECK(bindings.object[i].first == names[i]);
    for (const auto& [name, v] : bindings.object) {
        CHECK(v.object.size() == 5);
        for (const char* k : {"path", "expected_sha256", "observed_sha256", "status", "expected_source"}) {
            CHECK(v.find(k) != nullptr);
        }
    }
}

void check_initial(const Outcome& o, sh::IdentityLevel required) {
    check_identity_shape(o.initial_identity, required);
    CHECK(is_null(o.initial_identity, "level"));
    CHECK(is_null(o.initial_identity, "allowlist_sha256"));
    for (const auto& [name, v] : at(o.initial_identity, "bindings").object) CHECK(str(v, "status") == "unchecked");
}

// A member binding after VL1: its absolute path, the manifest's expectation.
void check_member(const Bundle& b, const JsonValue& id, const char* name, std::size_t index, const char* status,
                  bool observed) {
    const JsonValue& v = binding(id, name);
    CHECK(str(v, "path") == fs::canonical(b.member_path(index).parent_path()).string() + "/" +
                                b.member_path(index).filename().string());
    CHECK(str(v, "expected_sha256") == str(b.member(index), "sha256"));
    CHECK(str(v, "status") == status);
    if (observed) CHECK(str(v, "observed_sha256") == sha(read_file(b.member_path(index))));
    else CHECK(is_null(v, "observed_sha256"));
    const JsonValue& src = at(v, "expected_source");
    CHECK(str(src, "path") == fs::canonical(b.bundle).string() + "/model_bundle.json");
    CHECK(str(src, "json_pointer") == "/members/" + std::to_string(index) + "/sha256");
}

void expect_pass(const char* name, const Bundle& b, const Outcome& o, const char* level, const char* entry) {
    if (!o.error.empty() || !o.other.empty()) {
        std::fprintf(stderr, "%s: expected to pass: %s%s\n", name, o.error.c_str(), o.other.c_str());
    }
    CHECK(o.result.has_value());
    CHECK(str(o.journal, "state") == "running");
    CHECK(is_null(o.journal, "load_verification"));
    check_initial(o, b.required);
    const JsonValue& id = identity(o);
    check_identity_shape(id, b.required);
    CHECK(str(id, "level") == level);
    CHECK(str(id, "allowlist_entry") == entry);
    CHECK(str(id, "allowlist_sha256") == b.pin);
    CHECK(str(id, "bundle_manifest_sha256") == b.manifest_sha());
    if (std::string(level) == "expected_source_verified") CHECK(str(id, "expected_source") == "runtime_allowlist");
    else CHECK(is_null(id, "expected_source"));
    const Ids ids;
    check_member(b, id, "config", ids.config, "matched", true);
    check_member(b, id, "lineage", ids.lineage, "matched", true);
    check_member(b, id, "attestation", ids.attestation, "matched", true);
    check_member(b, id, "op_library", ids.op, "matched", true);
    check_member(b, id, "head", ids.head, "matched", true);
    check_member(b, id, "engine", ids.backbone, "matched", true);
    const JsonValue& mb = binding(id, "bundle_manifest");
    CHECK(str(mb, "observed_sha256") == b.manifest_sha());
    CHECK(str(mb, "status") == (std::string(entry) == "absent" ? "unchecked" : "matched"));
    if (!o.result) return;
    const auto& r = o.result->detector.resolved;
    CHECK(r != nullptr);
    if (r == nullptr) return;
    auto check_resolved = [&](const sh::ResolvedBinding& rb, std::size_t index, const char* role, const char* root) {
        CHECK(rb.role == role);
        CHECK(rb.root_kind == root);
        CHECK(rb.absolute_path == str(binding(id, index == ids.op ? "op_library" : index == ids.head ? "head" : "engine"),
                                      "path"));
        CHECK(rb.bytes == static_cast<std::int64_t>(read_file(b.member_path(index)).size()));
        CHECK(rb.sha256 == sha(read_file(b.member_path(index))));
    };
    check_resolved(r->op_library, ids.op, "scan_operator", "runtime_package");
    check_resolved(r->head_artifact, ids.head, "head_torchscript", "model_bundle");
    check_resolved(r->backbone_engine, ids.backbone, "backbone_engine", "model_bundle");
    CHECK(fs::path(r->op_library.absolute_path).is_absolute());
    CHECK(o.result->detector.op_library_from_attestation);
}

void expect_refused(const char* name, const Bundle& b, const Outcome& o, const std::string& needle,
                    const char* level = nullptr) {
    const bool ok = !o.result && o.other.empty() && o.error.rfind("preflight: ", 0) == 0 &&
                    o.error.find(needle) != std::string::npos;
    if (!ok) {
        std::fprintf(stderr, "%s: expected a preflight refusal naming '%s', got error='%s' other='%s'%s\n", name,
                     needle.c_str(), o.error.c_str(), o.other.c_str(), o.result ? " (passed)" : "");
    }
    CHECK(ok);
    CHECK(str(o.journal, "state") == "failed");
    CHECK(str(at(o.journal, "failure"), "message") == o.error);
    CHECK(is_null(o.journal, "load_verification"));
    for (const JsonValue& s : at(o.journal, "sequences").array) CHECK(str(s, "state") == "pending");
    check_initial(o, b.required);
    check_identity_shape(identity(o), b.required);
    if (level == nullptr) {
        CHECK(is_null(identity(o), "level"));
        CHECK(is_null(identity(o), "expected_source"));
    } else {
        CHECK(str(identity(o), "level") == level);  // a policy refusal keeps the proven level
    }
    CHECK(!fs::exists(b.report));
}

struct Case {
    const char* name;
    std::function<void(Bundle&)> setup;
    const char* needle;  // nullptr: Gate A passes
    const char* level;   // pass: the level; refusal: the kept level (policy) or nullptr
    const char* entry;   // pass: allowlist_entry
    std::function<void(const Bundle&, const Outcome&)> extra;
};

void check_frozen_model(const Fixture& fx, const fs::path& dir, const fs::path& root) {
    // The real N01-N06: the example manifest as published, the members copied
    // from the repository paths it names (the operator from the build tree).
    Bundle b;
    b.dir = dir;
    b.bundle = dir / "bundle";
    b.runtime = dir / "runtime";
    b.out = dir / "out";
    b.report = dir / "report.json";
    b.trace = dir / "trace";
    b.model_bundle = b.bundle.string();
    b.manifest_json = sh::parse_strict_json(fx.manifest_text);
    b.allowlist_json = sh::parse_strict_json(fx.allowlist_text);  // the published example allowlist
    const JsonValue& members = at(b.manifest_json, "members");
    for (std::size_t i = 0; i < members.array.size(); ++i) {
        fs::create_directories(b.member_path(i).parent_path());
        fs::copy_file(root / str(members.array[i], "path"), b.member_path(i), fs::copy_options::overwrite_existing);
    }
    write_file(b.bundle / sh::kModelBundleManifestName, fx.manifest_text);
    write_allowlist(b);
    for (const std::string& s : kSeqs) {
        write_sequence(dir / "seqs" / s, 3);
        b.sequences.push_back((dir / "seqs" / s).string());
    }
    const Outcome o = run_gate(b);
    // The example entry names this exact manifest; it is not an approval.
    expect_pass("frozen_model", b, o, "checksum_matched", "example");
    std::fprintf(stderr, "frozen_model: %s\n", o.error.empty() ? "passed" : o.error.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
                     "usage: %s <resolved config> <lineage> <attestation> <example manifest> [--keep DIR] "
                     "[--frozen-model-root ROOT]\n",
                     argv[0]);
        return 2;
    }
    fs::path keep, frozen_model_root;
    for (int i = 5; i < argc; i += 2) {
        if (i + 1 >= argc) return 2;
        const std::string option(argv[i]);
        if (option == "--keep") keep = argv[i + 1];
        else if (option == "--frozen-model-root") frozen_model_root = argv[i + 1];
        else return 2;
    }
    const fs::path manifest_path = argv[4];
    const Fixture fx{read_file(argv[1]), read_file(argv[2]), read_file(argv[3]), read_file(manifest_path),
                     read_file(manifest_path.parent_path() / "trusted_model_bundles.example.json")};
    char tmpl[] = "/tmp/saccade_preflight_manifest_test.XXXXXX";
    const fs::path base = keep.empty() ? fs::path(::mkdtemp(tmpl)) : keep;
    fs::create_directories(base);
    const Ids ids;
    using L = sh::IdentityLevel;

    const std::vector<Case> cases = {
        {"pass_unlisted", [](Bundle&) {}, nullptr, "checksum_matched", "absent", nullptr},
        {"pass_require_checksum", [](Bundle& b) { b.required = L::ChecksumMatched; }, nullptr, "checksum_matched",
         "absent", nullptr},
        // MB-30
        {"approved", [](Bundle& b) { allow(b, "approved"); }, nullptr, "expected_source_verified", "approved",
         [](const Bundle& b, const Outcome& o) {
             const JsonValue& mb = binding(identity(o), "bundle_manifest");
             CHECK(str(mb, "expected_sha256") == b.manifest_sha());
             CHECK(str(at(mb, "expected_source"), "path") ==
                   fs::canonical(b.runtime).string() + "/trusted_model_bundles.json");
             CHECK(str(at(mb, "expected_source"), "json_pointer") == "/entries/0/manifest_sha256");
             // Identity carries no load claim.
             CHECK(sh::dump_python_json(identity(o)).find("loaded") == std::string::npos);
         }},
        {"approved_required", [](Bundle& b) { allow(b, "approved"); b.required = L::ExpectedSourceVerified; },
         nullptr, "expected_source_verified", "approved", nullptr},
        // MB-33
        {"example_entry", [](Bundle& b) { allow(b, "example"); }, nullptr, "checksum_matched", "example", nullptr},
        {"revoked_entry", [](Bundle& b) { allow(b, "revoked"); }, nullptr, "checksum_matched", "revoked", nullptr},
        {"example_entry_required", [](Bundle& b) { allow(b, "example"); b.required = L::ExpectedSourceVerified; },
         "identity level checksum_matched is below --require-identity expected_source_verified", "checksum_matched",
         nullptr, nullptr},
        {"revoked_entry_required", [](Bundle& b) { allow(b, "revoked"); b.required = L::ExpectedSourceVerified; },
         "identity level checksum_matched is below --require-identity expected_source_verified", "checksum_matched",
         nullptr, nullptr},
        {"unlisted_required", [](Bundle& b) { b.required = L::ExpectedSourceVerified; },
         "is below --require-identity expected_source_verified", "checksum_matched", nullptr, nullptr},
        // MB-31: the original manifest is approved; a self-consistent
        // replacement of engine, lineage, attestation and manifest is not.
        {"self_consistent_substitution",
         [&](Bundle& b) {
             allow(b, "approved");
             write_file(b.member_path(ids.backbone), "substituted engine");
             at(at(at(b.lineage_json, "companions"), "backbone_engine"), "sha256").string = sha("substituted engine");
             write_lineage_and_attestation(b, ids);
             write_manifest(b);
         },
         nullptr, "checksum_matched", "absent", nullptr},
        {"self_consistent_substitution_required",
         [&](Bundle& b) {
             allow(b, "approved");
             write_file(b.member_path(ids.backbone), "substituted engine");
             at(at(at(b.lineage_json, "companions"), "backbone_engine"), "sha256").string = sha("substituted engine");
             write_lineage_and_attestation(b, ids);
             write_manifest(b);
             b.required = L::ExpectedSourceVerified;
         },
         "is below --require-identity expected_source_verified", "checksum_matched", nullptr, nullptr},
        // MB-32: allowlist bytes that are not the pin (an approval added after
        // the build), and no allowlist at all.
        {"allowlist_not_pinned",
         [](Bundle& b) {
             const std::string pin = b.pin;
             allow(b, "approved");
             b.pin = pin;
         },
         "is not the one this entrypoint was built with", nullptr, nullptr,
         [](const Bundle& b, const Outcome& o) {
             CHECK(str(identity(o), "allowlist_sha256") == sha(read_file(b.runtime / "trusted_model_bundles.json")));
             CHECK(is_null(identity(o), "allowlist_entry"));
             CHECK(str(binding(identity(o), "op_library"), "status") == "unchecked");
         }},
        {"allowlist_missing", [](Bundle& b) { fs::remove(b.runtime / "trusted_model_bundles.json"); },
         "trusted model bundles", nullptr, nullptr, nullptr},
        {"allowlist_invalid",
         [](Bundle& b) {
             at(b.allowlist_json, "entries").array.push_back(JsonValue::make_object());
             write_allowlist(b);
         },
         "trusted model bundles: /entries/0: required", nullptr, nullptr, nullptr},
        // MB-35
        {"config_edited_field_consistent",
         [&](Bundle& b) { write_file(b.member_path(ids.config), read_file(b.member_path(ids.config)) + "\n"); },
         "resolved config", nullptr, nullptr,
         [&](const Bundle& b, const Outcome& o) {
             check_member(b, identity(o), "config", ids.config, "size_mismatch", false);
         }},
        {"config_edited_same_size",
         [&](Bundle& b) {
             std::string t = read_file(b.member_path(ids.config));
             t.back() = t.back() == '\n' ? ' ' : '\n';
             write_file(b.member_path(ids.config), t);
         },
         "(manifest)", nullptr, nullptr,
         [&](const Bundle& b, const Outcome& o) {
             check_member(b, identity(o), "config", ids.config, "mismatch", true);
         }},
        // MB-36: size first, never hashed.
        {"engine_size_differs", [&](Bundle& b) { write_file(b.member_path(ids.backbone), kEngine + "x"); },
         "the manifest says", nullptr, nullptr,
         [&](const Bundle& b, const Outcome& o) {
             check_member(b, identity(o), "engine", ids.backbone, "size_mismatch", false);
             CHECK(str(binding(identity(o), "head"), "status") == "matched");
         }},
        {"op_library_replaced_same_size",
         [&](Bundle& b) {
             std::string t = kOpLibrary;
             t[0] = 'O';
             write_file(b.member_path(ids.op), t);
         },
         "operator library", nullptr, nullptr,
         [&](const Bundle& b, const Outcome& o) {
             check_member(b, identity(o), "op_library", ids.op, "mismatch", true);
             CHECK(str(binding(identity(o), "head"), "status") == "unchecked");
         }},
        // MB-37
        {"head_missing", [&](Bundle& b) { fs::remove(b.member_path(ids.head)); }, "does not exist", nullptr, nullptr,
         [&](const Bundle& b, const Outcome& o) {
             CHECK(str(binding(identity(o), "head"), "status") == "missing");
             CHECK(is_null(binding(identity(o), "head"), "observed_sha256"));
             (void)b;
         }},
        {"op_library_missing", [&](Bundle& b) { fs::remove(b.member_path(ids.op)); }, "does not exist", nullptr,
         nullptr, [](const Bundle&, const Outcome& o) {
             CHECK(str(binding(identity(o), "op_library"), "status") == "missing");
         }},
        {"attestation_missing", [&](Bundle& b) { fs::remove(b.member_path(ids.attestation)); }, "does not exist",
         nullptr, nullptr, [](const Bundle&, const Outcome& o) {
             CHECK(str(binding(identity(o), "attestation"), "status") == "missing");
         }},
        {"head_not_regular",
         [&](Bundle& b) {
             fs::remove(b.member_path(ids.head));
             fs::create_directory(b.member_path(ids.head));
         },
         "is not a regular file", nullptr, nullptr, nullptr},
        // MB-38
        {"engine_symlink_inside",
         [&](Bundle& b) {
             const fs::path p = b.member_path(ids.backbone);
             fs::rename(p, b.bundle / "real.engine");
             fs::create_symlink(b.bundle / "real.engine", p);
         },
         "symbolic link", nullptr, nullptr,
         [](const Bundle&, const Outcome& o) {
             CHECK(str(binding(identity(o), "engine"), "status") == "unsafe_path");
             CHECK(is_null(binding(identity(o), "engine"), "observed_sha256"));
         }},
        {"lineage_symlink_outside",
         [&](Bundle& b) {
             const fs::path p = b.member_path(ids.lineage);
             fs::rename(p, b.dir / "outside.lineage.json");
             fs::create_symlink(b.dir / "outside.lineage.json", p);
         },
         "symbolic link", nullptr, nullptr,
         [](const Bundle&, const Outcome& o) {
             CHECK(str(binding(identity(o), "lineage"), "status") == "unsafe_path");
         }},
        {"op_library_symlink_in_runtime",
         [&](Bundle& b) {
             const fs::path p = b.member_path(ids.op);
             fs::rename(p, b.dir / "outside_op.so");
             fs::create_symlink(b.dir / "outside_op.so", p);
         },
         "symbolic link", nullptr, nullptr,
         [](const Bundle&, const Outcome& o) {
             CHECK(str(binding(identity(o), "op_library"), "status") == "unsafe_path");
         }},
        // MB-39
        {"directory_symlink_escape",
         [&](Bundle& b) {
             fs::rename(b.bundle / "models", b.dir / "models_outside");
             fs::create_symlink(b.dir / "models_outside", b.bundle / "models");
         },
         "symbolic link", nullptr, nullptr, nullptr},
        {"manifest_symlink",
         [](Bundle& b) {
             fs::rename(b.bundle / "model_bundle.json", b.dir / "outside_manifest.json");
             fs::create_symlink(b.dir / "outside_manifest.json", b.bundle / "model_bundle.json");
         },
         "symbolic link", nullptr, nullptr,
         [](const Bundle&, const Outcome& o) {
             CHECK(str(binding(identity(o), "bundle_manifest"), "status") == "unsafe_path");
             CHECK(str(binding(identity(o), "config"), "status") == "unchecked");
         }},
        // MB-40: the same bytes elsewhere keep their level; the root itself
        // may be named through a symlink (resolved once).
        {"relocated_bundle",
         [](Bundle& b) {
             allow(b, "approved");
             fs::rename(b.bundle, b.dir / "elsewhere");
             b.bundle = b.dir / "elsewhere";
             b.model_bundle = b.bundle.string();
         },
         nullptr, "expected_source_verified", "approved", nullptr},
        {"bundle_root_via_symlink",
         [](Bundle& b) {
             fs::create_symlink(b.bundle, b.dir / "bundle_link");
             b.model_bundle = (b.dir / "bundle_link").string();
         },
         nullptr, "checksum_matched", "absent", nullptr},
        // VL0
        {"manifest_unknown_schema",
         [](Bundle& b) {
             at(b.manifest_json, "schema").string = "saccade.model_bundle/v2";
             write_manifest(b);
         },
         "model bundle manifest: /schema: const", nullptr, nullptr,
         [](const Bundle& b, const Outcome& o) {
             CHECK(str(binding(identity(o), "bundle_manifest"), "observed_sha256") == b.manifest_sha());
             CHECK(str(identity(o), "bundle_manifest_sha256") == b.manifest_sha());
             CHECK(is_null(identity(o), "allowlist_sha256"));
         }},
        {"manifest_pairing_wrong_role",
         [](Bundle& b) {
             at(at(b.manifest_json, "pairing"), "head").string = "backbone";
             write_manifest(b);
         },
         "R-03 pairing.head does not name a head_torchscript member", nullptr, nullptr, nullptr},
        {"manifest_missing", [](Bundle& b) { fs::remove(b.bundle / "model_bundle.json"); }, "does not exist",
         nullptr, nullptr, [](const Bundle&, const Outcome& o) {
             CHECK(str(binding(identity(o), "bundle_manifest"), "status") == "missing");
         }},
        {"manifest_malformed", [](Bundle& b) { write_file(b.bundle / "model_bundle.json", "{bad"); },
         "json syntax", nullptr, nullptr, nullptr},
        {"bundle_dir_missing", [](Bundle& b) { b.model_bundle = (b.dir / "no_such_bundle").string(); },
         "model bundle directory", nullptr, nullptr, nullptr},
        {"runtime_root_missing", [](Bundle& b) { fs::remove_all(b.runtime); }, "runtime package root", nullptr,
         nullptr, nullptr},
        {"legacy_option_given", [](Bundle& b) { b.legacy_config = "config.json"; },
         "--model-bundle takes no --config", nullptr, nullptr, nullptr},
        // The manifest and the lineage must bind the same files.
        {"manifest_head_elsewhere",
         [&](Bundle& b) {
             const fs::path moved = b.bundle / "models" / "other_head.pt";
             fs::copy_file(b.member_path(ids.head), moved);
             at(at(b.manifest_json, "members").array[ids.head], "path").string = "models/other_head.pt";
             write_manifest(b);
         },
         "is not the head artifact the lineage / attestation bind", nullptr, nullptr, nullptr},
        {"sequence_missing", [](Bundle& b) { b.sequences.back() += "_missing"; }, "seqinfo.ini", nullptr, nullptr,
         [](const Bundle&, const Outcome& o) {
             // Metadata members were checked; the load members not yet.
             CHECK(str(binding(identity(o), "config"), "status") == "matched");
             CHECK(str(binding(identity(o), "op_library"), "status") == "unchecked");
         }},
    };

    for (const Case& c : cases) {
        const fs::path dir = base / c.name;
        fs::remove_all(dir);
        Bundle b = make_bundle(fx, dir);
        write_file(b.report, "earlier report");
        c.setup(b);
        const Outcome o = run_gate(b);
        if (c.needle == nullptr) expect_pass(c.name, b, o, c.level, c.entry);
        else expect_refused(c.name, b, o, c.needle, c.level);
        if (c.extra) c.extra(b, o);
        std::fprintf(stderr, "%s: %s\n", c.name, o.error.empty() ? "passed" : o.error.c_str());
    }
    if (!frozen_model_root.empty()) check_frozen_model(fx, base / "frozen_model", frozen_model_root);
    if (keep.empty()) fs::remove_all(base);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
