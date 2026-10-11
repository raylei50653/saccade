// Gate A of saccade_track (#536 CC-536-01-02). See saccade_shipping/preflight.hpp.
#include "saccade_shipping/preflight.hpp"

#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <system_error>
#include <utility>

#include "saccade_shipping/ingest_plan.hpp"
#include "saccade_shipping/model_bundle.hpp"
#include "saccade_shipping/native_config.hpp"
#include "saccade_shipping/post_detector_plan.hpp"
#include "saccade_shipping/sequence_output.hpp"
#include "saccade_shipping/sha256.hpp"

namespace saccade::shipping {

// Gate A's only access to RunCompletion's identity (run_completion.hpp).
class GateAIdentityWriter {
public:
    static bool finalized(const RunCompletion& c) { return c.identity_finalized_; }
    static void record(RunCompletion& c, const JsonValue& identity, bool final) {
        c.record_gate_a_identity(identity, final);
    }
    static void seal(RunCompletion& c) { c.identity_finalized_ = true; }
};

namespace {

[[noreturn]] void preflight_error(const std::string& what) { throw PreflightError("preflight: " + what); }

// One step of Gate A: any error it throws becomes a PreflightError.
template <class F>
auto step(const std::string& context, F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const PreflightError&) {
        throw;
    } catch (const std::exception& e) {
        preflight_error(context + e.what());
    }
}

JsonValue& binding(JsonValue& identity, const char* name) {
    return *identity.find("bindings")->find(name);
}

void put(JsonValue& o, const char* key, JsonValue value) { *o.find(key) = std::move(value); }

void path_for(JsonValue& b, const std::string& path) {
    put(b, "path", path.empty() ? JsonValue::make_null() : JsonValue::make_string(path));
}

void expected_from(JsonValue& b, const std::string& sha, const std::string& path, const char* pointer) {
    put(b, "expected_sha256", JsonValue::make_string(sha));
    JsonValue source = JsonValue::make_object();
    source.set("path", JsonValue::make_string(path));
    source.set("json_pointer", JsonValue::make_string(pointer));
    put(b, "expected_source", std::move(source));
}

void compare_binding(JsonValue& b) {
    const JsonValue& expected = *b.find("expected_sha256");
    const JsonValue& observed = *b.find("observed_sha256");
    if (expected.kind == JsonValue::Kind::String && observed.kind == JsonValue::Kind::String) {
        put(b, "status", JsonValue::make_string(expected.string == observed.string ? "matched" : "mismatch"));
    }
}

// Hash the exact text returned to the existing parser; never re-open metadata
// to manufacture an observation of different bytes after parsing.
std::string read_metadata(JsonValue& b, const std::string& path, const char* what) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        if (!ec || ec == std::errc::no_such_file_or_directory || ec == std::errc::not_a_directory)
            put(b, "status", JsonValue::make_string("missing"));
        throw ConfigError(std::string("cannot open ") + what + path + " (not a regular file)");
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) throw ConfigError(std::string("cannot open ") + what + path);
    std::ostringstream text;
    text << in.rdbuf();
    if (!in.good() && !in.eof()) throw ConfigError(std::string("cannot read ") + what + path);
    const std::string bytes = text.str();
    put(b, "observed_sha256", JsonValue::make_string(sha256_hex(bytes.data(), bytes.size())));
    return bytes;
}

// A diagnostic expected value is available only from a recognized attestation
// schema and an actual, well-formed SHA field. plan_detector still owns all
// semantic acceptance and the original rejection reason.
void lineage_expected(JsonValue& b, const JsonValue& att, const std::string& path) {
    if (att.kind != JsonValue::Kind::Object) return;
    const JsonValue* schema = att.find("schema");
    if (schema == nullptr || schema->kind != JsonValue::Kind::String || schema->string != kHeadRealizationSchema)
        return;
    const JsonValue* frozen = att.find("frozen_lineage");
    if (frozen == nullptr || frozen->kind != JsonValue::Kind::Object) return;
    const JsonValue* sha = frozen->find("sha256");
    if (sha == nullptr || sha->kind != JsonValue::Kind::String || sha->string.size() != 64) return;
    for (char c : sha->string) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return;
    }
    expected_from(b, sha->string, path, "/frozen_lineage/sha256");
    compare_binding(b);
}

void artifact_expected(JsonValue& identity, const PreflightInputs& in, const DetectorPlan& plan) {
    path_for(binding(identity, "op_library"), resolve_model_path(in.model_root, plan.op_library.path));
    path_for(binding(identity, "head"), resolve_model_path(in.model_root, plan.head_artifact.path));
    path_for(binding(identity, "engine"), resolve_model_path(in.model_root, plan.backbone_engine.path));
    expected_from(binding(identity, "op_library"), plan.op_library.sha256,
                  plan.op_library_from_attestation ? in.attestation : in.lineage, "/op_library/sha256");
    expected_from(binding(identity, "head"), plan.head_artifact.sha256, in.lineage, "/torchscript/sha256");
    expected_from(binding(identity, "engine"), plan.backbone_engine.sha256, in.lineage,
                  "/companions/backbone_engine/sha256");
}

void check_artifact(const char* what, JsonValue& b) {
    const std::string& path = b.find("path")->string;
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        if (ec && ec != std::errc::no_such_file_or_directory && ec != std::errc::not_a_directory)
            preflight_error(std::string(what) + " " + path + " cannot be inspected: " + ec.message());
        put(b, "status", JsonValue::make_string("missing"));
        preflight_error(std::string(what) + " " + path + " is missing (not a regular file)");
    }
    const std::string got = step("", [&] { return sha256_file_hex(path); });
    put(b, "observed_sha256", JsonValue::make_string(got));
    compare_binding(b);
    if (b.find("status")->string == "mismatch") {
        preflight_error(std::string(what) + " " + path + " sha256 " + got + " != " +
                        b.find("expected_sha256")->string);
    }
}

// The config's strict loader, then every CUDA-free plan the runtime builds
// from it (step 1). Shared by both modes: `config_text` is the observed
// config buffer.
std::pair<ResolvedShippingConfig, Schedule> plan_config(const std::string& config_text, bool serial_requested) {
    const ResolvedShippingConfig cfg = step("", [&] { return parse_resolved_shipping_config(config_text); });
    const Schedule schedule = step("", [&] {
        plan_ingest(cfg);
        if (!plan_sequence_output(cfg).write_output) {
            throw ConfigError("the config writes no MOT output (steps.tail.write_output)");
        }
        plan_post_detector(cfg);
        planned_pipeline_snapshot(cfg);
        return select_schedule(cfg, serial_requested);
    });
    return {cfg, schedule};
}

void check_sequences_and_outputs(const PreflightInputs& in, RunCompletion& completion) {
    // 2. Every sequence's input, as run_sequence will read it.
    for (const std::string& dir : in.sequences) {
        step("sequence " + dir + ": ", [&] { return read_sequence_input(dir, in.max_frames); });
    }
    // 3. Every directory the run publishes into.
    step("", [&] { completion.check_writable(); });
}

PreflightResult legacy_gate_a(const PreflightInputs& in, RunCompletion& completion, JsonValue& identity) {
    auto publish = [&] { GateAIdentityWriter::record(completion, identity, false); };
    path_for(binding(identity, "config"), in.config);
    path_for(binding(identity, "lineage"), in.lineage);
    path_for(binding(identity, "attestation"), in.attestation);
    publish();
    // 1. Existing config/plan checks, using the observed metadata buffers.
    const std::string config_text = read_metadata(binding(identity, "config"), in.config, "resolved config ");
    publish();
    const auto planned = plan_config(config_text, in.serial_requested);
    const ResolvedShippingConfig& cfg = planned.first;
    const Schedule schedule = planned.second;
    const std::string lineage_text = read_metadata(binding(identity, "lineage"), in.lineage, "");
    publish();
    const JsonValue lineage = step("", [&] { return parse_strict_json(lineage_text); });
    const std::string lineage_sha = binding(identity, "lineage").find("observed_sha256")->string;
    DetectorPlan detector = step("", [&] { return plan_detector(cfg, lineage, lineage_sha, nullptr); });
    artifact_expected(identity, in, detector);
    publish();
    if (!in.attestation.empty()) {
        const std::string attestation_text = read_metadata(binding(identity, "attestation"), in.attestation, "");
        publish();
        const JsonValue att = step("", [&] { return parse_strict_json(attestation_text); });
        lineage_expected(binding(identity, "lineage"), att, in.attestation);
        publish();
        detector = step("", [&] { return plan_detector(cfg, lineage, lineage_sha, &att); });
        artifact_expected(identity, in, detector);
        publish();
    }

    check_sequences_and_outputs(in, completion);

    // 4. The three files Gate B loads, by the plan's sha256.
    for (const auto& item : {std::pair{"operator library", "op_library"},
                            std::pair{"head artifact", "head"}, std::pair{"backbone engine", "engine"}}) {
        check_artifact(item.first, binding(identity, item.second));
        publish();
    }
    // Legacy mode's ceiling: the expected hashes come from caller files.
    put(identity, "level", JsonValue::make_string("checksum_matched"));
    return PreflightResult{cfg, schedule, std::move(detector)};
}

// ── manifest mode (#549 S2-1) ─────────────────────────────────────────────────

void member_expected(JsonValue& b, const BundleRoot& root, const BundleMember& m, const std::string& manifest_path) {
    path_for(b, root.absolute(m.path));
    expected_from(b, m.sha256, manifest_path, ("/members/" + std::to_string(m.index) + "/sha256").c_str());
}

// Opens `rel` beneath `root` for binding `b`. A refusal records the binding's
// status (missing / unsafe_path) and throws.
BeneathOpen open_member(const BundleRoot& root, const std::string& rel, JsonValue& b, const std::string& what) {
    BeneathOpen f = step("", [&] { return open_beneath(root, rel); });
    switch (f.result) {
        case BeneathOpen::Result::Opened: return f;
        case BeneathOpen::Result::Missing:
        case BeneathOpen::Result::NotRegular:
            put(b, "status", JsonValue::make_string("missing"));
            preflight_error(what + " " + f.reason);
        case BeneathOpen::Result::Unsafe:
            put(b, "status", JsonValue::make_string("unsafe_path"));
            preflight_error(what + " " + f.reason);
    }
    preflight_error(what + " cannot be opened");
}

void require_size(const BeneathOpen& f, const BundleMember& m, JsonValue& b, const std::string& what) {
    if (f.size != m.bytes) {
        put(b, "status", JsonValue::make_string("size_mismatch"));
        preflight_error(what + " " + b.find("path")->string + " has " + std::to_string(f.size) +
                        " bytes, the manifest says " + std::to_string(m.bytes));
    }
}

void require_match(JsonValue& b, const std::string& what) {
    compare_binding(b);
    if (b.find("status")->string == "mismatch") {
        preflight_error(what + " " + b.find("path")->string + " sha256 " + b.find("observed_sha256")->string +
                        " != " + b.find("expected_sha256")->string + " (manifest)");
    }
}

// A JSON member: size, then the sha256 of the bytes read, which are the
// bytes returned to the parser.
std::string read_json_member(const BundleRoot& root, const BundleMember& m, JsonValue& b, const std::string& what) {
    const BeneathOpen f = open_member(root, m.path, b, what);
    require_size(f, m, b, what);
    const std::string bytes = step("", [&] { return read_all(f.fd.get(), f.size, b.find("path")->string); });
    put(b, "observed_sha256", JsonValue::make_string(sha256_hex(bytes.data(), bytes.size())));
    require_match(b, what);
    return bytes;
}

ResolvedBinding hash_load_member(const BundleRoot& root, const BundleMember& m, JsonValue& b, const std::string& what) {
    const BeneathOpen f = open_member(root, m.path, b, what);
    require_size(f, m, b, what);
    const std::string got = step("", [&] { return sha256_fd(f.fd.get(), f.size, b.find("path")->string); });
    put(b, "observed_sha256", JsonValue::make_string(got));
    require_match(b, what);
    return ResolvedBinding{m.role, root_kind_name(root.kind()), root.absolute(m.path), m.bytes, got};
}

// The lineage / attestation bind the same file (path and sha256) as the
// manifest member the pairing names for it.
void require_paired(const FileBinding& plan, const BundleMember& m, const char* what) {
    if (plan.path != m.path || plan.sha256 != m.sha256) {
        preflight_error(std::string("model bundle member ") + m.id + " (" + m.path + ", " + m.sha256 + ") is not the " +
                        what + " the lineage / attestation bind (" + plan.path + ", " + plan.sha256 + ")");
    }
}

PreflightResult manifest_gate_a(const PreflightInputs& in, RunCompletion& completion, JsonValue& identity) {
    auto publish = [&] { GateAIdentityWriter::record(completion, identity, false); };
    publish();
    if (!in.config.empty() || !in.lineage.empty() || !in.attestation.empty()) {
        preflight_error("--model-bundle takes no --config, --lineage or --attestation");
    }
    // The two roots, each resolved once.
    const BundleRoot bundle =
        step("", [&] { return BundleRoot::open(in.model_bundle, RootKind::ModelBundle, "model bundle directory"); });
    const BundleRoot runtime = step(
        "", [&] { return BundleRoot::open(in.runtime_root, RootKind::RuntimePackage, "runtime package root"); });

    // 0a. VL0: the manifest, by the bytes the parser reads.
    JsonValue& mb = binding(identity, "bundle_manifest");
    const std::string manifest_path = bundle.absolute(kModelBundleManifestName);
    path_for(mb, manifest_path);
    publish();
    const BeneathOpen mf = open_member(bundle, kModelBundleManifestName, mb, "model bundle manifest");
    const std::string manifest_text = step("", [&] { return read_all(mf.fd.get(), mf.size, manifest_path); });
    const std::string manifest_sha = sha256_hex(manifest_text.data(), manifest_text.size());
    put(mb, "observed_sha256", JsonValue::make_string(manifest_sha));
    put(identity, "bundle_manifest_sha256", JsonValue::make_string(manifest_sha));
    publish();
    const ModelBundleManifest manifest =
        step("", [&] { return parse_model_bundle(parse_strict_json(manifest_text)); });

    // 0b. The allowlist this entrypoint was built with (TR-1b).
    const std::string allowlist_path = runtime.absolute(kTrustedModelBundlesName);
    const BeneathOpen af = step("", [&] { return open_beneath(runtime, kTrustedModelBundlesName); });
    if (af.result != BeneathOpen::Result::Opened) preflight_error("trusted model bundles " + af.reason);
    const std::string allowlist_text = step("", [&] { return read_all(af.fd.get(), af.size, allowlist_path); });
    const std::string allowlist_sha = sha256_hex(allowlist_text.data(), allowlist_text.size());
    put(identity, "allowlist_sha256", JsonValue::make_string(allowlist_sha));
    publish();
    if (allowlist_sha != in.allowlist_sha256) {
        preflight_error("trusted model bundles " + allowlist_path + " sha256 " + allowlist_sha +
                        " is not the one this entrypoint was built with (" + in.allowlist_sha256 + ")");
    }
    const TrustedModelBundles allowlist =
        step("", [&] { return parse_trusted_bundles(parse_strict_json(allowlist_text)); });
    if (allowlist.detector_contract != manifest.detector_contract) {
        preflight_error("the manifest's detector contract " + manifest.detector_contract +
                        " is not the allowlist's " + allowlist.detector_contract);
    }
    std::size_t entry = 0;
    const AllowlistState state = allowlist_state(allowlist, manifest_sha, &entry);
    put(identity, "allowlist_entry", JsonValue::make_string(allowlist_state_name(state)));
    if (state != AllowlistState::Absent) {
        expected_from(mb, manifest_sha, allowlist_path, ("/entries/" + std::to_string(entry) + "/manifest_sha256").c_str());
        compare_binding(mb);
    }

    // Every member's expected value is the manifest's.
    const auto& members = manifest.members;
    auto root_of = [&](const BundleMember& m) -> const BundleRoot& {
        return m.carried_by == RootKind::ModelBundle ? bundle : runtime;
    };
    const std::pair<const char*, std::size_t> slots[] = {
        {"config", manifest.config},         {"lineage", manifest.lineage}, {"attestation", manifest.attestation},
        {"op_library", manifest.op_library}, {"head", manifest.head},       {"engine", manifest.backbone_engine}};
    for (const auto& [name, index] : slots) {
        member_expected(binding(identity, name), root_of(members[index]), members[index], manifest_path);
    }
    publish();

    // 1. VL1 for the three JSON members, then the existing config / plan
    //    checks on the bytes read; the attestation is required (ADR 028 D6).
    const BundleMember& config_m = members[manifest.config];
    const BundleMember& lineage_m = members[manifest.lineage];
    const BundleMember& att_m = members[manifest.attestation];
    const std::string config_text =
        read_json_member(root_of(config_m), config_m, binding(identity, "config"), "resolved config");
    publish();
    const auto planned = plan_config(config_text, in.serial_requested);
    const ResolvedShippingConfig& cfg = planned.first;
    const Schedule schedule = planned.second;
    const std::string lineage_text =
        read_json_member(root_of(lineage_m), lineage_m, binding(identity, "lineage"), "head lineage");
    publish();
    const std::string attestation_text =
        read_json_member(root_of(att_m), att_m, binding(identity, "attestation"), "realization attestation");
    publish();
    const JsonValue lineage = step("", [&] { return parse_strict_json(lineage_text); });
    const JsonValue att = step("", [&] { return parse_strict_json(attestation_text); });
    DetectorPlan detector = step("", [&] { return plan_detector(cfg, lineage, lineage_m.sha256, &att); });
    require_paired(detector.op_library, members[manifest.op_library], "operator library");
    require_paired(detector.head_artifact, members[manifest.head], "head artifact");
    require_paired(detector.backbone_engine, members[manifest.backbone_engine], "backbone engine");

    check_sequences_and_outputs(in, completion);

    // 4. VL1 for the three files Gate B loads: the resolved bindings.
    ResolvedLoadBindings resolved;
    const BundleMember& op_m = members[manifest.op_library];
    const BundleMember& head_m = members[manifest.head];
    const BundleMember& engine_m = members[manifest.backbone_engine];
    resolved.op_library = hash_load_member(root_of(op_m), op_m, binding(identity, "op_library"), "operator library");
    publish();
    resolved.head_artifact = hash_load_member(root_of(head_m), head_m, binding(identity, "head"), "head artifact");
    publish();
    resolved.backbone_engine =
        hash_load_member(root_of(engine_m), engine_m, binding(identity, "engine"), "backbone engine");
    publish();
    detector.resolved = std::make_shared<const ResolvedLoadBindings>(std::move(resolved));

    // VL2: only an approved entry for this exact manifest names an expected
    // source independent of the bundle.
    if (state == AllowlistState::Approved) {
        put(identity, "level", JsonValue::make_string("expected_source_verified"));
        put(identity, "expected_source", JsonValue::make_string("runtime_allowlist"));
    } else {
        put(identity, "level", JsonValue::make_string("checksum_matched"));
    }
    return PreflightResult{cfg, schedule, std::move(detector)};
}

IdentityLevel level_of(const JsonValue& identity) {
    const JsonValue& level = *identity.find("level");
    if (level.kind != JsonValue::Kind::String) return IdentityLevel::None;
    return parse_identity_level(level.string);
}

}  // namespace

PreflightResult run_preflight(const PreflightInputs& in, RunCompletion& completion) {
    if (GateAIdentityWriter::finalized(completion)) preflight_error("identity already finalized");
    const bool manifest_mode = !in.model_bundle.empty();
    JsonValue identity = unverified_identity(manifest_mode, in.required);
    std::optional<PreflightResult> result;
    try {
        result.emplace(manifest_mode ? manifest_gate_a(in, completion, identity)
                                     : legacy_gate_a(in, completion, identity));
        // The only promotion point, after the final Gate A check. Gate B and
        // completion cannot write identity; reports copy the sealed record.
        GateAIdentityWriter::record(completion, identity, true);
    } catch (const std::exception& e) {
        put(identity, "level", JsonValue::make_null());
        put(identity, "expected_source", JsonValue::make_null());
        try {
            GateAIdentityWriter::record(completion, identity, true);
        } catch (...) {
            // Preserve the original refusal if journal publication also fails.
            // fail() will retry with the diagnostics already held by completion.
        }
        GateAIdentityWriter::seal(completion);
        if (dynamic_cast<const PreflightError*>(&e) != nullptr) throw;
        preflight_error(e.what());
    }
    // --require-identity, against the sealed level (kept as it is).
    const IdentityLevel level = level_of(identity);
    if (level < in.required) {
        preflight_error("identity level " + std::string(identity_level_name(level)) + " is below --require-identity " +
                        identity_level_name(in.required) +
                        (manifest_mode ? "" : " (legacy mode cannot exceed checksum_matched)"));
    }
    return std::move(*result);
}

}  // namespace saccade::shipping
