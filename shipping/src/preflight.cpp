// Gate A of saccade_track (#536 CC-536-01-02). See saccade_shipping/preflight.hpp.
#include "saccade_shipping/preflight.hpp"

#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>
#include <utility>

#include "saccade_shipping/ingest_plan.hpp"
#include "saccade_shipping/native_config.hpp"
#include "saccade_shipping/post_detector_plan.hpp"
#include "saccade_shipping/sequence_output.hpp"
#include "saccade_shipping/sha256.hpp"

namespace saccade::shipping {
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

}  // namespace

PreflightResult run_preflight(const PreflightInputs& in, RunCompletion& completion) {
    if (completion.identity_finalized_) preflight_error("identity already finalized");
    JsonValue identity = unverified_identity();
    path_for(binding(identity, "config"), in.config);
    path_for(binding(identity, "lineage"), in.lineage);
    path_for(binding(identity, "attestation"), in.attestation);
    try {
        completion.record_gate_a_identity(identity, false);
        // 1. Existing config/plan checks, using the observed metadata buffers.
        const std::string config_text = read_metadata(binding(identity, "config"), in.config, "resolved config ");
        completion.record_gate_a_identity(identity, false);
        const ResolvedShippingConfig cfg = step("", [&] { return parse_resolved_shipping_config(config_text); });
        const Schedule schedule = step("", [&] {
            plan_ingest(cfg);
            if (!plan_sequence_output(cfg).write_output) {
                throw ConfigError("the config writes no MOT output (steps.tail.write_output)");
            }
            plan_post_detector(cfg);
            planned_pipeline_snapshot(cfg);
            return select_schedule(cfg, in.serial_requested);
        });
        const std::string lineage_text = read_metadata(binding(identity, "lineage"), in.lineage, "");
        completion.record_gate_a_identity(identity, false);
        const JsonValue lineage = step("", [&] { return parse_strict_json(lineage_text); });
        const std::string lineage_sha = binding(identity, "lineage").find("observed_sha256")->string;
        DetectorPlan detector = step("", [&] { return plan_detector(cfg, lineage, lineage_sha, nullptr); });
        artifact_expected(identity, in, detector);
        completion.record_gate_a_identity(identity, false);
        if (!in.attestation.empty()) {
            const std::string attestation_text = read_metadata(binding(identity, "attestation"), in.attestation, "");
            completion.record_gate_a_identity(identity, false);
            const JsonValue att = step("", [&] { return parse_strict_json(attestation_text); });
            lineage_expected(binding(identity, "lineage"), att, in.attestation);
            completion.record_gate_a_identity(identity, false);
            detector = step("", [&] { return plan_detector(cfg, lineage, lineage_sha, &att); });
            artifact_expected(identity, in, detector);
            completion.record_gate_a_identity(identity, false);
        }

        // 2. Every sequence's input, as run_sequence will read it.
        for (const std::string& dir : in.sequences) {
            step("sequence " + dir + ": ", [&] { return read_sequence_input(dir, in.max_frames); });
        }

        // 3. Every directory the run publishes into.
        step("", [&] { completion.check_writable(); });

        // 4. The three files Gate B loads, by the plan's sha256.
        for (const auto& item : {std::pair{"operator library", "op_library"},
                                std::pair{"head artifact", "head"}, std::pair{"backbone engine", "engine"}}) {
            check_artifact(item.first, binding(identity, item.second));
            completion.record_gate_a_identity(identity, false);
        }
        // The only promotion point, after the final Gate A check. Gate B and
        // completion cannot write identity; reports copy the sealed record.
        put(identity, "level", JsonValue::make_string("checksum_matched"));
        completion.record_gate_a_identity(identity, true);
        return PreflightResult{cfg, schedule, std::move(detector)};
    } catch (const std::exception& e) {
        put(identity, "level", JsonValue::make_null());
        try {
            completion.record_gate_a_identity(identity, true);
        } catch (...) {
            // Preserve the original refusal if journal publication also fails.
            // fail() will retry with the diagnostics already held by completion.
        }
        completion.identity_finalized_ = true;
        if (dynamic_cast<const PreflightError*>(&e) != nullptr) throw;
        preflight_error(e.what());
    }
}

}  // namespace saccade::shipping
