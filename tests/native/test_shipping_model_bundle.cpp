// The model bundle reader of saccade_track (#549 S2-1,
// saccade_shipping/model_bundle.hpp). CPU only; CI job `shipping-config-loader`.
//
// Usage: saccade_shipping_model_bundle_test <verification_matrix.json> <repo root>
//        <shipping/trusted_model_bundles.json>
//
// * the published examples (the matrix's `examples`) pass VL0 / R-09, and the
//   reader's view of the example manifest is its N01-N06 pairing;
// * every matrix row whose layer is schema or semantic -- the design rows of
//   tests/contract/test_model_bundle_contract_549.py -- runs against this
//   reader: the mutated document fails at the row's instance path and keyword
//   (schema), or passes the schema and fails exactly the row's rule
//   (semantic). A row this reader does not reject is a failure here;
// * keywords the matrix has no row for: additionalProperties, required,
//   oneOf, uniqueItems, multipleOf, maxItems, an integral float, a path with a
//   trailing newline (this reader is stricter than Python's re.search);
// * the production allowlist is schema-valid and EMPTY (no approved bundle);
//   allowlist_state for absent / example / revoked / approved entries;
// * open_beneath: a regular file opens; a missing one, a symlinked member, a
//   symlinked directory below the root, an escape and a FIFO are refused with
//   their reason; read_all / sha256_fd refuse a size other than fstat's.
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "saccade_shipping/model_bundle.hpp"
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

JsonValue load(const fs::path& p) { return sh::parse_strict_json(read_file(p)); }

// RFC 6901 tokens of a JSON pointer.
std::vector<std::string> tokens(const std::string& pointer) {
    std::vector<std::string> out;
    std::size_t i = 1;
    while (i <= pointer.size()) {
        const std::size_t j = pointer.find('/', i);
        std::string t = pointer.substr(i, j == std::string::npos ? std::string::npos : j - i);
        for (std::size_t k; (k = t.find("~1")) != std::string::npos;) t.replace(k, 2, "/");
        for (std::size_t k; (k = t.find("~0")) != std::string::npos;) t.replace(k, 2, "~");
        out.push_back(t);
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return out;
}

JsonValue& child(JsonValue& v, const std::string& t) {
    if (v.kind == JsonValue::Kind::Array) return v.array.at(std::stoul(t));
    JsonValue* c = v.find(t);
    if (c == nullptr) throw std::runtime_error("no member " + t);
    return *c;
}

// The matrix's mutation ops (replace / remove / add / copy), as the Python
// reference applies them.
void apply(JsonValue& doc, const JsonValue& ops) {
    for (const JsonValue& op : ops.array) {
        const std::string kind = op.find("op")->string;
        const std::vector<std::string> path = tokens(op.find("path")->string);
        JsonValue* parent = &doc;
        for (std::size_t i = 0; i + 1 < path.size(); ++i) parent = &child(*parent, path[i]);
        const std::string& key = path.back();
        JsonValue value;
        if (kind == "copy") {
            const std::vector<std::string> from = tokens(op.find("from")->string);
            JsonValue* src = &doc;
            for (const std::string& t : from) src = &child(*src, t);
            value = *src;
        } else if (const JsonValue* v = op.find("value")) {
            value = *v;
        }
        if (kind == "remove") {
            if (parent->kind == JsonValue::Kind::Array) {
                parent->array.erase(parent->array.begin() + static_cast<long>(std::stoul(key)));
            } else {
                parent->erase(key);
            }
        } else if (parent->kind == JsonValue::Kind::Array) {
            if (key == "-") parent->array.push_back(value);
            else if (kind == "replace") parent->array.at(std::stoul(key)) = value;
            else parent->array.insert(parent->array.begin() + static_cast<long>(std::stoul(key)), value);
        } else if (JsonValue* existing = parent->find(key)) {
            *existing = value;
        } else {
            parent->set(key, value);
        }
    }
}

bool has_issue(const std::vector<sh::BundleIssue>& issues, const std::string& path, const std::string& keyword) {
    for (const auto& i : issues) {
        if (i.instance_path == path && i.keyword == keyword) return true;
    }
    return false;
}

std::set<std::string> rules(const std::vector<sh::BundleIssue>& issues) {
    std::set<std::string> out;
    for (const auto& i : issues) out.insert(i.rule);
    return out;
}

void dump(const char* what, const std::vector<sh::BundleIssue>& issues) {
    for (const auto& i : issues) std::fprintf(stderr, "  %s: %s\n", what, sh::describe(i).c_str());
}

JsonValue approval() {
    JsonValue a = JsonValue::make_object();
    a.set("decision_ref", JsonValue::make_string("https://github.com/raylei50653/saccade/issues/549#issuecomment-1"));
    a.set("date", JsonValue::make_string("2026-10-11"));
    return a;
}

void test_matrix(const JsonValue& matrix, const JsonValue& manifest, const JsonValue& allowlist) {
    int run = 0;
    for (const JsonValue& row : matrix.find("rows")->array) {
        const std::string layer = row.find("layer")->string;
        const JsonValue* mutation = row.find("mutation");
        if ((layer != "schema" && layer != "semantic") || mutation->kind == JsonValue::Kind::Null) continue;
        ++run;
        const std::string id = row.find("id")->string;
        const bool is_manifest = row.find("target")->string == "manifest";
        JsonValue doc = is_manifest ? manifest : allowlist;
        apply(doc, *mutation);
        const auto schema = is_manifest ? sh::model_bundle_schema_issues(doc) : sh::trusted_bundles_schema_issues(doc);
        const JsonValue& want = *row.find("expect_error");
        bool ok = false;
        if (layer == "schema") {
            ok = has_issue(schema, want.find("instance_path")->string, want.find("keyword")->string);
        } else {
            const auto rule_issues = schema.empty() ? (is_manifest ? sh::model_bundle_rule_issues(doc)
                                                                   : sh::trusted_bundles_rule_issues(doc))
                                                    : std::vector<sh::BundleIssue>{};
            ok = schema.empty() && rules(rule_issues) == std::set<std::string>{want.find("rule")->string};
            if (!ok) dump(id.c_str(), rule_issues);
        }
        if (!ok) {
            std::fprintf(stderr, "matrix row %s not rejected as expected\n", id.c_str());
            dump(id.c_str(), schema);
        }
        CHECK(ok);
        // The parse entry points refuse it too (exit 2 in Gate A).
        bool refused = false;
        try {
            if (is_manifest) sh::parse_model_bundle(doc);
            else sh::parse_trusted_bundles(doc);
        } catch (const sh::ConfigError&) {
            refused = true;
        }
        CHECK(refused);
    }
    std::fprintf(stderr, "matrix: %d schema/semantic rows run\n", run);
    CHECK(run >= 23);  // the S1 matrix's schema / semantic rows
}

void expect_schema(const JsonValue& base, const char* ops_json, const char* path, const char* keyword) {
    JsonValue doc = base;
    apply(doc, sh::parse_strict_json(ops_json));
    const auto issues = sh::model_bundle_schema_issues(doc);
    const bool ok = has_issue(issues, path, keyword);
    if (!ok) {
        std::fprintf(stderr, "expected %s %s for %s\n", path, keyword, ops_json);
        dump("got", issues);
    }
    CHECK(ok);
}

void test_more_keywords(const JsonValue& manifest) {
    expect_schema(manifest, R"([{"op": "add", "path": "/extra", "value": 1}])", "", "additionalProperties");
    expect_schema(manifest, R"([{"op": "remove", "path": "/members/0/bytes"}])", "/members/0", "required");
    expect_schema(manifest, R"([{"op": "replace", "path": "/members/0/inventory_id", "value": "X1"}])",
                  "/members/0/inventory_id", "oneOf");
    expect_schema(manifest, R"([{"op": "replace", "path": "/migration/rollback_targets", "value": [
        "0000000000000000000000000000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000000"]}])",
                  "/migration/rollback_targets", "uniqueItems");
    expect_schema(manifest, R"([{"op": "replace", "path": "/preprocessing/resize/width", "value": 650}])",
                  "/preprocessing/resize/width", "multipleOf");
    expect_schema(manifest, R"([{"op": "replace", "path": "/members/0/bytes", "value": 0}])", "/members/0/bytes",
                  "minimum");
    expect_schema(manifest, R"([{"op": "replace", "path": "/members/0/bytes", "value": "1"}])", "/members/0/bytes",
                  "type");
    expect_schema(manifest, R"([{"op": "replace", "path": "/members/0/path", "value": "models/x.engine\n"}])",
                  "/members/0/path", "pattern");
    expect_schema(manifest, R"([{"op": "replace", "path": "/rights/channels/private/state", "value": "approved"}])",
                  "/rights/channels/private/decision_ref", "type");
    {
        JsonValue doc = manifest;
        JsonValue& members = *doc.find("members");
        for (int i = 0; i < 11; ++i) members.array.push_back(members.array[0]);
        CHECK(has_issue(sh::model_bundle_schema_issues(doc), "/members", "maxItems"));
    }
    // An integral float is a JSON Schema integer (as Python's validator has it).
    {
        JsonValue doc = manifest;
        apply(doc, sh::parse_strict_json(R"([{"op": "replace", "path": "/preprocessing/resize/width", "value": 640.0}])"));
        CHECK(sh::model_bundle_issues(doc).empty());
    }
    // No R-xx fires on a schema-valid document that breaks no rule; a
    // schema-invalid one is never given to the rules.
    bool threw = false;
    try {
        JsonValue doc = manifest;
        doc.erase("io");
        sh::model_bundle_rule_issues(doc);
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw);
}

void test_examples(const JsonValue& manifest, const JsonValue& allowlist, const std::string& manifest_sha) {
    const auto issues = sh::model_bundle_issues(manifest);
    dump("example manifest", issues);
    CHECK(issues.empty());
    CHECK(sh::trusted_bundles_issues(allowlist).empty());
    const sh::ModelBundleManifest m = sh::parse_model_bundle(manifest);
    CHECK(m.members.size() == 6);
    CHECK(m.members[m.backbone_engine].role == "backbone_engine");
    CHECK(m.members[m.head].role == "head_torchscript");
    CHECK(m.members[m.op_library].role == "scan_operator");
    CHECK(m.members[m.op_library].carried_by == sh::RootKind::RuntimePackage);
    CHECK(m.members[m.config].json_schema == "saccade.resolved_shipping_config/v1");
    CHECK(m.members[m.lineage].carried_by == sh::RootKind::ModelBundle);
    CHECK(m.members[m.attestation].role == "realization_attestation");
    CHECK(m.members[m.backbone_engine].bytes == 20298516);
    CHECK(m.detector_contract == sh::kNativeDetectorContract);

    // The example allowlist names the example manifest -- as an example.
    const sh::TrustedModelBundles a = sh::parse_trusted_bundles(allowlist);
    std::size_t index = 99;
    CHECK(sh::allowlist_state(a, manifest_sha, &index) == sh::AllowlistState::Example);
    CHECK(index == 0);
    CHECK(sh::allowlist_state(a, std::string(64, '0')) == sh::AllowlistState::Absent);

    // A synthetic approved entry (test fixture only; never shipped).
    JsonValue approved = allowlist;
    JsonValue& e = approved.find("entries")->array[0];
    *e.find("state") = JsonValue::make_string("approved");
    *e.find("approval") = approval();
    CHECK(sh::trusted_bundles_issues(approved).empty());
    CHECK(sh::allowlist_state(sh::parse_trusted_bundles(approved), manifest_sha) == sh::AllowlistState::Approved);
    CHECK(sh::allowlist_state(sh::parse_trusted_bundles(approved), std::string(64, '1')) ==
          sh::AllowlistState::Absent);

    JsonValue revoked = approved;
    JsonValue& r = revoked.find("entries")->array[0];
    *r.find("state") = JsonValue::make_string("revoked");
    JsonValue rev = approval();
    rev.set("reason", JsonValue::make_string("test"));
    *r.find("revocation") = rev;
    CHECK(sh::trusted_bundles_issues(revoked).empty());
    CHECK(sh::allowlist_state(sh::parse_trusted_bundles(revoked), manifest_sha) == sh::AllowlistState::Revoked);

    // An approved entry with a revocation record is not schema-valid.
    JsonValue torn = approved;
    *torn.find("entries")->array[0].find("revocation") = rev;
    CHECK(has_issue(sh::trusted_bundles_schema_issues(torn), "/entries/0/revocation", "type"));
}

void test_production_allowlist(const fs::path& path) {
    const JsonValue a = load(path);
    const auto issues = sh::trusted_bundles_issues(a);
    dump("production allowlist", issues);
    CHECK(issues.empty());
    CHECK(a.find("entries")->array.empty());  // owner authorization: starts EMPTY
}

void test_open_beneath(const fs::path& tmp) {
    fs::remove_all(tmp);
    const fs::path root = tmp / "root";
    write_file(root / "a" / "file.bin", "0123456789");
    write_file(tmp / "outside.bin", "outside");
    fs::create_directories(root / "real");
    write_file(root / "real" / "x.bin", "x");
    fs::create_symlink(root / "a" / "file.bin", root / "link.bin");
    fs::create_symlink(root / "real", root / "dirlink");
    fs::create_symlink(tmp, root / "escape");
    CHECK(::mkfifo((root / "fifo").c_str(), 0600) == 0);

    // The root itself may be reached through a symlink: it is resolved once.
    fs::create_symlink(root, tmp / "root_link");
    const sh::BundleRoot r = sh::BundleRoot::open((tmp / "root_link").string(), sh::RootKind::ModelBundle, "root");
    CHECK(r.real_path() == fs::canonical(root).string());
    CHECK(r.absolute("a/file.bin") == fs::canonical(root).string() + "/a/file.bin");

    sh::BeneathOpen f = sh::open_beneath(r, "a/file.bin");
    CHECK(f.result == sh::BeneathOpen::Result::Opened);
    CHECK(f.size == 10);
    CHECK(sh::read_all(f.fd.get(), f.size, "file") == "0123456789");
    CHECK(sh::sha256_fd(f.fd.get(), f.size, "file") == sh::sha256_hex("0123456789", 10));
    bool size_refused = false;
    try {
        sh::sha256_fd(f.fd.get(), 9, "file");
    } catch (const sh::ConfigError& e) {
        size_refused = std::string(e.what()).find("changed while it was read") != std::string::npos;
    }
    CHECK(size_refused);

    auto refused = [&](const char* rel, sh::BeneathOpen::Result want, const char* reason) {
        const sh::BeneathOpen g = sh::open_beneath(r, rel);
        const bool ok = g.result == want && g.fd.get() < 0 && g.reason.find(reason) != std::string::npos;
        if (!ok) std::fprintf(stderr, "open_beneath(%s): %d %s\n", rel, static_cast<int>(g.result), g.reason.c_str());
        CHECK(ok);
    };
    refused("missing.bin", sh::BeneathOpen::Result::Missing, "does not exist");
    refused("a/file.bin/x", sh::BeneathOpen::Result::Missing, "does not exist");
    refused("link.bin", sh::BeneathOpen::Result::Unsafe, "symbolic link");
    refused("dirlink/x.bin", sh::BeneathOpen::Result::Unsafe, "symbolic link");
    refused("escape/outside.bin", sh::BeneathOpen::Result::Unsafe, "symbolic link");
    refused("../outside.bin", sh::BeneathOpen::Result::Unsafe, "outside its root");
    refused("/etc/hostname", sh::BeneathOpen::Result::Unsafe, "outside its root");
    refused("fifo", sh::BeneathOpen::Result::NotRegular, "not a regular file");
    refused("a", sh::BeneathOpen::Result::NotRegular, "not a regular file");

    bool missing_root = false;
    try {
        sh::BundleRoot::open((tmp / "nope").string(), sh::RootKind::RuntimePackage, "runtime package root");
    } catch (const sh::ConfigError& e) {
        missing_root = std::string(e.what()).find("runtime package root") != std::string::npos;
    }
    CHECK(missing_root);
    bool file_root = false;
    try {
        sh::BundleRoot::open((root / "a" / "file.bin").string(), sh::RootKind::ModelBundle, "model bundle directory");
    } catch (const sh::ConfigError& e) {
        file_root = std::string(e.what()).find("not an openable directory") != std::string::npos;
    }
    CHECK(file_root);
    fs::remove_all(tmp);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <verification_matrix.json> <repo root> <trusted_model_bundles.json>\n", argv[0]);
        return 2;
    }
    try {
        const fs::path repo = argv[2];
        const JsonValue matrix = load(argv[1]);
        const fs::path manifest_path = repo / matrix.find("examples")->find("manifest")->string;
        const std::string manifest_text = read_file(manifest_path);
        const JsonValue manifest = sh::parse_strict_json(manifest_text);
        const JsonValue allowlist = load(repo / matrix.find("examples")->find("allowlist")->string);
        test_examples(manifest, allowlist, sh::sha256_hex(manifest_text.data(), manifest_text.size()));
        test_matrix(matrix, manifest, allowlist);
        test_more_keywords(manifest);
        test_production_allowlist(argv[3]);
        test_open_beneath(fs::temp_directory_path() / ("saccade_model_bundle_test." + std::to_string(::getpid())));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    std::fprintf(stderr, "%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
