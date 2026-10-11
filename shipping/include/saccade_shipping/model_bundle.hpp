// Model bundle manifest and release-side allowlist (#549 S2-1; ADR 028,
// docs/architecture/model_bundle_contract_549.md). CUDA-free.
//
// Three parts, all used by Gate A (preflight.hpp) in manifest mode:
//
//   VL0  `model_bundle_issues` / `parse_model_bundle`: the exact
//        saccade.model_bundle/v1 schema
//        (docs/architecture/model_bundle_549/saccade.model_bundle.v1.schema.json)
//        and the rules it cannot state, R-01..R-08 (contract section 3.2), as
//        the reference implementation in
//        tests/contract/test_model_bundle_contract_549.py states them. The
//        schema is hand-coded here, keyword for keyword; every issue names the
//        instance path and keyword (or rule) a JSON Schema validator would, so
//        the verification matrix's schema / semantic rows run against this code
//        (tests/native/test_shipping_model_bundle.cpp). One deliberate
//        difference: a pattern's `$` matches only at the end of the string,
//        where Python's `re.search` also matches before a final newline (this
//        reader is stricter).
//   VL1  `BundleRoot` / `open_beneath`: a root directory is resolved once
//        (realpath) and opened once; every member is opened beneath it with
//        openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS),
//        so a symlink anywhere below the root, an escape or a non-regular file
//        is refused. A kernel without openat2 is refused
//        (SecureOpenUnsupported); there is no weaker fallback.
//   VL2  `trusted_bundles_issues` / `parse_trusted_bundles` / `allowlist_state`:
//        the saccade.trusted_model_bundles/v1 allowlist (TR-1b) and R-09. Only
//        an `approved` entry for the manifest's exact sha256 can yield
//        expected_source_verified; `example`, `revoked` and an unlisted manifest
//        cannot. The allowlist's own sha256 is pinned in the entrypoint build
//        (track_driver.hpp), not here: this library trusts no file by itself.
//
// Nothing here reads the process environment, downloads, or looks anywhere but
// the two roots it is given.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "saccade_shipping/strict_json.hpp"

namespace saccade::shipping {

inline constexpr const char* kModelBundleSchema = "saccade.model_bundle/v1";
inline constexpr const char* kTrustedModelBundlesSchema = "saccade.trusted_model_bundles/v1";
inline constexpr const char* kNativeDetectorContract = "saccade.native_detector_contract/v1";
// The manifest's name in a bundle directory, and the allowlist's in the
// runtime package's share/saccade/.
inline constexpr const char* kModelBundleManifestName = "model_bundle.json";
inline constexpr const char* kTrustedModelBundlesName = "trusted_model_bundles.json";

// One violated schema keyword (`instance_path` + `keyword`, as a JSON Schema
// validator reports it; the root is "") or one violated rule (`rule`, "R-01"..
// "R-09"; `keyword` empty).
struct BundleIssue {
    std::string instance_path, keyword, rule, message;
};

// "<path>: <keyword>: <message>" or "R-xx <message>".
std::string describe(const BundleIssue& issue);

// Every schema issue of a manifest document (empty = schema-valid).
std::vector<BundleIssue> model_bundle_schema_issues(const JsonValue& manifest);
// R-01..R-08 of a schema-valid manifest (throws std::logic_error otherwise).
std::vector<BundleIssue> model_bundle_rule_issues(const JsonValue& manifest);
// Schema issues, then (only when there are none) rule issues.
std::vector<BundleIssue> model_bundle_issues(const JsonValue& manifest);

std::vector<BundleIssue> trusted_bundles_schema_issues(const JsonValue& allowlist);
std::vector<BundleIssue> trusted_bundles_rule_issues(const JsonValue& allowlist);  // R-09
std::vector<BundleIssue> trusted_bundles_issues(const JsonValue& allowlist);

enum class RootKind { ModelBundle, RuntimePackage };
const char* root_kind_name(RootKind k);  // "model_bundle" / "runtime_package"

struct BundleMember {
    std::string id, role, path, sha256;
    std::string json_schema;  // empty for a binary role
    RootKind carried_by = RootKind::ModelBundle;
    std::int64_t bytes = 0;
    std::size_t index = 0;  // in members[], for the expected-source pointer
};

// A VL0-valid manifest. The six pairing slots name members by index.
struct ModelBundleManifest {
    std::string bundle_name, bundle_version, detector_contract;
    std::vector<BundleMember> members;
    std::size_t backbone_engine = 0, head = 0, op_library = 0, config = 0, lineage = 0, attestation = 0;
};

// VL0: throws ConfigError "model bundle manifest: <first issue>" (all issues
// are listed) when the document fails the schema or a rule.
ModelBundleManifest parse_model_bundle(const JsonValue& manifest);

enum class AllowlistState { Absent, Example, Revoked, Approved };
const char* allowlist_state_name(AllowlistState s);  // "absent" / "example" / ...

struct TrustedBundleEntry {
    std::string manifest_sha256, state;
    std::size_t index = 0;
};
struct TrustedModelBundles {
    std::string detector_contract;
    std::vector<TrustedBundleEntry> entries;
};

// Throws ConfigError "trusted model bundles: <first issue>".
TrustedModelBundles parse_trusted_bundles(const JsonValue& allowlist);

// The allowlist's state for `manifest_sha256` (R-09: at most one entry); the
// entry's index in `*index` when there is one.
AllowlistState allowlist_state(const TrustedModelBundles& allowlist, const std::string& manifest_sha256,
                               std::size_t* index = nullptr);

// ── VL1: confined, symlink-free opens ─────────────────────────────────────────

// The kernel has no openat2 (or refuses it). Manifest mode stops (exit 2).
class SecureOpenUnsupported : public ConfigError {
public:
    using ConfigError::ConfigError;
};

class UniqueFd {
public:
    UniqueFd() = default;
    explicit UniqueFd(int fd) : fd_(fd) {}
    ~UniqueFd();
    UniqueFd(UniqueFd&& o) noexcept : fd_(o.release()) {}
    UniqueFd& operator=(UniqueFd&& o) noexcept;
    UniqueFd(const UniqueFd&) = delete;
    UniqueFd& operator=(const UniqueFd&) = delete;
    int get() const { return fd_; }
    int release() {
        const int fd = fd_;
        fd_ = -1;
        return fd;
    }

private:
    int fd_ = -1;
};

// A root directory, resolved with realpath once and opened once (O_DIRECTORY).
// Relocating a whole bundle is allowed: members are relative to it.
class BundleRoot {
public:
    // Throws ConfigError naming `what` when `dir` does not resolve to a
    // directory.
    static BundleRoot open(const std::string& dir, RootKind kind, const std::string& what);
    const std::string& real_path() const { return real_; }
    RootKind kind() const { return kind_; }
    int fd() const { return fd_.get(); }
    // real_path() + "/" + rel: what Gate A hashed and Gate B rehashes.
    std::string absolute(const std::string& rel) const;

private:
    std::string real_;
    RootKind kind_ = RootKind::ModelBundle;
    UniqueFd fd_;
};

struct BeneathOpen {
    enum class Result { Opened, Missing, Unsafe, NotRegular } result = Result::Missing;
    UniqueFd fd;
    std::int64_t size = 0;  // st_size of the opened regular file
    std::string reason;     // why it was not opened
};

// openat2(root, rel, O_RDONLY | O_NONBLOCK | O_NOCTTY | O_CLOEXEC,
// RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS), then fstat:
// only a regular file is Opened. Missing: ENOENT / ENOTDIR. Unsafe: a symlink
// (ELOOP) or an escape (EXDEV). Throws SecureOpenUnsupported when the kernel
// has no openat2, ConfigError on any other error.
BeneathOpen open_beneath(const BundleRoot& root, const std::string& rel);

// All of an opened file's bytes from offset 0; throws ConfigError when fewer
// or more than `expected_size` bytes are read (the file changed after fstat).
std::string read_all(int fd, std::int64_t expected_size, const std::string& what);
// The sha256 of an opened file's bytes from offset 0, with the same size rule.
std::string sha256_fd(int fd, std::int64_t expected_size, const std::string& what);

}  // namespace saccade::shipping
