// Run completion for saccade_track (#536 CC-536-01-01,
// docs/architecture/ship_export_contracts_536.md). CUDA-free: C++ std
// filesystem, POSIX flock / rename / fsync, strict_json and sha256 only.
//
// One invocation = one run, named by a random run id. Before anything is
// computed the run takes `<out>` (a non-blocking exclusive flock on
// `<out>/saccade_track.lock`, held until the process ends), installs a new
// journal `<out>/saccade_track.journal.json` (temp + rename, which replaces
// the previous run's), and only then removes the paths this run will write:
// the --report file, `<out>/<seq>.txt` and `<trace>/<seq>/detector.bin` of its
// own sequences. Nothing else in `<out>` is touched; the new journal and the
// lock file never are.
//
// Each sequence's trace and txt are written to a temp file in the same
// directory and renamed into place; only then is the sequence `written` in
// the journal, with its sha256. After every sequence the report is published
// the same way, and the last write is the journal's `state=complete`: the
// only commit point. A caught error writes `state=failed` (best effort);
// a kill or abort leaves `running`.
//
// Caller rule: a run is complete only when the journal's run_id is this
// invocation's and its state is `complete`; a txt / trace / report counts as
// this run's only when the journal records it with a sha256 equal to the
// file's. `pending` means "not confirmed": after a kill between a rename and
// the journal update the file can be this run's complete output, and it is
// still not evidence that the run produced it. State transitions and failure
// classes belong to #537. Gate A alone updates identity; its final record is
// immutable through sequence, report, complete and failed writes (S2-1a).
// Gate B alone writes load_verification, once, after Gate A passed (S2-1):
// `verified` when the detector's three files are loaded and checked,
// `failed` when that load throws (DetectorLoadError); an uncatchable end
// (the loader auditor's _exit(127), SIGKILL, abort) leaves it null with the
// state running. Its byte_scope is always hashed_before_load: Gate B rehashes
// each file before it loads it from its path, so the loaded bytes are not
// proven to be the hashed ones (TOCTOU, #549 S2-2). Nothing here can write
// loaded_buffer.
//
// Journal (format saccade.native_track_journal/v3; historical v1 / v2 are
// unchanged):
//   {"format", "run_id", "entrypoint", "state": running|failed|complete,
//    "identity": {"level": null|checksum_matched|expected_source_verified,
//                 "expected_source": null|"runtime_allowlist",
//                 "publisher_authentication": "not_checked_by_runtime",
//                 "mode": legacy|model_bundle,
//                 "required": none|checksum_matched|expected_source_verified,
//                 "allowlist_sha256": null|sha256,
//                 "allowlist_entry": null|absent|example|revoked|approved,
//                 "bundle_manifest_sha256": null|sha256,
//                 "bindings": {[bundle_manifest,] config, lineage, attestation,
//                              op_library, head, engine}},
//    "load_verification": null | {"status": verified|failed,
//                                 "byte_scope": "hashed_before_load"|null},
//    "sequences": [{"name", "state": pending|written, "txt", "txt_sha256",
//                   "trace", "trace_sha256"}...]   (argv order),
//    "report": null | {"path", "sha256"},
//    "failure": null | {"sequence": name|null, "message"}}
// A binding is {"path", "expected_sha256", "observed_sha256", "status":
// matched|mismatch|size_mismatch|missing|unsafe_path|unchecked,
// "expected_source": null|{"path", "json_pointer"}}; bundle_manifest exists
// in manifest mode only (preflight.hpp).
#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "saccade_shipping/strict_json.hpp"

namespace saccade::shipping {

inline constexpr const char* kRunJournalFormat = "saccade.native_track_journal/v3";
inline constexpr const char* kRunJournalName = "saccade_track.journal.json";
inline constexpr const char* kRunLockName = "saccade_track.lock";

// 128 random bits (getrandom) as 32 lowercase hex digits.
std::string new_run_id();

// --require-identity: the least identity level a run accepts (Gate A exits 2
// below it). Ordered: none < checksum_matched < expected_source_verified.
enum class IdentityLevel { None, ChecksumMatched, ExpectedSourceVerified };
const char* identity_level_name(IdentityLevel level);  // "none" / "checksum_matched" / ...
// Throws std::invalid_argument for any other name.
IdentityLevel parse_identity_level(const std::string& name);

// Initial identity: no level, unvisited bindings (bundle_manifest too in
// manifest mode), the requested policy, no recognized independent expected
// source or publisher authentication.
JsonValue unverified_identity(bool manifest_mode = false, IdentityLevel required = IdentityLevel::None);

class GateAIdentityWriter;  // preflight.cpp: Gate A's only access to identity

struct RunOutputs {
    std::filesystem::path out;             // --out
    std::filesystem::path report;          // --report; empty = none
    std::filesystem::path trace;           // --trace; empty = none
    std::vector<std::string> sequences;    // sequence names, argv order, distinct
};

class RunCompletion {
public:
    // Refuses an output path that would land on the journal, the lock or
    // another output of this run (before anything is created); creates
    // `<out>`, takes the lock (busy: throws, nothing written), installs the
    // journal (identity: unverified_identity(manifest_mode, required)),
    // removes this run's earlier outputs.
    RunCompletion(std::string run_id, std::string entrypoint, RunOutputs outputs, bool manifest_mode = false,
                  IdentityLevel required = IdentityLevel::None);
    ~RunCompletion();
    RunCompletion(const RunCompletion&) = delete;
    RunCompletion& operator=(const RunCompletion&) = delete;

    const std::string& run_id() const { return run_id_; }
    const JsonValue& identity() const { return identity_; }
    const JsonValue& load_verification() const { return load_verification_; }

    // Gate B (the detector load), once, after Gate A has passed: `verified`
    // with byte_scope hashed_before_load, or `failed` (byte_scope null).
    // Throws std::logic_error when Gate A has not passed or it was already
    // recorded; a journal write failure leaves it null and throws.
    void record_load_verification(bool verified);

    // Gate A's output check (preflight.hpp): in every directory this run
    // publishes into -- `<out>`, the --report file's directory (which must
    // exist; it is not created), each `<trace>/<seq>/` (created, as run()
    // does) -- a probe file `.saccade_track.preflight.<run_id>.tmp` is created
    // and removed. Throws when one cannot be.
    void check_writable() const;

    // Sequence i in order: `sequence(i, trace_tmp)` computes it, writing its
    // trace to trace_tmp when the run has --trace (empty path otherwise), and
    // returns its MOT text; each is committed before the next starts. Then,
    // with --report, `report()` gives the report text. Then state=complete.
    using SequenceFn = std::function<std::string(std::size_t, const std::filesystem::path&)>;
    using ReportFn = std::function<std::string()>;
    void run(const SequenceFn& sequence, const ReportFn& report);

    // state=failed with the sequence being run (or null) and `message`; never
    // throws (a journal that cannot be written stays as it was).
    void fail(const std::string& message) noexcept;

private:
    friend class GateAIdentityWriter;
    // Only Gate A may publish observations and seal its final identity. The
    // caller cannot install an identity or mutate the const report snapshot.
    void record_gate_a_identity(const JsonValue& identity, bool final);
    struct Sequence {
        std::string name;
        std::filesystem::path txt, trace;  // trace empty without --trace
        bool written = false;
        std::string txt_sha256, trace_sha256;
    };
    std::filesystem::path temp_for(const std::filesystem::path& final) const;
    void write_temp(const std::filesystem::path& final, const std::string& bytes) const;
    void publish(const std::filesystem::path& final, const std::string& bytes) const;
    void write_journal() const;
    void commit_sequence(std::size_t i, const std::string& text);

    std::string run_id_, entrypoint_;
    JsonValue identity_ = unverified_identity();
    bool identity_finalized_ = false;
    JsonValue load_verification_ = JsonValue::make_null();
    std::filesystem::path out_, report_, journal_;
    std::vector<Sequence> sequences_;
    std::string state_ = "running", report_sha256_;
    std::optional<std::size_t> current_;
    bool failed_ = false;
    std::string failure_message_;
    std::optional<std::size_t> failure_sequence_;
    int lock_fd_ = -1;
};

}  // namespace saccade::shipping
