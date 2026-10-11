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
//
// Journal (format saccade.native_track_journal/v2; historical v1 is unchanged):
//   {"format", "run_id", "entrypoint", "state": running|failed|complete,
//    "identity": {"level": null|checksum_matched, "expected_source": null,
//                 "publisher_authentication": "not_checked_by_runtime",
//                 "bindings": {config,lineage,attestation,op_library,head,engine}},
//    "sequences": [{"name", "state": pending|written, "txt", "txt_sha256",
//                   "trace", "trace_sha256"}...]   (argv order),
//    "report": null | {"path", "sha256"},
//    "failure": null | {"sequence": name|null, "message"}}
#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "saccade_shipping/strict_json.hpp"

namespace saccade::shipping {

inline constexpr const char* kRunJournalFormat = "saccade.native_track_journal/v2";
inline constexpr const char* kRunJournalName = "saccade_track.journal.json";
inline constexpr const char* kRunLockName = "saccade_track.lock";

// 128 random bits (getrandom) as 32 lowercase hex digits.
std::string new_run_id();

// Initial legacy identity: no level, six unvisited bindings, no recognized
// independent expected source or publisher authentication.
JsonValue unverified_identity();

struct PreflightInputs;
struct PreflightResult;

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
    // journal, removes this run's earlier outputs.
    RunCompletion(std::string run_id, std::string entrypoint, RunOutputs outputs);
    ~RunCompletion();
    RunCompletion(const RunCompletion&) = delete;
    RunCompletion& operator=(const RunCompletion&) = delete;

    const std::string& run_id() const { return run_id_; }
    const JsonValue& identity() const { return identity_; }

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
    friend PreflightResult run_preflight(const PreflightInputs&, RunCompletion&);
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
