// Run completion for saccade_track (#536 CC-536-01-01): see
// run_completion.hpp.
#include "saccade_shipping/run_completion.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/random.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <set>
#include <stdexcept>
#include <system_error>
#include <utility>

#include "saccade_shipping/sha256.hpp"

// Test builds only (tests/native/test_shipping_run_completion.cpp compiles
// this file with the definition): the points where a test kills the process.
// The shipping and measurement builds compile the points to nothing.
#ifdef SACCADE_RUN_COMPLETION_TEST_FAULTS
void saccade_run_completion_test_fault(const char* point, std::size_t sequence);
#define SACCADE_COMPLETION_POINT(point, sequence) saccade_run_completion_test_fault(point, sequence)
#else
#define SACCADE_COMPLETION_POINT(point, sequence) ((void)0)
#endif

namespace saccade::shipping {

namespace fs = std::filesystem;

namespace {

[[noreturn]] void fail_errno(const std::string& what) {
    throw std::runtime_error(what + ": " + std::strerror(errno));
}

// An existing file's data on disk before it is renamed into place.
void fsync_file(const fs::path& p) {
    const int fd = ::open(p.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) fail_errno("cannot open " + p.string());
    const int rc = ::fsync(fd);
    ::close(fd);
    if (rc != 0) fail_errno("cannot fsync " + p.string());
}

// A rename in `dir` on disk.
void fsync_dir(const fs::path& dir) {
    const int fd = ::open((dir.empty() ? fs::path(".") : dir).c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) fail_errno("cannot open " + dir.string());
    const int rc = ::fsync(fd);
    ::close(fd);
    if (rc != 0) fail_errno("cannot fsync " + dir.string());
}

void rename_into_place(const fs::path& from, const fs::path& to) {
    if (::rename(from.c_str(), to.c_str()) != 0) fail_errno("cannot rename " + from.string() + " to " + to.string());
    fsync_dir(to.parent_path());
}

// The same file whatever spelling (symlinks, "..", relative): the key the
// collision check compares.
fs::path key(const fs::path& p) { return fs::weakly_canonical(fs::absolute(p)); }

// One of this run's earlier outputs: removed when it is a file (or a
// symlink); a directory there is refused, never removed.
void remove_output(const fs::path& p) {
    const fs::file_status st = fs::symlink_status(p);
    if (!fs::exists(st)) return;
    if (fs::is_directory(st)) throw std::runtime_error(p.string() + " is a directory, not an output file");
    fs::remove(p);
}

JsonValue str_or_null(const std::string& s) {
    return s.empty() ? JsonValue::make_null() : JsonValue::make_string(s);
}

}  // namespace

std::string new_run_id() {
    unsigned char b[16];
    std::size_t got = 0;
    while (got < sizeof b) {
        const ssize_t n = ::getrandom(b + got, sizeof b - got, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            fail_errno("getrandom");
        }
        got += static_cast<std::size_t>(n);
    }
    static const char* hex = "0123456789abcdef";
    std::string id;
    for (unsigned char c : b) {
        id += hex[c >> 4];
        id += hex[c & 15];
    }
    return id;
}

JsonValue unverified_identity() {
    JsonValue o = JsonValue::make_object();
    o.set("level", JsonValue::make_null());
    o.set("expected_source", JsonValue::make_null());
    o.set("publisher_authentication", JsonValue::make_string("not_checked_by_runtime"));
    JsonValue bindings = JsonValue::make_object();
    for (const char* name : {"config", "lineage", "attestation", "op_library", "head", "engine"}) {
        JsonValue b = JsonValue::make_object();
        b.set("path", JsonValue::make_null());
        b.set("expected_sha256", JsonValue::make_null());
        b.set("observed_sha256", JsonValue::make_null());
        b.set("status", JsonValue::make_string("unchecked"));
        b.set("expected_source", JsonValue::make_null());
        bindings.set(name, std::move(b));
    }
    o.set("bindings", std::move(bindings));
    return o;
}

void RunCompletion::record_gate_a_identity(const JsonValue& identity, bool final) {
    if (identity_finalized_) throw std::logic_error("preflight: identity already finalized");
    identity_ = identity;
    try {
        write_journal();
    } catch (...) {
        // Failed publication must not leave an in-memory promotion that a
        // later fail() could publish as a successful Gate A result.
        *identity_.find("level") = JsonValue::make_null();
        throw;
    }
    identity_finalized_ = final;
    SACCADE_COMPLETION_POINT("identity_published", 0);
}

RunCompletion::RunCompletion(std::string run_id, std::string entrypoint, RunOutputs outputs)
    : run_id_(std::move(run_id)), entrypoint_(std::move(entrypoint)), out_(std::move(outputs.out)),
      report_(std::move(outputs.report)), journal_(out_ / kRunJournalName) {
    const fs::path lock = out_ / kRunLockName;
    for (const std::string& name : outputs.sequences) {
        Sequence s;
        s.name = name;
        s.txt = out_ / (name + ".txt");
        if (!outputs.trace.empty()) s.trace = outputs.trace / name / "detector.bin";
        sequences_.push_back(std::move(s));
    }

    // 1. Every path this run writes is distinct from the others and from the
    //    journal and the lock (before anything is created).
    std::set<fs::path> seen{key(journal_), key(lock)};
    auto claim = [&](const fs::path& p, const char* what) {
        if (!seen.insert(key(p)).second) {
            throw std::runtime_error(std::string(what) + " " + p.string() +
                                     " is the journal, the lock or another output of this run");
        }
    };
    for (const Sequence& s : sequences_) {
        claim(s.txt, "sequence output");
        if (!s.trace.empty()) claim(s.trace, "trace");
    }
    if (!report_.empty()) claim(report_, "--report");

    // 2. <out> (created when it does not exist), then its lock. A busy lock
    //    stops here: nothing has been written or removed.
    fs::create_directories(out_);
    lock_fd_ = ::open(lock.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    if (lock_fd_ < 0) fail_errno("cannot open " + lock.string());
    if (::flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
        const int e = errno;
        ::close(lock_fd_);
        lock_fd_ = -1;
        if (e == EWOULDBLOCK) {
            throw std::runtime_error(out_.string() + " is in use by another run (" + lock.string() +
                                     " is locked); nothing was changed");
        }
        errno = e;
        fail_errno("cannot lock " + lock.string());
    }

    // 3. The new journal replaces the previous run's: running, every sequence
    //    pending, identity level null.
    write_journal();

    // 4. Only now this run's earlier outputs go (never the journal or lock:
    //    step 1 keeps them out of this list). A failure here is the run's:
    //    the journal says failed. The lock stays held until the process ends
    //    (the destructor does not run for a constructor that throws).
    try {
        if (!report_.empty()) remove_output(report_);
        for (const Sequence& s : sequences_) {
            remove_output(s.txt);
            if (!s.trace.empty()) remove_output(s.trace);
        }
    } catch (const std::exception& e) {
        fail(e.what());
        throw;
    }
}

void RunCompletion::check_writable() const {
    std::vector<fs::path> dirs{out_};
    if (!report_.empty()) {
        const fs::path dir = report_.parent_path().empty() ? fs::path(".") : report_.parent_path();
        if (!fs::is_directory(dir)) {
            throw std::runtime_error("the directory of --report " + report_.string() + " does not exist");
        }
        dirs.push_back(dir);
    }
    for (const Sequence& s : sequences_) {
        if (s.trace.empty()) continue;
        fs::create_directories(s.trace.parent_path());
        dirs.push_back(s.trace.parent_path());
    }
    for (const fs::path& dir : dirs) {
        const fs::path probe = dir / (".saccade_track.preflight." + run_id_ + ".tmp");
        const int fd = ::open(probe.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
        if (fd < 0) fail_errno("cannot write in " + dir.string() + " (" + probe.filename().string() + ")");
        ::close(fd);
        if (::unlink(probe.c_str()) != 0) fail_errno("cannot remove " + probe.string());
    }
}

RunCompletion::~RunCompletion() {
    if (lock_fd_ >= 0) ::close(lock_fd_);  // releases the flock
}

fs::path RunCompletion::temp_for(const fs::path& final) const {
    return final.parent_path() / ("." + final.filename().string() + "." + run_id_ + ".tmp");
}

void RunCompletion::publish(const fs::path& final, const std::string& bytes) const {
    write_temp(final, bytes);
    rename_into_place(temp_for(final), final);
}

void RunCompletion::write_temp(const fs::path& final, const std::string& bytes) const {
    const fs::path tmp = temp_for(final);
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (fd < 0) fail_errno("cannot write " + tmp.string());
    std::size_t done = 0;
    while (done < bytes.size()) {
        const ssize_t n = ::write(fd, bytes.data() + done, bytes.size() - done);
        if (n < 0) {
            if (errno == EINTR) continue;
            const int e = errno;
            ::close(fd);
            errno = e;
            fail_errno("cannot write " + tmp.string());
        }
        done += static_cast<std::size_t>(n);
    }
    if (::fsync(fd) != 0) {
        const int e = errno;
        ::close(fd);
        errno = e;
        fail_errno("cannot fsync " + tmp.string());
    }
    if (::close(fd) != 0) fail_errno("cannot write " + tmp.string());
}

void RunCompletion::write_journal() const {
    JsonValue j = JsonValue::make_object();
    j.set("format", JsonValue::make_string(kRunJournalFormat));
    j.set("run_id", JsonValue::make_string(run_id_));
    j.set("entrypoint", JsonValue::make_string(entrypoint_));
    j.set("state", JsonValue::make_string(state_));
    j.set("identity", identity_);
    std::vector<JsonValue> seqs;
    for (const Sequence& s : sequences_) {
        JsonValue o = JsonValue::make_object();
        o.set("name", JsonValue::make_string(s.name));
        o.set("state", JsonValue::make_string(s.written ? "written" : "pending"));
        o.set("txt", JsonValue::make_string(s.txt.string()));
        o.set("txt_sha256", str_or_null(s.written ? s.txt_sha256 : ""));
        o.set("trace", str_or_null(s.trace.string()));
        o.set("trace_sha256", str_or_null(s.written ? s.trace_sha256 : ""));
        seqs.push_back(std::move(o));
    }
    j.set("sequences", JsonValue::make_array(std::move(seqs)));
    if (report_.empty()) {
        j.set("report", JsonValue::make_null());
    } else {
        JsonValue r = JsonValue::make_object();
        r.set("path", JsonValue::make_string(report_.string()));
        r.set("sha256", str_or_null(report_sha256_));
        j.set("report", std::move(r));
    }
    if (!failed_) {
        j.set("failure", JsonValue::make_null());
    } else {
        JsonValue f = JsonValue::make_object();
        f.set("sequence", failure_sequence_ ? JsonValue::make_string(sequences_[*failure_sequence_].name)
                                            : JsonValue::make_null());
        f.set("message", JsonValue::make_string(failure_message_));
        j.set("failure", std::move(f));
    }
    publish(journal_, dump_python_json(j) + "\n");
}

void RunCompletion::commit_sequence(std::size_t i, const std::string& text) {
    Sequence& s = sequences_[i];
    std::string trace_sha;
    if (!s.trace.empty()) {
        const fs::path tmp = temp_for(s.trace);
        fsync_file(tmp);
        trace_sha = sha256_file_hex(tmp.string());
        rename_into_place(tmp, s.trace);
    }
    write_temp(s.txt, text);
    SACCADE_COMPLETION_POINT("txt_written", i);
    rename_into_place(temp_for(s.txt), s.txt);
    SACCADE_COMPLETION_POINT("txt_published", i);
    s.written = true;
    s.txt_sha256 = sha256_hex(text.data(), text.size());
    s.trace_sha256 = trace_sha;
    write_journal();
}

void RunCompletion::run(const SequenceFn& sequence, const ReportFn& report) {
    for (std::size_t i = 0; i < sequences_.size(); ++i) {
        current_ = i;
        fs::path trace_tmp;
        if (!sequences_[i].trace.empty()) {
            fs::create_directories(sequences_[i].trace.parent_path());
            trace_tmp = temp_for(sequences_[i].trace);
        }
        const std::string text = sequence(i, trace_tmp);
        commit_sequence(i, text);
    }
    current_.reset();
    if (!report_.empty()) {
        const std::string bytes = report();
        publish(report_, bytes);
        SACCADE_COMPLETION_POINT("report_published", sequences_.size());
        report_sha256_ = sha256_hex(bytes.data(), bytes.size());
    }
    state_ = "complete";
    write_journal();  // the commit point
}

void RunCompletion::fail(const std::string& message) noexcept {
    try {
        failed_ = true;
        state_ = "failed";
        failure_message_ = message;
        failure_sequence_ = current_;
        if (current_) {
            // The interrupted sequence's temp files (its committed files stay).
            std::error_code ec;
            const Sequence& s = sequences_[*current_];
            fs::remove(temp_for(s.txt), ec);
            if (!s.trace.empty()) fs::remove(temp_for(s.trace), ec);
        }
        if (!report_.empty()) {
            std::error_code ec;
            fs::remove(temp_for(report_), ec);
        }
        write_journal();
    } catch (...) {
        // The journal stays as last written (running, or an earlier state).
    }
}

}  // namespace saccade::shipping
