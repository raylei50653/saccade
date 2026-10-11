// Run completion of saccade_track (#536 CC-536-01-01,
// docs/architecture/ship_export_contracts_536.md). CPU only; CI job
// `shipping-config-loader`.
//
// Usage: saccade_shipping_run_completion_test [--keep DIR]
//
// Each case runs RunCompletion in a forked child with a fake sequence
// function (the same RunCompletion::run that saccade_track's run_sequences
// calls), so a kill is a real SIGKILL of that process and a second run is a
// second process. This file is compiled together with run_completion.cpp
// with SACCADE_RUN_COMPLETION_TEST_FAULTS: the child can be killed after a
// txt's temp file is written and before its rename ("txt_written"), right
// after the rename and before the journal records it ("txt_published"), and
// after the report is renamed and before state=complete ("report_published").
// Cases:
//   * fresh:          <out> (and its parents) do not exist; complete run with
//                     --report and --trace outside <out>; every hash in the
//                     journal is the file's; no temp file is left;
//   * failed_rerun:   a complete run, then a rerun that throws in its second
//                     sequence: journal failed with that sequence and the
//                     message, the first sequence written with the rerun's
//                     bytes, the earlier txt / trace / report of the rerun's
//                     sequences gone, other files in <out> untouched;
//   * killed_in_first_sequence: SIGKILL in the first sequence, after the
//                     earlier outputs were removed: the new journal is there
//                     (running, this run's id, every sequence pending) and
//                     none of the earlier run's outputs is;
//   * killed_in_sequence: SIGKILL inside a sequence: journal running, that
//                     sequence pending and its txt absent; the lock is free
//                     again for the next run;
//   * killed_before_rename: SIGKILL with the txt in its temp file: no txt at
//                     the sequence's path (only the temp file, left behind);
//   * killed_after_rename: SIGKILL between the txt rename and the journal
//                     update: the txt is the run's full output but pending;
//                     the caller rule does not count it;
//   * killed_after_report: SIGKILL between the report rename and
//                     state=complete: report present, run not complete;
//   * lock_busy:      a second process on a held <out> fails and changes no
//                     byte, name or mtime under the run's paths; the journal
//                     keeps the holder's run id;
//   * collisions:     --report on the journal, the lock or a txt is refused
//                     before <out> is created; a directory at --report fails
//                     the run and is not removed.
// --keep DIR leaves each case's files in DIR/<case>/ with the run id of its
// last invocation in DIR/<case>/invocation.run_id, for the reader check in
// tests/unit/test_native_track_parity.py.

#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "saccade_shipping/run_completion.hpp"
#include "saccade_shipping/preflight.hpp"
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

const char* g_fault_point = nullptr;
std::size_t g_fault_sequence = 0;
bool g_metadata_replaced = false;
const char* g_metadata_key = nullptr;
fs::path g_metadata_path, g_metadata_journal;
const std::string kReplacedMetadata = "metadata replaced after its observed buffer was published";
std::string read_file(const fs::path& p);
void write_file(const fs::path& p, const std::string& s);

}  // namespace

// The kill points in run_completion.cpp (SACCADE_RUN_COMPLETION_TEST_FAULTS).
void saccade_run_completion_test_fault(const char* point, std::size_t sequence) {
    if (std::strcmp(point, "identity_published") == 0 && g_metadata_key != nullptr && !g_metadata_replaced) {
        const JsonValue j = sh::parse_strict_json(read_file(g_metadata_journal));
        const JsonValue& b = *j.find("identity")->find("bindings")->find(g_metadata_key);
        if (b.find("observed_sha256")->kind == JsonValue::Kind::String) {
            write_file(g_metadata_path, kReplacedMetadata);
            g_metadata_replaced = true;
        }
    }
    if (g_fault_point != nullptr && std::strcmp(point, g_fault_point) == 0 && sequence == g_fault_sequence) {
        ::raise(SIGKILL);
    }
}

namespace {

const std::vector<std::string> kSeqs = {"SEQ-A", "SEQ-B", "SEQ-C"};

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + p.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void write_file(const fs::path& p, const std::string& s) {
    std::ofstream f(p, std::ios::binary);
    f << s;
    if (!f) throw std::runtime_error("cannot write " + p.string());
}

std::string sha(const std::string& s) { return sh::sha256_hex(s.data(), s.size()); }

// What generation `gen` of the fake runtime computes for sequence i.
std::string mot_text(int gen, std::size_t i) {
    std::string t;
    for (int f = 1; f <= 3; ++f) {
        if (f > 1) t += "\n";
        t += std::to_string(f) + "," + std::to_string(i + 1) + "," + std::to_string(gen) +
             ".00,20.00,5.00,6.00,0.9000,-1,-1,-1";
    }
    return t;
}
std::string trace_bytes(int gen, std::size_t i) { return "trace gen " + std::to_string(gen) + " seq " + std::to_string(i); }

struct Paths {
    fs::path out, report, trace;
};

struct Child {
    Paths p;
    int gen = 1;
    int throw_at = -1;       // the sequence function throws in sequence i
    bool load_failure = false;
    bool report_failure = false;
    int kill_in = -1;        // SIGKILL inside sequence i (after half its trace)
    int block_in = -1;       // sequence i: touch `ready`, then wait to be killed
    const char* fault = nullptr;
    std::size_t fault_sequence = 0;
    fs::path ready;
    fs::path run_id_file;    // the child's run id (saccade_track: stderr's first line)
    sh::PreflightInputs gate_a;
    fs::path identity_dir;
    const char* replace_metadata = nullptr;
};

struct Status {
    bool exited = false;
    int code = -1, signal = 0;
};

[[noreturn]] void child_main(const Child& c) {
    g_fault_point = c.fault;
    g_fault_sequence = c.fault_sequence;
    g_metadata_replaced = false;
    g_metadata_key = c.replace_metadata;
    g_metadata_journal = c.p.out / sh::kRunJournalName;
    if (c.replace_metadata != nullptr) {
        g_metadata_path = std::strcmp(c.replace_metadata, "config") == 0 ? c.gate_a.config :
                          std::strcmp(c.replace_metadata, "lineage") == 0 ? c.gate_a.lineage : c.gate_a.attestation;
    }
    std::optional<sh::RunCompletion> completion;
    try {
        const std::string id = sh::new_run_id();
        if (!c.run_id_file.empty()) write_file(c.run_id_file, id);
        completion.emplace(id, "fake_track", sh::RunOutputs{c.p.out, c.p.report, c.p.trace, kSeqs});
        // Exercise real CUDA-free Gate A, rather than forging a success identity.
        sh::run_preflight(c.gate_a, *completion);
        write_file(c.identity_dir / (id + ".json"), sh::dump_python_json(completion->identity()));
        if (c.load_failure) throw std::runtime_error("injected model load failure after Gate A");
        completion->run(
            [&](std::size_t i, const fs::path& trace_tmp) {
                const std::string tb = trace_bytes(c.gen, i);
                if (!trace_tmp.empty()) {
                    std::ofstream t(trace_tmp, std::ios::binary);
                    t << tb.substr(0, tb.size() / 2);
                    t.flush();
                    if (static_cast<int>(i) == c.kill_in) ::raise(SIGKILL);
                    t << tb.substr(tb.size() / 2);
                } else if (static_cast<int>(i) == c.kill_in) {
                    ::raise(SIGKILL);
                }
                if (static_cast<int>(i) == c.block_in) {
                    write_file(c.ready, "ready");
                    for (;;) ::pause();
                }
                if (static_cast<int>(i) == c.throw_at) {
                    throw std::runtime_error("injected failure in " + kSeqs[i]);
                }
                return mot_text(c.gen, i);
            },
            [&]() {
                if (c.report_failure) throw std::runtime_error("injected report failure after sequences");
                JsonValue report = JsonValue::make_object();
                report.set("format", JsonValue::make_string("saccade.native_track_report/v4"));
                report.set("run_id", JsonValue::make_string(completion->run_id()));
                report.set("identity", completion->identity());
                report.set("gen", JsonValue::make_int(c.gen));
                return sh::dump_python_json(report) + "\n";
            });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fake_track: %s\n", e.what());
        if (completion) completion->fail(e.what());
        std::fflush(stderr);
        ::_exit(2);
    }
    ::_exit(0);
}

pid_t spawn(const Child& c) {
    std::fflush(stdout);
    std::fflush(stderr);
    const pid_t pid = ::fork();
    if (pid < 0) throw std::runtime_error("fork failed");
    if (pid == 0) child_main(c);
    return pid;
}

Status wait_for(pid_t pid) {
    int st = 0;
    if (::waitpid(pid, &st, 0) != pid) throw std::runtime_error("waitpid failed");
    Status s;
    if (WIFEXITED(st)) {
        s.exited = true;
        s.code = WEXITSTATUS(st);
    } else if (WIFSIGNALED(st)) {
        s.signal = WTERMSIG(st);
    }
    return s;
}

Status run_child(const Child& c) { return wait_for(spawn(c)); }

JsonValue journal(const Paths& p) { return sh::parse_strict_json(read_file(p.out / sh::kRunJournalName)); }

std::string str(const JsonValue& o, const char* k) {
    const JsonValue* v = o.find(k);
    return v != nullptr && v->kind == JsonValue::Kind::String ? v->string : std::string("<not a string>");
}

bool is_null(const JsonValue& o, const char* k) {
    const JsonValue* v = o.find(k);
    return v != nullptr && v->kind == JsonValue::Kind::Null;
}

const JsonValue& seq_entry(const JsonValue& j, std::size_t i) { return j.find("sequences")->array.at(i); }

// The caller rule (run_completion.hpp): the sequences of `run_id` whose txt
// is committed.
std::vector<std::string> committed(const Paths& p, const std::string& run_id) {
    std::vector<std::string> out;
    const JsonValue j = journal(p);
    if (str(j, "run_id") != run_id) return out;
    for (const JsonValue& s : j.find("sequences")->array) {
        if (str(s, "state") != "written") continue;
        const fs::path txt = p.out / (str(s, "name") + ".txt");
        if (fs::exists(txt) && sha(read_file(txt)) == str(s, "txt_sha256")) out.push_back(str(s, "name"));
    }
    return out;
}

bool complete(const Paths& p, const std::string& run_id) {
    const JsonValue j = journal(p);
    return str(j, "run_id") == run_id && str(j, "state") == "complete";
}

std::vector<std::string> temp_files(const fs::path& root) {
    std::vector<std::string> out;
    if (!fs::exists(root)) return out;
    for (const auto& e : fs::recursive_directory_iterator(root)) {
        const std::string n = e.path().filename().string();
        if (n.size() > 4 && n[0] == '.' && n.substr(n.size() - 4) == ".tmp") out.push_back(e.path().string());
    }
    return out;
}

// name -> sha256 + mtime of every file under `roots`.
std::map<std::string, std::string> snapshot(const std::vector<fs::path>& roots) {
    std::map<std::string, std::string> m;
    for (const fs::path& r : roots) {
        if (!fs::exists(r)) continue;
        std::vector<fs::path> files;
        if (fs::is_directory(r)) {
            for (const auto& e : fs::recursive_directory_iterator(r)) files.push_back(e.path());
        } else {
            files.push_back(r);
        }
        for (const fs::path& f : files) {
            struct stat st {};
            ::lstat(f.c_str(), &st);
            std::string v = std::to_string(st.st_mtim.tv_sec) + "." + std::to_string(st.st_mtim.tv_nsec);
            if (fs::is_regular_file(f)) v += " " + sha(read_file(f));
            m[f.string()] = v;
        }
    }
    return m;
}

struct Case {
    fs::path dir;
    Paths p;
    fs::path run_id_file;
    sh::PreflightInputs gate_a;
    fs::path identity_dir;
};

JsonValue& at(JsonValue& object, const char* key) {
    JsonValue* value = object.find(key);
    if (value == nullptr) throw std::runtime_error(std::string("fixture has no ") + key);
    return *value;
}

// Immutable case inputs: the same real Gate A parser and checks used by the
// CLI, on a self-consistent stand-in model. No model-loading/GPU claim.
sh::PreflightInputs make_gate_a_inputs(const fs::path& dir) {
    const fs::path repo(SACCADE_TEST_REPO_ROOT);
    const fs::path root = dir / "gate_a_model";
    fs::create_directories(root);
    const fs::path config = root / "config.json";
    write_file(config, read_file(repo / "configs/shipping/mamba_whole_graph.resolved.json"));
    JsonValue lineage = sh::parse_strict_json(read_file(repo / "tests/native/fixtures/shipping_head_lineage.json"));
    for (const char* key : {"op_library", "torchscript", "backbone_engine"}) {
        JsonValue& entry = std::strcmp(key, "backbone_engine") == 0 ?
                           at(at(lineage, "companions"), key) : at(lineage, key);
        const fs::path file = root / at(entry, "path").string;
        fs::create_directories(file.parent_path());
        const std::string bytes = std::string("completion test stand-in ") + key;
        write_file(file, bytes);
        at(entry, "sha256").string = sha(bytes);
    }
    const fs::path lineage_path = root / "lineage.json";
    write_file(lineage_path, sh::dump_python_json(lineage));
    std::vector<std::string> sequences;
    for (const std::string& name : kSeqs) {
        const fs::path sequence = root / name;
        fs::create_directories(sequence / "img1");
        write_file(sequence / "seqinfo.ini", "[Sequence]\nname=" + name +
                   "\nimWidth=8\nimHeight=6\nseqLength=3\n");
        for (int frame = 1; frame <= 3; ++frame) {
            char filename[16];
            std::snprintf(filename, sizeof filename, "%06d.jpg", frame);
            write_file(sequence / "img1" / filename, "not decoded by Gate A");
        }
        sequences.push_back(sequence.string());
    }
    return {config.string(), lineage_path.string(), "", root.string(), sequences};
}

Case make_case(const fs::path& base, const char* name) {
    Case c;
    c.dir = base / name;
    fs::create_directories(c.dir);
    c.p = {c.dir / "out", c.dir / "track_report.json", c.dir / "trace"};
    c.run_id_file = c.dir / "invocation.run_id";
    c.gate_a = make_gate_a_inputs(c.dir);
    c.identity_dir = c.dir / "gate_a_identity";
    fs::create_directories(c.identity_dir);
    return c;
}

Child child_of(const Case& c, int gen) {
    Child ch;
    ch.p = c.p;
    ch.gen = gen;
    ch.run_id_file = c.run_id_file;
    ch.gate_a = c.gate_a;
    ch.identity_dir = c.identity_dir;
    return ch;
}

std::string last_run_id(const Case& c) { return read_file(c.run_id_file); }

void check_preserved_identity(const Case& c, const std::string& id) {
    const JsonValue j = journal(c.p);
    const JsonValue sealed = sh::parse_strict_json(read_file(c.identity_dir / (id + ".json")));
    CHECK(str(sealed, "level") == "checksum_matched");
    CHECK(*j.find("identity") == sealed);
    if (fs::is_regular_file(c.p.report)) {
        const JsonValue report = sh::parse_strict_json(read_file(c.p.report));
        CHECK(str(report, "format") == "saccade.native_track_report/v4");
        CHECK(str(report, "run_id") == id);
        CHECK(*report.find("identity") == sealed);
    }
}

void check_fully_committed(const Case& c, int gen) {
    const std::string id = last_run_id(c);
    const JsonValue j = journal(c.p);
    CHECK(str(j, "format") == sh::kRunJournalFormat);
    CHECK(str(j, "run_id") == id);
    CHECK(str(j, "state") == "complete");
    CHECK(is_null(j, "failure"));
    check_preserved_identity(c, id);
    for (std::size_t i = 0; i < kSeqs.size(); ++i) {
        const JsonValue& s = seq_entry(j, i);
        CHECK(str(s, "name") == kSeqs[i]);
        CHECK(str(s, "state") == "written");
        CHECK(read_file(c.p.out / (kSeqs[i] + ".txt")) == mot_text(gen, i));
        CHECK(str(s, "txt_sha256") == sha(mot_text(gen, i)));
        CHECK(read_file(c.p.trace / kSeqs[i] / "detector.bin") == trace_bytes(gen, i));
        CHECK(str(s, "trace_sha256") == sha(trace_bytes(gen, i)));
    }
    const std::string rep = read_file(c.p.report);
    CHECK(rep.find(id) != std::string::npos);
    CHECK(str(*j.find("report"), "sha256") == sha(rep));
    CHECK(committed(c.p, id) == kSeqs);
    CHECK(complete(c.p, id));
}

void case_fresh(const fs::path& base) {
    Case c = make_case(base, "fresh");
    c.p.out = c.dir / "missing" / "parent" / "out";  // nothing of it exists
    CHECK(!fs::exists(c.dir / "missing"));
    const Status s = run_child(child_of(c, 1));
    CHECK(s.exited && s.code == 0);
    check_fully_committed(c, 1);
    CHECK(fs::exists(c.p.out / sh::kRunLockName));
    CHECK(temp_files(c.dir).empty());
}

void case_failed_rerun(const fs::path& base) {
    Case c = make_case(base, "failed_rerun");
    CHECK(run_child(child_of(c, 1)).code == 0);
    check_fully_committed(c, 1);
    const std::string first = last_run_id(c);
    write_file(c.p.out / "OTHER.txt", "another sequence, not in this run");
    write_file(c.p.out / "notes", "unrelated");
    Child ch = child_of(c, 2);
    ch.throw_at = 1;
    const Status s = run_child(ch);
    CHECK(s.exited && s.code == 2);
    const std::string id = last_run_id(c);
    CHECK(id != first);
    const JsonValue j = journal(c.p);
    CHECK(str(j, "run_id") == id);
    CHECK(str(j, "state") == "failed");
    check_preserved_identity(c, id);
    CHECK(str(*j.find("failure"), "sequence") == kSeqs[1]);
    CHECK(str(*j.find("failure"), "message") == "injected failure in " + kSeqs[1]);
    CHECK(str(seq_entry(j, 0), "state") == "written");
    CHECK(read_file(c.p.out / (kSeqs[0] + ".txt")) == mot_text(2, 0));
    CHECK(read_file(c.p.trace / kSeqs[0] / "detector.bin") == trace_bytes(2, 0));
    for (std::size_t i = 1; i < kSeqs.size(); ++i) {
        CHECK(str(seq_entry(j, i), "state") == "pending");
        CHECK(is_null(seq_entry(j, i), "txt_sha256"));
        CHECK(!fs::exists(c.p.out / (kSeqs[i] + ".txt")));           // the first run's is gone
        CHECK(!fs::exists(c.p.trace / kSeqs[i] / "detector.bin"));
    }
    CHECK(!fs::exists(c.p.report));                                  // no stale report
    CHECK(is_null(*j.find("report"), "sha256"));
    CHECK(read_file(c.p.out / "OTHER.txt") == "another sequence, not in this run");
    CHECK(read_file(c.p.out / "notes") == "unrelated");
    CHECK(committed(c.p, id) == std::vector<std::string>{kSeqs[0]});
    CHECK(!complete(c.p, id));
    CHECK(committed(c.p, first).empty());  // the earlier run id no longer names anything
    CHECK(temp_files(c.dir).empty());
}

void case_load_failure(const fs::path& base) {
    Case c = make_case(base, "load_failure");
    Child child = child_of(c, 1);
    child.load_failure = true;
    const Status status = run_child(child);
    CHECK(status.exited && status.code == 2);
    const std::string id = last_run_id(c);
    const JsonValue j = journal(c.p);
    check_preserved_identity(c, id);
    CHECK(str(j, "state") == "failed");
    CHECK(is_null(*j.find("failure"), "sequence"));
    CHECK(str(*j.find("failure"), "message") == "injected model load failure after Gate A");
    for (std::size_t i = 0; i < kSeqs.size(); ++i) CHECK(str(seq_entry(j, i), "state") == "pending");
    CHECK(!fs::exists(c.p.report));
    CHECK(committed(c.p, id).empty());
    CHECK(!complete(c.p, id));
}

void case_report_failure(const fs::path& base) {
    Case c = make_case(base, "report_failure");
    Child child = child_of(c, 1);
    child.report_failure = true;
    const Status status = run_child(child);
    CHECK(status.exited && status.code == 2);
    const std::string id = last_run_id(c);
    const JsonValue j = journal(c.p);
    check_preserved_identity(c, id);
    CHECK(str(j, "state") == "failed");
    CHECK(is_null(*j.find("failure"), "sequence"));
    CHECK(str(*j.find("failure"), "message") == "injected report failure after sequences");
    CHECK(!fs::exists(c.p.report));
    CHECK(committed(c.p, id) == kSeqs);
    CHECK(!complete(c.p, id));
}

void case_metadata_buffer(const fs::path& base, const char* key) {
    Case c = make_case(base, (std::string("observed_buffer_") + key).c_str());
    if (std::strcmp(key, "attestation") == 0) {
        const JsonValue lineage = sh::parse_strict_json(read_file(c.gate_a.lineage));
        JsonValue a = sh::parse_strict_json(read_file(fs::path(SACCADE_TEST_REPO_ROOT) /
                                                    "configs/shipping/mamba_head_realization.attestation.json"));
        JsonValue& frozen = at(a, "frozen_lineage");
        at(frozen, "sha256").string = sha(read_file(c.gate_a.lineage));
        at(frozen, "torchscript_sha256").string = str(*lineage.find("torchscript"), "sha256");
        at(frozen, "torchscript_content_sha256").string = str(*lineage.find("torchscript"), "content_sha256");
        at(frozen, "op_library_sha256").string = str(*lineage.find("op_library"), "sha256");
        at(at(a, "op_library"), "sha256").string = str(*lineage.find("op_library"), "sha256");
        c.gate_a.attestation = (c.dir / "gate_a_model" / "attestation.json").string();
        write_file(c.gate_a.attestation, sh::dump_python_json(a));
    }
    const fs::path path = std::strcmp(key, "config") == 0 ? c.gate_a.config :
                          std::strcmp(key, "lineage") == 0 ? c.gate_a.lineage : c.gate_a.attestation;
    const std::string original_hash = sha(read_file(path));
    Child child = child_of(c, 1);
    child.replace_metadata = key;
    const Status status = run_child(child);
    CHECK(status.exited && status.code == 0);
    const JsonValue j = journal(c.p);
    CHECK(str(j, "state") == "complete");
    check_preserved_identity(c, last_run_id(c));
    const JsonValue& b = *j.find("identity")->find("bindings")->find(key);
    CHECK(str(b, "observed_sha256") == original_hash);
    CHECK(read_file(path) == kReplacedMetadata);
    CHECK(str(b, "observed_sha256") != sha(read_file(path)));
    CHECK(str(b, "status") == "unchecked");
    // Invalid replacement bytes make a second read fail. Gate A succeeding
    // proves the parser consumed the exact original buffer whose SHA it kept.
    CHECK(complete(c.p, last_run_id(c)));
}

void case_killed_in_first_sequence(const fs::path& base) {
    Case c = make_case(base, "killed_in_first_sequence");
    CHECK(run_child(child_of(c, 1)).code == 0);
    Child ch = child_of(c, 2);
    ch.kill_in = 0;
    const Status s = run_child(ch);
    CHECK(!s.exited && s.signal == SIGKILL);
    const std::string id = last_run_id(c);
    CHECK(fs::exists(c.p.out / sh::kRunJournalName));  // the cleanup did not take it
    const JsonValue j = journal(c.p);
    CHECK(str(j, "run_id") == id);
    CHECK(str(j, "state") == "running");
    check_preserved_identity(c, id);
    for (std::size_t i = 0; i < kSeqs.size(); ++i) {
        CHECK(str(seq_entry(j, i), "state") == "pending");
        CHECK(!fs::exists(c.p.out / (kSeqs[i] + ".txt")));
        CHECK(!fs::exists(c.p.trace / kSeqs[i] / "detector.bin"));
    }
    CHECK(!fs::exists(c.p.report));
    CHECK(committed(c.p, id).empty());
}

void case_killed_in_sequence(const fs::path& base) {
    Case c = make_case(base, "killed_in_sequence");
    CHECK(run_child(child_of(c, 1)).code == 0);
    Child ch = child_of(c, 2);
    ch.kill_in = 1;
    const Status s = run_child(ch);
    CHECK(!s.exited && s.signal == SIGKILL);
    const std::string id = last_run_id(c);
    const JsonValue j = journal(c.p);
    CHECK(str(j, "run_id") == id);
    CHECK(str(j, "state") == "running");
    check_preserved_identity(c, id);
    CHECK(str(seq_entry(j, 0), "state") == "written");
    CHECK(str(seq_entry(j, 1), "state") == "pending");
    CHECK(!fs::exists(c.p.out / (kSeqs[1] + ".txt")));
    CHECK(!fs::exists(c.p.trace / kSeqs[1] / "detector.bin"));
    CHECK(!fs::exists(c.p.report));
    CHECK(committed(c.p, id) == std::vector<std::string>{kSeqs[0]});
    CHECK(!complete(c.p, id));
    // The kernel released the killed run's lock: the next run takes <out>.
    CHECK(run_child(child_of(c, 3)).code == 0);
    check_fully_committed(c, 3);
}

void case_killed_before_rename(const fs::path& base) {
    Case c = make_case(base, "killed_before_rename");
    CHECK(run_child(child_of(c, 1)).code == 0);
    Child ch = child_of(c, 2);
    ch.fault = "txt_written";
    ch.fault_sequence = 1;
    const Status s = run_child(ch);
    CHECK(!s.exited && s.signal == SIGKILL);
    const std::string id = last_run_id(c);
    CHECK(!fs::exists(c.p.out / (kSeqs[1] + ".txt")));  // neither run's txt
    CHECK(temp_files(c.p.out).size() == 1);              // the killed run's temp file
    check_preserved_identity(c, id);
    CHECK(str(seq_entry(journal(c.p), 1), "state") == "pending");
    CHECK(committed(c.p, id) == std::vector<std::string>{kSeqs[0]});
    CHECK(!complete(c.p, id));
}

void case_killed_after_rename(const fs::path& base) {
    Case c = make_case(base, "killed_after_rename");
    CHECK(run_child(child_of(c, 1)).code == 0);
    Child ch = child_of(c, 2);
    ch.fault = "txt_published";
    ch.fault_sequence = 1;
    const Status s = run_child(ch);
    CHECK(!s.exited && s.signal == SIGKILL);
    const std::string id = last_run_id(c);
    const JsonValue j = journal(c.p);
    CHECK(str(j, "state") == "running");
    check_preserved_identity(c, id);
    // The txt is this run's whole output, renamed into place ...
    CHECK(read_file(c.p.out / (kSeqs[1] + ".txt")) == mot_text(2, 1));
    // ... and the journal never confirmed it: pending, not committed.
    CHECK(str(seq_entry(j, 1), "state") == "pending");
    CHECK(is_null(seq_entry(j, 1), "txt_sha256"));
    CHECK(committed(c.p, id) == std::vector<std::string>{kSeqs[0]});
    CHECK(!complete(c.p, id));
}

void case_killed_after_report(const fs::path& base) {
    Case c = make_case(base, "killed_after_report");
    Child ch = child_of(c, 1);
    ch.fault = "report_published";
    ch.fault_sequence = kSeqs.size();
    const Status s = run_child(ch);
    CHECK(!s.exited && s.signal == SIGKILL);
    const std::string id = last_run_id(c);
    const JsonValue j = journal(c.p);
    CHECK(str(j, "state") == "running");
    check_preserved_identity(c, id);
    CHECK(read_file(c.p.report).find(id) != std::string::npos);  // this run's report ...
    CHECK(is_null(*j.find("report"), "sha256"));                  // ... not recorded
    CHECK(committed(c.p, id) == kSeqs);
    CHECK(!complete(c.p, id));
}

void case_lock_busy(const fs::path& base) {
    Case c = make_case(base, "lock_busy");
    CHECK(run_child(child_of(c, 1)).code == 0);
    Child holder = child_of(c, 2);
    holder.block_in = 1;
    holder.ready = c.dir / "ready";
    holder.run_id_file = c.dir / "holder.run_id";
    const pid_t pid = spawn(holder);
    for (int i = 0; i < 2000 && !fs::exists(holder.ready); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(fs::exists(holder.ready));
    const std::vector<fs::path> roots{c.p.out, c.p.report, c.p.trace};
    const auto before = snapshot(roots);
    const Status s = run_child(child_of(c, 3));
    CHECK(s.exited && s.code == 2);
    CHECK(snapshot(roots) == before);  // no name, byte or mtime changed
    const std::string holder_id = read_file(holder.run_id_file);
    check_preserved_identity(c, holder_id);
    CHECK(str(journal(c.p), "run_id") == holder_id);
    CHECK(str(journal(c.p), "run_id") != last_run_id(c));
    ::kill(pid, SIGKILL);
    CHECK(wait_for(pid).signal == SIGKILL);
    CHECK(str(journal(c.p), "state") == "running");
    CHECK(committed(c.p, holder_id) == std::vector<std::string>{kSeqs[0]});
    fs::remove(holder.ready);
}

bool refused_before_out(const Paths& p) {
    try {
        sh::RunCompletion rc("0123456789abcdef0123456789abcdef", "fake_track", sh::RunOutputs{p.out, p.report, p.trace, kSeqs});
    } catch (const std::exception& e) {
        return std::string(e.what()).find("is the journal, the lock or another output") != std::string::npos &&
               !fs::exists(p.out);
    }
    return false;
}

void case_collisions(const fs::path& base) {
    Case c = make_case(base, "collisions");
    const fs::path out = c.p.out;
    for (const fs::path& report : {out / sh::kRunJournalName, out / sh::kRunLockName, out / "SEQ-B.txt",
                                   out / "sub" / ".." / sh::kRunJournalName, c.p.trace / "SEQ-A" / "detector.bin"}) {
        CHECK(refused_before_out({out, report, c.p.trace}));
    }
    // A directory where the report goes: the run fails, the directory stays.
    fs::create_directories(c.p.report);
    write_file(c.p.report / "keep", "x");
    const Status s = run_child(child_of(c, 1));
    CHECK(s.exited && s.code == 2);
    CHECK(fs::exists(c.p.report / "keep"));
    const JsonValue j = journal(c.p);
    CHECK(str(j, "state") == "failed");
    CHECK(is_null(*j.find("failure"), "sequence"));
    CHECK(str(*j.find("failure"), "message").find("is a directory") != std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
    fs::path keep;
    if (argc == 3 && std::string(argv[1]) == "--keep") {
        keep = argv[2];
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--keep DIR]\n", argv[0]);
        return 2;
    }
    try {
        fs::path base = keep;
        if (base.empty()) {
            std::string tmpl = (fs::temp_directory_path() / "saccade_run_completion.XXXXXX").string();
            if (::mkdtemp(tmpl.data()) == nullptr) throw std::runtime_error("mkdtemp failed");
            base = tmpl;
        } else {
            fs::remove_all(base);
            fs::create_directories(base);
        }
        CHECK(sh::new_run_id().size() == 32);
        CHECK(sh::new_run_id() != sh::new_run_id());
        case_fresh(base);
        case_failed_rerun(base);
        case_load_failure(base);
        case_report_failure(base);
        case_metadata_buffer(base, "config");
        case_metadata_buffer(base, "lineage");
        case_metadata_buffer(base, "attestation");
        case_killed_in_first_sequence(base);
        case_killed_in_sequence(base);
        case_killed_before_rename(base);
        case_killed_after_rename(base);
        case_killed_after_report(base);
        case_lock_busy(base);
        case_collisions(base);
        if (keep.empty()) fs::remove_all(base);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
