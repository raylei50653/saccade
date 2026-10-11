// CPU controls for the N03 operator load (#549 S2-2, P-N03 rev 3c; D-N03-2..5):
// saccade_shipping/operator_loader.hpp on a real dynamic loader, with two
// stand-in operator objects (fixtures/operator_fixture.cpp, values 1 and 2).
//
// Usage: saccade_shipping_operator_loader_test <fixture value 1> <fixture value 2>
//            <origin runpath> <origin rpath> <absolute runpath>
//        saccade_shipping_operator_loader_test --print-entrypoint-operator
//   (the second form prints entrypoint_runtime_operator_path(), or the
//   refusal and exit 2; tests/unit/test_shipping_n03_auditor.py runs it from
//   an install-shaped prefix, directly and through the launcher's loader).
//
// One verified operator may be loaded per process, so every load case runs in
// its own child process:
//   * maps records: complete records of any length (8191/8192/65535 bytes,
//     with and without a final newline), decoys in earlier oversized records;
//   * D-N03-5 path: <prefix>/libexec/<name> -> <prefix>/share/saccade/build/
//     libsaccade_scan_torchop.so; anything else is refused;
//   * loaded bytes: the object is mapped from the sealed memfd (the kernel's
//     record), and replacing or rewriting the source after prepare, or after
//     load, does not change what runs (value 1);
//   * registry: the same SHA-256 and path reuse the handle (also a request
//     prepared while an identical one loaded); another SHA-256 or path is
//     refused; the post-check refuses a memfd that is not the mapping;
//   * descriptor reuse through load(): an object dlopen'ed earlier as
//     /proc/self/fd/<K> is returned by name when the operator memfd gets the
//     number K; load() refuses it, and the process is poisoned;
//   * seals: F_SEAL_FUTURE_WRITE without F_SEAL_WRITE does not count;
//   * search paths: $ORIGIN in DT_RUNPATH or DT_RPATH, or a non-ELF object, is
//     refused before any memfd; an absolute DT_RUNPATH is accepted; malformed
//     dynamic tables are refused, among them a PT_DYNAMIC whose file offset
//     names an empty table while the mapped one (what the loader reads)
//     carries $ORIGIN;
//   * refusals: wrong SHA-256 or size, a symlink (final or directory
//     component), a directory, a FIFO, a relative or missing path.
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>

#include "saccade_shipping/detector_plan.hpp"
#include "saccade_shipping/operator_loader.hpp"
#include "saccade_shipping/sha256.hpp"

namespace fs = std::filesystem;
namespace sh = saccade::shipping;

namespace {
int failures = 0;
int checks = 0;
#define CHECK(condition) do { ++checks; if (!(condition)) { ++failures; \
    std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition); } } while (0)

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream o;
    o << in.rdbuf();
    return o.str();
}

void write_file(const fs::path& p, const std::string& v) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(v.data(), static_cast<std::streamsize>(v.size()));
    if (!out) throw std::runtime_error("cannot write " + p.string());
}

// In place: same inode, new bytes.
void overwrite(const fs::path& p, const std::string& v) {
    const int fd = ::open(p.c_str(), O_WRONLY | O_TRUNC);
    if (fd < 0 || ::write(fd, v.data(), v.size()) != static_cast<ssize_t>(v.size())) {
        throw std::runtime_error("cannot overwrite " + p.string());
    }
    ::close(fd);
}

std::string refusal(const std::function<void()>& f) {
    try {
        f();
    } catch (const sh::DetectorLoadError& e) {
        return e.what();
    } catch (const std::exception& e) {
        return std::string("not a DetectorLoadError: ") + e.what();
    }
    return "";
}

bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

// Malformed variants of an ELF object (program header / dynamic table edits).
Elf64_Phdr* phdrs(std::string& elf, int& count) {
    Elf64_Ehdr eh;
    std::memcpy(&eh, elf.data(), sizeof eh);
    count = eh.e_phnum;
    return reinterpret_cast<Elf64_Phdr*>(elf.data() + eh.e_phoff);
}

Elf64_Phdr* dynamic_phdr(std::string& elf) {
    int count = 0;
    Elf64_Phdr* ph = phdrs(elf, count);
    for (int i = 0; i < count; ++i) {
        if (ph[i].p_type == PT_DYNAMIC) return &ph[i];
    }
    return nullptr;
}

std::string dynamic_offset_elsewhere(std::string elf) {  // file offset -> appended zeros
    const std::size_t zeros = elf.size();
    elf.append(4096, '\0');
    dynamic_phdr(elf)->p_offset = zeros;
    return elf;
}

std::string dynamic_unterminated(std::string elf) {
    dynamic_phdr(elf)->p_filesz = sizeof(Elf64_Dyn);  // one non-null entry, no DT_NULL
    return elf;
}

std::string strsz_oversized(std::string elf) {
    const Elf64_Phdr* d = dynamic_phdr(elf);
    for (std::uint64_t at = d->p_offset; at + sizeof(Elf64_Dyn) <= d->p_offset + d->p_filesz; at += sizeof(Elf64_Dyn)) {
        auto* e = reinterpret_cast<Elf64_Dyn*>(elf.data() + at);
        if (e->d_tag == DT_STRSZ) e->d_un.d_val = 0x7fffffff;
    }
    return elf;
}

// Review round 3: an earlier PT_LOAD extended over the dynamic table's pages,
// PT_DYNAMIC's file offset moved to where that segment would read zeros
// (DT_NULL); the loader maps the later, original segment over it (MAP_FIXED)
// and reads the real table.
std::string overlapping_dynamic(std::string elf) {
    Elf64_Phdr* d = dynamic_phdr(elf);
    const std::uint64_t dyn_vaddr = d->p_vaddr, dyn_size = d->p_filesz;
    int count = 0;
    Elf64_Phdr* ph = phdrs(elf, count);
    Elf64_Phdr* container = nullptr;
    Elf64_Phdr* prev = nullptr;
    for (int i = 0; i < count; ++i) {
        if (ph[i].p_type == PT_LOAD && dyn_vaddr >= ph[i].p_vaddr && dyn_vaddr < ph[i].p_vaddr + ph[i].p_memsz) container = &ph[i];
    }
    for (int i = 0; i < count; ++i) {
        if (ph[i].p_type == PT_LOAD && container != nullptr && ph[i].p_vaddr < container->p_vaddr &&
            (prev == nullptr || ph[i].p_vaddr > prev->p_vaddr)) {
            prev = &ph[i];
        }
    }
    if (prev == nullptr) return elf;
    const std::uint64_t span = dyn_vaddr + dyn_size - prev->p_vaddr;
    const std::uint64_t fake = prev->p_offset + (dyn_vaddr - prev->p_vaddr);
    prev->p_filesz = prev->p_memsz = span;
    d->p_offset = fake;
    const std::uint64_t needed = prev->p_offset + span;
    if (elf.size() < needed) elf.resize(needed, '\0');     // may reallocate: no pointer use after
    std::memset(elf.data() + fake, 0, sizeof(Elf64_Dyn));  // DT_NULL there
    return elf;
}

std::string two_dynamic(std::string elf) {
    int count = 0;
    Elf64_Phdr* ph = phdrs(elf, count);
    for (int i = 0; i < count; ++i) {
        if (ph[i].p_type == PT_NOTE) {
            ph[i].p_type = PT_DYNAMIC;
            break;
        }
    }
    return elf;
}

int value_of(void* handle) {
    using Fn = int (*)();
    const auto fn = reinterpret_cast<Fn>(::dlsym(handle, "saccade_operator_fixture_value"));
    return fn ? fn() : -1;
}

// Runs `body` in a child process (its own registry); a failed CHECK there
// fails the case.
void in_child(const char* name, const std::function<void()>& body) {
    std::fflush(nullptr);
    const pid_t pid = ::fork();
    if (pid == 0) {
        failures = 0;  // the child's verdict is its own checks only
        try {
            body();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s: unexpected %s\n", name, e.what());
            ++failures;
        }
        std::fflush(nullptr);
        ::_exit(failures == 0 ? 0 : 1);
    }
    int status = 0;
    ++checks;
    if (pid < 0 || ::waitpid(pid, &status, 0) != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        ++failures;
        std::fprintf(stderr, "case %s failed\n", name);
    } else {
        std::fprintf(stderr, "case %s ok\n", name);
    }
}

// /proc/self/maps texts: the target record covers 0x5000 with inode 777.
std::string target(std::size_t len, bool newline) {
    std::string r = "1000-9000 r--p 00000000 08:30 777 /";
    r.resize(len, 'p');
    return r + (newline ? "\n" : "");
}

std::string decoy(std::size_t len) {  // header-shaped text claiming inode 666 inside a long path
    const std::string fake = "1000-9000 r--p 00000000 08:30 666 /fake";
    std::string r = "a000-b000 r--p 00000000 08:30 1 /";
    r.resize(len, 'q');
    for (std::size_t at : {std::size_t(4096), std::size_t(8191), std::size_t(65535)}) {
        if (at + fake.size() < len) r.replace(at, fake.size(), fake);
    }
    return r + "\n";
}

void maps_parser() {
    const std::string pre = "0400-0800 r--p 00000000 08:30 5 /pre\n";
    for (std::size_t len : {std::size_t(60), std::size_t(8191), std::size_t(8192), std::size_t(65535),
                            std::size_t(70000)}) {
        for (bool newline : {true, false}) {
            std::istringstream in(pre + target(len, newline));
            const auto m = sh::mapped_object_at(in, 0x5000);
            CHECK(m && m->inode == 777 && m->major == 8 && m->minor == 0x30 && m->path.size() == len - 34);
        }
    }
    for (std::size_t len : {std::size_t(8191), std::size_t(70000)}) {
        std::istringstream in(pre + decoy(len) + target(60, true));
        const auto m = sh::mapped_object_at(in, 0x5000);
        CHECK(m && m->inode == 777);
    }
    std::istringstream none(pre + "c000-d000 r--p 00000000 08:30 9 /other\n");
    CHECK(!sh::mapped_object_at(none, 0x5000));
    std::istringstream anon("4000-6000 rw-p 00000000 00:00 0 \n");
    const auto a = sh::mapped_object_at(anon, 0x5000);
    CHECK(a && a->inode == 0 && a->path.empty());
}

void entrypoint_paths() {
    CHECK(sh::runtime_operator_path_for_entrypoint("/opt/saccade/libexec/saccade_track") ==
          "/opt/saccade/share/saccade/build/libsaccade_scan_torchop.so");
    CHECK(sh::runtime_operator_path_for_entrypoint("/libexec/saccade_track") ==
          "/share/saccade/build/libsaccade_scan_torchop.so");
    for (const char* bad : {"/opt/saccade/bin/saccade_track", "/opt/saccade/libexec2/saccade_track",
                            "/opt/saccade/libexec/", "libexec/saccade_track", "", "/saccade_track"}) {
        const std::string e = refusal([&] { (void)sh::runtime_operator_path_for_entrypoint(bad); });
        CHECK(has(e, "shipping detector load: the entrypoint"));
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--print-entrypoint-operator") {
        try {
            std::printf("%s\n", sh::entrypoint_runtime_operator_path().c_str());
            return 0;
        } catch (const sh::DetectorLoadError& e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 2;
        }
    }
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s <value 1> <value 2> <origin runpath> <origin rpath> <absolute runpath>\n",
                     argv[0]);
        return 2;
    }
    const std::string one = read_file(argv[1]), two = read_file(argv[2]);
    const std::string sha1 = sh::sha256_hex(one.data(), one.size());
    const std::string sha2 = sh::sha256_hex(two.data(), two.size());
    if (one.empty() || two.empty() || sha1 == sha2) {
        std::fprintf(stderr, "the two fixtures must exist and differ\n");
        return 2;
    }
    char temp[] = "/tmp/saccade_operator_loader.XXXXXX";
    char* created = ::mkdtemp(temp);
    if (!created) return 2;
    const fs::path dir = fs::canonical(created);
    try {
        maps_parser();
        entrypoint_paths();
        const fs::path a = dir / "a" / "libsaccade_scan_torchop.so";
        const fs::path b = dir / "b" / "libsaccade_scan_torchop.so";
        fs::create_directories(a.parent_path());
        fs::create_directories(b.parent_path());
        write_file(a, one);
        write_file(b, one);

        in_child("load_from_sealed_memfd", [&] {
            sh::VerifiedOperator op = sh::VerifiedOperator::prepare(a.string(), sha1, static_cast<std::int64_t>(one.size()));
            CHECK(!op.reused() && op.sha256() == sha1 && op.source() == a.string());
            void* h = op.load();
            CHECK(h != nullptr && value_of(h) == 1);
            const std::string& mapped = op.mapped().path;
            CHECK(mapped.rfind(std::string("/memfd:") + sh::kOperatorMemfdLabel, 0) == 0);
            CHECK(mapped.size() > 10 && mapped.compare(mapped.size() - 10, 10, " (deleted)") == 0);
            // The post-check refuses a memfd that is not the mapping (P-N03 F5),
            // even one holding the same bytes.
            const int other = ::memfd_create("saccade-n03:src-fd=0", MFD_CLOEXEC);
            CHECK(other >= 0 && ::write(other, one.data(), one.size()) == static_cast<ssize_t>(one.size()));
            CHECK(!sh::mapped_from_memfd(h, other));
            CHECK(!sh::mapped_from_memfd(nullptr, other));
            ::close(other);
        });
        in_child("registry_reuse_and_refusals", [&] {
            void* h = sh::VerifiedOperator::prepare(a.string(), sha1).load();
            sh::VerifiedOperator again = sh::VerifiedOperator::prepare(a.string(), sha1);
            CHECK(again.reused() && again.load() == h);
            CHECK(has(refusal([&] { (void)sh::VerifiedOperator::prepare(a.string(), sha2); }),
                      "is already loaded; operator library"));
            CHECK(has(refusal([&] { (void)sh::VerifiedOperator::prepare(b.string(), sha1); }),
                      "already loaded from " + a.string()));
        });
        in_child("source_replaced_after_prepare", [&] {
            sh::VerifiedOperator op = sh::VerifiedOperator::prepare(a.string(), sha1);
            write_file(dir / "two", two);
            fs::rename(dir / "two", a);  // rename-replace after the hash (P-N03 B1)
            void* h = op.load();
            CHECK(value_of(h) == 1);
            overwrite(a, two);  // in place after the load (P-N03 B3)
            CHECK(value_of(h) == 1);
        });
        write_file(a, one);
        in_child("overlapping_identical_requests", [&] {
            sh::VerifiedOperator first = sh::VerifiedOperator::prepare(a.string(), sha1);
            sh::VerifiedOperator second = sh::VerifiedOperator::prepare(a.string(), sha1);
            void* h = first.load();
            CHECK(second.load() == h && second.reused() && value_of(h) == 1);
        });
        in_child("descriptor_reuse_refused_by_load", [&] {
            // Occupy the lowest free number L with a placeholder, dlopen the
            // value-2 object as /proc/self/fd/<L+1>, then free both: prepare's
            // source descriptor takes L and its memfd L+1, whose name glibc
            // resolves to the object already loaded under it.
            const int placeholder = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
            const int other = ::open(argv[2], O_RDONLY | O_CLOEXEC);
            CHECK(placeholder >= 0 && other == placeholder + 1);
            void* stale = ::dlopen(("/proc/self/fd/" + std::to_string(other)).c_str(), RTLD_NOW | RTLD_LOCAL);
            CHECK(stale != nullptr && value_of(stale) == 2);
            ::close(other);
            ::close(placeholder);
            sh::VerifiedOperator op = sh::VerifiedOperator::prepare(a.string(), sha1);
            const std::string e = refusal([&] { (void)op.load(); });
            CHECK(has(e, "the loaded operator object is not the verified memfd"));
            CHECK(has(refusal([&] { (void)sh::VerifiedOperator::prepare(a.string(), sha1); }),
                      "an earlier operator library load failed in this process"));
        });
        in_child("source_rewritten_after_prepare", [&] {
            sh::VerifiedOperator op = sh::VerifiedOperator::prepare(a.string(), sha1);
            overwrite(a, two);  // in place after the hash (P-N03 B2)
            CHECK(value_of(op.load()) == 1);
        });
        write_file(a, one);

        // Seals: F_SEAL_FUTURE_WRITE does not stand in for F_SEAL_WRITE.
        for (int with_write : {0, 1}) {
            const int m = ::memfd_create("seal-check", MFD_CLOEXEC | MFD_ALLOW_SEALING);
            const int seals = (with_write ? F_SEAL_WRITE : 0x0010 /* F_SEAL_FUTURE_WRITE */) | F_SEAL_GROW |
                              F_SEAL_SHRINK | F_SEAL_SEAL;
            CHECK(m >= 0 && ::fcntl(m, F_ADD_SEALS, seals) == 0);
            CHECK(sh::sealed_against_writes(m) == (with_write == 1));
            ::close(m);
        }
        // Search paths ($ORIGIN would expand under /proc/self/fd) and non-ELF.
        CHECK(refusal([&] { sh::require_no_origin_search_path(one, "fixture"); }).empty());
        CHECK(refusal([&] { sh::require_no_origin_search_path(read_file(argv[5]), "absolute"); }).empty());
        CHECK(has(refusal([&] { sh::require_no_origin_search_path(read_file(argv[3]), "runpath"); }),
                  "runpath DT_RUNPATH $ORIGIN/../lib uses $ORIGIN"));
        CHECK(has(refusal([&] { sh::require_no_origin_search_path(read_file(argv[4]), "rpath"); }),
                  "rpath DT_RPATH $ORIGIN uses $ORIGIN"));
        CHECK(has(refusal([&] { sh::require_no_origin_search_path("not an elf object at all, padded out to the header "
                                                                  "size of an ELF64 file", "text"); }),
                  "text is not a 64-bit little-endian ELF object"));
        {
            const std::string origin = read_file(argv[3]);
            const std::string elsewhere = dynamic_offset_elsewhere(origin);
            CHECK(has(refusal([&] { sh::require_no_origin_search_path(elsewhere, "split"); }),
                      "split has a dynamic section whose file and memory locations disagree"));
            CHECK(has(refusal([&] { sh::require_no_origin_search_path(dynamic_unterminated(origin), "short"); }),
                      "short has an unterminated dynamic section"));
            CHECK(has(refusal([&] { sh::require_no_origin_search_path(strsz_oversized(origin), "strsz"); }),
                      "strsz has a string table outside its file-backed segment"));
            CHECK(has(refusal([&] { sh::require_no_origin_search_path(two_dynamic(origin), "twice"); }),
                      "twice has more than one dynamic segment"));
            CHECK(has(refusal([&] { sh::require_no_origin_search_path(overlapping_dynamic(origin), "overlap"); }),
                      "overlap has overlapping or unordered loadable segments"));
        }
        {
            const fs::path origin = dir / "origin" / "libsaccade_scan_torchop.so";
            fs::create_directories(origin.parent_path());
            const std::string bytes = read_file(argv[3]);
            write_file(origin, bytes);
            CHECK(has(refusal([&] {
                          (void)sh::VerifiedOperator::prepare(origin.string(), sh::sha256_hex(bytes.data(), bytes.size()));
                      }),
                      "uses $ORIGIN"));
        }

        // Refusals before anything is mapped.
        CHECK(has(refusal([&] { (void)sh::VerifiedOperator::prepare(a.string(), sha2); }), "sha256 " + sha1 + " != " + sha2));
        CHECK(has(refusal([&] { (void)sh::VerifiedOperator::prepare(a.string(), sha1, 1); }),
                  "bytes " + std::to_string(one.size()) + " != 1"));
        fs::create_symlink(a, dir / "link.so");
        CHECK(has(refusal([&] { (void)sh::VerifiedOperator::prepare((dir / "link.so").string(), sha1); }),
                  "has a symbolic link in its path"));
        fs::create_directory_symlink(dir / "a", dir / "linkdir");
        CHECK(has(refusal([&] {
                      (void)sh::VerifiedOperator::prepare((dir / "linkdir" / "libsaccade_scan_torchop.so").string(), sha1);
                  }),
                  "has a symbolic link in its path"));
        CHECK(has(refusal([&] { (void)sh::VerifiedOperator::prepare((dir / "a").string(), sha1); }), "is not a regular file"));
        CHECK(::mkfifo((dir / "fifo").c_str(), 0600) == 0);
        CHECK(has(refusal([&] { (void)sh::VerifiedOperator::prepare((dir / "fifo").string(), sha1); }), "is not a regular file"));
        CHECK(has(refusal([&] { (void)sh::VerifiedOperator::prepare("a/libsaccade_scan_torchop.so", sha1); }),
                  "is not an absolute path"));
        CHECK(has(refusal([&] { (void)sh::VerifiedOperator::prepare((dir / "missing.so").string(), sha1); }),
                  "cannot open operator library"));
        // Nothing above loaded anything into this process.
        std::ifstream maps("/proc/self/maps");
        std::string line, all;
        while (std::getline(maps, line)) all += line + "\n";
        CHECK(!has(all, std::string("memfd:") + sh::kOperatorMemfdLabel));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "unexpected: %s\n", e.what());
        ++failures;
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
