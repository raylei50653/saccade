// N03 operator library load (#549 S2-2). See saccade_shipping/operator_loader.hpp.
#include "saccade_shipping/operator_loader.hpp"

#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <limits.h>
#include <link.h>
#include <linux/openat2.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "saccade_shipping/detector_plan.hpp"
#include "saccade_shipping/sha256.hpp"

#ifndef MFD_NOEXEC_SEAL
#define MFD_NOEXEC_SEAL 0x0008U
#endif
#ifndef F_SEAL_FUTURE_WRITE
#define F_SEAL_FUTURE_WRITE 0x0010
#endif

namespace saccade::shipping {
namespace {

[[noreturn]] void fail(const std::string& message) {
    throw DetectorLoadError("shipping detector load: " + message);
}

constexpr int kSeals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
constexpr const char kDeleted[] = " (deleted)";

bool ends_with(const std::string& s, const char* tail) {
    const std::size_t n = std::strlen(tail);
    return s.size() >= n && s.compare(s.size() - n, n, tail) == 0;
}

// The whole of `fd` from offset 0 (pread; the descriptor's offset is not used).
std::string read_whole(int fd, std::uint64_t size, const std::string& what) {
    if (size > std::string().max_size()) fail(what + " size is not representable");
    std::string bytes(static_cast<std::size_t>(size), '\0');
    std::size_t done = 0;
    while (done < bytes.size()) {
        const ssize_t n = ::pread(fd, bytes.data() + done, bytes.size() - done, static_cast<off_t>(done));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) fail("cannot read " + what + ": " + std::strerror(errno));
        if (n == 0) fail(what + " changed size while reading");
        done += static_cast<std::size_t>(n);
    }
    char extra;
    ssize_t n;
    do { n = ::pread(fd, &extra, 1, static_cast<off_t>(done)); } while (n < 0 && errno == EINTR);
    if (n < 0) fail("cannot read " + what + ": " + std::strerror(errno));
    if (n != 0) fail(what + " changed size while reading");
    return bytes;
}

// The source: no symlink and no magic link in any component (D-N03-5 names a
// physical path). A kernel without openat2 is refused, with no weaker open.
int open_source(const std::string& path) {
    open_how how{};
    how.flags = O_RDONLY | O_NONBLOCK | O_NOCTTY | O_CLOEXEC;
    how.resolve = RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS;
    long fd = -1;
    for (int attempt = 0; attempt < 3; ++attempt) {
        fd = ::syscall(SYS_openat2, AT_FDCWD, path.c_str(), &how, sizeof how);
        if (fd >= 0 || errno != EAGAIN) break;  // EAGAIN: a concurrent rename; retry
    }
    if (fd < 0) {
        const int e = errno;
        if (e == ENOSYS || e == E2BIG || e == EPERM) {
            fail("secure open (openat2 RESOLVE_NO_SYMLINKS) of the operator library " + path +
                 " is not supported here: " + std::strerror(e) + "; there is no weaker fallback");
        }
        if (e == ELOOP) fail("operator library " + path + " has a symbolic link in its path");
        fail("cannot open operator library " + path + ": " + std::strerror(e));
    }
    return static_cast<int>(fd);
}

int create_memfd(const std::string& label) {
    // MFD_NOEXEC_SEAL stops execve, not mmap(PROT_EXEC) (P-N03 r2 C20), and is
    // what a vm.memfd_noexec=2 host forces; a kernel before 6.3 has no such
    // flag (EINVAL), where every memfd is created without it.
    int fd = ::memfd_create(label.c_str(), MFD_CLOEXEC | MFD_ALLOW_SEALING | MFD_NOEXEC_SEAL);
    if (fd < 0 && errno == EINVAL) fd = ::memfd_create(label.c_str(), MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) fail(std::string("memfd_create for the operator library failed: ") + std::strerror(errno));
    return fd;
}

void write_all(int fd, const std::string& bytes) {
    std::size_t done = 0;
    while (done < bytes.size()) {
        const ssize_t n = ::write(fd, bytes.data() + done, bytes.size() - done);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) fail(std::string("cannot write the operator memfd: ") + std::strerror(errno));
        done += static_cast<std::size_t>(n);
    }
}

std::string fd_link(int fd) {
    char link[PATH_MAX];
    const std::string proc = "/proc/self/fd/" + std::to_string(fd);
    const ssize_t n = ::readlink(proc.c_str(), link, sizeof link - 1);
    if (n <= 0) return "";
    return std::string(link, static_cast<std::size_t>(n));
}

void close_fd(int& fd) {
    if (fd >= 0) ::close(fd);
    fd = -1;
}

// The process-wide verified operator (one per process).
struct Registry {
    std::mutex mutex;
    std::string sha256, source;
    void* handle = nullptr;
    int memfd = -1;
    MappedObject mapped;
    // A load that failed after dlopen succeeded: its object may be mapped and
    // its memfd is kept open (the number is never reused); every later
    // request is refused.
    bool poisoned = false;
    int poisoned_memfd = -1;
};

Registry& registry() {
    static Registry r;
    return r;
}

int first_object(struct dl_phdr_info* info, std::size_t, void* data) {
    for (int i = 0; i < info->dlpi_phnum; ++i) {
        if (info->dlpi_phdr[i].p_type == PT_LOAD) {
            *static_cast<std::uintptr_t*>(data) = info->dlpi_addr + info->dlpi_phdr[i].p_vaddr;
            break;
        }
    }
    return 1;  // the main program only
}

}  // namespace

std::optional<MappedObject> mapped_object_at(std::istream& maps, std::uintptr_t addr) {
    std::string line;
    while (std::getline(maps, line)) {
        unsigned long lo = 0, hi = 0, off = 0, ino = 0;
        unsigned int maj = 0, mnr = 0;
        char perm[8];
        int pos = 0;
        if (std::sscanf(line.c_str(), "%lx-%lx %7s %lx %x:%x %lu %n", &lo, &hi, perm, &off, &maj, &mnr, &ino, &pos) <
                7 ||
            addr < lo || addr >= hi) {
            continue;
        }
        MappedObject m;
        m.major = maj;
        m.minor = mnr;
        m.inode = ino;
        std::size_t p = static_cast<std::size_t>(pos);
        while (p < line.size() && line[p] == ' ') ++p;
        m.path = line.substr(p);
        return m;
    }
    return std::nullopt;
}

std::optional<MappedObject> mapped_object_at(std::uintptr_t addr) {
    std::ifstream maps("/proc/self/maps");
    if (!maps) return std::nullopt;
    return mapped_object_at(maps, addr);
}

std::string runtime_operator_path_for_entrypoint(const std::string& entrypoint) {
    const std::size_t name = entrypoint.rfind('/');
    if (entrypoint.empty() || entrypoint[0] != '/' || name == std::string::npos || name + 1 >= entrypoint.size()) {
        fail("the entrypoint " + entrypoint + " is not an absolute file path");
    }
    const std::string dir = entrypoint.substr(0, name);  // <prefix>/libexec
    const std::size_t libexec = dir.rfind('/');
    if (libexec == std::string::npos || dir.compare(libexec + 1, std::string::npos, "libexec") != 0) {
        fail("the entrypoint " + entrypoint +
             " is not installed as <prefix>/libexec/<name>; the runtime package's operator library has no "
             "location (D-N03-5)");
    }
    const std::string prefix = dir.substr(0, libexec);
    return (prefix.empty() ? std::string() : prefix) + "/" + kRuntimeOperatorRel;
}

std::string entrypoint_runtime_operator_path() {
    std::uintptr_t addr = 0;
    ::dl_iterate_phdr(first_object, &addr);
    if (addr == 0) fail("cannot locate the entrypoint's mapping");
    const std::optional<MappedObject> m = mapped_object_at(addr);
    if (!m || m->path.empty() || m->path[0] != '/') fail("the entrypoint's mapping has no file path");
    if (ends_with(m->path, kDeleted)) fail("the entrypoint " + m->path + " was replaced while running");
    char real[PATH_MAX];
    struct stat st {};
    if (::realpath(m->path.c_str(), real) == nullptr || m->path != real || ::stat(real, &st) != 0 ||
        static_cast<unsigned long>(st.st_ino) != m->inode) {
        fail("the entrypoint's mapped path " + m->path + " does not name the mapped file");
    }
    return runtime_operator_path_for_entrypoint(m->path);
}

std::optional<MappedObject> memfd_mapping(void* handle, int fd) {
    struct link_map* lm = nullptr;
    struct stat st {};
    if (handle == nullptr || ::dlinfo(handle, RTLD_DI_LINKMAP, &lm) != 0 || lm == nullptr || lm->l_ld == nullptr ||
        ::fstat(fd, &st) != 0) {
        return std::nullopt;
    }
    std::optional<MappedObject> m = mapped_object_at(reinterpret_cast<std::uintptr_t>(lm->l_ld));
    if (!m || m->major != major(st.st_dev) || m->minor != minor(st.st_dev) ||
        m->inode != static_cast<unsigned long>(st.st_ino) || m->path != fd_link(fd)) {
        return std::nullopt;
    }
    return m;
}

bool mapped_from_memfd(void* handle, int fd) { return memfd_mapping(handle, fd).has_value(); }

bool sealed_against_writes(int fd) {
    const int seals = ::fcntl(fd, F_GET_SEALS);
    return seals >= 0 && (seals & kSeals) == kSeals;
}

namespace {

// The file offset of [vaddr, vaddr + size) when one PT_LOAD maps all of it
// from the file (within p_filesz), overflow-safe; nullopt otherwise.
std::optional<std::uint64_t> file_range(const std::vector<Elf64_Phdr>& ph, std::uint64_t file_size,
                                        std::uint64_t vaddr, std::uint64_t size) {
    for (const Elf64_Phdr& h : ph) {
        if (h.p_type != PT_LOAD || vaddr < h.p_vaddr) continue;
        const std::uint64_t rel = vaddr - h.p_vaddr;
        if (rel > h.p_filesz || size > h.p_filesz - rel) continue;
        if (h.p_offset > file_size || h.p_filesz > file_size - h.p_offset) return std::nullopt;
        return h.p_offset + rel;
    }
    return std::nullopt;
}

}  // namespace

void require_no_origin_search_path(const std::string& elf, const std::string& what) {
    const auto bad = [&](const std::string& why) { fail(what + " " + why); };
    const std::uint64_t n = elf.size();
    if (n < sizeof(Elf64_Ehdr) || std::memcmp(elf.data(), ELFMAG, SELFMAG) != 0 ||
        static_cast<unsigned char>(elf[EI_CLASS]) != ELFCLASS64 || static_cast<unsigned char>(elf[EI_DATA]) != ELFDATA2LSB) {
        bad("is not a 64-bit little-endian ELF object");
    }
    Elf64_Ehdr eh;
    std::memcpy(&eh, elf.data(), sizeof eh);
    if (eh.e_phentsize != sizeof(Elf64_Phdr) || eh.e_phoff > n || eh.e_phnum > (n - eh.e_phoff) / sizeof(Elf64_Phdr)) {
        bad("has malformed program headers");
    }
    std::vector<Elf64_Phdr> ph(eh.e_phnum);
    if (eh.e_phnum != 0) std::memcpy(ph.data(), elf.data() + eh.e_phoff, eh.e_phnum * sizeof(Elf64_Phdr));
    // The loader maps every PT_LOAD at its page-rounded range, later ones
    // MAP_FIXED over earlier ones. With ascending, page-disjoint segments whose
    // file offset is congruent to their address, each mapped byte comes from
    // exactly one segment, so reading the file through that segment is reading
    // what the loader maps. Anything else is refused.
    constexpr std::uint64_t kPage = 4096;
    std::uint64_t mapped_end = 0;  // page-rounded end of the previous segment
    bool first = true;
    for (const Elf64_Phdr& h : ph) {
        if (h.p_type != PT_LOAD || h.p_memsz == 0) continue;
        if (h.p_filesz > h.p_memsz || h.p_vaddr % kPage != h.p_offset % kPage ||
            h.p_memsz > UINT64_MAX - h.p_vaddr - (kPage - 1)) {
            bad("has a malformed loadable segment");
        }
        const std::uint64_t lo = h.p_vaddr - h.p_vaddr % kPage;
        const std::uint64_t hi = (h.p_vaddr + h.p_memsz + kPage - 1) / kPage * kPage;
        if (!first && lo < mapped_end) bad("has overlapping or unordered loadable segments");
        mapped_end = hi;
        first = false;
    }
    const Elf64_Phdr* dyn = nullptr;
    for (const Elf64_Phdr& h : ph) {
        if (h.p_type != PT_DYNAMIC) continue;
        if (dyn != nullptr) bad("has more than one dynamic segment");
        dyn = &h;
    }
    if (dyn == nullptr) bad("has no dynamic section");
    // The loader reads the dynamic table where it is mapped (p_vaddr). That
    // table, read from the file where a PT_LOAD maps it, is what is checked,
    // and its file location must be the one PT_DYNAMIC states.
    const std::optional<std::uint64_t> dyn_off = file_range(ph, n, dyn->p_vaddr, dyn->p_filesz);
    if (!dyn_off || *dyn_off != dyn->p_offset || dyn->p_filesz % sizeof(Elf64_Dyn) != 0) {
        bad("has a dynamic section whose file and memory locations disagree");
    }
    std::uint64_t strtab = 0, strsz = 0;
    bool have_strtab = false, have_strsz = false, terminated = false;
    std::vector<std::pair<std::int64_t, std::uint64_t>> paths;  // (tag, string offset)
    for (std::uint64_t at = *dyn_off; at < *dyn_off + dyn->p_filesz; at += sizeof(Elf64_Dyn)) {
        Elf64_Dyn d;
        std::memcpy(&d, elf.data() + at, sizeof d);
        if (d.d_tag == DT_NULL) {
            terminated = true;
            break;
        }
        if (d.d_tag == DT_STRTAB) strtab = d.d_un.d_ptr, have_strtab = true;
        if (d.d_tag == DT_STRSZ) strsz = d.d_un.d_val, have_strsz = true;
        // Every tag whose string the loader expands dynamic string tokens in
        // (last DT_STRTAB / DT_STRSZ win, as in the loader).
        if (d.d_tag == DT_RPATH || d.d_tag == DT_RUNPATH || d.d_tag == DT_NEEDED || d.d_tag == DT_AUXILIARY ||
            d.d_tag == DT_FILTER) {
            paths.emplace_back(d.d_tag, d.d_un.d_val);
        }
    }
    if (!terminated) bad("has an unterminated dynamic section");
    if (paths.empty()) return;
    if (!have_strtab || !have_strsz) bad("has dynamic strings but no string table");
    const std::optional<std::uint64_t> str_off = file_range(ph, n, strtab, strsz);
    if (!str_off) bad("has a string table outside its file-backed segment");
    for (const auto& [tag, off] : paths) {
        if (off >= strsz) bad("has a malformed dynamic string");
        const char* begin = elf.data() + *str_off + off;
        const void* stop = std::memchr(begin, '\0', strsz - off);
        if (stop == nullptr) bad("has an unterminated dynamic string");
        const std::string entry(begin, static_cast<const char*>(stop));
        if (entry.find("$ORIGIN") != std::string::npos || entry.find("${ORIGIN}") != std::string::npos) {
            const char* name = tag == DT_RPATH ? "DT_RPATH" : tag == DT_RUNPATH ? "DT_RUNPATH"
                             : tag == DT_NEEDED ? "DT_NEEDED" : tag == DT_AUXILIARY ? "DT_AUXILIARY" : "DT_FILTER";
            bad(std::string(name) + " " + entry + " uses $ORIGIN, which a load from its memfd would expand under /proc/self/fd");
        }
    }
}

VerifiedOperator VerifiedOperator::prepare(const std::string& source, const std::string& expected_sha256,
                                           std::int64_t expected_size) {
    const std::string what = "operator library " + source;
    if (source.empty() || source[0] != '/') fail(what + " is not an absolute path");
    VerifiedOperator op;
    op.source_ = source;
    {
        Registry& r = registry();
        std::lock_guard<std::mutex> lock(r.mutex);
        if (r.poisoned) fail("an earlier operator library load failed in this process; " + what + " is refused");
        if (r.handle != nullptr) {
            if (r.sha256 != expected_sha256) {
                fail("an operator library with sha256 " + r.sha256 + " is already loaded; " + what + " expects " +
                     expected_sha256);
            }
            if (r.source != source) fail("the operator library is already loaded from " + r.source + ", not " + source);
            op.sha256_ = r.sha256;
            op.reused_ = true;
            op.mapped_ = r.mapped;
            return op;
        }
    }
    op.source_fd_ = open_source(source);
    struct stat st {};
    if (::fstat(op.source_fd_, &st) != 0) fail("cannot stat " + what + ": " + std::strerror(errno));
    if (!S_ISREG(st.st_mode)) fail(what + " is not a regular file");
    if (st.st_size < 0) fail(what + " size is not representable");
    if (expected_size >= 0 && st.st_size != expected_size) {
        fail(what + " bytes " + std::to_string(st.st_size) + " != " + std::to_string(expected_size));
    }
    const std::string bytes = read_whole(op.source_fd_, static_cast<std::uint64_t>(st.st_size), what);
    const std::string got = sha256_hex(bytes.data(), bytes.size());
    if (got != expected_sha256) fail(what + " sha256 " + got + " != " + expected_sha256);
    // Its search paths keep their meaning when loaded from /proc/self/fd/<M>.
    require_no_origin_search_path(bytes, what);

    op.memfd_ = create_memfd(kOperatorMemfdLabel + std::to_string(op.source_fd_));
    write_all(op.memfd_, bytes);
    if (::fcntl(op.memfd_, F_ADD_SEALS, kSeals) != 0) {
        fail(std::string("cannot seal the operator memfd: ") + std::strerror(errno));
    }
    // F_SEAL_FUTURE_WRITE is not a substitute: F_SEAL_WRITE itself must be set.
    if (!sealed_against_writes(op.memfd_)) fail("the operator memfd does not carry its four seals");
    struct stat ms {};
    if (::fstat(op.memfd_, &ms) != 0 || ms.st_size != st.st_size) fail("the operator memfd size is not the source's");
    const std::string back = read_whole(op.memfd_, static_cast<std::uint64_t>(ms.st_size), "the operator memfd");
    if (sha256_hex(back.data(), back.size()) != expected_sha256 || back != bytes) {
        fail("the sealed operator memfd does not hold the verified bytes");
    }
    op.sha256_ = got;
    return op;
}

void* VerifiedOperator::load() {
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    if (loaded_) fail("the operator library was already loaded by this request");
    if (r.poisoned) fail("an earlier operator library load failed in this process");
    if (r.handle != nullptr) {
        // A reuse, or a request prepared while an identical one loaded.
        if (r.sha256 != sha256_ || r.source != source_) fail("another operator library was loaded since this request");
        mapped_ = r.mapped;
        reused_ = true;
        loaded_ = true;
        return r.handle;
    }
    if (reused_) fail("the operator registry changed");
    const std::string name = "/proc/self/fd/" + std::to_string(memfd_);
    void* handle = ::dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char* e = ::dlerror();
        fail("dlopen of the verified operator memfd (" + source_ + "): " + (e ? e : "unknown error"));
    }
    // From here an object may be mapped: any failure poisons the registry and
    // keeps the memfd open, so its number is never reused by a later load.
    std::optional<MappedObject> checked;
    MappedObject request_copy;
    std::string sha, source;
    try {
        checked = memfd_mapping(handle, memfd_);
        if (checked) {
            request_copy = *checked;
            sha = sha256_;
            source = source_;
        }
    } catch (...) {
        checked.reset();
    }
    if (!checked) {
        r.poisoned = true;
        r.poisoned_memfd = memfd_;
        memfd_ = -1;
        fail("the loaded operator object is not the verified memfd (" + source_ + ")");
    }
    // Publication: nonthrowing transfers, the handle last.
    r.sha256.swap(sha);
    r.source.swap(source);
    r.mapped = std::move(*checked);
    r.memfd = memfd_;  // kept open: its number is never reused for another operator
    r.handle = handle;
    memfd_ = -1;
    mapped_ = std::move(request_copy);
    close_fd(source_fd_);
    loaded_ = true;
    return handle;
}

VerifiedOperator::VerifiedOperator(VerifiedOperator&& other) noexcept
    : source_(std::move(other.source_)),
      sha256_(std::move(other.sha256_)),
      source_fd_(other.source_fd_),
      memfd_(other.memfd_),
      reused_(other.reused_),
      loaded_(other.loaded_),
      mapped_(std::move(other.mapped_)) {
    other.source_fd_ = -1;
    other.memfd_ = -1;
}

VerifiedOperator::~VerifiedOperator() {
    close_fd(source_fd_);
    close_fd(memfd_);
}

}  // namespace saccade::shipping
