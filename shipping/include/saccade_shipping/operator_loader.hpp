// N03 operator library load (#549 S2-2, P-N03 rev 3c as revised by rev 3d;
// ADR 028 D5; docs/reference/model_bundle_s2_2_n03_549.md and
// docs/reference/model_bundle_s2_2_n03_offline_proposal_549.md). The bytes the
// dynamic loader maps are the verified bytes, the source is the runtime
// package's fixed path (owner decision D-N03-5), and those bytes are the one
// admitted N03 the runtime was built with (n03_operator_pin(), rev 3d):
//
//   1. the only legal source is <prefix>/share/saccade/build/
//      libsaccade_scan_torchop.so, where the kernel maps this process's
//      entrypoint at <prefix>/libexec/<name> (entrypoint_runtime_operator_path;
//      the same under direct exec and the launcher's explicit loader);
//   2. prepare: require the plan's SHA-256 (and Gate A's size when known) to
//      be the pin's; open that path once (openat2, no symlink or magic link in
//      any component), require a regular file of the pin's size, read it
//      whole and require the pin's SHA-256; copy the buffer into a memfd
//      labeled "saccade-n03:src-fd=<S>" (S = the source descriptor, which the
//      loader auditor checks), add F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK |
//      F_SEAL_SEAL and read them back, then read the memfd back whole and
//      require the plan's size and SHA-256 again;
//   3. load: dlopen("/proc/self/fd/<M>") while S and M are open; the complete
//      /proc/self/maps record holding the returned object's dynamic section
//      must be M (device and inode); then S is closed and M stays open for the
//      process lifetime.
//
// The runtime does not parse the ELF object: its search-path rule ($ORIGIN
// would expand under /proc/self/fd) is checked once, at admission
// (saccade_shipping/n03_admission.hpp), for the digest the pin names.
//
// One verified operator per process: a later request with the same SHA-256
// and source path reuses it (also one prepared while it loaded); any other
// request is refused. A load that fails after dlopen succeeded poisons the
// process: its memfd stays open and every later request is refused. There is no
// pathname-load fallback; a host without openat2, memfd_create, /proc or an
// executable memfd mapping is refused (D-N03-3). Every refusal is a
// DetectorLoadError (catchable); the loader auditor's own refusals are its
// _exit(127).
#pragma once

#include <cstdint>
#include <istream>
#include <optional>
#include <string>

namespace saccade::shipping {

// The runtime package's operator library, relative to the install prefix
// (the loader auditor's kOpLibraryRel).
inline constexpr const char kRuntimeOperatorRel[] = "share/saccade/build/libsaccade_scan_torchop.so";

// The memfd label; /proc/self/fd/<M> reads "/memfd:<label><S> (deleted)".
inline constexpr const char kOperatorMemfdLabel[] = "saccade-n03:src-fd=";

// One /proc/self/maps record: device, inode and the kernel's path text.
struct MappedObject {
    unsigned int major = 0, minor = 0;
    unsigned long inode = 0;
    std::string path;
};

// The complete record (any length, the last one with or without '\n') whose
// address range contains `addr`; nullopt when none does.
std::optional<MappedObject> mapped_object_at(std::istream& maps, std::uintptr_t addr);
std::optional<MappedObject> mapped_object_at(std::uintptr_t addr);  // /proc/self/maps

// <prefix>/share/saccade/build/libsaccade_scan_torchop.so for an entrypoint
// at <prefix>/libexec/<name>; DetectorLoadError for any other location.
std::string runtime_operator_path_for_entrypoint(const std::string& entrypoint);

// D-N03-5: the entrypoint is the main program's mapping (dl_iterate_phdr's
// first object), named by the kernel. Its path must be absolute, not a
// deleted entry, resolve to itself and be the mapped inode.
std::string entrypoint_runtime_operator_path();

// The post-check of step 3: the mapping holding `handle`'s dynamic section is
// the memfd `fd` (device, inode and the memfd's own /proc name).
bool mapped_from_memfd(void* handle, int fd);

// Step 2's seal check: F_SEAL_WRITE | GROW | SHRINK | SEAL are all set
// (F_SEAL_FUTURE_WRITE does not count for F_SEAL_WRITE).
bool sealed_against_writes(int fd);

// The one N03 the runtime accepts (P-N03 rev 3d): its SHA-256 and size.
struct N03OperatorPin {
    const char* sha256;
    std::int64_t bytes;
};

// Defined by exactly one object in a program. Every runtime target gets the
// object generated from shipping/n03_operator_pin.json (library
// saccade_shipping_n03_pin, linked by the loader's library); only CPU tests
// link a fixture definition ahead of it. It has no setter, CLI or
// environment input.
const N03OperatorPin& n03_operator_pin();

class VerifiedOperator {
public:
    // Step 2 (or a registry reuse). `source` must be absolute; a SHA-256 or a
    // known size other than the pin's is refused before anything is opened.
    static VerifiedOperator prepare(const std::string& source, const std::string& expected_sha256,
                                    std::int64_t expected_size = -1);
    // Step 3; returns the dlopen handle (the registry's when reused).
    void* load();

    VerifiedOperator(VerifiedOperator&& other) noexcept;
    VerifiedOperator& operator=(VerifiedOperator&&) = delete;
    VerifiedOperator(const VerifiedOperator&) = delete;
    VerifiedOperator& operator=(const VerifiedOperator&) = delete;
    ~VerifiedOperator();

    const std::string& sha256() const { return sha256_; }
    const std::string& source() const { return source_; }
    bool reused() const { return reused_; }
    // After load(): the kernel's record of the loaded object (the memfd).
    const MappedObject& mapped() const { return mapped_; }

private:
    VerifiedOperator() = default;
    std::string source_, sha256_;
    int source_fd_ = -1, memfd_ = -1;
    bool reused_ = false, loaded_ = false;
    MappedObject mapped_;
};

}  // namespace saccade::shipping
