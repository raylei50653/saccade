// N03 admission (#549 S2-2, P-N03 rev 3d). See saccade_shipping/n03_admission.hpp.
// The ELF rule below moved here unchanged from the runtime loader (rev 3c,
// review rounds 1-4); only its error type changed.
#include "saccade_shipping/n03_admission.hpp"

#include <elf.h>

#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include "saccade_shipping/sha256.hpp"
#include "saccade_shipping/strict_json.hpp"

namespace saccade::shipping {
namespace {

[[noreturn]] void fail(const std::string& message) { throw N03AdmissionError("n03 admission: " + message); }

}  // namespace

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
        if (h.p_type != PT_LOAD) continue;
        // Every load command counts (the loader maps each one, MAP_FIXED);
        // a zero-size one is refused rather than reasoned about.
        if (h.p_memsz == 0 || h.p_filesz > h.p_memsz || h.p_vaddr % kPage != h.p_offset % kPage) {
            bad("has a malformed loadable segment");
        }
        if (h.p_memsz > UINT64_MAX - h.p_vaddr) bad("has a malformed loadable segment");
        const std::uint64_t end = h.p_vaddr + h.p_memsz;
        if (end > UINT64_MAX - (kPage - 1)) bad("has a malformed loadable segment");
        const std::uint64_t lo = h.p_vaddr - h.p_vaddr % kPage;
        const std::uint64_t hi = (end + (kPage - 1)) / kPage * kPage;
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

std::string admit_n03_operator(const std::string& bytes, const std::string& path, const std::string& pin_json,
                               const std::string& checker_source_sha256) {
    JsonValue pin;
    try {
        pin = parse_strict_json(pin_json);
    } catch (const ConfigError& e) {
        fail(std::string("the pin is not valid JSON: ") + e.what());
    }
    const JsonValue* sha = pin.kind == JsonValue::Kind::Object ? pin.find("sha256") : nullptr;
    const JsonValue* size = pin.kind == JsonValue::Kind::Object ? pin.find("bytes") : nullptr;
    const JsonValue* policy = pin.kind == JsonValue::Kind::Object ? pin.find("admission_policy") : nullptr;
    if (!sha || sha->kind != JsonValue::Kind::String || sha->string.size() != 64 ||
        sha->string.find_first_not_of("0123456789abcdef") != std::string::npos || !size ||
        size->kind != JsonValue::Kind::Int || size->integer <= 0 || !policy ||
        policy->kind != JsonValue::Kind::String) {
        fail("the pin has no sha256 / bytes / admission_policy");
    }
    if (policy->string != kN03AdmissionPolicy) {
        fail("the pin's admission_policy " + policy->string + " is not " + kN03AdmissionPolicy);
    }
    const std::string what = "operator library " + path;
    const std::string got = sha256_hex(bytes.data(), bytes.size());
    if (static_cast<std::int64_t>(bytes.size()) != size->integer) {
        fail(what + " bytes " + std::to_string(bytes.size()) + " != the pin's " + std::to_string(size->integer));
    }
    if (got != sha->string) fail(what + " sha256 " + got + " != the pin's " + sha->string);
    require_no_origin_search_path(bytes, what);

    JsonValue artifact = JsonValue::make_object();
    artifact.set("path", JsonValue::make_string(path));
    artifact.set("sha256", JsonValue::make_string(got));
    artifact.set("bytes", JsonValue::make_int(static_cast<std::int64_t>(bytes.size())));
    JsonValue pinned = JsonValue::make_object();
    pinned.set("sha256", JsonValue::make_string(sha->string));
    pinned.set("bytes", JsonValue::make_int(size->integer));
    pinned.set("file_sha256", JsonValue::make_string(sha256_hex(pin_json.data(), pin_json.size())));
    JsonValue record = JsonValue::make_object();
    record.set("schema", JsonValue::make_string("saccade.n03_admission_record/v1"));
    record.set("reading", JsonValue::make_string(
        "Evidence that these bytes are the pin's and satisfy the admission policy; it does not approve any "
        "other bytes, and it does not vouch for how the object's dependencies resolve."));
    record.set("artifact", std::move(artifact));
    record.set("pin", std::move(pinned));
    record.set("policy", JsonValue::make_string(kN03AdmissionPolicy));
    record.set("checker_source_sha256", JsonValue::make_string(checker_source_sha256));
    record.set("target", JsonValue::make_string(
        "x86-64 ELF64 little-endian, 4096-byte pages, glibc ld.so dynamic string token expansion"));
    record.set("result", JsonValue::make_string("admitted"));
    return dump_python_json(record) + "\n";
}

}  // namespace saccade::shipping
