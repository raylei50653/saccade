// CPU controls for N03 admission (#549 S2-2, P-N03 rev 3d):
// saccade_shipping/n03_admission.hpp. The ELF controls moved here from the
// runtime loader's test (rev 3c review rounds 1-4); each requires its reason.
//
// Usage: saccade_shipping_n03_admission_test <fixture value 1> <fixture value 2>
//            <origin runpath> <origin rpath> <absolute runpath>
//
//   * policy: $ORIGIN in DT_RUNPATH or DT_RPATH, ${ORIGIN}, and $ORIGIN under
//     each of the five expanded tags (DT_RPATH, DT_RUNPATH, DT_NEEDED,
//     DT_AUXILIARY, DT_FILTER) are refused; a non-ELF object is refused; an
//     absolute DT_RUNPATH and an object without search paths pass;
//   * layout: a PT_DYNAMIC whose file offset names an empty table while the
//     mapped one carries $ORIGIN (split), an unterminated table, an oversized
//     DT_STRSZ, two PT_DYNAMIC, overlapping PT_LOAD, a zero-size PT_LOAD over
//     the table and a segment ending past 2^64 are refused;
//   * pin pairing: the record is produced only for bytes whose size and
//     SHA-256 are the pin's and that pass the policy; another digest or size, a
//     pin with another policy, an unreadable pin, or a policy refusal of the
//     pinned bytes gives no record.
#include <elf.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>

#include "saccade_shipping/n03_admission.hpp"
#include "saccade_shipping/sha256.hpp"

namespace sh = saccade::shipping;

namespace {
int failures = 0;
int checks = 0;
#define CHECK(condition) do { ++checks; if (!(condition)) { ++failures; \
    std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition); } } while (0)

std::string read_file(const char* p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream o;
    o << in.rdbuf();
    return o.str();
}

std::string refusal(const std::function<void()>& f) {
    try {
        f();
    } catch (const sh::N03AdmissionError& e) {
        return e.what();
    } catch (const std::exception& e) {
        return std::string("not an N03AdmissionError: ") + e.what();
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

// Review round 4: the same, with the extended segment's p_memsz zero (the
// loader still maps its p_filesz bytes); and a segment ending past 2^64.
std::string zero_size_overlap(std::string elf) {
    elf = overlapping_dynamic(std::move(elf));
    int count = 0;
    Elf64_Phdr* ph = phdrs(elf, count);
    Elf64_Phdr* widest = nullptr;
    for (int i = 0; i < count; ++i) {
        if (ph[i].p_type == PT_LOAD && (widest == nullptr || ph[i].p_filesz > widest->p_filesz)) widest = &ph[i];
    }
    if (widest != nullptr) widest->p_memsz = 0;
    return elf;
}

std::string wrapping_segment(std::string elf) {
    int count = 0;
    Elf64_Phdr* ph = phdrs(elf, count);
    for (int i = 0; i < count; ++i) {
        if (ph[i].p_type == PT_LOAD) {
            ph[i].p_vaddr = UINT64_MAX - 4095 + (ph[i].p_offset % 4096);
            ph[i].p_memsz = 4096;
            break;
        }
    }
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

// The (consistent) dynamic table's first entry tagged `from`, retagged `to`.
std::string retag(std::string elf, std::int64_t from, std::int64_t to) {
    const Elf64_Phdr* d = dynamic_phdr(elf);
    for (std::uint64_t at = d->p_offset; at + sizeof(Elf64_Dyn) <= d->p_offset + d->p_filesz; at += sizeof(Elf64_Dyn)) {
        auto* e = reinterpret_cast<Elf64_Dyn*>(elf.data() + at);
        if (e->d_tag == from) {
            e->d_tag = to;
            break;
        }
    }
    return elf;
}

// Every occurrence of `from` replaced by `to` (same length).
std::string replace_bytes(std::string elf, const std::string& from, const std::string& to) {
    for (std::size_t at = elf.find(from); at != std::string::npos; at = elf.find(from, at + to.size())) {
        elf.replace(at, from.size(), to);
    }
    return elf;
}

std::string pin_json(const std::string& sha, std::size_t bytes, const char* policy = sh::kN03AdmissionPolicy) {
    return std::string("{\"sha256\": \"") + sha + "\", \"bytes\": " + std::to_string(bytes) +
           ", \"admission_policy\": \"" + policy + "\"}";
}

std::string pin_for(const std::string& bytes) { return pin_json(sh::sha256_hex(bytes.data(), bytes.size()), bytes.size()); }

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s <value 1> <value 2> <origin runpath> <origin rpath> <absolute runpath>\n",
                     argv[0]);
        return 2;
    }
    const std::string one = read_file(argv[1]), two = read_file(argv[2]);
    const std::string runpath = read_file(argv[3]), rpath = read_file(argv[4]), absolute = read_file(argv[5]);
    if (one.empty() || two.empty() || runpath.empty() || rpath.empty() || absolute.empty() || one == two) {
        std::fprintf(stderr, "the fixtures must exist and the two values differ\n");
        return 2;
    }
    const auto rule = [](const std::string& elf, const char* what) {
        return refusal([&] { sh::require_no_origin_search_path(elf, what); });
    };

    // Policy n03-no-origin-v1.
    CHECK(rule(one, "fixture").empty());
    CHECK(rule(absolute, "absolute").empty());
    CHECK(has(rule(runpath, "runpath"), "n03 admission: runpath DT_RUNPATH $ORIGIN/../lib uses $ORIGIN"));
    CHECK(has(rule(rpath, "rpath"), "rpath DT_RPATH $ORIGIN uses $ORIGIN"));
    CHECK(has(rule("not an elf object at all, padded out to the header size of an ELF64 file", "text"),
              "text is not a 64-bit little-endian ELF object"));
    {
        // "/opt/saccade-none/lib" -> "${ORIGIN}/saccade/lib" (same length) and
        // "$ORIGIN/saccade-x/lib", then the RUNPATH entry under each tag.
        const std::string braced = replace_bytes(absolute, "/opt/saccade-none/lib", "${ORIGIN}/saccade/lib");
        const std::string plain = replace_bytes(absolute, "/opt/saccade-none/lib", "$ORIGIN/saccade-x/lib");
        CHECK(braced != absolute && plain != absolute);
        CHECK(has(rule(braced, "braced"), "braced DT_RUNPATH ${ORIGIN}/saccade/lib uses $ORIGIN"));
        const std::pair<std::int64_t, const char*> tags[] = {{DT_RPATH, "DT_RPATH"}, {DT_RUNPATH, "DT_RUNPATH"},
                                                             {DT_NEEDED, "DT_NEEDED"}, {DT_AUXILIARY, "DT_AUXILIARY"},
                                                             {DT_FILTER, "DT_FILTER"}};
        for (const auto& [tag, name] : tags) {
            CHECK(has(rule(retag(plain, DT_RUNPATH, tag), "tag"),
                      std::string("tag ") + name + " $ORIGIN/saccade-x/lib uses $ORIGIN"));
            CHECK(has(rule(retag(braced, DT_RUNPATH, tag), "tag"),
                      std::string("tag ") + name + " ${ORIGIN}/saccade/lib uses $ORIGIN"));
        }
    }
    // Layout: what the checker reads must be what the loader maps.
    CHECK(has(rule(dynamic_offset_elsewhere(runpath), "split"),
              "split has a dynamic section whose file and memory locations disagree"));
    CHECK(has(rule(dynamic_unterminated(runpath), "short"), "short has an unterminated dynamic section"));
    CHECK(has(rule(strsz_oversized(runpath), "strsz"), "strsz has a string table outside its file-backed segment"));
    CHECK(has(rule(two_dynamic(runpath), "twice"), "twice has more than one dynamic segment"));
    CHECK(has(rule(overlapping_dynamic(runpath), "overlap"), "overlap has overlapping or unordered loadable segments"));
    CHECK(has(rule(zero_size_overlap(runpath), "zero"), "zero has a malformed loadable segment"));
    CHECK(has(rule(wrapping_segment(runpath), "wrap"), "wrap has a malformed loadable segment"));

    // Pin pairing: a record only for the pinned bytes that pass the policy.
    std::string record;
    CHECK(refusal([&] { record = sh::admit_n03_operator(one, "/x/op.so", pin_for(one), "c0ffee"); }).empty());
    const std::string sha1 = sh::sha256_hex(one.data(), one.size());
    CHECK(has(record, "\"result\": \"admitted\"") && has(record, "\"sha256\": \"" + sha1 + "\"") &&
          has(record, "\"path\": \"/x/op.so\"") && has(record, "\"checker_source_sha256\": \"c0ffee\"") &&
          has(record, "\"policy\": \"n03-no-origin-v1\""));
    const auto admit = [&](const std::string& bytes, const std::string& pin) {
        return refusal([&] { (void)sh::admit_n03_operator(bytes, "/x/op.so", pin, "c0ffee"); });
    };
    CHECK(has(admit(two, pin_json(sha1, two.size())), "operator library /x/op.so sha256 "));
    CHECK(has(admit(one, pin_json(sha1, one.size() + 1)), "operator library /x/op.so bytes "));
    CHECK(has(admit(one, pin_json(sha1, one.size(), "another-policy")), "admission_policy another-policy is not"));
    CHECK(has(admit(one, "{\"sha256\": 1}"), "the pin has no sha256 / bytes / admission_policy"));
    CHECK(has(admit(one, "not json"), "the pin is not valid JSON"));
    CHECK(has(admit(runpath, pin_for(runpath)), "DT_RUNPATH $ORIGIN/../lib uses $ORIGIN"));

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
