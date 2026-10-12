// N03 admission (#549 S2-2, P-N03 rev 3d;
// docs/reference/model_bundle_s2_2_n03_offline_proposal_549.md). The offline
// check that a candidate N03 operator library may become the runtime's pin
// (shipping/n03_operator_pin.json). No runtime target links it: the runtime
// accepts only the pinned digest (operator_loader.hpp) and does not parse ELF.
//
// Its parser decides only whether a reviewer is shown a passing record; a
// digest is accepted by review and by the pin, not by the record. A parser
// that accepts a hostile ELF therefore no longer loads anything.
#pragma once

#include <stdexcept>
#include <string>

namespace saccade::shipping {

class N03AdmissionError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline constexpr const char kN03AdmissionPolicy[] = "n03-no-origin-v1";

// Policy n03-no-origin-v1: a 64-bit little-endian ELF whose dynamic strings
// that the loader expands (DT_RPATH, DT_RUNPATH, DT_NEEDED, DT_AUXILIARY,
// DT_FILTER) do not use $ORIGIN or ${ORIGIN} (a load from /proc/self/fd/<M>
// would expand it there), read where the loader maps them: nonempty,
// ascending, page-disjoint PT_LOAD segments with congruent offsets (x86-64,
// 4096-byte pages), the dynamic table through its address with an agreeing
// file offset and a terminator, the string table within its segment.
// N03AdmissionError naming `what` otherwise (also for a non-ELF object or one
// these properties do not hold for). It is not a full ELF validator; other ELF
// acceptance checks (version, machine, type, ...) remain the loader's.
void require_no_origin_search_path(const std::string& elf, const std::string& what);

// The admission of one owned buffer: its size and SHA-256 must be the pin's
// (`pin_json`, the text of shipping/n03_operator_pin.json), then the policy
// holds for the same buffer. Returns the admission record (JSON text):
// artifact path / SHA-256 / size, the pin and its file SHA-256, the policy,
// `checker_source_sha256`, the target loader / ABI and the result.
// N03AdmissionError, and no record, otherwise.
std::string admit_n03_operator(const std::string& bytes, const std::string& path, const std::string& pin_json,
                               const std::string& checker_source_sha256);

}  // namespace saccade::shipping
