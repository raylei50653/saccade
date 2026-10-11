// Gate A of saccade_track (#536 CC-536-01-02, N-T2,
// docs/architecture/ship_export_contracts_536.md): every check that needs no
// GPU, after the run has taken `<out>` (run_completion.hpp) and before the
// first CUDA API call. CUDA-free: the library (saccade_shipping_preflight)
// links no CUDA, so nothing here can initialize a device.
//
// In order, cheapest first; the first failure throws PreflightError, whose
// message starts "preflight: ", and the entrypoint exits 2 with the journal
// `failed` (run_completion.hpp):
//   1. config: the strict loader, then every CUDA-free plan the runtime
//      builds from it -- ingest, sequence output (it must write MOT output),
//      post-detector, pipeline, schedule (select_schedule, with the developer
//      serial override) -- and the detector plan from the lineage and the
//      attestation (N-R4: lineage vs config, attestation bound to the
//      lineage's sha256);
//   2. sequences: seqinfo.ini and the img1 listing of every sequence, as the
//      runtime reads them (read_sequence_input, the same max_frames);
//   3. outputs: RunCompletion::check_writable (`<out>`, the --report
//      directory, each --trace/<sequence>/);
//   4. artifacts: the operator library, the head artifact and the backbone
//      engine are regular files under the model root and their sha256 is the
//      detector plan's (the detector load checks them again, Gate B).
//
// Manifest mode (#549 S2-1, --model-bundle DIR; model_bundle.hpp,
// docs/architecture/model_bundle_contract_549.md): there is no config,
// lineage, attestation or model root argument. Two roots are each resolved
// once: the bundle directory (every member carried_by model_bundle) and the
// runtime package's share/saccade/ (the operator library and the allowlist).
// Before step 1:
//   0a. VL0: <bundle>/model_bundle.json, opened beneath the bundle root, is
//       the exact saccade.model_bundle/v1 schema and R-01..R-08;
//   0b. VL2 input: <runtime root>/trusted_model_bundles.json has the sha256
//       the entrypoint was built with (`allowlist_sha256`), the
//       saccade.trusted_model_bundles/v1 schema and R-09, and the manifest's
//       detector contract; its entry for the manifest's sha256 (if any) is
//       recorded;
// then steps 1-4 as above, except that every file is a manifest member: each
// is opened beneath its root with openat2 (no symlink, no escape; a kernel
// without openat2 is refused), its size is checked before it is hashed, and
// its sha256 must be the manifest's -- the config, lineage and attestation
// too (the attestation is required). The lineage / attestation must bind the
// same three files (path and sha256) as the manifest's pairing. Gate A then
// gives the detector plan the resolved bindings of those three files
// (DetectorPlan::resolved): the only paths the detector load reads.
//
// Gate A decodes no frame, dlopens nothing, deserializes no engine and loads
// no head (Gate B, detector_host.hpp). Through RunCompletion, Gate A records
// the bindings and promotes the level only after every check passes:
// checksum_matched, or in manifest mode expected_source_verified when the
// allowlist has an `approved` entry for the manifest's exact sha256
// (`example`, `revoked` and an unlisted manifest stay checksum_matched;
// legacy mode never exceeds checksum_matched). Metadata observations hash the
// same buffers the existing parsers consume. The final Gate A identity is
// immutable, including on rejection; it establishes neither publisher
// authentication nor successful loading (Gate B writes load_verification).
// Last, the level is compared with --require-identity (`required`, recorded
// in the identity): a lower level exits 2, with the identity kept as it is.
// The runtime re-reads each sequence's input when
// it runs it and refuses what it refuses here, so a file changed after Gate A
// still fails closed, at that sequence.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "saccade_shipping/detector_plan.hpp"
#include "saccade_shipping/resolved_config.hpp"
#include "saccade_shipping/run_completion.hpp"
#include "saccade_shipping/schedule_plan.hpp"

namespace saccade::shipping {

class PreflightError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct PreflightInputs {
    std::string config, lineage, attestation;  // legacy mode; attestation may be empty
    std::string model_root = ".";              // legacy mode only
    std::vector<std::string> sequences;        // sequence directories, argv order
    int max_frames = 0;                        // developer build only; <= 0 = every frame
    bool serial_requested = false;             // developer build only (--schedule serial)
    IdentityLevel required = IdentityLevel::None;  // --require-identity
    // Manifest mode (non-empty model_bundle): the bundle directory, the
    // runtime package's share/saccade/ and the allowlist sha256 the
    // entrypoint was built with (track_driver.hpp; never a caller option).
    std::string model_bundle{}, runtime_root{}, allowlist_sha256{};
};

// What Gate A checked, for the runtime to build from: the same config and
// detector plan, so Gate B loads exactly the bindings whose hashes passed.
struct PreflightResult {
    ResolvedShippingConfig config;
    Schedule schedule = Schedule::Serial;
    DetectorPlan detector;
};

PreflightResult run_preflight(const PreflightInputs& in, RunCompletion& completion);

}  // namespace saccade::shipping
