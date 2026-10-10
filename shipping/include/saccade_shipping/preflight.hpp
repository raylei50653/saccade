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
// Gate A decodes no frame, dlopens nothing, deserializes no engine and loads
// no head (Gate B, detector_host.hpp). It writes no identity: journal and
// report keep {"level": null}; binding the checked hashes into `identity` is a
// later slice of CC-536-01-02. The runtime re-reads each sequence's input when
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
    std::string config, lineage, attestation;  // attestation may be empty
    std::string model_root = ".";
    std::vector<std::string> sequences;        // sequence directories, argv order
    int max_frames = 0;                        // developer build only; <= 0 = every frame
    bool serial_requested = false;             // developer build only (--schedule serial)
};

// What Gate A checked, for the runtime to build from: the same config and
// detector plan, so Gate B loads exactly the bindings whose hashes passed.
struct PreflightResult {
    ResolvedShippingConfig config;
    Schedule schedule = Schedule::Serial;
    DetectorPlan detector;
};

PreflightResult run_preflight(const PreflightInputs& in, const RunCompletion& completion);

}  // namespace saccade::shipping
