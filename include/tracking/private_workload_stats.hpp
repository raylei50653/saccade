#pragma once

#include "saccade/common.hpp"
#include <cuda_runtime.h>

namespace saccade {

// Default-off private-continuation workload counter (SACCADE_ASSOC_STATS=1).
// Lives outside tracker_gpu.{hpp,cu}: those two files are strict frozen
// inputs of the closed H0/GCTM packets, so measurement-only instrumentation
// must not touch them. Layout of `stats` (unsigned long long[8]):
//   [0] invocations, [1] sum candidate_count, [2] sum added,
//   [3] frames with added > 0, [4] sum num_private_priors.
void SACCADE_TRACKING_API accumulate_private_workload_cuda(
    const int* added_count,
    const int* candidate_count,
    int num_private_priors,
    unsigned long long* stats,
    cudaStream_t stream);

}  // namespace saccade
