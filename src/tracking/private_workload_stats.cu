#include "tracking/private_workload_stats.hpp"

namespace saccade {

// Diagnostic-only. Launched only while private workload stats are enabled.
__global__ void accumulate_private_workload_kernel(
    const int* added_count,
    const int* candidate_count,
    int num_private_priors,
    unsigned long long* stats)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int cands = candidate_count ? *candidate_count : 0;
    int added = added_count ? *added_count : 0;
    if (cands < 0) cands = 0;
    if (added < 0) added = 0;
    atomicAdd(&stats[0], 1ull);
    atomicAdd(&stats[1], (unsigned long long)cands);
    atomicAdd(&stats[2], (unsigned long long)added);
    if (added > 0) atomicAdd(&stats[3], 1ull);
    atomicAdd(&stats[4], (unsigned long long)max(num_private_priors, 0));
}

void accumulate_private_workload_cuda(
    const int* added_count,
    const int* candidate_count,
    int num_private_priors,
    unsigned long long* stats,
    cudaStream_t stream)
{
    if (stats == nullptr) return;
    accumulate_private_workload_kernel<<<1, 1, 0, stream>>>(
        added_count, candidate_count, num_private_priors, stats);
    checkCuda(cudaGetLastError());
}

}  // namespace saccade
