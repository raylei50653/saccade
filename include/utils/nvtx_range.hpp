#pragma once

#ifdef SACCADE_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>

struct NvtxRange {
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
    NvtxRange(NvtxRange&&) = delete;
    NvtxRange& operator=(NvtxRange&&) = delete;
};

#else

struct NvtxRange {
    explicit NvtxRange(const char* /*name*/) {}
};

#endif
