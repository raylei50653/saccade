#pragma once

#include <cstdint>

namespace saccade {

/**
 * @brief ReidCropStore — narrow interface the online handover uses to re-extract
 *        dense recent-tail embeddings for borderline re-query.
 *
 * Implemented by PerceptionPipeline, which owns a GPU crop ring keyed by the
 * never-reused track_uid. The relinker (SemanticRelinkerCpp) holds a pointer to
 * this interface (set via set_crop_store) and calls it synchronously at a
 * flippable handover decision to densify the top-K candidates. Keeping this a
 * narrow interface (not a full pipeline pointer) keeps the relinker decoupled
 * from CUDA/TensorRT and lets tests stub it.
 */
class ReidCropStore {
public:
    virtual ~ReidCropStore() = default;

    /// Embedding dimensionality produced by requery_extract.
    virtual int embed_dim() const = 0;

    /// Max crops retained per track_uid (the ring depth).
    virtual int ring_depth() const = 0;

    /**
     * @brief Extract dense embeddings for each uid's stashed ring crops.
     *
     * @param uids        [n_uids] never-reused track uids to densify.
     * @param n_uids      Number of uids.
     * @param out_embeds  Host buffer, filled contiguously (row-major
     *                    [total_crops, embed_dim]); caller sizes it to at least
     *                    ring_depth() * n_uids * embed_dim() floats.
     * @param out_counts  [n_uids] host — number of crop rows written for uid i
     *                    (0 if the uid has no stashed crops).
     * @return Total crop rows written across all uids (0 if none).
     */
    virtual int requery_extract(const uint64_t* uids, int n_uids,
                                float* out_embeds, int* out_counts) = 0;

    /// Drop a uid's stashed crops (called when the dead archive entry expires).
    virtual void evict(uint64_t uid) = 0;
};

}  // namespace saccade
