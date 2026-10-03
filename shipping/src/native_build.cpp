// Build the native tracking objects from the resolved shipping config
// (#465 Phase B PR-4b). See saccade_shipping/native_build.hpp.
#include "saccade_shipping/native_build.hpp"

#include "tracking/gmc.hpp"
#include "tracking/pipeline.hpp"
#include "tracking/tracker_gpu.hpp"

namespace saccade::shipping {

std::unique_ptr<GPUByteTracker> build_tracker(const ResolvedShippingConfig& cfg,
                                              SequenceGeometry geometry) {
    const auto& k = cfg.native_params.tracker.constructor;
    auto tracker = std::make_unique<GPUByteTracker>(native_int(k.max_objects, "max_objects"),
                                                    native_int(k.embedding_dim, "embedding_dim"),
                                                    native_int(k.max_assoc, "max_assoc"));
    tracker->set_assoc_dump_path("");  // native_env.SACCADE_ASSOC_DUMP: unset
    // No ReID in shipping: an update that brings embeddings fails closed, so
    // the embedding-association branch (and reid_min_candidates) is unreachable.
    tracker->forbid_embeddings();
    apply_tracker_config(*tracker, cfg, geometry);
    require_readback("GPUByteTracker",
                     readback_mismatches(expected_tracker_snapshot(cfg, geometry),
                                         tracker->snapshot()));
    return tracker;
}

std::unique_ptr<GMC> build_gmc(const ResolvedShippingConfig& cfg) {
    const GmcSnapshot plan = planned_gmc_snapshot(cfg);
    auto gmc = std::make_unique<GMC>(plan.downscale, plan.max_corners, plan.quality_level,
                                     plan.min_distance, plan.min_inliers, plan.ransac_threshold);
    gmc->set_pcr_thresh(plan.pcr_thresh);
    gmc->set_profiling_enabled(plan.profiling_enabled);
    require_readback("GMC", readback_mismatches(expected_gmc_snapshot(cfg), gmc->snapshot()));
    return gmc;
}

std::unique_ptr<PerceptionPipeline> build_perception_pipeline(const ResolvedShippingConfig& cfg) {
    const PerceptionPipelineSnapshot plan = planned_pipeline_snapshot(cfg);
    if (plan.reid_ptr != 0 || plan.cropper_ptr != 0) {
        throw ConfigError("shipping: PerceptionPipeline takes no ReID extractor or cropper");
    }
    auto pipeline = std::make_unique<PerceptionPipeline>(nullptr, nullptr, plan.config);
    pipeline->set_postprocess_profiling_enabled(plan.postprocess_profiling_enabled);
    pipeline->set_filter_compaction_mode(plan.filter_compaction);
    pipeline->set_private_workload_stats_enabled(plan.private_workload_stats_enabled);
    require_readback("PerceptionPipeline",
                     readback_mismatches(expected_pipeline_snapshot(cfg), pipeline->snapshot()));
    return pipeline;
}

}  // namespace saccade::shipping
