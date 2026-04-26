#include "saccade/common.hpp"
#include <vector>

namespace saccade {

/**
 * @brief GPU 追蹤結果結構體 (Zero-Sync)
 */
struct TrackResult {
    float x1, y1, x2, y2;
    int obj_id;
    float score;
    int class_id;
};

struct TrackStateSnapshot {
    int obj_id;
    int class_id;
    int age;
    float score;
    std::vector<float> state;      // [cx, cy, a, h, vx, vy, va, vh]
    std::vector<float> covariance; // row-major 8x8 covariance
};

struct TrackCandidateSnapshot {
    int obj_id;
    int class_id;
    int age;
    int hit_streak;
    int required_confirm_streak;
    float score;
    float x1, y1, x2, y2;
};

/**
 * @brief ITracker 接口
 */
class SACCADE_TRACKING_API ITracker {
public:
    virtual ~ITracker() = default;
    virtual std::vector<TrackResult> update(
        float* boxes_ptr, 
        float* scores_ptr, 
        int* classes_ptr, 
        int num_dets,
        cudaStream_t stream,
        float* embeddings_ptr = nullptr,
        float* gmc_ptr = nullptr,
        float light_factor = 0.0f,
        float mid_thresh_scale = 1.0f
    ) = 0;
};

/**
 * @brief GPUByteTracker 核心實作
 */
class SACCADE_TRACKING_API GPUByteTracker : public ITracker {
public:
    GPUByteTracker(int max_objects = 2048, int embedding_dim = 768);
    ~GPUByteTracker();

    void set_params(
        float track_thresh,
        float high_thresh,
        float match_thresh,
        int track_buffer,
        float mid_thresh = 0.40f,
        int confirm_streak = 3,
        float confirm_score_thresh = 0.50f,
        bool adaptive_confirmation = false
    );
    void set_reid_params(float cos_threshold, float iou_low, float iou_high, float weight);
    void update_reference_features(int* track_ids, float* features_ptr, int num, cudaStream_t stream);
    std::vector<TrackStateSnapshot> get_state_snapshots(cudaStream_t stream);
    std::vector<TrackCandidateSnapshot> get_tentative_candidates(cudaStream_t stream);

    std::vector<TrackResult> update(
        float* boxes_ptr, 
        float* scores_ptr, 
        int* classes_ptr, 
        int num_dets,
        cudaStream_t stream,
        float* embeddings_ptr = nullptr,
        float* gmc_ptr = nullptr,
        float light_factor = 0.0f,
        float mid_thresh_scale = 1.0f
    ) override;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

void SACCADE_TRACKING_API merge_cross_tile_duplicates_cuda(
    const float* boxes_ptr,
    const float* scores_ptr,
    const int* classes_ptr,
    int num_dets,
    int* anchor_indices_ptr,
    float* box_sums_ptr,
    float* score_sums_ptr,
    int* score_bits_max_ptr,
    int* cluster_counts_ptr,
    float* out_boxes_ptr,
    float* out_scores_ptr,
    int* out_classes_ptr,
    int* out_count_ptr,
    float iou_threshold,
    float center_threshold,
    float area_ratio_threshold,
    cudaStream_t stream
);

} // namespace saccade
