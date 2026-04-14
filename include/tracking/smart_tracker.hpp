#pragma once
#include "saccade/common.hpp"
#include <cuda_runtime.h>
#include <memory>

namespace saccade {

/**
 * @brief SmartTracker 智能追蹤與特徵排程器 (C++ 版本)
 * 
 * 負責維護物件狀態，計算 IoU 與速度變化，並透過 CUDA Kernel
 * 生成需要進行特徵提取的 boolean mask。
 */
class SACCADE_TRACKING_API SmartTracker {
public:
    SmartTracker(float iou_threshold = 0.7f, float velocity_angle_threshold = 45.0f, int max_objects = 2048);
    ~SmartTracker();

    /**
     * @brief 根據物件 ID 和 BBox 判斷是否需要提取特徵
     * 
     * @param d_obj_ids    輸入：物件 IDs (N)
     * @param d_boxes      輸入：物件 BBoxes (N x 4, [x1, y1, x2, y2])
     * @param d_extract_mask 輸出：是否需要提取特徵的遮罩 (N)
     * @param num_objs     物件數量
     * @param stream       CUDA Stream
     */
    void update_and_filter(
        const int* d_obj_ids,
        const float* d_boxes,
        bool* d_extract_mask,
        int num_objs,
        cudaStream_t stream
    );

    // 動態參數調整 (用於 L6 降級)
    void set_max_lost_frames(int max_lost);
    void set_min_confidence(float min_conf);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace saccade
