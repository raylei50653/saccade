#include "tracking/smart_tracker.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace saccade {

__global__ void update_and_filter_kernel(
    const int* obj_ids,
    const float* boxes,
    bool* extract_mask,
    int num_objs,
    float* state_tensor, // [max_objects, 12]
    bool* flag_tensor,   // [max_objects, 3]
    float iou_threshold,
    float cos_threshold,
    int max_objects
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objs) return;

    int obj_id = obj_ids[idx] % max_objects;
    float* state = state_tensor + obj_id * 12;
    bool* flags = flag_tensor + obj_id * 3;

    float bx1 = boxes[idx * 4 + 0];
    float by1 = boxes[idx * 4 + 1];
    float bx2 = boxes[idx * 4 + 2];
    float by2 = boxes[idx * 4 + 3];

    float cx = (bx1 + bx2) * 0.5f;
    float cy = (by1 + by2) * 0.5f;

    bool has_history = flags[0];
    bool has_ext_history = flags[1];
    bool has_velocity = flags[2];

    float last_ex1 = state[8];
    float last_ey1 = state[9];
    float last_ex2 = state[10];
    float last_ey2 = state[11];

    // IoU computation
    float ix1 = max(bx1, last_ex1);
    float iy1 = max(by1, last_ey1);
    float ix2 = min(bx2, last_ex2);
    float iy2 = min(by2, last_ey2);
    
    float inter = max(0.0f, ix2 - ix1) * max(0.0f, iy2 - iy1);
    float area1 = (bx2 - bx1) * (by2 - by1);
    float area2 = (last_ex2 - last_ex1) * (last_ey2 - last_ey1);
    float iou = inter / (area1 + area2 - inter + 1e-6f);

    // Velocity computation
    float vx = cx - state[4];
    float vy = cy - state[5];
    float last_vx = state[6];
    float last_vy = state[7];

    float norm1 = sqrt(vx * vx + vy * vy);
    float norm2 = sqrt(last_vx * last_vx + last_vy * last_vy);
    float cos_sim = (vx * last_vx + vy * last_vy) / (norm1 * norm2 + 1e-6f);

    bool should_extract = !has_history ||
                          (has_ext_history && (iou < iou_threshold)) ||
                          (has_velocity && (norm1 > 2.0f) && (norm2 > 2.0f) && (cos_sim < cos_threshold));

    extract_mask[idx] = should_extract;

    // Update state
    state[0] = bx1;
    state[1] = by1;
    state[2] = bx2;
    state[3] = by2;
    
    state[4] = cx;
    state[5] = cy;
    
    if (has_history) {
        state[6] = vx;
        state[7] = vy;
        flags[2] = true; // has_velocity
    }

    flags[0] = true; // has_history

    if (should_extract) {
        state[8] = bx1;
        state[9] = by1;
        state[10] = bx2;
        state[11] = by2;
        flags[1] = true; // has_ext_history
    }
}

class SmartTracker::Impl {
public:
    Impl(float iou_thresh, float vel_angle, int max_objs) 
        : iou_threshold_(iou_thresh), max_objects_(max_objs) {
        cos_threshold_ = cos(vel_angle * 3.14159265f / 180.0f);
        cudaMalloc(&d_state_tensor_, max_objects_ * 12 * sizeof(float));
        cudaMalloc(&d_flag_tensor_, max_objects_ * 3 * sizeof(bool));
        cudaMemset(d_state_tensor_, 0, max_objects_ * 12 * sizeof(float));
        cudaMemset(d_flag_tensor_, 0, max_objects_ * 3 * sizeof(bool));
    }

    ~Impl() {
        cudaFree(d_state_tensor_);
        cudaFree(d_flag_tensor_);
    }

    void update_and_filter(const int* d_obj_ids, const float* d_boxes, bool* d_extract_mask, int num_objs, cudaStream_t stream) {
        if (num_objs == 0) return;
        int threads = 256;
        int blocks = (num_objs + threads - 1) / threads;
        update_and_filter_kernel<<<blocks, threads, 0, stream>>>(
            d_obj_ids, d_boxes, d_extract_mask, num_objs, 
            d_state_tensor_, d_flag_tensor_, 
            iou_threshold_, cos_threshold_, max_objects_
        );
    }

private:
    float iou_threshold_;
    float cos_threshold_;
    int max_objects_;
    float* d_state_tensor_;
    bool* d_flag_tensor_;
};

SmartTracker::SmartTracker(float iou, float vel, int max_objs) : pimpl_(std::make_unique<Impl>(iou, vel, max_objs)) {}
SmartTracker::~SmartTracker() = default;

void SmartTracker::update_and_filter(const int* ids, const float* boxes, bool* mask, int num, cudaStream_t stream) {
    pimpl_->update_and_filter(ids, boxes, mask, num, stream);
}

} // namespace saccade
