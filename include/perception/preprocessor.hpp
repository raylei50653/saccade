#include "saccade/common.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cuda_runtime.h>

namespace saccade {

/**
 * @brief 預處理器 (Preprocessor)
 * 負責縮放、歸一化與格式轉換 (NV12 -> RGB, HWC -> CHW)。
 */
class SACCADE_PERCEPTION_API Preprocessor {
public:
    Preprocessor(int target_width, int target_height);
    ~Preprocessor();

    /**
     * @brief [Legacy] CPU 預處理 (用於備援，HWC BGR -> CHW RGB Float)
     */
    void process(void* input_ptr, int width, int height, void* output_cuda_ptr, cudaStream_t stream);

    /**
     * @brief [Industrial] 全 GPU 預處理 (NV12 -> RGB -> Resize -> CHW Float)
     * 利用 NPP (NVIDIA Performance Primitives) 進行極速轉換。
     * @param input_cuda_ptr 輸入 GPU NV12 指標 (由 GstClient 提供)
     * @param output_cuda_ptr 輸出 GPU CHW Float 指標 (供 TensorRT 推理)
     */
    void process_gpu(void* input_cuda_ptr, int src_width, int src_height, void* output_cuda_ptr, cudaStream_t stream);

private:
    int target_width_;
    int target_height_;
    
    // GPU 中間緩衝區 (用於 Resize 與 CHW 轉換)
    void* d_rgb_interleaved_ = nullptr;
    size_t rgb_buffer_size_ = 0;
};

/**
 * @brief 裁切器 (Cropper)
 * 負責從原始影格中提取多個 RoI。
 */
class SACCADE_PERCEPTION_API Cropper {
public:
    Cropper(int crop_width, int crop_height);
    ~Cropper();

    void process(void* input_ptr, int width, int height, 
                 float* boxes, int num_boxes, void* output_cuda_ptr, cudaStream_t stream);

    /**
     * @brief [Industrial] 全 GPU 批量裁切 (RoI 提取 + Resize + CHW)
     */
    void process_gpu(void* input_cuda_ptr, int src_width, int src_height, 
                     float* boxes, int num_boxes, void* output_cuda_ptr, cudaStream_t stream);

private:
    int crop_width_;
    int crop_height_;
};

} // namespace saccade
