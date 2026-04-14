#include "saccade/common.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace saccade {

/**
 * @brief 預處理器 (Preprocessor)
 * 負責縮放、歸一化與格式轉換 (BGR -> RGB, HWC -> CHW)。
 */
class SACCADE_PERCEPTION_API Preprocessor {
public:
    Preprocessor(int target_width, int target_height);
    ~Preprocessor();

    /**
     * @brief 執行預處理
     * @param input_ptr 輸入影格 (HWC, BGR, uint8)
     * @param output_cuda_ptr 輸出 Tensor (CHW, RGB, float32)
     */
    void process(void* input_ptr, int width, int height, void* output_cuda_ptr, cudaStream_t stream);

private:
    int target_width_;
    int target_height_;
};

/**
 * @brief 裁切器 (Cropper)
 * 負責從原始影格中提取多個 RoI。
 */
class SACCADE_PERCEPTION_API Cropper {
public:
    Cropper(int crop_width, int crop_height);
    ~Cropper();

    /**
     * @brief 執行批量裁切 (RoI Align 近似)
     * @param input_ptr 原始影格 (HWC, BGR, uint8)
     * @param boxes [N, 4] 座標
     * @param output_cuda_ptr [N, 3, crop_height, crop_width]
     */
    void process(void* input_ptr, int width, int height, 
                 float* boxes, int num_boxes, void* output_cuda_ptr, cudaStream_t stream);

private:
    int crop_width_;
    int crop_height_;
};

} // namespace saccade
