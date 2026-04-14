#include <iostream>
#include <memory>
#include <gst/gst.h>
#include <chrono>
#include <thread>
#include "perception/trt_engine.hpp"
#include "perception/preprocessor.hpp"
#include "media/gst_client.hpp"
#include "saccade/common.hpp"

int main(int argc, char* argv[]) {
    std::cout << "🚀 Saccade C++ Perception Node Starting..." << std::endl;

    // 1. 初始化基礎組件
    gst_init(&argc, &argv);
    cudaStream_t stream;
    saccade::checkCuda(cudaStreamCreate(&stream));

    try {
        // 2. 初始化感知組件
        const std::string yolo_path = "./models/yolo/yolo26n_native.engine";
        auto detector = std::make_unique<saccade::TRTEngine>(yolo_path);
        auto preprocessor = std::make_unique<saccade::Preprocessor>(640, 640);

        // 預分配 GPU 記憶體用於 YOLO 輸入 Tensor [1, 3, 640, 640]
        void* d_yolo_input = nullptr;
        saccade::checkCuda(cudaMalloc(&d_yolo_input, 1 * 3 * 640 * 640 * sizeof(float)));
        
        // 獲取模型輸出指標 (實際應用中應預分配輸出 Tensor)
        // 此處簡化處理
        void* d_yolo_output = nullptr;
        saccade::checkCuda(cudaMalloc(&d_yolo_output, 1 * 300 * 6 * sizeof(float)));

        std::vector<void*> bindings = { d_yolo_input, d_yolo_output };

        // 3. 初始化媒體管線
        // 使用 videotestsrc 進行初步測試，若有影片可換成 filesrc
        std::string pipeline_str = "videotestsrc ! video/x-raw,width=1920,height=1080,format=RGB ! "
                                   "cudaupload ! cudaconvert ! video/x-raw(memory:CUDAMemory),format=RGB ! appsink name=sink emit-signals=true";
        
        auto media = std::make_unique<saccade::GstClient>(pipeline_str);

        // 設定影格回調 (全流水線邏輯在此觸發)
        media->setFrameCallback([&](const saccade::FrameData& frame) {
            auto start = std::chrono::high_resolution_clock::now();

            // A. 預處理 (Zero-Copy)
            preprocessor->process(frame.cuda_ptr, frame.width, frame.height, d_yolo_input, stream);

            // B. 推理 (Async)
            detector->infer(bindings, stream);

            // 此處可以繼續銜接 Tracker 與 Feature Extractor
            
            // C. 為了測量延遲，我們進行一次同步 (實際運行時應盡量避免)
            cudaStreamSynchronize(stream);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            
            std::cout << "\r⚡ Frame Processed. Latency: " << duration << " ms" << std::flush;
        });

        if (!media->connect()) return -1;

        // 4. 運行主循環 (保持程式不退出)
        std::cout << "📡 Pipeline Running. Press Ctrl+C to exit." << std::endl;
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ Runtime Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
