#include "saccade/common.hpp"
#include <string>
#include <functional>
#include <memory>
#include <atomic>

namespace saccade {

/**
 * @brief 緩衝區狀態列舉 (State Machine)
 */
enum class BufferStatus {
    EMPTY = 0,      // C++ 可寫入
    WRITING = 1,    // C++ 正在進行 H2D 搬運
    READY = 2,      // 搬運指令已發出，等待 Python 處理
    PROCESSING = 3  // Python 正持有此 Buffer 進行計算
};

/**
 * @brief 影格封裝物件 (Industrial Grade)
 */
struct FrameData {
    void* cuda_ptr;    // GPU 顯存指標
    void* stream_ptr;  // 專屬 CUDA Stream 指標 (用於 ExternalStream 綁定)
    int buffer_index;  // 緩衝區索引
    int width;
    int height;
    int channels;
    long long timestamp;

    // 狀態管理物件，由 Impl 指向
    void* owner_ptr; 
};

/**
 * @brief 媒體客戶端接口
 */
class SACCADE_MEDIA_API IMediaClient {
public:
    virtual ~IMediaClient() = default;
    virtual bool connect() = 0;
    virtual void release() = 0;
    
    using FrameCallback = std::function<void(const FrameData&)>;
    virtual void setFrameCallback(FrameCallback cb) = 0;

    // 狀態手動控制介面 (可由 Python 調用，也可由 GC 自動觸發)
    virtual void markProcessing(int index) = 0;
    virtual void releaseBuffer(int index) = 0;
    virtual void syncBuffer(int index) = 0;
};

class SACCADE_MEDIA_API GstClient : public IMediaClient {
public:
    GstClient(const std::string& pipeline_str);
    ~GstClient();

    bool connect() override;
    void release() override;
    void setFrameCallback(FrameCallback cb) override;

    void markProcessing(int index) override;
    void releaseBuffer(int index) override;
    void syncBuffer(int index) override;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace saccade
