#include "saccade/common.hpp"
#include <string>
#include <functional>

namespace saccade {

/**
 * @brief 媒體管線結果 (Zero-Copy)
 */
struct FrameData {
    void* cuda_ptr;    // GPU 顯存指標
    int width;
    int height;
    int channels;
    long long timestamp;
};

/**
 * @brief 媒體客戶端接口
 */
class SACCADE_MEDIA_API IMediaClient {
public:
    virtual ~IMediaClient() = default;
    virtual bool connect() = 0;
    virtual void release() = 0;
    
    // 影格到達的回調函數
    using FrameCallback = std::function<void(const FrameData&)>;
    virtual void setFrameCallback(FrameCallback cb) = 0;
};

/**
 * @brief GStreamer 媒體客戶端實作
 */
class SACCADE_MEDIA_API GstClient : public IMediaClient {
public:
    GstClient(const std::string& pipeline_str);
    ~GstClient();

    bool connect() override;
    void release() override;
    void setFrameCallback(FrameCallback cb) override;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace saccade
