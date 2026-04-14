#include "media/gst_client.hpp"
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>

namespace saccade {

class GstClient::Impl {
public:
    static constexpr size_t POOL_SIZE = 5; // 循環緩衝池大小

    Impl(const std::string& pipeline_str) : pipeline_str_(pipeline_str) {
        cudaStreamCreate(&stream_);
    }

    ~Impl() {
        release();
        std::lock_guard<std::mutex> lock(pool_mutex_);
        for (void* ptr : d_buffers_) {
            if (ptr) cudaFree(ptr);
        }
        if (stream_) cudaStreamDestroy(stream_);
    }

    bool connect() {
        GError* error = nullptr;
        pipeline_ = gst_parse_launch(pipeline_str_.c_str(), &error);
        if (error) {
            std::cerr << "❌ [GstClient] Failed to parse pipeline: " << error->message << std::endl;
            g_error_free(error);
            return false;
        }

        GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
        if (!sink) {
            std::cerr << "❌ [GstClient] Missing appsink 'sink' in pipeline." << std::endl;
            return false;
        }

        // 設定 appsink 回調
        GstAppSinkCallbacks callbacks = {nullptr};
        callbacks.new_sample = on_new_sample;
        gst_app_sink_set_callbacks(GST_APP_SINK(sink), &callbacks, this, nullptr);
        gst_object_unref(sink);

        GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "❌ [GstClient] Failed to start pipeline." << std::endl;
            return false;
        }

        std::cout << "✅ [GstClient] GStreamer Pipeline Playing with 5-Buffer GPU Pool." << std::endl;
        return true;
    }

    void release() {
        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
    }

    void setFrameCallback(FrameCallback cb) {
        std::lock_guard<std::mutex> lock(cb_mutex_);
        frame_cb_ = cb;
    }

private:
    static GstFlowReturn on_new_sample(GstAppSink* sink, gpointer user_data) {
        auto* self = static_cast<Impl*>(user_data);
        GstSample* sample = gst_app_sink_pull_sample(sink);
        if (!sample) return GST_FLOW_ERROR;

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstCaps* caps = gst_sample_get_caps(sample);
        if (!caps) return GST_FLOW_ERROR;

        GstStructure* struct_caps = gst_caps_get_structure(caps, 0);
        int width, height;
        gst_structure_get_int(struct_caps, "width", &width);
        gst_structure_get_int(struct_caps, "height", &height);

        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            // 確保緩衝池大小適配解析度
            self->ensureBufferPool(map.size);
            
            // 獲取當前寫入位置
            size_t idx = self->write_idx_ % POOL_SIZE;
            void* target_ptr = self->d_buffers_[idx];
            
            // 非同步拷貝 CPU 解碼資料到指定的 GPU 緩衝池索引
            cudaMemcpyAsync(target_ptr, map.data, map.size, cudaMemcpyHostToDevice, self->stream_);
            
            FrameData data;
            data.cuda_ptr = target_ptr;
            data.width = width;
            data.height = height;
            data.channels = 3;
            data.timestamp = GST_BUFFER_TIMESTAMP(buffer);

            {
                std::lock_guard<std::mutex> lock(self->cb_mutex_);
                if (self->frame_cb_) {
                    self->frame_cb_(data);
                }
            }

            self->write_idx_++;
            gst_buffer_unmap(buffer, &map);
        }

        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    void ensureBufferPool(size_t size) {
        if (gpu_buffer_size_ < size) {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            for (void* ptr : d_buffers_) {
                if (ptr) cudaFree(ptr);
            }
            d_buffers_.clear();
            for (size_t i = 0; i < POOL_SIZE; ++i) {
                void* ptr = nullptr;
                cudaMalloc(&ptr, size);
                d_buffers_.push_back(ptr);
            }
            gpu_buffer_size_ = size;
            std::cout << "📈 [GstClient] GPU Pool Allocated: " << POOL_SIZE << " x " << (size/1024/1024) << " MB" << std::endl;
        }
    }

    std::string pipeline_str_;
    GstElement* pipeline_ = nullptr;
    FrameCallback frame_cb_;
    std::mutex cb_mutex_;
    std::mutex pool_mutex_;
    
    // GPU 循環緩衝池
    std::vector<void*> d_buffers_;
    std::atomic<size_t> write_idx_{0};
    size_t gpu_buffer_size_ = 0;
    cudaStream_t stream_ = nullptr;
};

GstClient::GstClient(const std::string& pipeline_str)
    : pimpl_(std::make_unique<Impl>(pipeline_str)) {}

GstClient::~GstClient() = default;

bool GstClient::connect() { return pimpl_->connect(); }
void GstClient::release() { pimpl_->release(); }
void GstClient::setFrameCallback(FrameCallback cb) { pimpl_->setFrameCallback(cb); }

} // namespace saccade
