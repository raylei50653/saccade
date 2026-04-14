#include "media/gst_client.hpp"
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <vector>
#include <atomic>

namespace saccade {

class GstClient::Impl {
public:
    static constexpr size_t POOL_SIZE = 5;

    Impl(const std::string& pipeline_str) : pipeline_str_(pipeline_str) {
        for (size_t i = 0; i < POOL_SIZE; ++i) {
            cudaStreamCreate(&streams_[i]);
            buffer_states_[i].store((int)BufferStatus::EMPTY);
        }
    }

    ~Impl() {
        release();
        std::lock_guard<std::mutex> lock(pool_mutex_);
        for (void* ptr : d_buffers_) {
            if (ptr) cudaFree(ptr);
        }
        for (size_t i = 0; i < POOL_SIZE; ++i) {
            if (streams_[i]) cudaStreamDestroy(streams_[i]);
        }
    }

    bool connect() {
        GError* error = nullptr;
        pipeline_ = gst_parse_launch(pipeline_str_.c_str(), &error);
        if (error) {
            std::cerr << "❌ [GstClient] Pipeline failed: " << error->message << std::endl;
            g_error_free(error);
            return false;
        }

        GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
        GstAppSinkCallbacks callbacks = {nullptr};
        callbacks.new_sample = on_new_sample;
        gst_app_sink_set_callbacks(GST_APP_SINK(sink), &callbacks, this, nullptr);
        gst_object_unref(sink);

        gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        std::cout << "🚀 [GstClient] 5-Stream State-Machine Enabled." << std::endl;
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

    // 當 Python 拿到影格並開始分析時調用
    void markProcessing(int index) {
        if (index >= 0 && index < (int)POOL_SIZE) {
            buffer_states_[index].store((int)BufferStatus::PROCESSING);
        }
    }

    // 當 Python 完成處理（GC 或手動）時調用
    void releaseBuffer(int index) {
        if (index >= 0 && index < (int)POOL_SIZE) {
            buffer_states_[index].store((int)BufferStatus::EMPTY);
        }
    }

    void syncBuffer(int index) {
        if (index >= 0 && index < (int)POOL_SIZE) {
            cudaStreamSynchronize(streams_[index]);
        }
    }

private:
    static GstFlowReturn on_new_sample(GstAppSink* sink, gpointer user_data) {
        auto* self = static_cast<Impl*>(user_data);
        GstSample* sample = gst_app_sink_pull_sample(sink);
        if (!sample) return GST_FLOW_ERROR;

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstCaps* caps = gst_sample_get_caps(sample);
        GstStructure* struct_caps = gst_caps_get_structure(caps, 0);
        int w, h;
        gst_structure_get_int(struct_caps, "width", &w);
        gst_structure_get_int(struct_caps, "height", &h);

        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            self->ensureBufferPool(map.size);
            
            // 找出下一個可用的 EMPTY Buffer
            int target_idx = -1;
            for (size_t i = 0; i < POOL_SIZE; ++i) {
                size_t check_idx = (self->write_idx_ + i) % POOL_SIZE;
                if (self->buffer_states_[check_idx].load() == (int)BufferStatus::EMPTY) {
                    target_idx = (int)check_idx;
                    break;
                }
            }

            if (target_idx == -1) {
                // 所有緩衝區都在 PROCESSING 或 WRITING，執行 Drop Frame
                // 這對即時系統非常重要，避免卡住解碼端
                gst_buffer_unmap(buffer, &map);
                gst_sample_unref(sample);
                return GST_FLOW_OK;
            }

            // 更新寫入索引
            self->write_idx_ = (target_idx + 1) % POOL_SIZE;

            // 狀態變更：EMPTY -> WRITING
            self->buffer_states_[target_idx].store((int)BufferStatus::WRITING);

            void* target_ptr = self->d_buffers_[target_idx];
            cudaMemcpyAsync(target_ptr, map.data, map.size, cudaMemcpyHostToDevice, self->streams_[target_idx]);
            
            // 狀態變更：WRITING -> READY (搬運指令已排隊)
            self->buffer_states_[target_idx].store((int)BufferStatus::READY);

            FrameData data;
            data.cuda_ptr = target_ptr;
            data.stream_ptr = (void*)self->streams_[target_idx];
            data.buffer_index = target_idx;
            data.width = w;
            data.height = h;
            data.channels = 3;
            data.timestamp = GST_BUFFER_TIMESTAMP(buffer);
            data.owner_ptr = (void*)self;

            {
                std::lock_guard<std::mutex> lock(self->cb_mutex_);
                if (self->frame_cb_) {
                    self->frame_cb_(data);
                }
            }

            gst_buffer_unmap(buffer, &map);
        }

        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    void ensureBufferPool(size_t size) {
        if (gpu_buffer_size_ < size) {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            for (void* ptr : d_buffers_) if (ptr) cudaFree(ptr);
            d_buffers_.clear();
            for (size_t i = 0; i < POOL_SIZE; ++i) {
                void* ptr = nullptr;
                cudaMalloc(&ptr, size);
                d_buffers_.push_back(ptr);
            }
            gpu_buffer_size_ = size;
        }
    }

    std::string pipeline_str_;
    GstElement* pipeline_ = nullptr;
    FrameCallback frame_cb_;
    std::mutex cb_mutex_;
    std::mutex pool_mutex_;
    
    std::vector<void*> d_buffers_;
    std::atomic<int> buffer_states_[POOL_SIZE];
    cudaStream_t streams_[POOL_SIZE];
    std::atomic<size_t> write_idx_{0};
    size_t gpu_buffer_size_ = 0;
};

GstClient::GstClient(const std::string& pipeline_str)
    : pimpl_(std::make_unique<Impl>(pipeline_str)) {}

GstClient::~GstClient() = default;

bool GstClient::connect() { return pimpl_->connect(); }
void GstClient::release() { pimpl_->release(); }
void GstClient::setFrameCallback(FrameCallback cb) { pimpl_->setFrameCallback(cb); }
void GstClient::markProcessing(int index) { pimpl_->markProcessing(index); }
void GstClient::releaseBuffer(int index) { pimpl_->releaseBuffer(index); }
void GstClient::syncBuffer(int index) { pimpl_->syncBuffer(index); }

} // namespace saccade
