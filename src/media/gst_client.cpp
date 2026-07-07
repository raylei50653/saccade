#include "media/gst_client.hpp"
#include "media/buffer_pool.hpp"
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>

namespace saccade {

class GstClient::Impl : public IMediaClient {
public:
    explicit Impl(const std::string& pipeline_str)
        : pipeline_str_(pipeline_str), pool_(std::make_unique<BufferPool>(0)) {}

    ~Impl() override {
        // 先停止 GStreamer pipeline 並等 callback 結束 (修 S5/S9):
        // release() 觸發 GST_STATE_NULL,但 appsink streaming thread 可能仍在
        // on_new_sample 中。持 cb_mutex_ 確保 in-flight callback 結束後才銷毀 pool。
        release();
        std::lock_guard<std::mutex> cb_lock(cb_mutex_);
        // reset() 在此觸發 ~BufferPool (sync 全部 stream 再 free,修 S3),
        // 並避免自動解構式再次銷毀 pool。
        pool_.reset();
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
        std::cout << "🚀 [GstClient] 5-Stream State-Machine Enabled (race-fixed)." << std::endl;
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

    // IMediaClient — 由 Python FrameData.mark_processing / release 呼叫
    void markProcessing(int index) override {
        // READY → PROCESSING (CAS);失敗代表狀態不符契約,忽略。
        pool_->mark_processing(index);
    }

    void releaseBuffer(int index) override {
        pool_->release(index);
    }

    void syncBuffer(int index) override {
        pool_->sync_slot(index);
    }

private:
    static GstFlowReturn on_new_sample(GstAppSink* sink, gpointer user_data) {
        auto* self = static_cast<Impl*>(user_data);
        GstSample* sample = gst_app_sink_pull_sample(sink);
        if (!sample) return GST_FLOW_ERROR;

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstCaps* caps = gst_sample_get_caps(sample);
        GstStructure* struct_caps = gst_caps_get_structure(caps, 0);
        int w = 0, h = 0;
        if (!gst_structure_get_int(struct_caps, "width", &w)) w = 0;
        if (!gst_structure_get_int(struct_caps, "height", &h)) h = 0;
        if (w <= 0 || h <= 0) {
            gst_sample_unref(sample);
            return GST_FLOW_ERROR;
        }

        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            // 原子取得 EMPTY 槽 (CAS EMPTY→WRITING),不足空間時只成長該槽 (修 S2/S6/S7)。
            int target_idx = self->pool_->acquire_empty_slot(map.size);
            if (target_idx < 0) {
                // 所有緩衝區都在 PROCESSING/READY/WRITING → drop frame。
                // 即時系統避免阻塞解碼端。
                gst_buffer_unmap(buffer, &map);
                gst_sample_unref(sample);
                return GST_FLOW_OK;
            }

            // 排隊 H2D 並標記 READY (修 S1):consumer 讀取前必須 sync_slot。
            self->pool_->submit_h2d(target_idx, map.data, map.size);

            FrameData data;
            data.cuda_ptr = self->pool_->device_ptr(target_idx);
            data.stream_ptr = (void*)self->pool_->stream(target_idx);
            data.buffer_index = target_idx;
            data.width = w;
            data.height = h;
            data.channels = 3;
            data.timestamp = GST_BUFFER_TIMESTAMP(buffer);
            data.owner_ptr = (void*)static_cast<IMediaClient*>(self);

            {
                // cb_mutex_ 保證 ~Impl 不會在 callback 進行中銷毀 pool (修 S9)。
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

    std::string pipeline_str_;
    GstElement* pipeline_ = nullptr;
    FrameCallback frame_cb_;
    std::mutex cb_mutex_;
    std::unique_ptr<BufferPool> pool_;
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
