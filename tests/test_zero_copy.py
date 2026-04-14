from perception.zero_copy import GstZeroCopyDecoder


def test_gst_decoder_initialization():
    """Test that the decoder can be initialized."""
    # Using a fake URL to test initialization logic
    decoder = GstZeroCopyDecoder("rtsp://localhost:8554/test")
    assert decoder.source_url == "rtsp://localhost:8554/test"
    # The decoder might be nvh264dec (GPU) or avdec_h264 (CPU) depending on the environment
    assert decoder.decoder_name in ["nvh264dec", "avdec_h264"]
    assert decoder.decoder_name in decoder.pipeline_str


def test_pipeline_construction():
    """Test that the pipeline string is generated correctly."""
    decoder_file = GstZeroCopyDecoder("/tmp/test.mp4")
    assert "filesrc location=/tmp/test.mp4" in decoder_file.pipeline_str

    decoder_rtsp = GstZeroCopyDecoder("rtsp://localhost:8554/live")
    assert "rtspsrc location=rtsp://localhost:8554/live" in decoder_rtsp.pipeline_str
