import torch
import tensorrt as trt
import gi
import sys

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402
from perception.detector_trt import TRTYoloDetector  # noqa: E402


def check_gpu_stack() -> bool:
    print("🔍 Initializing GPU Stack Deep Check...")
    passed = True

    # 1. PyTorch & CUDA
    if not torch.cuda.is_available():
        print("❌ ERROR: PyTorch cannot find CUDA.")
        passed = False
    else:
        print(f"✅ PyTorch CUDA: Found {torch.cuda.get_device_name(0)}")

    # 2. TensorRT Python Bindings
    print(f"✅ TensorRT Version: {trt.__version__}")

    # 3. GStreamer Hardware Decoder Check
    Gst.init(None)
    registry = Gst.Registry.get()
    if registry.find_feature("nvh264dec", Gst.ElementFactory.__gtype__):
        print("✅ GStreamer: nvh264dec (Hardware Decoder) is available.")
    else:
        print("⚠️ WARNING: nvh264dec not found. Zero-copy performance will be degraded.")
        # 這不一定算 Fail，但對於 Saccade 來說很重要

    # 4. Native Engine Load & Inference Test
    try:
        print("⏳ Testing YOLO26 TensorRT Inference...")
        detector = TRTYoloDetector(engine_path="models/yolo/yolo26n_native.engine")
        dummy_input = torch.randn(1, 3, 640, 640, device="cuda")
        # 執行一次推理
        _ = detector.detect(dummy_input)
        torch.cuda.synchronize()
        print("✅ YOLO26 Inference: Success (Zero-Sync Path Working)")
    except Exception as e:
        print(f"❌ ERROR: TensorRT Inference failed: {e}")
        passed = False

    return passed


if __name__ == "__main__":
    if check_gpu_stack():
        print("\n🟢 [GPU Check] All systems green.")
        sys.exit(0)
    else:
        print("\n🔴 [GPU Check] FAILED. Check your drivers or engine files.")
        sys.exit(1)
