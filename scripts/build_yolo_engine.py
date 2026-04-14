import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_file_path: str, engine_file_path: str) -> None:
    print(f"🚀 Starting TensorRT Build Process for {onnx_file_path} (FP16)...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 強制開啟 FP16 精度
    config.set_flag(trt.BuilderFlag.FP16)
    
    # 設定 Memory Pool 限制 (1GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    
    print("⏳ Parsing ONNX model...")
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print('❌ ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    # 設定 Dynamic Shapes (640x640 for YOLO26)
    profile = builder.create_optimization_profile()
    profile.set_shape("images", (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))
    config.add_optimization_profile(profile)
    
    print("⚙️ Building TensorRT Engine (this takes a few minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("❌ ERROR: Failed to build the engine.")
        return
        
    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)
        
    print(f"🎉 Successfully built and saved TRT Engine to: {engine_file_path}")

if __name__ == "__main__":
    onnx_path = "models/yolo/yolo26n.onnx"
    engine_path = "models/yolo/yolo26n_native.engine"
    
    if os.path.exists(onnx_path):
        build_engine(onnx_path, engine_path)
    else:
        print(f"❌ ONNX file not found: {onnx_path}")
