import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_file_path, engine_file_path):
    print(f"🚀 Starting TensorRT Build Process (FP16)...")
    builder = trt.Builder(TRT_LOGGER)
    # 允許使用動態 Batch
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 強制開啟 FP16 精度，以榨出極限吞吐量
    config.set_flag(trt.BuilderFlag.FP16)
    
    print("⏳ Parsing ONNX model...")
    if not parser.parse_from_file(onnx_file_path):
        print('❌ ERROR: Failed to parse the ONNX file.')
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return None
            
    # 設定 Dynamic Shapes 的最佳化設定：Min=1, Opt=8, Max=32
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 224, 224), (8, 3, 224, 224), (32, 3, 224, 224))
    config.add_optimization_profile(profile)
    
    print(f"⚙️ Building TensorRT Engine (this takes a few minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("❌ ERROR: Failed to build the engine.")
        return
        
    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)
        
    print(f"🎉 Successfully built and saved TRT Engine to: {engine_file_path}")

if __name__ == "__main__":
    onnx_path = "models/embedding/vit_so400m_patch14_siglip_224.onnx"
    engine_path = "models/embedding/vit_so400m_patch14_siglip_224.engine"
    
    if not os.path.exists(engine_path):
        build_engine(onnx_path, engine_path)
    else:
        print(f"✅ Engine already exists: {engine_path}")
