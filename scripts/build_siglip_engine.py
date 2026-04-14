import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(onnx_file_path: str, engine_file_path: str) -> None:
    print(f"🚀 Starting TensorRT Build Process for {onnx_file_path} (FP16)...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 強制開啟 FP16 精度
    config.set_flag(trt.BuilderFlag.FP16)

    # 設定 Memory Pool 限制 (例如 1GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    print("⏳ Parsing ONNX model...")
    # 切換工作目錄到 ONNX 檔案所在目錄，以解決 .onnx.data 檔案找不到的問題
    old_cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(onnx_file_path)))
    onnx_filename = os.path.basename(onnx_file_path)

    with open(onnx_filename, "rb") as model:
        if not parser.parse(model.read()):
            print("❌ ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            os.chdir(old_cwd)
            return None
    os.chdir(old_cwd)

    # 設定 Dynamic Shapes (224x224 for SigLIP 2 Base)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "pixel_values", (1, 3, 224, 224), (8, 3, 224, 224), (32, 3, 224, 224)
    )
    config.add_optimization_profile(profile)

    print("⚙️ Building TensorRT Engine (this may take 5-10 minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("❌ ERROR: Failed to build the engine.")
        return

    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)

    print(f"🎉 Successfully built and saved TRT Engine to: {engine_file_path}")


if __name__ == "__main__":
    onnx_path = "models/embedding/google_siglip2-base-patch16-224.onnx"
    engine_path = "models/embedding/google_siglip2-base-patch16-224.engine"

    if os.path.exists(onnx_path):
        build_engine(onnx_path, engine_path)
    else:
        print(f"❌ ONNX file not found: {onnx_path}")
