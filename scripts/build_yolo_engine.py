import tensorrt as trt
import os
import argparse

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

    # 設定 Memory Pool 限制 (1GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    print("⏳ Parsing ONNX model...")
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("❌ ERROR: Failed to parse the ONNX file.")
            for i in range(parser.num_errors):
                error = parser.get_error(i)
                print(f"Error {i}: {error.code()} - {error.desc()} at {error.file()}:{error.line()}")
            return None

    # 動態輸入支援 (YOLO11/YOLO26 通常使用 images 名稱)
    profile = builder.create_optimization_profile()
    
    # 取得輸入節點名稱
    input_name = network.get_input(0).name
    print(f"🔍 Input Node Name: {input_name}")
    
    profile.set_shape(input_name, (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))
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
    parser = argparse.ArgumentParser(description="Build TensorRT Engine from ONNX")
    parser.add_argument("--onnx", type=str, required=True, help="Input ONNX file path")
    parser.add_argument("--engine", type=str, required=True, help="Output Engine file path")
    
    args = parser.parse_args()

    if os.path.exists(args.onnx):
        build_engine(args.onnx, args.engine)
    else:
        print(f"❌ ONNX file not found: {args.onnx}")
