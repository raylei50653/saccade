import tensorrt as trt
import os
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class CacheCalibrator(trt.IInt8EntropyCalibrator2):
    """Load a pre-existing INT8 calibration cache — no data needed."""

    def __init__(self, cache_file: str):
        super().__init__()
        self.cache_file = cache_file

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names):  # type: ignore[override]
        return None

    def read_calibration_cache(self):
        with open(self.cache_file, "rb") as f:
            return f.read()

    def write_calibration_cache(self, cache):
        pass


def build_engine(
    onnx_file_path: str,
    engine_file_path: str,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 1,
    img_size: int = 640,
    precision: str = "fp16",
    int8_cache: str = "",
) -> None:
    print(f"🚀 Starting TensorRT Build Process for {onnx_file_path} ({precision.upper()})...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.set_flag(trt.BuilderFlag.FP16)  # always enable FP16 as fallback

    if precision == "int8":
        if not int8_cache or not os.path.exists(int8_cache):
            raise FileNotFoundError(f"INT8 cache not found: {int8_cache}")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = CacheCalibrator(int8_cache)
        print(f"📦 Using INT8 calibration cache: {int8_cache}")

    # 設定 Memory Pool 限制 (1GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    print("⏳ Parsing ONNX model...")
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("❌ ERROR: Failed to parse the ONNX file.")
            for i in range(parser.num_errors):
                error = parser.get_error(i)
                print(
                    f"Error {i}: {error.code()} - {error.desc()} at {error.file()}:{error.line()}"
                )
            return None

    # 動態輸入支援 (YOLO11/YOLO26 通常使用 images 名稱)
    profile = builder.create_optimization_profile()

    # 取得輸入節點名稱
    input_name = network.get_input(0).name
    print(f"🔍 Input Node Name: {input_name}")

    profile.set_shape(
        input_name,
        (min_batch, 3, img_size, img_size),
        (opt_batch, 3, img_size, img_size),
        (max_batch, 3, img_size, img_size),
    )
    config.add_optimization_profile(profile)

    print("⚙️ Building Dynamic TensorRT Engine (this takes a few minutes)...")
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
    parser.add_argument(
        "--engine", type=str, required=True, help="Output Engine file path"
    )
    parser.add_argument(
        "--min-batch", type=int, default=1, help="Minimum TensorRT profile batch size"
    )
    parser.add_argument(
        "--opt-batch", type=int, default=1, help="Optimal TensorRT profile batch size"
    )
    parser.add_argument(
        "--max-batch", type=int, default=1, help="Maximum TensorRT profile batch size"
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="Square input image size"
    )
    parser.add_argument(
        "--precision", type=str, default="fp16", choices=["fp16", "int8"],
        help="Build precision: fp16 or int8"
    )
    parser.add_argument(
        "--int8-cache", type=str, default="", help="Path to INT8 calibration cache file"
    )

    args = parser.parse_args()

    if os.path.exists(args.onnx):
        build_engine(
            args.onnx,
            args.engine,
            min_batch=args.min_batch,
            opt_batch=args.opt_batch,
            max_batch=args.max_batch,
            img_size=args.img_size,
            precision=args.precision,
            int8_cache=args.int8_cache,
        )
    else:
        print(f"❌ ONNX file not found: {args.onnx}")
