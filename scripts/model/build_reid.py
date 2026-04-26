"""Generic TensorRT FP16 engine builder for ReID models.

Usage:
    # Build DINOv2 (224×224):
    uv run python scripts/model/build_reid.py \\
        --onnx models/embedding/facebook_dinov2-base.onnx \\
        --engine models/embedding/facebook_dinov2-base.engine \\
        --input-hw 224 224

    # Build TransReID (256×128):
    uv run python scripts/model/build_reid.py \\
        --onnx models/embedding/vitb16_imagenet_256x128.onnx \\
        --engine models/embedding/transreid_256x128.engine \\
        --input-hw 256 128
"""
import argparse
import os
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(
    onnx_path: str,
    engine_path: str,
    input_hw: tuple[int, int] = (224, 224),
    min_batch: int = 1,
    opt_batch: int = 8,
    max_batch: int = 32,
    input_name: str = "pixel_values",
) -> None:
    h, w = input_hw
    print(f"Building TRT FP16 engine: {onnx_path} → {engine_path}  [{h}×{w}]")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    old_cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(onnx_path)))
    with open(os.path.basename(onnx_path), "rb") as f:
        ok = parser.parse(f.read())
    os.chdir(old_cwd)

    if not ok:
        for i in range(parser.num_errors):
            print("ONNX parse error:", parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX.")

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        (min_batch, 3, h, w),
        (opt_batch, 3, h, w),
        (max_batch, 3, h, w),
    )
    config.add_optimization_profile(profile)

    print("Building engine (may take a few minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Engine build failed.")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    print(f"Saved: {engine_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--input-hw", nargs=2, type=int, default=[224, 224], metavar=("H", "W"))
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=32)
    args = parser.parse_args()

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        input_hw=(args.input_hw[0], args.input_hw[1]),
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
    )
