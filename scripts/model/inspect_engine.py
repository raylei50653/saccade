import argparse
from pathlib import Path

import tensorrt as trt


def inspect_engine(engine_path: Path) -> None:
    logger = trt.Logger(trt.Logger.ERROR)
    with engine_path.open("rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

    print(f"Engine: {engine_path}")
    print(f"I/O tensors: {engine.num_io_tensors}")
    print(f"Optimization profiles: {engine.num_optimization_profiles}")

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        print(f"  [{i}] {mode.name:<6} {name}: dtype={dtype.name}, shape={shape}")

    for profile_idx in range(engine.num_optimization_profiles):
        print(f"Profile {profile_idx}:")
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            min_shape, opt_shape, max_shape = engine.get_tensor_profile_shape(
                name, profile_idx
            )
            print(
                f"  {name}: min={tuple(min_shape)}, opt={tuple(opt_shape)}, max={tuple(max_shape)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a TensorRT engine.")
    parser.add_argument(
        "engine",
        nargs="?",
        default="models/embedding/google_siglip2-base-patch16-224.engine",
        help="Path to a TensorRT .engine file.",
    )
    args = parser.parse_args()
    inspect_engine(Path(args.engine))


if __name__ == "__main__":
    main()
