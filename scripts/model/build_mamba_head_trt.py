#!/usr/bin/env python3
"""Build TensorRT engine from MambaHead ONNX with the SelectiveScan custom plugin.

Usage:
    uv run scripts/model/build_mamba_head_trt.py \
        --onnx models/yolo/mamba_head_26m.onnx \
        --engine models/yolo/mamba_head_26m.engine \
        --p3-channels 256 --p4-channels 512 --p5-channels 512

Prerequisites:
    - ONNX file (from export_mamba_head_onnx.py)
    - build/libsaccade_scan_plugin.so
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt

project_root = Path(__file__).resolve().parent.parent.parent


def build(
    onnx_path: str,
    engine_path: str,
    p3_channels: int = 128,
    p4_channels: int = 256,
    p5_channels: int = 512,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 4,
    fp16: bool = True,
) -> None:
    onnx = Path(onnx_path)
    engine = Path(engine_path)
    plugin = project_root / "build/libsaccade_scan_plugin.so"

    if not onnx.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx}")
    if not plugin.exists():
        raise FileNotFoundError(f"Plugin not found: {plugin}")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, logger)

    registry = trt.get_plugin_registry()
    print(f"Loading plugin: {plugin}")
    registry.load_library(str(plugin))

    creator = registry.get_plugin_creator("SelectiveScan", "1", "saccade")
    if creator is None:
        print("WARNING: SelectiveScan plugin not found in registry after load_library.")
        print(f"Available creators ({len(registry.all_creators)}):")
        for c in registry.all_creators:
            print(f"  {c.name} v{c.plugin_version} ns={c.plugin_namespace}")
    else:
        print(f"Plugin registered: {creator.name} v{creator.plugin_version}")

    print(f"Parsing ONNX: {onnx}")
    with open(onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "p3",
        (min_batch, p3_channels, 80, 80),
        (opt_batch, p3_channels, 80, 80),
        (max_batch, p3_channels, 80, 80),
    )
    profile.set_shape(
        "p4",
        (min_batch, p4_channels, 40, 40),
        (opt_batch, p4_channels, 40, 40),
        (max_batch, p4_channels, 40, 40),
    )
    profile.set_shape(
        "p5",
        (min_batch, p5_channels, 20, 20),
        (opt_batch, p5_channels, 20, 20),
        (max_batch, p5_channels, 20, 20),
    )
    config.add_optimization_profile(profile)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")

    print("Building TRT engine (this may take several minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    engine_data = serialized if isinstance(serialized, bytes) else bytes(serialized)
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_bytes(engine_data)
    print(f"TRT engine saved: {engine} ({len(engine_data) / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build MambaHead TensorRT engine from ONNX"
    )
    parser.add_argument(
        "--onnx", default="models/yolo/mamba_head_26m.onnx", help="Input ONNX path"
    )
    parser.add_argument(
        "--engine",
        default="models/yolo/mamba_head_26m.engine",
        help="Output engine path",
    )
    parser.add_argument("--p3-channels", type=int, default=256, help="FPN P3 channels")
    parser.add_argument("--p4-channels", type=int, default=512, help="FPN P4 channels")
    parser.add_argument("--p5-channels", type=int, default=512, help="FPN P5 channels")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    args = parser.parse_args()

    build(
        onnx_path=args.onnx,
        engine_path=args.engine,
        p3_channels=args.p3_channels,
        p4_channels=args.p4_channels,
        p5_channels=args.p5_channels,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        fp16=not args.no_fp16,
    )
