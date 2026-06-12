#!/usr/bin/env python3
"""Build TensorRT engine from MambaHead ONNX with the SelectiveScan custom plugin.

Usage:
    uv run scripts/model/build_mamba_head_trt.py

Prerequisites:
    - models/yolo/mamba_head.onnx  (from export_mamba_head_onnx.py)
    - build/libsaccade_scan_plugin.so
"""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt

project_root = Path(__file__).resolve().parent.parent.parent


def build() -> None:
    onnx_path = project_root / "models/yolo/mamba_head.onnx"
    engine_path = project_root / "models/yolo/mamba_head.engine"
    plugin_path = project_root / "build/libsaccade_scan_plugin.so"

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    if not plugin_path.exists():
        raise FileNotFoundError(f"Plugin not found: {plugin_path}")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    # Parse ONNX
    parser = trt.OnnxParser(network, logger)

    # Register custom plugin library before parsing
    registry = trt.get_plugin_registry()
    print(f"Loading plugin: {plugin_path}")
    registry.load_library(str(plugin_path))

    # Verify registration
    creator = registry.get_plugin_creator("SelectiveScan", "1", "saccade")
    if creator is None:
        print("WARNING: SelectiveScan plugin not found in registry after load_library.")
        print(f"Available creators ({len(registry.all_creators)}):")
        for c in registry.all_creators:
            print(f"  {c.name} v{c.plugin_version} ns={c.plugin_namespace}")
    else:
        print(f"Plugin registered: {creator.name} v{creator.plugin_version}")

    # Parse ONNX
    print(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB

    # FP16 + dynamic shapes
    profile = builder.create_optimization_profile()
    profile.set_shape("p3", (1, 128, 80, 80), (1, 128, 80, 80), (4, 128, 80, 80))
    profile.set_shape("p4", (1, 256, 40, 40), (1, 256, 40, 40), (4, 256, 40, 40))
    profile.set_shape("p5", (1, 512, 20, 20), (1, 512, 20, 20), (4, 512, 20, 20))
    config.add_optimization_profile(profile)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")

    # Build engine
    print("Building TRT engine (this may take several minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    # TRT 10.x returns IHostMemory; write bytes
    engine_data = serialized if isinstance(serialized, bytes) else bytes(serialized)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(engine_data)
    print(f"TRT engine saved: {engine_path} ({len(engine_data) / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
