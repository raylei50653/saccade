#!/usr/bin/env python3
"""Build the default OSNet TensorRT engine from an ONNX export.

This is a thin wrapper around scripts/model/build_reid.py so the repo has a
first-class, discoverable OSNet build path that matches the default runtime
engine lookup in perception/feature_extractor.py.

Expected ONNX input:
    models/embedding/osnet_x1_0_256x128.onnx

Default engine output:
    models/embedding/osnet_x1_0_256x128.engine

Usage:
    uv run python scripts/model/build_osnet.py
    uv run python scripts/model/build_osnet.py --onnx /path/to/osnet.onnx
"""
# status: stable

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.model.build_reid import build_engine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="models/embedding/osnet_x1_0_256x128.onnx")
    parser.add_argument(
        "--engine", default="models/embedding/osnet_x1_0_256x128.engine"
    )
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=32)
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise SystemExit(
            "OSNet ONNX not found at "
            f"{onnx_path}. Export or place an ONNX there, or pass --onnx explicitly."
        )

    engine_path = Path(args.engine)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    build_engine(
        onnx_path=str(onnx_path),
        engine_path=str(engine_path),
        input_hw=(256, 128),
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        input_name="input",
    )


if __name__ == "__main__":
    main()
