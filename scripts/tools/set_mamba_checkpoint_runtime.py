"""Create a weight-identical Mamba checkpoint with explicit runtime semantics."""
# status: stable

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scan-runtime",
        choices=("legacy-n1", "fixed-n16"),
        required=True,
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    mamba_args = dict(checkpoint.get("mamba_args", {}))
    legacy_n1 = args.scan_runtime == "legacy-n1"
    mamba_args["legacy_n1_scan"] = legacy_n1
    if legacy_n1:
        mamba_args["legacy_n1_source"] = "77fcc262^"
    else:
        mamba_args.pop("legacy_n1_source", None)
    checkpoint["mamba_args"] = mamba_args
    checkpoint["runtime_variant"] = {
        "scan_runtime": args.scan_runtime,
        "source_checkpoint": str(args.input),
        "source_sha256": sha256_file(args.input),
        "weights_modified": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Wrote {args.output}")
    print(f"source_sha256={checkpoint['runtime_variant']['source_sha256']}")


if __name__ == "__main__":
    main()
