"""Add a validated lineage manifest to a legacy feature-only Mamba cache."""
# status: stable

from __future__ import annotations

import argparse
import hashlib
import json
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
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--teacher-ckpt", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--sequences", required=True)
    args = parser.parse_args()

    sequences = [item.strip() for item in args.sequences.split(",") if item.strip()]
    frame_counts: dict[str, int] = {}
    sample: dict[str, torch.Tensor] | None = None
    for sequence in sequences:
        sequence_dir = args.cache_dir / sequence
        files = sorted(sequence_dir.glob("*.pt"))
        if not files:
            raise ValueError(f"No cache files found for {sequence_dir}")
        expected = [f"{index:06d}" for index in range(1, len(files) + 1)]
        actual = [path.stem for path in files]
        if actual != expected:
            raise ValueError(f"Cache frames are not contiguous for {sequence}")
        frame_counts[sequence] = len(files)
        if sample is None:
            sample = torch.load(files[0], map_location="cpu", weights_only=True)

    assert sample is not None
    required = ("p3", "p4", "p5")
    if any(key not in sample for key in required):
        raise ValueError(f"Legacy cache sample is missing FPN keys: {sample.keys()}")
    fpn_channels = [int(sample[key].shape[0]) for key in required]
    dtypes = {str(sample[key].dtype).removeprefix("torch.") for key in required}
    if len(dtypes) != 1:
        raise ValueError(f"Mixed FPN cache dtypes are unsupported: {dtypes}")

    manifest = {
        "schema": "mamba-teacher-cache-v2",
        "status": "complete",
        "migration": "legacy-feature-only-cache",
        "img_size": args.img_size,
        "resize_mode": "stretch",
        "gate_input": None,
        "dtype": next(iter(dtypes)),
        "fpn_channels": fpn_channels,
        "base_yolo_path": str(args.yolo_weights),
        "base_yolo_sha256": sha256_file(args.yolo_weights),
        "teacher_checkpoint_path": str(args.teacher_ckpt),
        "teacher_checkpoint_sha256": sha256_file(args.teacher_ckpt),
        "sequences": sequences,
        "frame_counts": frame_counts,
        "total_frames": sum(frame_counts.values()),
    }

    output = args.cache_dir / "manifest.json"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(f"Existing manifest differs from migration: {output}")
        print(f"Verified existing {output}")
        return
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output} ({manifest['total_frames']} frames)")


if __name__ == "__main__":
    main()
