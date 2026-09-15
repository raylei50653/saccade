"""Fingerprint resource-sweep presets, local models, native libraries and frames."""

# status: diagnostic
import argparse
import configparser
import hashlib
import json
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def fingerprint(preset, sequences, max_frames):
    preset_path = ROOT / "configs/presets" / f"{preset}.yaml"
    config = yaml.safe_load(preset_path.read_text())
    files = {preset_path, *list((ROOT / "build").glob("*.so"))}
    for value in config.values():
        if isinstance(value, str) and value and (ROOT / value).is_file():
            files.add(ROOT / value)
    sequence_lengths = {}
    for seq in sequences.split(","):
        directory = ROOT / "datasets/MOT17/train" / seq
        info = configparser.ConfigParser()
        info.read(directory / "seqinfo.ini")
        sequence_lengths[seq] = info.getint("Sequence", "seqLength")
        frames = sorted((directory / "img1").glob("*.jpg"))
        if not frames:
            raise ValueError(f"no input frames for {seq}")
        files.update(frames[:max_frames] if max_frames else frames)
        files.update(
            p
            for p in (directory / "seqinfo.ini", directory / "gt/gt.txt")
            if p.exists()
        )
    rows = {}
    for path in sorted(files):
        before = path.stat()
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        after = path.stat()
        if before.st_mtime_ns != after.st_mtime_ns or before.st_size != after.st_size:
            raise ValueError(f"file changed while fingerprinting: {path}")
        rows[str(path.relative_to(ROOT))] = {
            "sha256": digest,
            "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns,
        }
    return {
        "captured_epoch": time.time(),
        "sequence_lengths": sequence_lengths,
        "files": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="mamba_whole_graph_m")
    parser.add_argument("--sequences", default="MOT17-04-SDP")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(fingerprint(args.preset, args.sequences, args.max_frames), indent=2)
        + "\n"
    )
