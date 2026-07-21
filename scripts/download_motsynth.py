#!/usr/bin/env python3
"""Download and setup MOTSynth dataset (Mini/Full)."""
# status: diagnostic

import argparse
import subprocess
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
MOTSYNTH_ROOT = DATASETS_DIR / "MOTSynth"

URLS = {
    "videos_1": "https://motchallenge.net/data/MOTSynth_1.zip",
    "videos_2": "https://motchallenge.net/data/MOTSynth_2.zip",
    "videos_3": "https://motchallenge.net/data/MOTSynth_3.zip",
    "mot_anns": "https://motchallenge.net/data/MOTSynth_mot_annotations.zip",
}


def download_file(url: str, dest: Path) -> None:
    """Download a file using curl with resume support."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ⏭️  Already exists: {dest.name}")
        return
    print(f"  ⬇️  Downloading {dest.name}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Using curl -L (follow redirects) and -C - (resume)
    cmd = ["curl", "-L", "-C", "-", "-o", str(dest), url]
    subprocess.run(cmd, check=True)


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract a zip file."""
    print(f"  📦 Extracting {zip_path.name}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["unzip", "-q", "-o", str(zip_path), "-d", str(extract_dir)], check=True
    )
    print(f"  ✅ Extracted {zip_path.name}")


def extract_frames(motsynth_root: Path):
    """
    Extract frames from videos.
    Note: Requires opencv-python.
    """
    import cv2

    print("  🎥 Extracting frames from videos (this may take a LONG time)...")
    video_dirs = [
        motsynth_root / "MOTSynth_1",
        motsynth_root / "MOTSynth_2",
        motsynth_root / "MOTSynth_3",
    ]

    for vdir in video_dirs:
        if not vdir.exists():
            continue
        print(f"    Processing {vdir.name}...")
        for vfile in vdir.glob("*.mp4"):
            seq_name = vfile.stem
            out_dir = motsynth_root / seq_name / "img1"
            out_dir.mkdir(parents=True, exist_ok=True)

            if any(out_dir.iterdir()):
                print(f"      ⏭️  {seq_name} already extracted")
                continue

            cap = cv2.VideoCapture(str(vfile))
            count = 1
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(
                    str(out_dir / f"{count:06d}.jpg"),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                count += 1
            cap.release()
            print(f"      ✅ Extracted {seq_name} ({count - 1} frames)")


def main():
    parser = argparse.ArgumentParser(description="Download and Setup MOTSynth")
    parser.add_argument(
        "--part", type=int, choices=[1, 2, 3], help="Download specific part"
    )
    parser.add_argument(
        "--anns-only", action="store_true", help="Download annotations only"
    )
    parser.add_argument(
        "--extract-only", action="store_true", help="Extract frames only"
    )
    parser.add_argument(
        "--mini", action="store_true", help="Download only Part 1 (first 250 sequences)"
    )
    args = parser.parse_args()

    MOTSYNTH_ROOT.mkdir(parents=True, exist_ok=True)

    if args.extract_only:
        extract_frames(MOTSYNTH_ROOT)
        return

    # Download Annotations
    ann_zip = MOTSYNTH_ROOT / "MOTSynth_mot_annotations.zip"
    download_file(URLS["mot_anns"], ann_zip)
    extract_zip(ann_zip, MOTSYNTH_ROOT)

    # Download Videos
    parts = []
    if args.part:
        parts = [args.part]
    elif args.mini:
        parts = [1]
    elif not args.anns_only:
        parts = [1, 2, 3]

    for p in parts:
        v_zip = MOTSYNTH_ROOT / f"MOTSynth_{p}.zip"
        download_file(URLS[f"videos_{p}"], v_zip)
        extract_zip(v_zip, MOTSYNTH_ROOT)
        # Optional: unlink to save space immediately
        # v_zip.unlink()

    print("\n✅ Download and zip extraction done.")
    print(
        "👉 Next step: Run with --extract-only to convert .mp4 to .jpg (requires cv2)"
    )


if __name__ == "__main__":
    main()
