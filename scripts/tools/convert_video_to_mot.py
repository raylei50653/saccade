#!/usr/bin/env python3

"""Convert MP4 video into a MOT-compatible sequence folder."""

# status: diagnostic
import argparse
import configparser
from pathlib import Path
import cv2
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Convert an MP4 video into a MOT-compatible sequence folder."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("datasets/demo/15779246_3840_2160_60fps.mp4"),
        help="Path to input MP4 video",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/demo/custom_seq"),
        help="Path to output MOT sequence directory",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"❌ Input video not found: {args.video}")
        sys.exit(1)

    print(f"🎥 Reading video: {args.video}")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print("❌ Failed to open video file.")
        sys.exit(1)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30  # Fallback

    print(
        f"📊 Video stats: Resolution={width}x{height}, FPS={fps}, Total Frames={total_frames}"
    )

    img_dir = args.output_dir / "img1"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Extracting frames to: {img_dir}")
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as 000001.jpg, 000002.jpg, etc.
        out_path = img_dir / f"{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if frame_idx % 50 == 0 or frame_idx == total_frames:
            print(f"  Processed {frame_idx}/{total_frames} frames...", end="\r")
        frame_idx += 1

    cap.release()
    actual_frames = frame_idx - 1
    print(f"\n✅ Extracted {actual_frames} frames.")

    # Generate seqinfo.ini
    ini_path = args.output_dir / "seqinfo.ini"
    print(f"📝 Writing metadata to: {ini_path}")

    config = configparser.ConfigParser()
    config["Sequence"] = {
        "name": args.output_dir.name,
        "imDir": "img1",
        "frameRate": str(fps),
        "seqLength": str(actual_frames),
        "imWidth": str(width),
        "imHeight": str(height),
        "imExt": ".jpg",
    }

    with open(ini_path, "w") as f:
        config.write(f)

    print("\n🎉 Conversion completed successfully!")
    print("👉 You can now run tracking on this sequence using:")
    print(
        f"   uv run scripts/eval/mot17.py --data-root {args.output_dir.parent} --split . --sequences {args.output_dir.name} --visualize"
    )


if __name__ == "__main__":
    main()
