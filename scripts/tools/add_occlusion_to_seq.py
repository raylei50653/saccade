#!/usr/bin/env python3

"""Inject simulated occlusion boxes into MOT sequences."""

# status: diagnostic
import argparse
from pathlib import Path
import cv2
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Add a simulated occlusion box in the middle of all frames in a sequence."
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=Path("datasets/demo/custom_seq/img1"),
        help="Path to output img1 folder (occluded frames will be written here)",
    )
    parser.add_argument(
        "--source-img-dir",
        type=Path,
        default=None,
        help="Path to clean source img1 folder (if different from --img-dir). If not set, frames are read from --img-dir (in-place).",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="60,60,60",
        help="BGR color of the box (comma-separated)",
    )
    parser.add_argument(
        "--width-ratio",
        type=float,
        default=0.125,
        help="Ratio of occlusion box width to image width (default 0.125, half of original 0.25)",
    )
    parser.add_argument(
        "--height-ratio",
        type=float,
        default=0.55,
        help="Ratio of occlusion box height to image height (default 0.55)",
    )
    parser.add_argument(
        "--x-offset",
        type=int,
        default=0,
        help="Horizontal offset from center in pixels (positive = right, negative = left)",
    )
    parser.add_argument(
        "--y-offset",
        type=int,
        default=0,
        help="Vertical offset from center in pixels (positive = down, negative = up)",
    )
    args = parser.parse_args()

    if not args.img_dir.exists():
        print(f"❌ Image directory not found: {args.img_dir}")
        sys.exit(1)

    source_dir = args.source_img_dir if args.source_img_dir else args.img_dir
    if not source_dir.exists():
        print(f"❌ Source image directory not found: {source_dir}")
        sys.exit(1)

    args.img_dir.mkdir(parents=True, exist_ok=True)

    # Parse color
    try:
        color = tuple(map(int, args.color.split(",")))
        if len(color) != 3:
            raise ValueError()
    except Exception:
        print("❌ Invalid color format. Use BGR 'B,G,R' e.g. '60,60,60'")
        sys.exit(1)

    frame_files = sorted(source_dir.glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(source_dir.glob("*.png"))
    if not frame_files:
        print("❌ No images found in directory.")
        sys.exit(1)

    print(f"🖼️ Found {len(frame_files)} frames. Adding simulated occlusion box...")

    # Read first image to get dimensions
    first_img = cv2.imread(str(frame_files[0]))
    if first_img is None:
        print("❌ Failed to read first image.")
        sys.exit(1)

    h, w, _ = first_img.shape

    # Calculate central box coordinates
    # Box size: width = configured ratio of image width, height = 55% of image height
    box_w = int(w * args.width_ratio)
    box_h = int(h * args.height_ratio)

    x1 = (w - box_w) // 2 + args.x_offset
    y1 = (h - box_h) // 2 + args.y_offset
    x2 = x1 + box_w
    y2 = y1 + box_h

    print(f"📏 Frame dimensions: {w}x{h}")
    print(
        f"🔳 Occlusion box: coordinates=({x1}, {y1}) to ({x2}, {y2}), size={box_w}x{box_h}, color={color}"
    )

    for idx, fpath in enumerate(frame_files):
        img = cv2.imread(str(fpath))
        if img is None:
            continue

        # Draw solid rectangle in the center
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        # Write to output directory
        out_path = args.img_dir / fpath.name
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if (idx + 1) % 50 == 0 or (idx + 1) == len(frame_files):
            print(f"  Processed {idx + 1}/{len(frame_files)} frames...", end="\r")

    print(f"\n\n🎉 Successfully added occlusion box to all {len(frame_files)} frames!")
    print("👉 Now run the tracker to see how it handles the simulated occlusion!")


if __name__ == "__main__":
    main()
