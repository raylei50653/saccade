#!/usr/bin/env python3
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
        help="Path to sequence img1 folder",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="60,60,60",
        help="BGR color of the box (comma-separated)",
    )
    args = parser.parse_args()

    if not args.img_dir.exists():
        print(f"❌ Image directory not found: {args.img_dir}")
        sys.exit(1)

    # Parse color
    try:
        color = tuple(map(int, args.color.split(",")))
        if len(color) != 3:
            raise ValueError()
    except Exception:
        print("❌ Invalid color format. Use BGR 'B,G,R' e.g. '60,60,60'")
        sys.exit(1)

    frame_files = sorted(args.img_dir.glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(args.img_dir.glob("*.png"))
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
    # Box size: width = 25% of image width, height = 55% of image height
    box_w = int(w * 0.25)
    box_h = int(h * 0.55)

    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
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

        # Write back to the same file
        cv2.imwrite(str(fpath), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if (idx + 1) % 50 == 0 or (idx + 1) == len(frame_files):
            print(f"  Processed {idx + 1}/{len(frame_files)} frames...", end="\r")

    print(f"\n\n🎉 Successfully added occlusion box to all {len(frame_files)} frames!")
    print("👉 Now run the tracker to see how it handles the simulated occlusion!")


if __name__ == "__main__":
    main()
