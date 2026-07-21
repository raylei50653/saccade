#!/usr/bin/env python3
"""Download CrowdHuman dataset from Hugging Face."""
# status: diagnostic

import argparse
import shutil
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
REPO_ID = "Carles208AVL/CrowdHuman"


def download_crowdhuman_hf(dest_dir: Path):
    """Download and setup CrowdHuman dataset from Hugging Face."""
    print(f"🚀 Downloading {REPO_ID} from Hugging Face...")

    tmp_dir = dest_dir / "tmp_hf"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Download the repository
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=tmp_dir,
        local_dir_use_symlinks=False,
    )

    img_dir = dest_dir / "images"
    ann_dir = dest_dir / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    print("📦 Extracting archives...")

    # Handle zips
    zip_files = sorted(list(tmp_dir.glob("*.zip")))
    for zip_file in zip_files:
        print(f"  Extracting {zip_file.name} to {img_dir}...")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(img_dir)

    # Handle annotations
    odgt_files = list(tmp_dir.glob("*.odgt"))
    for odgt_file in odgt_files:
        print(f"  Copying {odgt_file.name} to {ann_dir}...")
        shutil.copy(odgt_file, ann_dir / odgt_file.name)

    # Clean up
    print("🧹 Cleaning up temporary downloads...")
    shutil.rmtree(tmp_dir)

    # Flatten structure if needed (moved to post-execution in my manual steps,
    # but I'll add it to the script for future use)
    if (img_dir / "Images").exists():
        print("📂 Flattening Images directory...")
        for f in (img_dir / "Images").glob("*"):
            shutil.move(str(f), str(img_dir / f.name))
        (img_dir / "Images").rmdir()

    if (img_dir / "images_test").exists():
        print("📂 Flattening images_test directory...")
        for f in (img_dir / "images_test").glob("*"):
            shutil.move(str(f), str(img_dir / f.name))
        (img_dir / "images_test").rmdir()

    print(f"\n✅ CrowdHuman dataset setup complete at {dest_dir}")

    # Verification
    print("\nVerification:")
    total_imgs = len(list(img_dir.glob("*.jpg")))
    anns = [f.name for f in ann_dir.glob("*.odgt")]
    print(f"  Total Images: {total_imgs}")
    print(f"  Annotations: {anns}")


def main():
    parser = argparse.ArgumentParser(
        description="Download CrowdHuman from Hugging Face"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATASETS_DIR / "CrowdHuman"),
        help="Output directory",
    )
    args = parser.parse_args()

    dest_dir = Path(args.output)
    download_crowdhuman_hf(dest_dir)


if __name__ == "__main__":
    main()
