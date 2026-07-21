#!/usr/bin/env python3
"""Download CrowdHuman and CityPersons datasets for FP classifier training."""
# status: diagnostic

import argparse
import subprocess
import sys
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

# CrowdHuman URLs (ETH Zurich)
CROWDHUMAN_URLS = {
    "images_train": "https://data.vision.ee.ethz.ch/cvl/rro/CrowdHuman/CrowdHuman.tar",
    "images_val": "https://data.vision.ee.ethz.ch/cvl/rro/CrowdHuman/CrowdHuman.tar",
    "annotations": "https://data.vision.ee.ethz.ch/cvl/rro/CrowdHuman/CrowdHuman.zip",
}

# CityPersons URLs (ETH Zurich)
CITYPERSONS_URLS = {
    "images_train": "https://data.vision.ee.ethz.ch/CD/684804/data/CityPersons.zip",
    "images_val": "https://data.vision.ee.ethz.ch/CD/684804/data/CityPersons.zip",
    "annotations": "https://data.vision.ee.ethz.ch/CD/684804/data/CityPersonsAnnotations.zip",
}


def download_file(url: str, dest: Path) -> None:
    """Download a file using curl with resume support."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ⏭️  Already exists: {dest.name} ({dest.stat().st_size / 1e9:.1f} GB)")
        return
    print(f"  ⬇️  Downloading {dest.name}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-L", "-C", "-", "-o", str(dest), url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ Download failed: {result.stderr}")
        sys.exit(1)
    size_gb = dest.stat().st_size / 1e9
    print(f"  ✅ Downloaded {dest.name} ({size_gb:.1f} GB)")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract a zip file."""
    print(f"  📦 Extracting {zip_path.name}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unzip", "-o", str(zip_path), "-d", str(extract_dir)], check=True)
    print(f"  ✅ Extracted to {extract_dir}")


def extract_tar(tar_path: Path, extract_dir: Path) -> None:
    """Extract a tar file."""
    print(f"  📦 Extracting {tar_path.name}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xf", str(tar_path), "-C", str(extract_dir)], check=True)
    print(f"  ✅ Extracted to {extract_dir}")


def setup_crowdhuman() -> None:
    """Download and setup CrowdHuman dataset."""
    print("\n=== CrowdHuman ===")
    base = DATASETS_DIR / "CrowdHuman"
    img_dir = base / "images"
    ann_dir = base / "annotations"

    # CrowdHuman has a single tar with train/val images and a zip with annotations
    tar_path = base / "CrowdHuman.tar"
    zip_path = base / "CrowdHuman.zip"

    download_file(CROWDHUMAN_URLS["images_train"], tar_path)
    download_file(CROWDHUMAN_URLS["annotations"], zip_path)

    extract_tar(tar_path, img_dir)
    extract_zip(zip_path, ann_dir)

    # Clean up archives
    print("  🧹 Cleaning up archives...")
    tar_path.unlink()
    zip_path.unlink()

    # Verify structure
    img_train = img_dir / "CrowdHuman_train"
    img_val = img_dir / "CrowdHuman_val"
    ann_file = ann_dir / "annotation_crowdhuman_train.odgt"
    ann_dir / "annotation_crowdhuman_val.odgt"

    print(
        f"  📁 images/train: {len(list(img_train.glob('*.jpg')))} images"
        if img_train.exists()
        else "  ❌ Missing train images"
    )
    print(
        f"  📁 images/val: {len(list(img_val.glob('*.jpg')))} images"
        if img_val.exists()
        else "  ❌ Missing val images"
    )
    print(f"  📁 annotations: {ann_file.exists()}")


def setup_citypersons() -> None:
    """Download and setup CityPersons dataset."""
    print("\n=== CityPersons ===")
    base = DATASETS_DIR / "CityPersons"
    img_dir = base / "images"
    ann_dir = base / "annotations"

    img_zip = base / "CityPersons.zip"
    ann_zip = base / "CityPersonsAnnotations.zip"

    download_file(CITYPERSONS_URLS["images_train"], img_zip)
    download_file(CITYPERSONS_URLS["annotations"], ann_zip)

    # CityPersons zip contains CityPersons/ subdirectory
    extract_zip(img_zip, img_dir.parent)
    extract_zip(ann_zip, ann_dir.parent)

    # Clean up archives
    print("  🧹 Cleaning up archives...")
    img_zip.unlink()
    ann_zip.unlink()

    # Verify structure
    img_train = img_dir / "train"
    img_val = img_dir / "val"
    ann_file = ann_dir / "CityPersons.train.odgt"
    ann_dir / "CityPersons.val.odgt"

    print(
        f"  📁 images/train: {len(list(img_train.glob('*.jpg')))} images"
        if img_train.exists()
        else "  ❌ Missing train images"
    )
    print(
        f"  📁 images/val: {len(list(img_val.glob('*.jpg')))} images"
        if img_val.exists()
        else "  ❌ Missing val images"
    )
    print(f"  📁 annotations: {ann_file.exists()}")


def main():
    parser = argparse.ArgumentParser(
        description="Download external datasets for FP classifier training"
    )
    parser.add_argument(
        "--crowdhuman", action="store_true", help="Download CrowdHuman only"
    )
    parser.add_argument(
        "--citypersons", action="store_true", help="Download CityPersons only"
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    args = parser.parse_args()

    if not any([args.crowdhuman, args.citypersons, args.all]):
        print("Usage: uv run scripts/download_external_datasets.py --all")
        print(
            "   or: uv run scripts/download_external_datasets.py --crowdhuman --citypersons"
        )
        sys.exit(1)

    if args.all or args.crowdhuman:
        setup_crowdhuman()
    if args.all or args.citypersons:
        setup_citypersons()

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
