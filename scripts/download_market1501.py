#!/usr/bin/env python3
"""Download and setup Market-1501 dataset from Hugging Face."""

import argparse
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
REPO_ID = "aveocr/Market-1501-v15.09.15.zip"
FILENAME = "Market-1501-v15.09.15.zip"


def download_and_extract(dest_dir: Path):
    """Download and setup Market-1501 dataset from Hugging Face."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / FILENAME

    print(f"🚀 Downloading {FILENAME} from HF dataset repo '{REPO_ID}'...")
    try:
        # Download the file from HF dataset hub
        downloaded_file = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="dataset",
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
        )
        print(f"✅ Downloaded successfully to: {downloaded_file}")
    except Exception as e:
        print(f"❌ Failed to download from Hugging Face: {e}")
        return False

    print("📦 Extracting archive...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # List some files to verify structure
            namelist = zip_ref.namelist()
            print(f"  Total items in zip: {len(namelist)}")
            # Extract all
            zip_ref.extractall(dest_dir)
        print("✅ Extraction complete!")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

    # Clean up the zip file
    if zip_path.exists():
        print("🧹 Cleaning up zip archive...")
        zip_path.unlink()

    # Let's verify the extracted folder
    # Normally the zip extracts to a folder named 'Market-1501-v15.09.15'
    extracted_dir = dest_dir / "Market-1501-v15.09.15"
    if not extracted_dir.exists():
        # Check if it was extracted directly or into another folder name
        # Let's search for folders inside dest_dir
        subdirs = [
            x for x in dest_dir.iterdir() if x.is_dir() and "market" in x.name.lower()
        ]
        if subdirs:
            extracted_dir = subdirs[0]
            print(f"ℹ️ Found extracted folder at: {extracted_dir}")
        else:
            print(
                "⚠️ Could not find standard Market-1501 directory structure. Please check the datasets/ folder."
            )
            return False

    # Let's rename the directory to a standard name or keep it as is
    # We will keep it as 'Market-1501-v15.09.15' as it is standard, but also create a symlink or check folders.
    print(f"\n📂 Dataset extracted to: {extracted_dir}")
    print("\nVerification of structure:")
    expected_folders = ["bounding_box_train", "bounding_box_test", "query", "gt_query"]
    for folder in expected_folders:
        f_path = extracted_dir / folder
        if f_path.exists():
            num_files = len(list(f_path.glob("*")))
            print(f"  ✅ {folder}/: {num_files} files found")
        else:
            print(f"  ❌ {folder}/ is missing")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download Market-1501 from Hugging Face"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATASETS_DIR),
        help="Output directory",
    )
    args = parser.parse_args()

    dest_dir = Path(args.output)
    success = download_and_extract(dest_dir)
    if success:
        print("\n🎉 Market-1501 dataset setup complete!")
    else:
        print("\n❌ Setup failed.")


if __name__ == "__main__":
    main()
