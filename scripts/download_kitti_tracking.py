#!/usr/bin/env python3
"""
Download the KITTI Tracking Dataset.

Requires a registered account at:
    https://www.cvlibs.net/datasets/kitti/eval_tracking.php

Usage:
    # Interactive: enter email & password when prompted
    python scripts/download_kitti_tracking.py

    # Non-interactive (set env vars or pass args)
    KITTI_EMAIL=user@example.com KITTI_PASSWORD=xxx python scripts/download_kitti_tracking.py
    python scripts/download_kitti_tracking.py --email user@example.com --password xxx

    # Download only specific components
    python scripts/download_kitti_tracking.py --components image_2,label_2,calib

Data layout after download:
    data/kitti/tracking/
        training/
            image_02/    # Left color images (15 GB)
            image_03/    # Right color images (15 GB) [optional]
            velodyne/    # Point clouds (35 GB) [optional]
            oxts/        # GPS/IMU data (8 MB) [optional]
            calib/       # Camera calibration (1 MB)
            label_02/    # Training labels (9 MB)
"""

import argparse
import getpass
import os
import sys
import zipfile
from pathlib import Path

# ── components ──────────────────────────────────────────────────────────────
COMPONENTS = {
    "image_2": {
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip",
        "size": "15 GB",
        "extract_dir": "training/image_02",
        "required": False,
    },
    "image_3": {
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_3.zip",
        "size": "15 GB",
        "extract_dir": "training/image_03",
        "required": False,
    },
    "velodyne": {
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_velodyne.zip",
        "size": "35 GB",
        "extract_dir": "training/velodyne",
        "required": False,
    },
    "oxts": {
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_oxts.zip",
        "size": "8 MB",
        "extract_dir": "training/oxts",
        "required": False,
    },
    "calib": {
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_calib.zip",
        "size": "1 MB",
        "extract_dir": "training/calib",
        "required": True,
    },
    "label_2": {
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip",
        "size": "9 MB",
        "extract_dir": "training/label_02",
        "required": True,
    },
}

# ── helpers ─────────────────────────────────────────────────────────────────


def _get_email(config: dict) -> str:
    email = config.get("email")
    if not email:
        email = os.environ.get("KITTI_EMAIL", "")
    if not email:
        email = input("KITTI registered email: ").strip()
    if not email:
        raise SystemExit("Email is required.")
    return email


def _get_password(config: dict) -> str:
    pwd = config.get("password")
    if not pwd:
        pwd = os.environ.get("KITTI_PASSWORD", "")
    if not pwd:
        pwd = getpass.getpass("KITTI password: ")
    if not pwd:
        raise SystemExit("Password is required.")
    return pwd


def _login(email: str, password: str) -> dict:
    """Login to cvlibs.net and return session cookies."""
    import requests

    session = requests.Session()
    login_url = "https://www.cvlibs.net/datasets/kitti/user_login_check.php"
    resp = session.post(
        login_url,
        data={"email": email, "password": password},
        headers={"Referer": "https://www.cvlibs.net/datasets/kitti/user_login.php"},
    )
    if "user_login.php" in resp.url or resp.status_code != 200:
        raise SystemExit(
            f"Login failed (status {resp.status_code}). Check email/password."
        )
    print("  ✓ Logged in successfully")
    return session.cookies.get_dict()


def _download(url: str, dest: Path, cookies: dict, desc: str) -> None:
    """Download a file with a progress bar using requests."""
    import requests
    from urllib3.exceptions import InsecureRequestWarning
    import warnings

    warnings.filterwarnings("ignore", category=InsecureRequestWarning)

    if dest.exists():
        print(f"  ✓ {desc} already exists ({dest}), skipping download")
        return

    print(f"  Downloading {desc} …")
    with requests.get(url, cookies=cookies, stream=True, timeout=30, verify=False) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    mb = downloaded / 1024 / 1024
                    sys.stdout.write(
                        f"\r    {pct:5.1f}%  ({mb:.0f} / {total / 1024 / 1024:.0f} MB)"
                    )
                    sys.stdout.flush()
        if total:
            print()
    print(f"  ✓ Downloaded {desc}")


def _extract(zip_path: Path, extract_to: Path, desc: str) -> None:
    """Extract a zip file."""
    # Check if already extracted
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"  ✓ {desc} already extracted, skipping")
        return

    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {desc} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"  ✓ Extracted {desc}")


# ── link for TrackEval ──────────────────────────────────────────────────────


def _link_trackeval(base_dir: Path, project_root: Path) -> None:
    """Create symlink so TrackEval can find the GT data."""
    gt_dir = project_root / "third_party" / "TrackEval" / "data" / "gt" / "kitti"
    gt_dir.mkdir(parents=True, exist_ok=True)

    link_target = gt_dir / "kitti_2d_box_train"
    link_source = base_dir / "tracking" / "training"

    if not link_source.exists():
        print(f"  ! Source not found: {link_source}, cannot create link")
        return

    if link_target.exists() or link_target.is_symlink():
        link_target.unlink()

    link_target.symlink_to(link_source.resolve())
    print(f"  ✓ Created symlink: {link_target} → {link_source}")


# ── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Download KITTI Tracking Dataset")
    parser.add_argument("--email", help="Registered email on cvlibs.net")
    parser.add_argument(
        "--password", help="Password (not recommended; use env var or prompt)"
    )
    parser.add_argument(
        "--components",
        default="image_2,calib,label_2",
        help="Comma-separated list of components to download "
        "(image_2,image_3,velodyne,oxts,calib,label_2). "
        "Default: image_2,calib,label_2",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory to store data. Default: <project_root>/data/kitti",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Only download, do not extract",
    )
    args = parser.parse_args()

    config = {"email": args.email, "password": args.password}

    # Determine paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "data" / "kitti"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Determine which components to download
    selected = [c.strip() for c in args.components.split(",")]
    for c in selected:
        if c not in COMPONENTS:
            print(f"  ! Unknown component: {c}. Valid: {', '.join(COMPONENTS.keys())}")
            sys.exit(1)

    # Summary
    print("KITTI Tracking Dataset Download")
    print(f"  Data directory: {data_dir}")
    print(f"  Components: {', '.join(selected)}")
    total_size = sum(float(COMPONENTS[c]["size"].split()[0]) for c in selected)
    print(f"  Estimated total: {total_size:.0f} GB")
    print()

    # Login
    email = _get_email(config)
    password = _get_password(config)
    print("Logging in …")
    cookies = _login(email, password)
    print()

    # Download & extract
    for comp_name in selected:
        comp = COMPONENTS[comp_name]
        zip_name = f"data_tracking_{comp_name}.zip"
        zip_path = data_dir / zip_name
        extract_dir = data_dir / comp["extract_dir"]

        print(f"[{comp_name}] ({comp['size']})")

        # Download
        _download(comp["url"], zip_path, cookies, desc=zip_name)

        # Extract
        if not args.no_extract:
            _extract(zip_path, extract_dir, desc=comp_name)

    # Create TrackEval symlink
    print()
    print("Setting up TrackEval link …")
    _link_trackeval(data_dir, project_root)

    print()
    print("Done! KITTI tracking data is ready.")
    print(f"  Data: {data_dir}")
    gt_link = project_root / "third_party/TrackEval/data/gt/kitti/kitti_2d_box_train"
    print(f"  GT link: {gt_link}")
    print()
    print("To verify, run:")
    print(f"  ls {data_dir / 'tracking' / 'training' / 'image_02'} | head -5")
    print(f"  ls {data_dir / 'tracking' / 'training' / 'label_02'} | head -5")


if __name__ == "__main__":
    main()
