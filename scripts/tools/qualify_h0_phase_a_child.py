#!/usr/bin/env python3
"""Synthetic no-capture child used only by H0 substrate qualification."""
# status: stable

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    args = parser.parse_args(argv)
    if os.environ.get("H0_QUALIFICATION_MODE") != "1":
        parser.error("synthetic qualification child requires H0_QUALIFICATION_MODE=1")
    build = args.build_dir.resolve(strict=True)
    extension = args.extension.resolve(strict=True)
    plugin = args.plugin.resolve(strict=True)
    if extension.parent != build or plugin.parent != build:
        parser.error("qualification artifacts must be direct members of --build-dir")
    sys.path.insert(0, build.as_posix())
    import saccade_tracking_ext

    loaded = Path(saccade_tracking_ext.__file__).resolve(strict=True)
    if loaded != extension:
        parser.error("synthetic runner loaded a different tracking extension")
    ctypes.CDLL(plugin.as_posix(), mode=ctypes.RTLD_LOCAL)
    print("h0 qualification synthetic runner: ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
