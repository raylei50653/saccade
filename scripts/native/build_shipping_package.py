#!/usr/bin/env python3
"""Build the shipping package from an installed shipping tree (#465 Phase C PR-C3).

docs/reference/native_runtime_resolved_config.md §19. The tree is what
``cmake --install <build> --component shipping`` wrote; it must pass
``check_shipping_bundle.py static`` (run here, report kept); the repository
must be clean and its runtime-identity publication must describe HEAD
(``check_runtime_identity_staleness.py --mode attested``), since the MANIFEST
records that coordinate. ``--trial`` packages anyway and records which of the
two did not hold; ``check_shipping_package.py tarball`` fails such a package.

Writes three files to OUT (created; must not hold anything):

* ``<name>.tar.gz``: ``<name>/MANIFEST.json`` and the tree under ``<name>/``.
  Deterministic: USTAR, members sorted by path, owner 0 without names, every
  mtime the source commit's time, modes 0755 (directories, and files with an
  execute bit) or 0644, gzip without a name or time. The same tree at the same
  commit gives the same bytes (with this host's zlib).
* ``<name>.install.sh``: ``shipping/package/install.sh`` at the source commit.
* ``<name>.sha256``: the package digest, ``sha256sum`` lines for the two above.

``<name>`` = ``saccade-<version>-linux-x86_64-cu<x.y>-trt<x.y>-sm120-glibc<x.y>``
(``check_shipping_package.package_name``). Every MANIFEST key but ``files`` is
derived from the repository at the source commit
(``check_shipping_package.manifest_head``), which also refuses files that are
not the repository's pins.

Usage::

    build_shipping_package.py --tree TREE --out DIST --static-report static.json

Exit 0: written; 1: the tree or the repository is not packageable; 2: error.
"""
# status: active

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, BinaryIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_shipping_package as pkg  # noqa: E402

CHECK_BUNDLE = Path(__file__).resolve().parent / "check_shipping_bundle.py"
CHECK_IDENTITY = pkg.REPO / "scripts/tools/check_runtime_identity_staleness.py"


class _HashingReader:
    """Reads a file for tarfile while hashing what was read, so a file that
    changed after the manifest pass is caught."""

    def __init__(self, f: BinaryIO) -> None:
        self._f = f
        self.sha = hashlib.sha256()

    def read(self, n: int = -1) -> bytes:
        data = self._f.read(n)
        self.sha.update(data)
        return data


def _member(
    name: str, mode: str, mtime: int, size: int = 0, is_dir: bool = False
) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.type = tarfile.DIRTYPE if is_dir else tarfile.REGTYPE
    ti.mode = int(mode, 8)
    ti.mtime = mtime
    ti.size = size
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    return ti


def write_package(
    tree: Path,
    out: Path,
    head: dict[str, Any],
    files: dict[str, dict[str, Any]],
    installer: bytes,
    compresslevel: int = 6,
) -> dict[str, str]:
    """Write the three release files; returns their sha256 by file name."""
    name = head["package"]
    doc = {**head, "files": [files[k] for k in sorted(files)]}
    manifest = pkg.manifest_bytes(doc)
    _, problems = pkg.parse_manifest(manifest)
    if problems:
        raise pkg.PackageError(f"the manifest is not well formed: {problems}")
    mtime = head["source"]["commit_time"] if "source" in head else 0

    entries = set(files) | {pkg.MANIFEST}
    dirs = {""}
    for e in entries:
        parts = e.split("/")[:-1]
        dirs |= {"/".join(parts[: i + 1]) for i in range(len(parts))}
    order = sorted(dirs | entries)

    tarball = out / f"{name}.tar.gz"
    partial = out / f".{name}.tar.gz.partial"
    with partial.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=compresslevel
        ) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tf:
                for rel in order:
                    arc = f"{name}/{rel}" if rel else name
                    if rel in dirs:
                        tf.addfile(_member(arc, "0755", mtime, is_dir=True))
                    elif rel == pkg.MANIFEST:
                        tf.addfile(
                            _member(arc, "0644", mtime, len(manifest)),
                            io.BytesIO(manifest),
                        )
                    else:
                        f = files[rel]
                        with (tree / rel).open("rb") as src:
                            reader = _HashingReader(src)
                            tf.addfile(
                                _member(arc, f["mode"], mtime, f["bytes"]), reader
                            )  # type: ignore[arg-type]
                        if reader.sha.hexdigest() != f["sha256"]:
                            raise pkg.PackageError(
                                f"{rel} changed while it was packaged"
                            )
    os.replace(partial, tarball)

    inst = out / f"{name}.install.sh"
    inst.write_bytes(installer)
    inst.chmod(0o755)
    sums = {
        tarball.name: pkg.g2.sha256_file(tarball),
        inst.name: pkg.sha256_bytes(installer),
    }
    (out / f"{name}.sha256").write_text("".join(f"{v}  {k}\n" for k, v in sums.items()))
    return sums


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--static-report", type=Path, required=True)
    ap.add_argument(
        "--trial",
        action="store_true",
        help="package a dirty tree or a stale runtime identity; recorded, and the package check fails it",
    )
    ap.add_argument("--compresslevel", type=int, default=6)
    args = ap.parse_args(argv)
    tree = args.tree.resolve()
    try:
        if args.out.exists() and any(args.out.iterdir()):
            print(f"build_shipping_package: {args.out} is not empty", file=sys.stderr)
            return 2
        if (tree / pkg.MANIFEST).exists():
            print(
                f"build_shipping_package: {tree} already holds {pkg.MANIFEST}",
                file=sys.stderr,
            )
            return 1
        static = subprocess.run(
            [
                sys.executable,
                str(CHECK_BUNDLE),
                "static",
                "--tree",
                str(tree),
                "--report",
                str(args.static_report),
            ]
        )
        if static.returncode != 0:
            print(
                f"build_shipping_package: {tree} fails check_shipping_bundle.py static",
                file=sys.stderr,
            )
            return 1
        commit = pkg._git("rev-parse", "HEAD").strip()
        clean = pkg._git("status", "--porcelain").strip() == ""
        identity = subprocess.run(
            [sys.executable, str(CHECK_IDENTITY), "--mode", "attested"],
            cwd=pkg.REPO,
            capture_output=True,
            text=True,
        )
        current = identity.returncode == 0
        for ok, what in (
            (clean, "the working tree is not clean"),
            (current, "the runtime-identity publication does not describe HEAD"),
        ):
            if not ok:
                print(f"build_shipping_package: {what}", file=sys.stderr)
        if not (clean and current) and not args.trial:
            return 1
        files, problems = pkg.tree_entries(tree)
        if problems:
            print(f"build_shipping_package: {problems}", file=sys.stderr)
            return 1
        head = pkg.manifest_head(
            commit=commit, tree_clean=clean, identity_current=current, files=files
        )
        installer = pkg._git_show(
            commit, pkg.INSTALLER_SOURCE.relative_to(pkg.REPO).as_posix()
        )
        args.out.mkdir(parents=True, exist_ok=True)
        sums = write_package(tree, args.out, head, files, installer, args.compresslevel)
    except pkg.PackageError as exc:
        print(f"build_shipping_package: {exc}", file=sys.stderr)
        return 1
    except (OSError, subprocess.CalledProcessError, tarfile.TarError) as exc:
        print(f"build_shipping_package: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "package": head["package"],
                "commit": commit,
                "tree_clean": clean,
                "identity_current": current,
                "sha256": sums,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
