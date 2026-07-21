"""Generate the slim release tree from a git ref.

The slim release is a deployable subset of the repo (deps, src, tests, build/
export/ops scripts) with research/ablation cruft stripped. Rather than keep it
as a divergent mass-deletion branch that rots against main, this script
materializes it on demand from a manifest, so main stays the single source of
truth.

Usage:
    # Materialize the slim tree from HEAD into ./slim-out
    build_slim_release.py --out slim-out

    # Materialize from a specific ref
    build_slim_release.py --ref main --out slim-out

    # Print the resolved file list (no output written)
    build_slim_release.py --list

    # Verify the manifest reproduces an existing tree/branch exactly
    build_slim_release.py --check release/slim-main
"""
# status: diagnostic

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "slim_manifest.yaml"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _path_exists_in_ref(ref: str, path: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{ref}:{path}"],
            capture_output=True,
        ).returncode
        == 0
    )


def load_manifest(manifest_path: Path) -> dict[str, list[str]]:
    data = yaml.safe_load(manifest_path.read_text())
    for key in ("keep_dirs", "keep_root_files", "keep_files"):
        data.setdefault(key, [])
        if not isinstance(data[key], list):
            raise ValueError(f"manifest key {key!r} must be a list")
    return data


def resolve_files(
    ref: str, manifest: dict[str, list[str]]
) -> tuple[set[str], list[str]]:
    """Return (resolved file set, list of missing explicit paths)."""
    files: set[str] = set()
    missing: list[str] = []

    for path in (*manifest["keep_root_files"], *manifest["keep_files"]):
        if _path_exists_in_ref(ref, path):
            files.add(path)
        else:
            missing.append(path)

    for directory in manifest["keep_dirs"]:
        listed = _git("ls-tree", "-r", "--name-only", ref, "--", directory)
        entries = [line for line in listed.splitlines() if line]
        if not entries:
            missing.append(f"{directory}/ (empty or absent)")
        files.update(entries)

    return files, missing


def build(ref: str, files: set[str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # Stream the selected paths out of the ref as a tar and unpack into out/.
    archive = subprocess.Popen(
        ["git", "archive", "--format=tar", ref, "--", *sorted(files)],
        stdout=subprocess.PIPE,
    )
    extract = subprocess.Popen(["tar", "-x", "-C", str(out)], stdin=archive.stdout)
    if archive.stdout is not None:
        archive.stdout.close()
    extract.communicate()
    archive.wait()
    if archive.returncode or extract.returncode:
        raise RuntimeError("git archive | tar pipeline failed")


def check(ref: str, resolved: set[str], compare_ref: str) -> int:
    actual_listing = _git("ls-tree", "-r", "--name-only", compare_ref)
    actual = {line for line in actual_listing.splitlines() if line}
    missing = sorted(actual - resolved)  # in compare_ref but not produced
    extra = sorted(resolved - actual)  # produced but not in compare_ref

    if not missing and not extra:
        print(f"OK: manifest reproduces {compare_ref} exactly ({len(resolved)} files)")
        return 0

    if missing:
        print(f"MISSING ({len(missing)}) — in {compare_ref} but not in manifest:")
        for path in missing:
            print(f"  - {path}")
    if extra:
        print(f"EXTRA ({len(extra)}) — produced by manifest but not in {compare_ref}:")
        for path in extra:
            print(f"  + {path}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="git ref to build the slim tree from (default: HEAD)",
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST, help="manifest YAML path"
    )
    parser.add_argument(
        "--out", type=Path, help="materialize the slim tree into this directory"
    )
    parser.add_argument(
        "--list", action="store_true", help="print resolved file list and exit"
    )
    parser.add_argument(
        "--check", metavar="REF", help="verify the manifest reproduces REF exactly"
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    files, missing = resolve_files(args.ref, manifest)

    if missing:
        print(
            f"WARNING: {len(missing)} manifest path(s) absent in {args.ref}:",
            file=sys.stderr,
        )
        for path in missing:
            print(f"  ! {path}", file=sys.stderr)

    if args.list:
        for path in sorted(files):
            print(path)
        return 0

    if args.check:
        return check(args.ref, files, args.check)

    if args.out:
        build(args.ref, files, args.out)
        print(f"Wrote {len(files)} files to {args.out} (from {args.ref})")
        return 0

    print(
        f"Resolved {len(files)} files from {args.ref}. Pass --out, --list, or --check."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
