#!/usr/bin/env python3
"""Checks of the shipping package (#465 Phase C PR-C3).

The package is three files (docs/reference/native_runtime_resolved_config.md
§19; a fourth, the minisign signature of the digest, since PR-C4 §20), written by ``build_shipping_package.py`` from a tree that passed
``check_shipping_bundle.py static``:

* ``<name>.tar.gz``: one directory ``<name>/`` holding the tree and
  ``MANIFEST.json`` (every file's sha256, size and mode; version pins, SM list,
  glibc baseline, source commit, attestation / lineage references, the
  runtime-identity coordinate);
* ``<name>.install.sh``: ``shipping/package/install.sh``, byte for byte;
* ``<name>.sha256``: the package digest, ``sha256sum`` lines for the other two.

This module also holds the MANIFEST format the builder writes and the
installer parses: canonical JSON with one file object per line in a fixed
form, so ``install.sh`` can read it with ``sed`` (no JSON parser in the base
system). ``check_shipping_bundle.py static --manifest`` uses it on an installed
tree.

Subcommand:

``tarball``  the release set is exactly those three files; the digest names
             both, once, with their sha256; the installer is the repository's;
             the tarball's members are one top directory, regular files and
             directories only, no links, no ``..``, owner 0, one mtime, fixed
             modes, no empty directory, the gzip header carries no name or time;
             MANIFEST.json is canonical and lists exactly the other members with
             their sha256, size and mode; the tree inside is the expected layout
             with the pinned bytes (vendor set, entrypoint, launcher, operator
             library, THIRD_PARTY.md, README.txt); the name and metadata agree with the
             repository at the recorded source commit, which was clean and
             whose runtime-identity publication was current
             (``check_runtime_identity_staleness.py --mode attested``).

             With ``--pubkey`` (PR-C4, §20) the release set is four files,
             the fourth ``<name>.sha256.minisig``, and ``signature`` checks it:
             ``minisign -V`` and ``sign_shipping_package``'s own reader both
             verify it under the key, and its trusted comment is the one the
             tarball's MANIFEST gives (package, commit, MANIFEST sha256).

``install-trace`` an installation recorded with ``strace -ff -yy``
             (``run_package_container.sh install-strace``): the target is
             touched by exactly one successful ``renameat2(<staging>/x/<name>,
             TARGET, RENAME_NOREPLACE)`` and by no other mutating call, even a
             failed one; every other mutating call (open for writing, mkdir,
             unlink, rename, link, chmod, chown, utimens, truncate) is inside
             the one staging directory (or /dev/null); the staging directory is
             removed. A mutating call whose path cannot be resolved fails.

Usage::

    check_shipping_package.py tarball --dist DIST --report package.json [--pubkey minisign.pub]
    check_shipping_package.py install-trace --strace-prefix DIR/i \
        --target /install/saccade --report trace.json

Exit 0: every check passes; 1: a check fails (named in the report); 2: error.
"""
# status: diagnostic

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_shipping_tree as g2  # noqa: E402
import sign_shipping_package as signing  # noqa: E402

SCHEMA = "saccade.shipping_package_check/v1"
MANIFEST_SCHEMA = "saccade.shipping_manifest/v1"
MANIFEST = "MANIFEST.json"
REPO = Path(__file__).resolve().parents[2]
INSTALLER_SOURCE = REPO / "shipping/package/install.sh"
LICENSE_AUDIT = REPO / "shipping/license_audit.json"
# The MANIFEST "reading" as of a source commit: a commit before PR-C4 (no
# licence audit) keeps the PR-C3 text, so its MANIFEST still derives exactly.
READING_PRE_C4 = (
    "The Saccade native tracker for Linux x86_64 (#465 Phase C, "
    "docs/reference/native_runtime_resolved_config.md §19): the shipping "
    "tree with its bundled third-party set. Every file of the package "
    "is listed below with its sha256, size and mode; the installer "
    "refuses a tree that differs. Not a signature: the package digest "
    "<name>.sha256 is checked by the installer, signing is PR-C4."
)
READING = (
    "The Saccade native tracker for Linux x86_64 (#465 Phase C, "
    "docs/reference/native_runtime_resolved_config.md §19): the shipping "
    "tree with its bundled third-party set. Every file of the package "
    "is listed below with its sha256, size and mode; the installer "
    "refuses a tree that differs. Not a signature: the package digest "
    "<name>.sha256 is checked by the installer; its minisign signature "
    "<name>.sha256.minisig (PR-C4, §20) names this file's sha256 in the "
    "trusted comment."
)
RUNTIME_IDENTITY = "docs/reference/runtime_identity.generated.json"
# The GPUs the package runs on: the operator library and the backbone engine
# are sm_120 only (owner decision C-D2); the entrypoint carries more SASS.
SUPPORTED_SM = ("sm_120",)
PLATFORM = {"os": "linux", "arch": "x86_64"}
MODES = ("0644", "0755")
# A path component of a package file: no leading '.', no separator, nothing
# the installer's sed pattern does not accept.
_SEGMENT = r"[A-Za-z0-9_+-][A-Za-z0-9._+-]*"
PATH_RE = re.compile(rf"^{_SEGMENT}(/{_SEGMENT})*$")
NAME_RE = re.compile(r"^saccade-[A-Za-z0-9._+-]+$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_FILE_KEYS = ("path", "sha256", "bytes", "mode")
# The model root's files by role (the operator library is in "pins"); the
# tracked ones must equal the repository's at the source commit.
MODEL_ROOT_ROLES = {
    "resolved_config": "configs/shipping/mamba_whole_graph.resolved.json",
    "attestation": g2.ATTESTATION,
    "lineage": "models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json",
    "head": "models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.pt",
    "backbone_engine": "models/yolo/yolo26s_backbone_640_best.engine",
}
TRACKED_ROLES = ("resolved_config", "attestation")


class PackageError(Exception):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package_name(version: str, wheels: Iterable[str]) -> str:
    """saccade-<version>-linux-x86_64-cu<major.minor>-trt<major.minor>-sm120-glibc<x.y>,
    with the CUDA and TensorRT versions read from the bundled wheels."""

    def minor(prefix: str) -> str:
        hits = [w[len(prefix) :] for w in wheels if w.startswith(prefix)]
        if len(hits) != 1:
            raise PackageError(f"expected one {prefix}* wheel, found {hits}")
        return ".".join(hits[0].split(".")[:2])

    cuda = minor("nvidia_cuda_runtime-")
    trt = minor("tensorrt_cu12_libs-")
    sm = "".join(s.replace("_", "") for s in SUPPORTED_SM)
    glibc = ".".join(map(str, g2.VERSION_BASELINE["GLIBC"]))
    name = f"saccade-{version}-linux-x86_64-cu{cuda}-trt{trt}-{sm}-glibc{glibc}"
    if not NAME_RE.match(name):
        raise PackageError(f"package name {name!r} has unexpected characters")
    return name


def file_mode(st_mode: int) -> str:
    """The package mode of an installed file: 0755 if any execute bit is set."""
    return "0755" if st_mode & 0o111 else "0644"


def manifest_bytes(doc: dict[str, Any]) -> bytes:
    """The one serialization of a MANIFEST: indent-2 JSON whose last two keys
    are "file_count" and "files", with each file object on one line in the
    order path, sha256, bytes, mode. install.sh depends on this form."""
    keys = list(doc)
    if keys[:2] != ["schema", "package"] or keys[-2:] != ["file_count", "files"]:
        raise PackageError(f"MANIFEST keys out of order: {keys}")
    head = {k: v for k, v in doc.items() if k != "files"}
    text = json.dumps(head, indent=2)
    if '"path": ' in text:
        raise PackageError('a MANIFEST key other than a file entry is named "path"')
    lines = []
    for f in doc["files"]:
        if list(f) != list(_FILE_KEYS):
            raise PackageError(f"file entry keys {list(f)} != {list(_FILE_KEYS)}")
        lines.append("    " + json.dumps(f, separators=(", ", ": ")))
    return (text[:-2] + ',\n  "files": [\n' + ",\n".join(lines) + "\n  ]\n}\n").encode()


def parse_manifest(data: bytes) -> tuple[dict[str, Any], list[str]]:
    """The MANIFEST and the problems with it (empty when it is well formed and
    canonical)."""
    problems: list[str] = []
    try:
        doc = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, [f"{MANIFEST} is not JSON: {exc}"]
    if not isinstance(doc, dict):
        return {}, [f"{MANIFEST} is not an object"]
    if doc.get("schema") != MANIFEST_SCHEMA:
        problems.append(f"schema is not {MANIFEST_SCHEMA}")
    files = doc.get("files")
    if not isinstance(files, list):
        return doc, problems + ["no files list"]
    for f in files:
        if not isinstance(f, dict) or list(f) != list(_FILE_KEYS):
            problems.append(f"malformed file entry {f!r}")
            continue
        if not isinstance(f["path"], str) or not PATH_RE.match(f["path"]):
            problems.append(f"bad path {f['path']!r}")
        if f["path"] == MANIFEST:
            problems.append(f"{MANIFEST} lists itself")
        if not isinstance(f["sha256"], str) or not _SHA.match(f["sha256"]):
            problems.append(f"{f['path']}: bad sha256")
        if type(f["bytes"]) is not int or f["bytes"] < 0:
            problems.append(f"{f['path']}: bad size")
        if f["mode"] not in MODES:
            problems.append(f"{f['path']}: mode {f['mode']!r} is not one of {MODES}")
    paths = [f.get("path") for f in files if isinstance(f, dict)]
    if paths != sorted(set(paths), key=str):
        problems.append("file paths are not sorted and unique")
    if doc.get("file_count") != len(files):
        problems.append(f"file_count {doc.get('file_count')} != {len(files)} entries")
    if not problems:
        try:
            if manifest_bytes(doc) != data:
                problems.append(f"{MANIFEST} is not in the canonical form")
        except PackageError as exc:
            problems.append(str(exc))
    return doc, problems


def tree_entries(root: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Regular files under `root` as manifest entries (without MANIFEST.json),
    and every other non-directory entry or empty directory as a problem."""
    files: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        st = p.lstat()
        if stat.S_ISDIR(st.st_mode):
            if not any(p.iterdir()):
                problems.append(f"{rel}: empty directory")
            continue
        if not stat.S_ISREG(st.st_mode):
            problems.append(f"{rel}: not a regular file or directory")
            continue
        if rel == MANIFEST:
            continue
        files[rel] = {
            "path": rel,
            "sha256": g2.sha256_file(p),
            "bytes": st.st_size,
            "mode": file_mode(st.st_mode),
        }
    return files, problems


def manifest_matches_tree(root: Path) -> dict[str, Any]:
    """An installed tree against its own MANIFEST.json: canonical, and exactly
    the tree's other files with their sha256, size and mode (modes as the
    installer sets them: files 0644 / 0755, directories 0755)."""
    m = root / MANIFEST
    if not m.is_file() or m.is_symlink():
        return {"pass": False, "problems": [f"no {MANIFEST}"]}
    doc, problems = parse_manifest(m.read_bytes())
    have, tree_problems = tree_entries(root)
    problems += tree_problems
    listed = {f["path"]: f for f in doc.get("files", []) if isinstance(f, dict)}
    for rel in sorted(set(listed) | set(have)):
        if rel not in have:
            problems.append(f"{rel}: listed but not in the tree")
        elif rel not in listed:
            problems.append(f"{rel}: in the tree but not listed")
        elif listed[rel] != have[rel]:
            problems.append(f"{rel}: {have[rel]} != listed {listed[rel]}")
    for p in [root, *root.rglob("*")]:
        st = p.lstat()
        if stat.S_ISDIR(st.st_mode) and st.st_mode & 0o7777 != 0o755:
            problems.append(
                f"{p.relative_to(root).as_posix() or '.'}: directory mode {oct(st.st_mode & 0o7777)}"
            )
        elif stat.S_ISREG(st.st_mode) and st.st_mode & 0o7777 not in (0o644, 0o755):
            problems.append(
                f"{p.relative_to(root).as_posix()}: mode {oct(st.st_mode & 0o7777)}"
            )
    return {
        "pass": not problems,
        "problems": problems,
        "package": doc.get("package"),
        "files": len(listed),
    }


# ---------------------------------------------------------------------------
# tarball


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True
    ).stdout


def _git_show(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO), "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
    ).stdout


def digest_lines(text: str) -> list[tuple[str, str]]:
    """``sha256sum`` text-mode lines ``<hex>  <name>``; anything else is an error."""
    out = []
    for line in text.splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._+-]+)", line)
        if not m:
            raise PackageError(f"malformed digest line {line!r}")
        out.append((m.group(1), m.group(2)))
    return out


def gzip_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        head = f.read(10)
    if head[:3] != b"\x1f\x8b\x08":
        return {"gzip": False}
    return {
        "gzip": True,
        "flags": head[3],
        "mtime": int.from_bytes(head[4:8], "little"),
    }


def read_members(tarball: Path) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Every member's header fields and sha256 (MANIFEST.json's bytes kept)."""
    members: list[dict[str, Any]] = []
    kept: dict[str, bytes] = {}
    with tarfile.open(tarball, "r:gz") as tf:
        for ti in tf:
            m = {
                "name": ti.name,
                "type": ti.type.decode()
                if isinstance(ti.type, bytes)
                else str(ti.type),
                "mode": f"{ti.mode:04o}",
                "uid": ti.uid,
                "gid": ti.gid,
                "uname": ti.uname,
                "gname": ti.gname,
                "mtime": ti.mtime,
                "size": ti.size,
                "linkname": ti.linkname,
                "pax": dict(ti.pax_headers),
            }
            if ti.isreg():
                h = hashlib.sha256()
                fo = tf.extractfile(ti)
                assert fo is not None
                data = b""
                keep = ti.name.endswith("/" + MANIFEST) and ti.name.count("/") == 1
                while chunk := fo.read(1 << 20):
                    h.update(chunk)
                    if keep:
                        data += chunk
                m["sha256"] = h.hexdigest()
                if keep:
                    kept[ti.name] = data
            members.append(m)
    return members, kept


def check_members(
    members: list[dict[str, Any]], name: str
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Structure of the tarball's members; returns the problems and the files
    (relative to the top directory, with their manifest entry)."""
    problems: list[str] = []
    files: dict[str, dict[str, Any]] = {}
    dirs: set[str] = set()
    seen: set[str] = set()
    mtimes = {m["mtime"] for m in members}
    if len(mtimes) != 1:
        problems.append(f"members carry {len(mtimes)} different mtimes")
    for m in members:
        n = m["name"]
        if n in seen:
            problems.append(f"duplicate member {n}")
        seen.add(n)
        if n != name and not n.startswith(name + "/"):
            problems.append(f"member {n} is outside {name}/")
            continue
        rel = n[len(name) + 1 :]
        if rel and not PATH_RE.match(rel):
            problems.append(f"member {n}: path not allowed")
        if m["uid"] or m["gid"] or m["uname"] or m["gname"]:
            problems.append(f"member {n}: owner is not 0/0 without names")
        if m["linkname"] or m["pax"]:
            problems.append(f"member {n}: link name or pax header")
        if m["type"] == tarfile.DIRTYPE.decode():
            if m["mode"] != "0755":
                problems.append(f"directory {n}: mode {m['mode']}")
            dirs.add(rel)
        elif m["type"] in (tarfile.REGTYPE.decode(), tarfile.AREGTYPE.decode()):
            if m["mode"] not in MODES:
                problems.append(f"file {n}: mode {m['mode']}")
            if rel == "":
                problems.append(f"the top entry {n} is a file")
                continue
            if rel != MANIFEST:
                files[rel] = {
                    "path": rel,
                    "sha256": m["sha256"],
                    "bytes": m["size"],
                    "mode": m["mode"],
                }
            elif m["mode"] != "0644":
                problems.append(f"{MANIFEST}: mode {m['mode']}")
        else:
            problems.append(
                f"member {n}: type {m['type']!r} is not a file or directory"
            )
    if "" not in dirs:
        problems.append(f"no top directory {name}/")
    entries = set(files) | {MANIFEST}
    for d in dirs - {""}:
        if not any(e.startswith(d + "/") for e in entries):
            problems.append(f"directory {d} is empty")
    for e in entries:
        parent = e.rsplit("/", 1)[0] if "/" in e else ""
        if parent not in dirs:
            problems.append(f"{e}: parent directory {parent or name} has no member")
    return problems, files


def _check(checks: dict[str, Any], key: str, problems: list[str], **extra: Any) -> None:
    checks[key] = {"pass": not problems, "problems": problems, **extra}


def cmd_tarball(args: argparse.Namespace) -> int:
    import check_shipping_bundle as bundle

    dist: Path = args.dist.resolve()
    checks: dict[str, Any] = {}
    present = sorted(p.name for p in dist.iterdir())
    tars = [p for p in present if p.endswith(".tar.gz")]
    if len(tars) != 1:
        raise PackageError(f"{dist}: expected one .tar.gz, found {tars}")
    name = tars[0][: -len(".tar.gz")]
    tarball = dist / f"{name}.tar.gz"
    installer = dist / f"{name}.install.sh"
    digest = dist / f"{name}.sha256"
    signature = dist / f"{name}.sha256{signing.SIG_SUFFIX}"
    want = sorted(
        [tarball.name, installer.name, digest.name]
        + ([signature.name] if args.pubkey else [])
    )
    _check(
        checks,
        "release_set",
        [] if present == want else [f"{dist} holds {present}, expected {want}"],
    )

    sums = {"tarball": g2.sha256_file(tarball)}
    if installer.is_file():
        sums["installer"] = g2.sha256_file(installer)
    d_bad: list[str] = []
    try:
        lines = digest_lines(digest.read_text()) if digest.is_file() else []
        if not digest.is_file():
            d_bad.append(f"no {digest.name}")
        expect = [
            (sums.get("tarball"), tarball.name),
            (sums.get("installer"), installer.name),
        ]
        if lines != expect:
            d_bad.append(f"{digest.name} is {lines}, expected {expect}")
    except PackageError as exc:
        d_bad.append(str(exc))
    _check(checks, "package_digest", d_bad, sha256=sums)

    i_bad = []
    if (
        not installer.is_file()
        or installer.read_bytes() != args.installer_source.read_bytes()
    ):
        i_bad.append(f"{installer.name} is not {args.installer_source}")
    _check(checks, "installer_exact", i_bad)

    gz = gzip_header(tarball)
    gz_bad = []
    if not gz["gzip"] or gz["flags"] != 0 or gz["mtime"] != 0:
        gz_bad.append(f"gzip header {gz} carries a name, a time or other flags")
    members, kept = read_members(tarball)
    m_bad, files = check_members(members, name)
    _check(
        checks, "tar_members", gz_bad + m_bad, members=len(members), files=len(files)
    )

    data = kept.get(f"{name}/{MANIFEST}")
    doc, man_bad = (
        parse_manifest(data) if data is not None else ({}, [f"no {MANIFEST}"])
    )
    if doc.get("package") != name:
        man_bad.append(
            f"package {doc.get('package')!r} is not the tarball's name {name}"
        )
    listed = {f["path"]: f for f in doc.get("files", []) if isinstance(f, dict)}
    for rel in sorted(set(listed) | set(files)):
        if rel not in files:
            man_bad.append(f"{rel}: listed but not in the tarball")
        elif rel not in listed:
            man_bad.append(f"{rel}: in the tarball but not listed")
        elif listed[rel] != files[rel]:
            man_bad.append(f"{rel}: {files[rel]} != listed {listed[rel]}")
    _check(checks, "manifest_exact", man_bad, sha256=data and sha256_bytes(data))

    # The tree inside: the bundle's layout and pinned bytes.
    set_ = bundle._load(args.third_party_set)
    pin = bundle._load(args.entrypoint_pin)
    p_bad: list[str] = []
    want_files = bundle.expected_files(set_)
    if set(files) != want_files:
        p_bad.append(
            f"layout: missing {sorted(want_files - set(files))}, extra {sorted(set(files) - want_files)}"
        )
    for e in set_["entries"]:
        rel = f"{bundle.VENDOR}/{e['soname']}"
        if files.get(rel, {}).get("sha256") != e["sha256"]:
            p_bad.append(f"{rel}: not the pinned {e['sha256']}")
    if files.get(bundle.ENTRYPOINT, {}).get("sha256") != pin["sha256"]:
        p_bad.append(f"{bundle.ENTRYPOINT}: not the pinned {pin['sha256']}")
    for rel, src in (
        (bundle.LAUNCHER, args.launcher_source),
        ("licenses/THIRD_PARTY.md", bundle.NOTICE_SOURCE),
        (bundle.README, bundle.README_SOURCE),
    ):
        if files.get(rel, {}).get("sha256") != g2.sha256_file(src):
            p_bad.append(f"{rel}: not {src}")
    if files.get(bundle.LAUNCHER, {}).get("mode") != "0755":
        p_bad.append(f"{bundle.LAUNCHER}: not executable")
    if files.get(bundle.ENTRYPOINT, {}).get("mode") != "0755":
        p_bad.append(f"{bundle.ENTRYPOINT}: not executable")
    _check(checks, "pinned_tree", p_bad)

    # Metadata: the head the builder must have written at the recorded commit.
    md_bad: list[str] = []
    source = doc.get("source") if isinstance(doc.get("source"), dict) else {}
    try:
        head = manifest_head(
            commit=source.get("commit", ""),
            tree_clean=source.get("tree_clean"),
            identity_current=source.get("identity_current"),
            files=files,
        )
    except (subprocess.CalledProcessError, PackageError, KeyError, ValueError) as exc:
        head = {}
        md_bad.append(f"cannot derive the metadata at {source.get('commit')!r}: {exc}")
    for key, value in head.items():
        if doc.get(key) != value:
            md_bad.append(f"{key}: {doc.get(key)!r} != {value!r}")
    if head and name != head["package"]:
        md_bad.append(f"name {name} != {head['package']}")
    if head and head["pins"]["installer"]["sha256"] != sums.get("installer"):
        md_bad.append("the released installer is not the one at the source commit")
    if source.get("tree_clean") is not True:
        md_bad.append("built from a working tree that was not clean")
    if source.get("identity_current") is not True:
        md_bad.append(
            "the runtime-identity publication did not describe the source commit"
        )
    _check(checks, "metadata", md_bad, commit=source.get("commit"))

    if args.pubkey:
        sig = signing.check_signed_digest(dist, args.pubkey)
        _check(
            checks,
            "signature",
            sig.pop("problems"),  # type: ignore[arg-type]
            **sig,
        )

    report = {
        "schema": SCHEMA,
        "kind": "tarball",
        "dist": str(dist),
        "package": name,
        "sha256": sums,
        "signed": bool(args.pubkey),
        "checks": checks,
        "pass": all(c["pass"] for c in checks.values()),
    }
    g2._write(args.report, report)
    for key, c in checks.items():
        print(f"{key}: {'PASS' if c['pass'] else 'FAIL'}")
    return 0 if report["pass"] else 1


def manifest_head(
    commit: str,
    tree_clean: Any,
    identity_current: Any,
    files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Every MANIFEST key but "files", derived from the repository at `commit`
    and the package's files; the builder writes it and ``tarball`` recomputes
    it. Raises if the files disagree with the repository's pins at `commit`
    (vendor set, entrypoint pin, launcher, attested operator library, the
    tracked model-root files)."""
    import tomllib

    import check_shipping_bundle as bundle

    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PackageError(f"{commit!r} is not a full commit id")

    def repo(path: str) -> bytes:
        return _git_show(commit, path)

    def rel(path: Path) -> str:
        return path.relative_to(REPO).as_posix()

    def entry(path: str) -> dict[str, Any]:
        if path not in files:
            raise PackageError(f"{path} is not in the package")
        return {"file": path, "sha256": files[path]["sha256"]}

    version = tomllib.loads(repo("pyproject.toml").decode())["project"]["version"]
    set_bytes = repo(rel(bundle.THIRD_PARTY_SET))
    set_ = json.loads(set_bytes)
    pin = json.loads(repo(rel(bundle.ENTRYPOINT_PIN)))
    wheels = sorted({e["wheel"] for e in set_["entries"]})

    def wheel_version(prefix: str) -> str:
        (hit,) = [w[len(prefix) :] for w in wheels if w.startswith(prefix)]
        return hit

    mr = g2.MODEL_ROOT
    att = json.loads(repo(g2.ATTESTATION))
    op_file = f"{mr}/{att['op_library']['path']}"
    if set(MODEL_ROOT_ROLES.values()) | {att["op_library"]["path"]} != set(
        bundle.MODEL_ROOT_FILES
    ):
        raise PackageError("MODEL_ROOT_ROLES does not cover the model root")
    model_root: dict[str, Any] = {"dir": mr}
    for role, path in MODEL_ROOT_ROLES.items():
        model_root[role] = entry(f"{mr}/{path}")
        if role in TRACKED_ROLES and model_root[role]["sha256"] != sha256_bytes(
            repo(path)
        ):
            raise PackageError(f"{mr}/{path} is not {path} at {commit[:12]}")
    if entry(op_file)["sha256"] != att["op_library"]["sha256"]:
        raise PackageError(f"{op_file} is not the attested operator library")
    for e in set_["entries"]:
        if entry(f"{bundle.VENDOR}/{e['soname']}")["sha256"] != e["sha256"]:
            raise PackageError(
                f"{bundle.VENDOR}/{e['soname']} is not the pinned object"
            )
    if entry(bundle.ENTRYPOINT)["sha256"] != pin["sha256"]:
        raise PackageError(f"{bundle.ENTRYPOINT} is not the pinned entrypoint")
    launcher = repo(rel(bundle.LAUNCHER_SOURCE))
    if entry(bundle.LAUNCHER)["sha256"] != sha256_bytes(launcher):
        raise PackageError(f"{bundle.LAUNCHER} is not the launcher at {commit[:12]}")
    rid = repo(RUNTIME_IDENTITY)
    try:
        audit_bytes: bytes | None = repo(rel(LICENSE_AUDIT))
    except subprocess.CalledProcessError:
        audit_bytes = None  # a source commit before PR-C4: no "licenses" key
    head: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "package": package_name(version, wheels),
        "reading": READING if audit_bytes is not None else READING_PRE_C4,
        "version": version,
        "platform": {
            **PLATFORM,
            "loader": bundle.SYSTEM_LOADER,
            "glibc_baseline": {
                k: ".".join(map(str, v)) for k, v in g2.VERSION_BASELINE.items()
            },
        },
        "gpu": {
            "supported": list(SUPPORTED_SM),
            "entrypoint_sass": sorted(g2.SHIPPING_SASS, key=lambda s: int(s[3:])),
            "entrypoint_ptx": sorted(g2.SHIPPING_PTX),
            "why": "the operator library and the backbone engine are sm_120 only (owner decision C-D2)",
        },
        "versions": {
            "cuda_runtime": wheel_version("nvidia_cuda_runtime-"),
            "tensorrt": wheel_version("tensorrt_cu12_libs-"),
            "torch": wheel_version("torch-"),
            "cudnn": wheel_version("nvidia_cudnn_cu13-"),
            "wheels": wheels,
        },
        "source": {
            "commit": commit,
            "commit_time": int(_git("show", "-s", "--format=%ct", commit).strip()),
            "tree_clean": tree_clean,
            "identity_current": identity_current,
        },
        "pins": {
            "third_party_set": {
                "repo_file": rel(bundle.THIRD_PARTY_SET),
                "sha256": sha256_bytes(set_bytes),
                "objects": len(set_["entries"]),
                "dir": bundle.VENDOR,
            },
            "entrypoint": {
                "repo_file": rel(bundle.ENTRYPOINT_PIN),
                "file": bundle.ENTRYPOINT,
                "sha256": pin["sha256"],
            },
            "launcher": {
                "repo_file": rel(bundle.LAUNCHER_SOURCE),
                "file": bundle.LAUNCHER,
                "sha256": sha256_bytes(launcher),
            },
            "auditor": entry(bundle.AUDITOR),
            "operator_library": {
                "file": op_file,
                "sha256": att["op_library"]["sha256"],
                "attestation": f"{mr}/{g2.ATTESTATION}",
            },
            "installer": {
                "repo_file": rel(INSTALLER_SOURCE),
                "sha256": sha256_bytes(repo(rel(INSTALLER_SOURCE))),
            },
        },
        "model_root": model_root,
        "runtime_identity": {
            "repo_file": RUNTIME_IDENTITY,
            "sha256": sha256_bytes(rid),
            "coordinate": json.loads(rid)["coordinate"],
        },
    }
    if audit_bytes is not None:
        head["licenses"] = {
            "repo_file": rel(LICENSE_AUDIT),
            "sha256": sha256_bytes(audit_bytes),
            "distribution": json.loads(audit_bytes)["distribution"]["status"],
            "notice": "licenses/THIRD_PARTY.md",
        }
    head["file_count"] = len(files)
    return head


# ---------------------------------------------------------------------------
# install-trace

_LINE = re.compile(r"^(\w+)\((.*)\)\s+=\s+(-?\d+|\?)(.*)$")
_ANNOT = re.compile(r"^(AT_FDCWD|-?\d+)<(.*)>$")
_WRITE_FLAGS = re.compile(r"O_WRONLY|O_RDWR|O_CREAT|O_TRUNC|O_APPEND")
_STAGING = re.compile(r"^(.*)/\.saccade-install\.[A-Za-z0-9]{6}$")
# syscall -> how to read the paths it mutates: ("path", i) a path argument,
# ("at", i, j) dirfd i + path j, ("fd", i) the file a descriptor names.
_MUTATING: dict[str, list[tuple[Any, ...]]] = {
    "mkdir": [("path", 0)],
    "mkdirat": [("at", 0, 1)],
    "unlink": [("path", 0)],
    "rmdir": [("path", 0)],
    "unlinkat": [("at", 0, 1)],
    "rename": [("path", 0), ("path", 1)],
    "renameat": [("at", 0, 1), ("at", 2, 3)],
    "renameat2": [("at", 0, 1), ("at", 2, 3)],
    "link": [("path", 1)],
    "linkat": [("at", 2, 3)],
    "symlink": [("path", 1)],
    "symlinkat": [("at", 1, 2)],
    "chmod": [("path", 0)],
    "fchmodat": [("at", 0, 1)],
    "fchmodat2": [("at", 0, 1)],
    "chown": [("path", 0)],
    "lchown": [("path", 0)],
    "fchownat": [("at", 0, 1)],
    "truncate": [("path", 0)],
    "creat": [("path", 0)],
    "fchmod": [("fd", 0)],
    "fchown": [("fd", 0)],
    "ftruncate": [("fd", 0)],
    "fallocate": [("fd", 0)],
    "utimensat": [("at", 0, 1)],
    "utimes": [("path", 0)],
    "utime": [("path", 0)],
}


def split_args(text: str) -> list[str]:
    """strace's top-level comma-separated arguments (strings, <annotations>,
    {structs}, [arrays] kept whole)."""
    out: list[str] = []
    depth = 0
    cur = ""
    in_str = False
    i = 0
    while i < len(text):
        c = text[i]
        if in_str:
            cur += c
            if c == "\\" and i + 1 < len(text):
                cur += text[i + 1]
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            cur += c
        elif c in "([{<":
            depth += 1
            cur += c
        elif c in ")]}>":
            depth -= 1
            cur += c
        elif c == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += c
        i += 1
    if cur.strip():
        out.append(cur.strip())
    return out


def _unquote(arg: str) -> str | None:
    if len(arg) >= 2 and arg[0] == '"' and arg.endswith('"'):
        body = arg[1:-1]
        return body.encode().decode("unicode_escape") if "\\" in body else body
    return None


def _join(base: str, rel: str) -> str:
    import posixpath

    return posixpath.normpath(rel if rel.startswith("/") else f"{base}/{rel}")


def mutated_paths(name: str, args: list[str]) -> tuple[list[str], list[str]]:
    """The absolute paths a call mutates, and why any could not be resolved."""
    if name in ("open", "openat"):
        flags = args[1] if name == "open" else (args[2] if len(args) > 2 else "")
        if not _WRITE_FLAGS.search(flags):
            return [], []
        spec = [("path", 0)] if name == "open" else [("at", 0, 1)]
    elif name in _MUTATING:
        spec = _MUTATING[name]
    else:
        return [], []
    paths: list[str] = []
    unresolved: list[str] = []
    for kind, *idx in spec:
        try:
            if kind == "fd":
                m = _ANNOT.match(args[idx[0]])
                if m is None:
                    unresolved.append(f"{name}: fd {args[idx[0]]} has no target")
                else:
                    paths.append(m.group(2))
                continue
            raw = args[idx[-1]]
            if raw == "NULL" and kind == "at":
                m = _ANNOT.match(args[idx[0]])
                if m is None:
                    unresolved.append(f"{name}: fd {args[idx[0]]} has no target")
                else:
                    paths.append(m.group(2))
                continue
            rel = _unquote(raw)
            if rel is None:
                unresolved.append(f"{name}: path argument {raw!r}")
                continue
            if rel.startswith("/"):
                paths.append(_join("/", rel))
                continue
            if kind == "at":
                m = _ANNOT.match(args[idx[0]])
                if m is None:
                    unresolved.append(
                        f"{name}: dirfd {args[idx[0]]} has no target for {rel!r}"
                    )
                    continue
                paths.append(_join(m.group(2), rel))
            else:
                unresolved.append(f"{name}: relative path {rel!r} without a dirfd")
        except IndexError:
            unresolved.append(f"{name}: too few arguments {args}")
    return paths, unresolved


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def cmd_install_trace(args: argparse.Namespace) -> int:
    prefix: Path = args.strace_prefix
    logs = sorted(prefix.parent.glob(prefix.name + ".*"))
    if not logs:
        raise PackageError(f"no strace logs {prefix}.*")
    target = args.target.rstrip("/")
    calls: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for log in logs:
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith(("+++", "---")) or not line.strip():
                continue
            if "<unfinished ...>" in line or "resumed>" in line:
                incomplete.append(f"{log.name}: {line[:200]}")
                continue
            m = _LINE.match(line)
            if m is None:
                incomplete.append(f"{log.name}: {line[:200]}")
                continue
            name, argtext, ret, rest = m.groups()
            argv = split_args(argtext)
            paths, unresolved = mutated_paths(name, argv)
            if unresolved:
                incomplete.extend(f"{log.name}: {u}" for u in unresolved)
            if paths:
                calls.append(
                    {
                        "log": log.name,
                        "call": name,
                        "paths": paths,
                        "ok": ret != "?" and not ret.startswith("-"),
                        "flags": argv[4]
                        if name == "renameat2" and len(argv) > 4
                        else "",
                        "line": line[:300],
                    }
                )
    checks: dict[str, Any] = {}
    stagings = sorted(
        {
            p
            for c in calls
            for p in c["paths"]
            if _STAGING.match(p) and c["call"] in ("mkdir", "mkdirat")
        }
    )
    staging = stagings[0] if len(stagings) == 1 else None
    s_bad = [] if staging else [f"expected one staging directory, found {stagings}"]
    if staging and _STAGING.match(staging).group(1) != target.rsplit("/", 1)[0]:  # type: ignore[union-attr]
        s_bad.append(f"staging {staging} is not next to {target}")
    _check(checks, "one_staging_directory", s_bad, staging=staging)

    on_target = [c for c in calls if any(_under(p, target) for p in c["paths"])]
    t_bad: list[str] = []
    want_src = f"{staging}/x/{args.package}" if staging else None
    ok_renames = [
        c
        for c in on_target
        if c["call"] == "renameat2"
        and c["ok"]
        and c["paths"] == [want_src, target]
        and "RENAME_NOREPLACE" in c["flags"]
    ]
    if len(ok_renames) != 1:
        t_bad.append(
            f"{len(ok_renames)} renameat2({want_src}, {target}, RENAME_NOREPLACE) = 0"
        )
    for c in on_target:
        if c not in ok_renames:
            t_bad.append(f"another call on the target: {c['line']}")
    _check(checks, "target_only_by_one_noreplace_rename", t_bad, calls=len(on_target))

    o_bad = [
        c["line"]
        for c in calls
        if c not in on_target
        and not all(
            (staging and _under(p, staging)) or p == "/dev/null" for p in c["paths"]
        )
    ]
    _check(
        checks, "other_mutations_in_staging", o_bad, calls=len(calls) - len(on_target)
    )

    removed = any(
        c["ok"]
        and c["paths"] == [staging]
        and (
            c["call"] == "rmdir"
            or (c["call"] == "unlinkat" and "AT_REMOVEDIR" in c["line"])
        )
        for c in calls
    )
    _check(checks, "staging_removed", [] if removed else [f"{staging} was not removed"])
    _check(checks, "trace_complete", incomplete[:50], incomplete=len(incomplete))

    report = {
        "schema": SCHEMA,
        "kind": "install-trace",
        "strace_prefix": str(prefix),
        "logs": len(logs),
        "target": target,
        "checks": checks,
        "pass": all(c["pass"] for c in checks.values()),
    }
    g2._write(args.report, report)
    for key, c in checks.items():
        print(f"{key}: {'PASS' if c['pass'] else 'FAIL'}")
    return 0 if report["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    import check_shipping_bundle as bundle

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tarball")
    t.add_argument("--dist", type=Path, required=True)
    t.add_argument("--report", type=Path, required=True)
    t.add_argument("--third-party-set", type=Path, default=bundle.THIRD_PARTY_SET)
    t.add_argument("--entrypoint-pin", type=Path, default=bundle.ENTRYPOINT_PIN)
    t.add_argument("--launcher-source", type=Path, default=bundle.LAUNCHER_SOURCE)
    t.add_argument("--installer-source", type=Path, default=INSTALLER_SOURCE)
    t.add_argument(
        "--pubkey",
        type=Path,
        help="minisign public key: the release set must also hold <name>.sha256.minisig, which must verify (PR-C4)",
    )
    it = sub.add_parser("install-trace")
    it.add_argument("--strace-prefix", type=Path, required=True)
    it.add_argument("--target", required=True, help="TARGET as the installer saw it")
    it.add_argument("--package", required=True, help="the package name <name>")
    it.add_argument("--report", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        return {"tarball": cmd_tarball, "install-trace": cmd_install_trace}[args.cmd](
            args
        )
    except (
        PackageError,
        signing.SignatureError,
        g2.CheckError,
        OSError,
        tarfile.TarError,
    ) as exc:
        print(f"check_shipping_package: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
