"""The shipping package: MANIFEST format, builder, installer (#465 PR-C3).

docs/reference/native_runtime_resolved_config.md §19. The real package is
3.7 GiB and its acceptance runs in a clean container; these pin the logic on a
synthetic tree in the ordinary pytest job:

* ``scripts/native/check_shipping_package.py``: the MANIFEST's one canonical
  form (one file object per line, the form ``install.sh`` parses), path and
  mode rules, the package name, the tar member rules, the installed-tree
  match used by ``check_shipping_bundle.py static --manifest``;
* ``scripts/native/build_shipping_package.py``: the tarball is deterministic
  (same input, same bytes; no name or time in the gzip header; owner 0);
* ``shipping/package/install.sh`` (run with ``dash`` when present, else
  ``sh``): installs, and ``--verify`` passes; a corrupt tarball, a changed,
  extra or missing file, a symlink or ``..`` member, a renamed package, a
  malformed MANIFEST line, an existing target (directory, empty directory,
  file, symlink) and a prefix with ``:`` all fail with the target never
  created (or left as it was) and no staging directory left behind;
* the unsigned local install (#546, §21): the installer never reads a
  signature, so an unsigned package installs through the same checks, a
  ``.minisig`` (even a broken one) changes nothing, and its output claims no
  authentication.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
NATIVE = REPO / "scripts/native"
INSTALLER = REPO / "shipping/package/install.sh"
SH = shutil.which("dash") or shutil.which("sh")
NAME = "saccade-0.0.0-linux-x86_64-cu13.0-trt10.16-sm120-glibc2.39"
MTIME = 1_700_000_000


def _load(name: str) -> ModuleType:
    sys.path.insert(0, str(NATIVE))
    spec = importlib.util.spec_from_file_location(name, NATIVE / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pkg = _load("check_shipping_package")
builder = _load("build_shipping_package")


def _tree(root: Path) -> Path:
    files = {
        "bin/saccade_track": (b"#!/bin/sh\nexit 0\n", 0o755),
        "lib/vendor/libx.so.1": (b"\x7fELF fake object" * 100, 0o644),
        "lib/vendor/liby.so.2": (b"another object", 0o755),
        "share/saccade/configs/a.json": (b'{"a": 1}\n', 0o644),
        "licenses/x/License.txt": (b"license text\n", 0o644),
    }
    for rel, (data, mode) in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        p.chmod(mode)
    return root


def _head(n: int, name: str = NAME) -> dict:
    return {
        "schema": pkg.MANIFEST_SCHEMA,
        "package": name,
        "source": {"commit": "0" * 40, "commit_time": MTIME, "tree_clean": True},
        "pins": {"third_party_set": {"repo_file": "shipping/third_party_set.json"}},
        "file_count": n,
    }


def _package(tmp: Path, tree: Path | None = None, name: str = NAME) -> Path:
    tree = tree or _tree(tmp / "tree")
    files, problems = pkg.tree_entries(tree)
    assert not problems
    out = tmp / "dist"
    out.mkdir()
    builder.write_package(
        tree, out, _head(len(files), name), files, INSTALLER.read_bytes()
    )
    return out


def _install(dist: Path, target: Path, name: str = NAME) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            SH,
            str(dist / f"{name}.install.sh"),
            str(dist / f"{name}.tar.gz"),
            str(target),
        ],
        capture_output=True,
        text=True,
    )


def _staging_left(parent: Path) -> list[str]:
    return [p.name for p in parent.iterdir() if p.name.startswith(".saccade-install.")]


def _members(tarball: Path) -> list[tuple[tarfile.TarInfo, bytes | None]]:
    out = []
    with tarfile.open(tarball, "r:gz") as tf:
        for ti in tf:
            fo = tf.extractfile(ti) if ti.isreg() else None
            out.append((ti, fo.read() if fo else None))
    return out


def _repack(
    dist: Path, members: list[tuple[tarfile.TarInfo, bytes | None]], name: str = NAME
) -> None:
    """Rewrite the tarball from `members` and its digest to match: a package
    whose digest is consistent but whose content is not what MANIFEST says."""
    tarball = dist / f"{name}.tar.gz"
    with tarfile.open(tarball, "w:gz", format=tarfile.USTAR_FORMAT) as tf:
        for ti, data in members:
            tf.addfile(ti, io.BytesIO(data) if data is not None else None)
    _redigest(dist, name)


def _redigest(dist: Path, name: str = NAME) -> None:
    lines = []
    for f in (f"{name}.tar.gz", f"{name}.install.sh"):
        lines.append(f"{hashlib.sha256((dist / f).read_bytes()).hexdigest()}  {f}\n")
    (dist / f"{name}.sha256").write_text("".join(lines))


# ---------------------------------------------------------------------------
# MANIFEST format


def _doc(files: list[dict]) -> dict:
    return {**_head(len(files)), "files": files}


def _f(path: str, mode: str = "0644") -> dict:
    return {"path": path, "sha256": "a" * 64, "bytes": 3, "mode": mode}


def test_manifest_round_trip_and_one_file_per_line() -> None:
    data = pkg.manifest_bytes(_doc([_f("bin/x", "0755"), _f("lib/y.so")]))
    doc, problems = pkg.parse_manifest(data)
    assert problems == [] and doc["file_count"] == 2
    lines = data.decode().splitlines()
    assert (
        '    {"path": "bin/x", "sha256": "'
        + "a" * 64
        + '", "bytes": 3, "mode": "0755"},'
        in lines
    )
    assert (
        '  "file_count": 2,' in lines
        and '  "schema": "saccade.shipping_manifest/v1",' in lines
    )


@pytest.mark.parametrize(
    "files, problem",
    [
        ([_f("../x")], "bad path"),
        ([_f(".hidden")], "bad path"),
        ([_f("a b")], "bad path"),
        ([_f("a//b")], "bad path"),
        ([_f("x", "0777")], "mode"),
        ([_f("MANIFEST.json")], "lists itself"),
        ([_f("b"), _f("a")], "not sorted"),
        ([_f("a"), _f("a")], "not sorted"),
    ],
)
def test_manifest_rejects(files: list[dict], problem: str) -> None:
    data = pkg.manifest_bytes(_doc(files))
    _, problems = pkg.parse_manifest(data)
    assert any(problem in p for p in problems), problems


def test_manifest_must_be_canonical() -> None:
    data = pkg.manifest_bytes(_doc([_f("a")]))
    _, problems = pkg.parse_manifest(data.replace(b'", "bytes"', b'",  "bytes"'))
    assert any("canonical" in p for p in problems)


def test_manifest_head_must_not_use_the_path_key() -> None:
    doc = _doc([_f("a")])
    doc["pins"] = {"x": {"path": "y"}}
    with pytest.raises(pkg.PackageError, match='"path"'):
        pkg.manifest_bytes(doc)


def test_package_name_reads_the_wheels() -> None:
    wheels = [
        "nvidia_cuda_runtime-13.0.96",
        "tensorrt_cu12_libs-10.16.1.11",
        "torch-2.11.0",
    ]
    assert (
        pkg.package_name("0.1.0", wheels)
        == "saccade-0.1.0-linux-x86_64-cu13.0-trt10.16-sm120-glibc2.39"
    )
    with pytest.raises(pkg.PackageError):
        pkg.package_name("0.1.0", wheels[1:])


# ---------------------------------------------------------------------------
# builder


def test_package_is_deterministic_and_well_formed(tmp_path: Path) -> None:
    tree = _tree(tmp_path / "tree")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _package(tmp_path / "a", tree)
    b = _package(tmp_path / "b", tree)
    for f in (f"{NAME}.tar.gz", f"{NAME}.install.sh", f"{NAME}.sha256"):
        assert (a / f).read_bytes() == (b / f).read_bytes()
    assert sorted(p.name for p in a.iterdir()) == sorted(
        [f"{NAME}.tar.gz", f"{NAME}.install.sh", f"{NAME}.sha256"]
    )
    assert pkg.gzip_header(a / f"{NAME}.tar.gz") == {
        "gzip": True,
        "flags": 0,
        "mtime": 0,
    }
    members, kept = pkg.read_members(a / f"{NAME}.tar.gz")
    problems, files = pkg.check_members(members, NAME)
    assert problems == []
    assert set(files) == set(pkg.tree_entries(tree)[0])
    assert (
        files["bin/saccade_track"]["mode"] == "0755"
        and files["lib/vendor/libx.so.1"]["mode"] == "0644"
    )
    assert {m["mtime"] for m in members} == {MTIME}
    _, mproblems = pkg.parse_manifest(kept[f"{NAME}/MANIFEST.json"])
    assert mproblems == []
    lines = pkg.digest_lines((a / f"{NAME}.sha256").read_text())
    assert [n for _, n in lines] == [f"{NAME}.tar.gz", f"{NAME}.install.sh"]


def test_builder_refuses_a_file_that_changes_while_packaged(tmp_path: Path) -> None:
    tree = _tree(tmp_path / "tree")
    files, _ = pkg.tree_entries(tree)
    (tree / "share/saccade/configs/a.json").write_bytes(b'{"a": 2}\n')
    out = tmp_path / "dist"
    out.mkdir()
    with pytest.raises(pkg.PackageError, match="changed while"):
        builder.write_package(
            tree, out, _head(len(files)), files, INSTALLER.read_bytes()
        )
    assert not (out / f"{NAME}.tar.gz").exists()


@pytest.mark.parametrize(
    "edit, problem",
    [
        (lambda ti: setattr(ti, "uid", 1000), "owner"),
        (lambda ti: setattr(ti, "type", tarfile.SYMTYPE), "not a file or directory"),
        (lambda ti: setattr(ti, "name", "elsewhere/x"), "outside"),
        (lambda ti: setattr(ti, "mode", 0o777), "mode"),
        (lambda ti: setattr(ti, "mtime", 1), "mtimes"),
    ],
)
def test_check_members_rejects(tmp_path: Path, edit, problem: str) -> None:
    dist = _package(tmp_path)
    members = _members(dist / f"{NAME}.tar.gz")
    ti, data = members[-1]
    edit(ti)
    _repack(dist, members)
    got, _ = pkg.check_members(pkg.read_members(dist / f"{NAME}.tar.gz")[0], NAME)
    assert any(problem in p for p in got), got


# ---------------------------------------------------------------------------
# installer

needs_sh = pytest.mark.skipif(SH is None, reason="no POSIX sh")


@needs_sh
def test_install_and_verify(tmp_path: Path) -> None:
    dist = _package(tmp_path)
    target = tmp_path / "opt" / "saccade"
    target.parent.mkdir()
    r = _install(dist, target)
    assert r.returncode == 0, r.stderr
    assert (target / "MANIFEST.json").is_file()
    assert os.access(target / "bin/saccade_track", os.X_OK)
    assert (target / "lib/vendor/libx.so.1").stat().st_mode & 0o777 == 0o644
    assert _staging_left(target.parent) == []
    assert pkg.manifest_matches_tree(target)["pass"]
    v = subprocess.run(
        [SH, str(INSTALLER), "--verify", str(target)], capture_output=True, text=True
    )
    assert v.returncode == 0, v.stderr
    # --verify sees a changed file (replaced, not written through).
    p = target / "lib/vendor/libx.so.1"
    data = p.read_bytes()
    p.unlink()
    p.write_bytes(data + b"\0")
    v = subprocess.run(
        [SH, str(INSTALLER), "--verify", str(target)], capture_output=True, text=True
    )
    assert v.returncode == 1 and "size is not" in v.stderr
    assert not pkg.manifest_matches_tree(target)["pass"]


def _assert_refused(
    tmp_path: Path, dist: Path, rc: int, message: str, name: str = NAME
) -> None:
    parent = tmp_path / "opt"
    parent.mkdir(exist_ok=True)
    target = parent / "saccade"
    r = _install(dist, target, name)
    assert r.returncode == rc, (r.returncode, r.stderr)
    assert message in r.stderr, r.stderr
    assert not target.exists() and not target.is_symlink()
    assert _staging_left(parent) == []


@needs_sh
def test_corrupt_tarball_is_refused(tmp_path: Path) -> None:
    dist = _package(tmp_path)
    t = dist / f"{NAME}.tar.gz"
    data = bytearray(t.read_bytes())
    data[len(data) // 2] ^= 0xFF
    t.write_bytes(bytes(data))
    _assert_refused(tmp_path, dist, 1, "sha256 is not the one in")


def _edit_member(dist: Path, rel: str, data: bytes | None) -> None:
    members = []
    for ti, old in _members(dist / f"{NAME}.tar.gz"):
        if ti.name == f"{NAME}/{rel}":
            if data is None:
                continue
            ti.size = len(data)
            old = data
        members.append((ti, old))
    _repack(dist, members)


@needs_sh
def test_changed_file_is_refused_even_with_a_consistent_digest(tmp_path: Path) -> None:
    dist = _package(tmp_path)
    _edit_member(
        dist,
        "lib/vendor/libx.so.1",
        b"\x7fELF fake object" * 99 + b"\x7fELF fake objecT",
    )
    _assert_refused(tmp_path, dist, 1, "lib/vendor/libx.so.1: sha256 is not")


@needs_sh
def test_missing_file_is_refused(tmp_path: Path) -> None:
    dist = _package(tmp_path)
    _edit_member(dist, "licenses/x/License.txt", None)
    _assert_refused(tmp_path, dist, 1, "not exactly MANIFEST.json's")


@needs_sh
def test_extra_file_is_refused(tmp_path: Path) -> None:
    dist = _package(tmp_path)
    members = _members(dist / f"{NAME}.tar.gz")
    extra = builder._member(f"{NAME}/share/saccade/helper.py", "0644", MTIME, 3)
    _repack(dist, [*members, (extra, b"x=1")])
    _assert_refused(tmp_path, dist, 1, "share/saccade/helper.py")


@needs_sh
def test_symlink_member_is_refused(tmp_path: Path) -> None:
    dist = _package(tmp_path)
    members = [
        (ti, d)
        for ti, d in _members(dist / f"{NAME}.tar.gz")
        if not ti.name.endswith("libx.so.1")
    ]
    link = tarfile.TarInfo(f"{NAME}/lib/vendor/libx.so.1")
    link.type, link.linkname = tarfile.SYMTYPE, "liby.so.2"
    _repack(dist, [*members, (link, None)])
    _assert_refused(tmp_path, dist, 1, "")


@needs_sh
def test_dotdot_member_writes_nothing_outside(tmp_path: Path) -> None:
    dist = _package(tmp_path)
    members = _members(dist / f"{NAME}.tar.gz")
    evil = builder._member(f"{NAME}/../../escaped", "0644", MTIME, 3)
    _repack(dist, [*members, (evil, b"bad")])
    _assert_refused(tmp_path, dist, 1, "")
    assert not any(p.name == "escaped" for p in tmp_path.rglob("escaped"))


@needs_sh
def test_renamed_package_is_refused(tmp_path: Path) -> None:
    dist = _package(tmp_path)
    other = NAME.replace("0.0.0", "0.0.1")
    for suffix in (".tar.gz", ".install.sh"):
        (dist / f"{NAME}{suffix}").rename(dist / f"{other}{suffix}")
    (dist / f"{NAME}.sha256").unlink()
    _redigest(dist, other)
    _assert_refused(tmp_path, dist, 1, "does not hold exactly one directory", other)
    # The top directory renamed too: MANIFEST.json still names the old package.
    members = []
    for ti, data in _members(dist / f"{other}.tar.gz"):
        ti.name = other + ti.name[len(NAME) :]
        members.append((ti, data))
    _repack(dist, members, other)
    _assert_refused(tmp_path, dist, 1, f"package is not {other}", other)


@needs_sh
def test_malformed_manifest_line_is_refused(tmp_path: Path) -> None:
    dist = _package(tmp_path)
    m = dict(_members(dist / f"{NAME}.tar.gz"))
    (ti, data) = next((k, v) for k, v in m.items() if k.name.endswith("MANIFEST.json"))
    assert data is not None
    _edit_member(
        dist, "MANIFEST.json", data.replace(b'"mode": "0644"}', b'"mode": "0666"}', 1)
    )
    _assert_refused(tmp_path, dist, 1, "well formed")


@needs_sh
def test_missing_or_ambiguous_digest(tmp_path: Path) -> None:
    dist = _package(tmp_path)
    digest = dist / f"{NAME}.sha256"
    text = digest.read_text()
    digest.write_text(text + text.splitlines()[0] + "\n")
    _assert_refused(tmp_path, dist, 1, "exactly once")
    digest.unlink()
    _assert_refused(tmp_path, dist, 2, "no package digest")


@needs_sh
@pytest.mark.parametrize("kind", ["dir", "empty_dir", "file", "symlink", "dangling"])
def test_existing_target_is_left_alone(tmp_path: Path, kind: str) -> None:
    dist = _package(tmp_path)
    parent = tmp_path / "opt"
    parent.mkdir()
    target = parent / "saccade"
    if kind == "dir":
        target.mkdir()
        (target / "keep").write_text("mine")
    elif kind == "empty_dir":
        target.mkdir()
    elif kind == "file":
        target.write_text("mine")
    elif kind == "symlink":
        (parent / "elsewhere").mkdir()
        target.symlink_to(parent / "elsewhere")
    else:
        target.symlink_to(parent / "nowhere")
    before = sorted(
        (p.relative_to(parent).as_posix(), p.is_symlink()) for p in parent.rglob("*")
    )
    r = _install(dist, target)
    assert r.returncode == 2 and "exists; nothing was changed" in r.stderr
    after = sorted(
        (p.relative_to(parent).as_posix(), p.is_symlink()) for p in parent.rglob("*")
    )
    assert after == before
    if kind == "dir":
        assert (target / "keep").read_text() == "mine"


@needs_sh
def test_prefix_with_a_colon_is_refused(tmp_path: Path) -> None:
    dist = _package(tmp_path)
    parent = tmp_path / "a:b"
    parent.mkdir()
    r = _install(dist, parent / "saccade")
    assert r.returncode == 2 and "must not contain" in r.stderr
    assert list(parent.iterdir()) == []


@needs_sh
def test_unsigned_local_install_claims_no_authentication(tmp_path: Path) -> None:
    """#546 §21: signing is optional and the installer is integrity-only. It
    installs an unsigned package through the same checks, a signature file
    next to the package (here a broken one) changes nothing, and nothing it
    prints claims a signature or an authenticated publisher."""
    assert "minisig" not in INSTALLER.read_text()
    outs = []
    for signed in (False, True):
        work = tmp_path / ("b" if signed else "a")
        work.mkdir()
        dist = _package(work)
        if signed:
            (dist / f"{NAME}.sha256.minisig").write_text("not a signature\n")
        target = work / "opt" / "saccade"
        target.parent.mkdir()
        r = _install(dist, target)
        assert r.returncode == 0, r.stderr
        assert pkg.manifest_matches_tree(target)["pass"]
        out = r.stderr.replace(str(work), "<work>")
        assert "sign" not in out.lower() and "authentic" not in out.lower(), out
        outs.append(out)
    assert re.sub(r"saccade-install\.\w+", "S", outs[0]) == re.sub(
        r"saccade-install\.\w+", "S", outs[1]
    )


def test_installer_has_no_test_hook() -> None:
    """Only the documented arguments change what install.sh does: it reads no
    environment variable but TMPDIR (--verify's scratch) and LC_ALL (set)."""
    text = INSTALLER.read_text()
    used = set(re.findall(r"\$\{?([A-Z][A-Z0-9_]*)", text))
    assert used <= {"SCHEMA", "TMPDIR", "LC_ALL"}, used


# ---------------------------------------------------------------------------
# install-trace (strace -ff -yy of an installation)

_STG = "/install/.saccade-install.AbC123"
_GOOD = [
    f'mkdir("{_STG}", 0700) = 0',
    f'mkdir("{_STG}/x", 0777) = 0',
    f'openat(4</{_STG[1:]}/x>, "{NAME}/bin/saccade_track", O_WRONLY|O_CREAT|O_EXCL, 0755) = 5<{_STG}/x/{NAME}/bin/saccade_track>',
    f'fchmodat(AT_FDCWD</>, "{_STG}/x/{NAME}/bin/saccade_track", 0755) = 0',
    'openat(AT_FDCWD</>, "/dev/null", O_WRONLY|O_CREAT|O_TRUNC, 0666) = 3</dev/null<char 1:3>>',
    'newfstatat(AT_FDCWD</>, "/install/saccade", 0x7ffd, 0) = -1 ENOENT (No such file or directory)',
    f'renameat2(AT_FDCWD</>, "{_STG}/x/{NAME}", AT_FDCWD</>, "/install/saccade", RENAME_NOREPLACE) = 0',
    f'unlinkat(4<{_STG}>, "x", AT_REMOVEDIR) = 0',
    f'unlinkat(AT_FDCWD</>, "{_STG}", AT_REMOVEDIR) = 0',
]


def _trace(tmp_path: Path, lines: list[str]) -> dict:
    d = tmp_path / "strace"
    d.mkdir(exist_ok=True)
    for p in d.iterdir():
        p.unlink()
    (d / "i.1").write_text("\n".join(lines) + "\n+++ exited with 0 +++\n")
    report = tmp_path / "trace.json"
    rc = pkg.main(
        [
            "install-trace",
            "--strace-prefix",
            str(d / "i"),
            "--target",
            "/install/saccade",
            "--package",
            NAME,
            "--report",
            str(report),
        ]
    )
    r = json.loads(report.read_text())
    assert rc == (0 if r["pass"] else 1)
    return {k: c["pass"] for k, c in r["checks"].items()}


def test_install_trace_accepts_the_staged_install(tmp_path: Path) -> None:
    assert all(_trace(tmp_path, _GOOD).values())


@pytest.mark.parametrize(
    "edit, failing",
    [
        # A write into the target before the rename (the PR-12 install shape).
        (
            lambda ls: [
                *ls[:2],
                'openat(AT_FDCWD</>, "/install/saccade/bin/x", O_WRONLY|O_CREAT, 0644) = 3</install/saccade/bin/x>',
                *ls[2:],
            ],
            "target_only_by_one_noreplace_rename",
        ),
        # A failed attempt counts too.
        (
            lambda ls: [
                *ls,
                'mkdir("/install/saccade", 0755) = -1 EEXIST (File exists)',
            ],
            "target_only_by_one_noreplace_rename",
        ),
        # A rename that may replace.
        (
            lambda ls: [x.replace("RENAME_NOREPLACE", "0") for x in ls],
            "target_only_by_one_noreplace_rename",
        ),
        (
            lambda ls: [x for x in ls if not x.startswith("renameat2")],
            "target_only_by_one_noreplace_rename",
        ),
        (
            lambda ls: [
                *ls,
                'openat(AT_FDCWD</>, "/etc/passwd", O_WRONLY) = 3</etc/passwd>',
            ],
            "other_mutations_in_staging",
        ),
        (
            lambda ls: [x for x in ls if f'"{_STG}", AT_REMOVEDIR' not in x],
            "staging_removed",
        ),
        (lambda ls: [*ls, 'mkdir("relative", 0755) = 0'], "trace_complete"),
        (lambda ls: [*ls, 'unlinkat(4, "y", 0) = 0'], "trace_complete"),
        (
            lambda ls: [*ls, 'mkdir("/install/.saccade-install.ZzZ999", 0700) = 0'],
            "one_staging_directory",
        ),
    ],
)
def test_install_trace_rejects(tmp_path: Path, edit, failing: str) -> None:
    got = _trace(tmp_path, edit(list(_GOOD)))
    assert got[failing] is False, got


@needs_sh
def test_leftover_staging_is_reported_not_removed(tmp_path: Path) -> None:
    dist = _package(tmp_path)
    parent = tmp_path / "opt"
    parent.mkdir()
    left = parent / ".saccade-install.Left01"
    left.mkdir()
    (left / "partial").write_text("x")
    r = _install(dist, parent / "saccade")
    assert r.returncode == 0, r.stderr
    assert f"note: {left} is left from another installation" in r.stderr
    assert (left / "partial").read_text() == "x"
    assert _staging_left(parent) == [left.name]
