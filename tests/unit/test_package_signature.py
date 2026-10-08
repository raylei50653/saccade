"""The minisign signature of the package digest (#465 PR-C4).

docs/reference/native_runtime_resolved_config.md §20. Pins
``scripts/native/sign_shipping_package.py`` and ``check_shipping_package.py
tarball --pubkey`` on a synthetic package. Keys and signatures are made here
in minisign's format with Ed25519 (no secret key of the release is involved):

* the reader verifies ``ED`` (BLAKE2b-512 prehashed) and legacy ``Ed``
  signatures, and refuses a changed message, a changed trusted comment, a
  signature by another key, and malformed files;
* the trusted comment is ``package=<name> commit=<commit>
  manifest_sha256=<sha256>`` read from the tarball's MANIFEST;
* ``tarball --pubkey`` requires the fourth file and fails ``signature`` when
  the digest, the trusted comment or the key is not the expected one;
* with the ``minisign`` binary present, a key made by ``minisign -G`` and a
  signature made by ``sign_shipping_package.py sign`` verify with both
  ``minisign -V`` and the reader, and both refuse the same tampered digest;
* the signature is optional (#546, §21): an unsigned release keeps every
  integrity check and reports ``authentication`` ``none``; a ``.minisig``
  without ``--pubkey`` fails ``release_set``; a bad signature never reports
  the publisher as authenticated.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

REPO = Path(__file__).resolve().parents[2]
NATIVE = REPO / "scripts/native"
INSTALLER = REPO / "shipping/package/install.sh"
NAME = "saccade-0.0.0-linux-x86_64-cu13.0-trt10.16-sm120-glibc2.39"
COMMIT = "1" * 40
MINISIGN = shutil.which("minisign")


def _load(name: str) -> ModuleType:
    sys.path.insert(0, str(NATIVE))
    spec = importlib.util.spec_from_file_location(name, NATIVE / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


signing = _load("sign_shipping_package")
pkg = _load("check_shipping_package")
builder = _load("build_shipping_package")


def _keypair(seed: int) -> tuple[Ed25519PrivateKey, bytes, str]:
    sk = Ed25519PrivateKey.from_private_bytes(bytes([seed]) * 32)
    key_id = bytes([seed, 2, 3, 4, 5, 6, 7, 8])
    raw = sk.public_key().public_bytes_raw()
    pub = (
        "untrusted comment: test key\n"
        + base64.b64encode(b"Ed" + key_id + raw).decode()
        + "\n"
    )
    return sk, key_id, pub


def _sign(
    sk: Ed25519PrivateKey, key_id: bytes, message: bytes, tc: str, alg: bytes = b"ED"
) -> str:
    signed = (
        hashlib.blake2b(message, digest_size=64).digest() if alg == b"ED" else message
    )
    sig = sk.sign(signed)
    glob = sk.sign(sig + tc.encode())
    return (
        "untrusted comment: test signature\n"
        + base64.b64encode(alg + key_id + sig).decode()
        + f"\ntrusted comment: {tc}\n"
        + base64.b64encode(glob).decode()
        + "\n"
    )


# ---------------------------------------------------------------------------
# the reader


@pytest.mark.parametrize("alg", [b"ED", b"Ed"])
def test_reader_verifies(alg: bytes) -> None:
    sk, kid, pub = _keypair(1)
    sig = _sign(sk, kid, b"message", "tc", alg)
    assert (
        signing.verify(
            signing.parse_public_key(pub), signing.parse_signature(sig), b"message"
        )
        == []
    )


def test_reader_refuses() -> None:
    sk, kid, pub = _keypair(1)
    p = signing.parse_public_key(pub)
    sig = _sign(sk, kid, b"message", "tc")
    assert signing.verify(p, signing.parse_signature(sig), b"messagE") == [
        "the signature does not verify"
    ]
    lines = sig.splitlines()
    lines[2] = "trusted comment: other"
    assert signing.verify(p, signing.parse_signature("\n".join(lines)), b"message") == [
        "the trusted comment's signature does not verify"
    ]
    sk2, kid2, _ = _keypair(2)
    other = signing.verify(
        p, signing.parse_signature(_sign(sk2, kid2, b"message", "tc")), b"message"
    )
    assert len(other) == 3 and "key id" in other[0]
    for bad in ("", "x\ny\n", sig.replace("trusted comment:", "comment:")):
        with pytest.raises(signing.SignatureError):
            signing.parse_signature(bad)
    with pytest.raises(signing.SignatureError):
        signing.parse_public_key("untrusted comment: k\nAAAA\n")


# ---------------------------------------------------------------------------
# a synthetic signed release


def _dist(tmp: Path) -> Path:
    tree = tmp / "tree"
    (tree / "bin").mkdir(parents=True)
    (tree / "bin/saccade_track").write_bytes(b"#!/bin/sh\n")
    (tree / "bin/saccade_track").chmod(0o755)
    files, problems = pkg.tree_entries(tree)
    assert not problems
    head = {
        "schema": pkg.MANIFEST_SCHEMA,
        "package": NAME,
        "source": {"commit": COMMIT, "commit_time": 1_700_000_000, "tree_clean": True},
        "file_count": len(files),
    }
    out = tmp / "dist"
    out.mkdir()
    builder.write_package(tree, out, head, files, INSTALLER.read_bytes())
    return out


def _sign_dist(dist: Path, seed: int = 1, tc: str | None = None) -> Path:
    sk, kid, pub = _keypair(seed)
    _, want = signing.expected_trusted_comment(dist)
    digest = dist / f"{NAME}.sha256"
    Path(f"{digest}.minisig").write_text(
        _sign(sk, kid, digest.read_bytes(), tc or want)
    )
    key = dist.parent / f"key{seed}.pub"
    key.write_text(pub)
    return key


def test_trusted_comment_reads_the_manifest(tmp_path: Path) -> None:
    dist = _dist(tmp_path)
    name, tc = signing.expected_trusted_comment(dist)
    manifest = signing.manifest_from_tarball(dist / f"{NAME}.tar.gz", NAME)
    assert name == NAME
    assert (
        tc
        == f"package={NAME} commit={COMMIT} manifest_sha256={hashlib.sha256(manifest).hexdigest()}"
    )
    assert json.loads(manifest)["package"] == NAME


@pytest.fixture
def no_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reader alone (minisign -V is exercised in the binary test below)."""
    monkeypatch.setattr(signing, "minisign_verify", lambda *a: (0, ""))


def test_signed_digest_verifies(tmp_path: Path, no_binary: None) -> None:
    dist = _dist(tmp_path)
    key = _sign_dist(dist)
    assert signing.check_signed_digest(dist, key)["problems"] == []


def test_signed_digest_refuses(tmp_path: Path, no_binary: None) -> None:
    dist = _dist(tmp_path)
    key = _sign_dist(dist)
    digest = dist / f"{NAME}.sha256"
    good = digest.read_bytes()
    digest.write_bytes(good.replace(b"a", b"b", 1) if b"a" in good else good + b"\n")
    assert signing.check_signed_digest(dist, key)["problems"] == [
        "reader: the signature does not verify"
    ]
    digest.write_bytes(good)

    other = _sign_dist(_dist(tmp_path / "o"), seed=2)
    assert any(
        "key id" in p for p in signing.check_signed_digest(dist, other)["problems"]
    )

    sig = Path(f"{digest}.minisig")
    sig.unlink()
    _sign_dist(dist, tc=f"package={NAME} commit={COMMIT} manifest_sha256={'0' * 64}")
    problems = signing.check_signed_digest(dist, key)["problems"]
    assert len(problems) == 1 and problems[0].startswith("trusted comment")


def _tarball_report(dist: Path, tmp: Path, pubkey: Path | None) -> dict:
    report = tmp / "package.json"
    argv = ["tarball", "--dist", str(dist), "--report", str(report)]
    if pubkey:
        argv += ["--pubkey", str(pubkey)]
    pkg.main(argv)
    return json.loads(report.read_text())


def test_tarball_check_release_set_and_signature(
    tmp_path: Path, no_binary: None
) -> None:
    dist = _dist(tmp_path)
    unsigned = _tarball_report(dist, tmp_path, None)
    assert (
        unsigned["checks"]["release_set"]["pass"]
        and "signature" not in unsigned["checks"]
    )
    assert unsigned["signed"] is False

    key = _sign_dist(dist)
    signed = _tarball_report(dist, tmp_path, key)
    assert (
        signed["checks"]["release_set"]["pass"]
        and signed["checks"]["signature"]["pass"]
    )
    assert signed["signed"] is True
    # the synthetic tree is not the real one, so the other checks fail as before
    assert not signed["pass"]

    Path(dist / f"{NAME}.sha256.minisig").unlink()
    missing = _tarball_report(dist, tmp_path, key)
    assert not missing["checks"]["release_set"]["pass"]
    assert not missing["checks"]["signature"]["pass"]


# ---------------------------------------------------------------------------
# optional signature (#546, §21)

INTEGRITY_CHECKS = {
    "release_set",
    "package_digest",
    "installer_exact",
    "tar_members",
    "manifest_exact",
    "pinned_tree",
    "metadata",
}


def test_unsigned_release_keeps_every_integrity_check(
    tmp_path: Path, no_binary: None
) -> None:
    dist = _dist(tmp_path)
    report = _tarball_report(dist, tmp_path, None)
    assert set(report["checks"]) == INTEGRITY_CHECKS
    assert report["checks"]["release_set"]["pass"]
    assert report["checks"]["package_digest"]["pass"]
    assert report["authentication"] == {
        "method": "none",
        "publisher_authenticated": False,
        "reading": "integrity only; the publisher is not authenticated",
    }


def test_unsigned_release_still_refuses_a_changed_digest(
    tmp_path: Path, no_binary: None
) -> None:
    dist = _dist(tmp_path)
    digest = dist / f"{NAME}.sha256"
    digest.write_text("0" * 64 + digest.read_text()[64:])
    report = _tarball_report(dist, tmp_path, None)
    assert not report["checks"]["package_digest"]["pass"]
    assert not report["pass"]
    assert report["authentication"]["publisher_authenticated"] is False


def test_signature_without_pubkey_is_not_ignored(
    tmp_path: Path, no_binary: None
) -> None:
    dist = _dist(tmp_path)
    _sign_dist(dist)
    report = _tarball_report(dist, tmp_path, None)
    problems = report["checks"]["release_set"]["problems"]
    assert any("present but no --pubkey" in p for p in problems)
    assert "signature" not in report["checks"]
    assert report["authentication"]["method"] == "none"
    assert not report["pass"]


def test_bad_signature_is_never_authenticated(tmp_path: Path, no_binary: None) -> None:
    dist = _dist(tmp_path)
    key = _sign_dist(dist)
    good = _tarball_report(dist, tmp_path, key)
    assert good["checks"]["signature"]["pass"]
    assert good["authentication"]["method"] == "minisign"
    assert good["authentication"]["publisher_authenticated"] is True

    other = _sign_dist(_dist(tmp_path / "o"), seed=2)
    wrong_key = _tarball_report(dist, tmp_path, other)
    assert not wrong_key["checks"]["signature"]["pass"]
    assert wrong_key["authentication"] == {
        "method": "minisign",
        "publisher_authenticated": False,
        "reading": "a signature was requested and did not verify",
    }

    sig = dist / f"{NAME}.sha256.minisig"
    lines = sig.read_text().splitlines()
    lines[2] = lines[2][:-1] + ("0" if lines[2][-1] != "0" else "1")
    sig.write_text("\n".join(lines) + "\n")
    tampered = _tarball_report(dist, tmp_path, key)
    assert not tampered["checks"]["signature"]["pass"]
    assert tampered["authentication"]["publisher_authenticated"] is False


def test_authentication_reading() -> None:
    assert pkg.authentication(False, None)["publisher_authenticated"] is False
    assert pkg.authentication(False, True)["method"] == "none"
    assert pkg.authentication(True, None)["publisher_authenticated"] is False
    assert pkg.authentication(True, False)["publisher_authenticated"] is False
    assert pkg.authentication(True, True)["publisher_authenticated"] is True


# ---------------------------------------------------------------------------
# the minisign binary


@pytest.mark.skipif(not MINISIGN, reason="needs minisign")
def test_minisign_binary_and_reader_agree(tmp_path: Path) -> None:
    dist = _dist(tmp_path)
    pub, sec = tmp_path / "test.pub", tmp_path / "test.key"
    subprocess.run(
        [MINISIGN, "-G", "-W", "-p", str(pub), "-s", str(sec)],
        check=True,
        capture_output=True,
    )
    r = subprocess.run(
        [
            sys.executable,
            str(NATIVE / "sign_shipping_package.py"),
            "sign",
            "--dist",
            str(dist),
            "--secret-key",
            str(sec),
        ],
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    assert r.returncode == 0, r.stderr
    res = signing.check_signed_digest(dist, pub)
    assert res["problems"] == [] and res["minisign_exit"] == 0

    digest = dist / f"{NAME}.sha256"
    digest.write_bytes(digest.read_bytes() + b"\n")
    res = signing.check_signed_digest(dist, pub)
    assert res["minisign_exit"] != 0
    assert "reader: the signature does not verify" in res["problems"]


def test_pre_pr_c4_manifest_keeps_its_reading() -> None:
    """A source commit before PR-C4 (no licence audit) derives the PR-C3
    MANIFEST reading, byte for byte (the PR-C3 package's MANIFEST)."""
    assert hashlib.sha256(pkg.READING_PRE_C4.encode()).hexdigest() == (
        hashlib.sha256(
            (
                "The Saccade native tracker for Linux x86_64 (#465 Phase C, "
                "docs/reference/native_runtime_resolved_config.md \u00a719): the shipping "
                "tree with its bundled third-party set. Every file of the package "
                "is listed below with its sha256, size and mode; the installer "
                "refuses a tree that differs. Not a signature: the package digest "
                "<name>.sha256 is checked by the installer, signing is PR-C4."
            ).encode()
        ).hexdigest()
    )
    assert pkg.READING != pkg.READING_PRE_C4
