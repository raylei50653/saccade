#!/usr/bin/env python3
# status: active
"""Sign the shipping package's digest with minisign (#465 Phase C PR-C4).

docs/reference/native_runtime_resolved_config.md §20; owner decision C-D5
(minisign). The signed file is ``<name>.sha256`` (the package digest, which
names the tarball and the installer); the signature is
``<name>.sha256.minisig`` next to it, a fourth release file. Signing is
optional for a local-only package (#546, §21): an unsigned package keeps every
integrity check and is not authenticated. The trusted comment, which minisign
signs too, is

    package=<name> commit=<source commit> manifest_sha256=<sha256 of MANIFEST.json>

read from the tarball's MANIFEST, so the signature also vouches for the
``MANIFEST.json`` an installed tree carries (``install.sh --verify`` trusts
it). The user verifies before installing (``minisign -Vm <name>.sha256 -p
minisign.pub``, then ``sha256sum -c``); the installer does not change.

The release key is the owner's: it is generated and kept outside this
repository, and only ``shipping/package/minisign.pub`` is committed. This tool
calls ``minisign`` for signing (it prompts for the key's password); it never
reads the secret key itself.

This module also holds an independent reader of minisign public keys and
signatures (Ed25519; ``ED`` = BLAKE2b-512 prehashed, ``Ed`` = legacy), which
``check_shipping_package.py tarball --pubkey`` uses next to ``minisign -V``.

Subcommands::

    sign_shipping_package.py trusted-comment --dist DIST
    sign_shipping_package.py sign --dist DIST --secret-key KEY
    sign_shipping_package.py verify --dist DIST --pubkey minisign.pub

Exit 0: done / verified; 1: verification fails; 2: error.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

MANIFEST = "MANIFEST.json"
SIG_SUFFIX = ".minisig"


class SignatureError(Exception):
    pass


@dataclass(frozen=True)
class PublicKey:
    key_id: bytes
    key: bytes


@dataclass(frozen=True)
class Signature:
    algorithm: bytes
    key_id: bytes
    signature: bytes
    trusted_comment: str
    global_signature: bytes


def _b64(line: str, size: int, what: str) -> bytes:
    try:
        raw = base64.b64decode(line.strip(), validate=True)
    except ValueError as exc:
        raise SignatureError(f"{what}: not base64") from exc
    if len(raw) != size:
        raise SignatureError(f"{what}: {len(raw)} bytes, expected {size}")
    return raw


def parse_public_key(text: str) -> PublicKey:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) != 2 or not lines[0].startswith("untrusted comment:"):
        raise SignatureError("public key: expected a comment line and a key line")
    raw = _b64(lines[1], 42, "public key")
    if raw[:2] != b"Ed":
        raise SignatureError(f"public key: algorithm {raw[:2]!r} is not Ed")
    return PublicKey(key_id=raw[2:10], key=raw[10:])


def parse_signature(text: str) -> Signature:
    lines = text.splitlines()
    if (
        len(lines) != 4
        or not lines[0].startswith("untrusted comment:")
        or not lines[2].startswith("trusted comment: ")
    ):
        raise SignatureError("signature: expected four lines (minisign format)")
    raw = _b64(lines[1], 74, "signature")
    if raw[:2] not in (b"ED", b"Ed"):
        raise SignatureError(f"signature: algorithm {raw[:2]!r}")
    return Signature(
        algorithm=raw[:2],
        key_id=raw[2:10],
        signature=raw[10:],
        trusted_comment=lines[2][len("trusted comment: ") :],
        global_signature=_b64(lines[3], 64, "global signature"),
    )


def verify(pub: PublicKey, sig: Signature, message: bytes) -> list[str]:
    """Problems with `sig` over `message` under `pub`; empty if it verifies."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    key = Ed25519PublicKey.from_public_bytes(pub.key)
    bad = []
    if sig.key_id != pub.key_id:
        bad.append(
            f"key id {sig.key_id[::-1].hex().upper()} is not the public key's {pub.key_id[::-1].hex().upper()}"
        )
    signed = (
        hashlib.blake2b(message, digest_size=64).digest()
        if sig.algorithm == b"ED"
        else message
    )
    try:
        key.verify(sig.signature, signed)
    except InvalidSignature:
        bad.append("the signature does not verify")
    try:
        key.verify(sig.global_signature, sig.signature + sig.trusted_comment.encode())
    except InvalidSignature:
        bad.append("the trusted comment's signature does not verify")
    return bad


def release_name(dist: Path) -> str:
    tars = sorted(p.name for p in dist.glob("*.tar.gz"))
    if len(tars) != 1:
        raise SignatureError(f"{dist}: expected one .tar.gz, found {tars}")
    return tars[0][: -len(".tar.gz")]


def manifest_from_tarball(tarball: Path, name: str) -> bytes:
    with tarfile.open(tarball, "r:gz") as tf:
        try:
            member = tf.getmember(f"{name}/{MANIFEST}")
        except KeyError as exc:
            raise SignatureError(f"{tarball} has no {name}/{MANIFEST}") from exc
        f = tf.extractfile(member)
        if f is None:
            raise SignatureError(f"{name}/{MANIFEST} is not a regular file")
        return f.read()


def trusted_comment(name: str, manifest: bytes) -> str:
    import json

    doc = json.loads(manifest)
    commit = doc.get("source", {}).get("commit", "")
    if doc.get("package") != name or not commit:
        raise SignatureError(
            f"MANIFEST names package {doc.get('package')!r}, commit {commit!r}; expected {name}"
        )
    return (
        f"package={name} commit={commit} "
        f"manifest_sha256={hashlib.sha256(manifest).hexdigest()}"
    )


def expected_trusted_comment(dist: Path) -> tuple[str, str]:
    name = release_name(dist)
    return name, trusted_comment(
        name, manifest_from_tarball(dist / f"{name}.tar.gz", name)
    )


def minisign_verify(pubkey: Path, message: Path, sig: Path) -> tuple[int, str]:
    exe = shutil.which("minisign")
    if exe is None:
        return 127, "minisign not found"
    r = subprocess.run(
        [exe, "-V", "-p", str(pubkey), "-m", str(message), "-x", str(sig)],
        capture_output=True,
        text=True,
    )
    return r.returncode, (r.stdout + r.stderr).strip()


def check_signed_digest(dist: Path, pubkey: Path) -> dict[str, object]:
    """The signature checks of a release set: minisign -V, this module's
    reader, and the trusted comment against the tarball's MANIFEST."""
    name, want_tc = expected_trusted_comment(dist)
    digest = dist / f"{name}.sha256"
    sig_path = dist / f"{name}.sha256{SIG_SUFFIX}"
    bad: list[str] = []
    rc, out = minisign_verify(pubkey, digest, sig_path)
    if rc != 0:
        bad.append(f"minisign -V exit {rc}: {out[-300:]}")
    got_tc = None
    try:
        sig = parse_signature(sig_path.read_text())
        got_tc = sig.trusted_comment
        bad += [
            f"reader: {p}"
            for p in verify(
                parse_public_key(pubkey.read_text()), sig, digest.read_bytes()
            )
        ]
    except (OSError, SignatureError, UnicodeDecodeError) as exc:
        bad.append(f"reader: {exc}")
    if got_tc != want_tc:
        bad.append(f"trusted comment {got_tc!r} is not {want_tc!r}")
    return {
        "problems": bad,
        "minisign_exit": rc,
        "trusted_comment": got_tc,
        "pubkey_sha256": hashlib.sha256(pubkey.read_bytes()).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trusted-comment")
    t.add_argument("--dist", type=Path, required=True)
    s = sub.add_parser("sign")
    s.add_argument("--dist", type=Path, required=True)
    s.add_argument("--secret-key", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--dist", type=Path, required=True)
    v.add_argument("--pubkey", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "trusted-comment":
            print(expected_trusted_comment(args.dist)[1])
            return 0
        if args.cmd == "sign":
            name, tc = expected_trusted_comment(args.dist)
            digest = args.dist / f"{name}.sha256"
            sig = Path(f"{digest}{SIG_SUFFIX}")
            if sig.exists():
                print(f"sign_shipping_package: {sig} exists", file=sys.stderr)
                return 2
            exe = shutil.which("minisign")
            if exe is None:
                print("sign_shipping_package: minisign not found", file=sys.stderr)
                return 2
            r = subprocess.run(
                [
                    exe,
                    "-S",
                    "-s",
                    str(args.secret_key),
                    "-m",
                    str(digest),
                    "-x",
                    str(sig),
                    "-t",
                    tc,
                    "-c",
                    f"signature from {name}.sha256",
                ]
            )
            if r.returncode != 0:
                return 2
            print(f"{sig}\ntrusted comment: {tc}")
            return 0
        res = check_signed_digest(args.dist, args.pubkey)
        for p in res["problems"]:  # type: ignore[union-attr]
            print(p)
        print("verified" if not res["problems"] else "NOT verified")
        return 0 if not res["problems"] else 1
    except (OSError, SignatureError, tarfile.TarError, ValueError) as exc:
        print(f"sign_shipping_package: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
