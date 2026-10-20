#!/usr/bin/env python3
"""Fetch a model file from a user-chosen https URL, and verify it offline later.

Issue #610 slice 2 (ADR 028 ``#decision-610`` R4/R6;
docs/reference/model_fetch_610.md). Repo-side developer tooling, Python
standard library only, Linux. SACCADE hosts, mirrors and relays no model: the
user names the upstream URL, the expected sha256 and where that hash came from
in a source file (``saccade.model_source/v1``) and records that they accepted
the upstream terms. The runtime package does not carry this tool and the
runtime never downloads.

Subcommands::

    model_fetch.py fetch  --source SOURCE.json --dest DIR [--allow-no-expected-hash]
                          [--ca-bundle PEM] [--max-redirects N] [--timeout S]
                          [--max-bytes N]
    model_fetch.py verify --source SOURCE.json --dest DIR

``fetch`` downloads into a hidden staging file in DIR (same file system),
hashing while it writes, checks size and sha256, then publishes
``DIR/<name>`` with a hard link that refuses to replace anything and writes
``DIR/<name>.fetch.json`` (``saccade.model_fetch_record/v1``) **last**: the
record is the publication marker, a file without it is unpublished. A failure
or a signal removes the staging files; an existing file is never replaced or
modified. A rerun with the same source and a published pair checks it offline
and sends no request. A file left without its record (interrupted between the
two links) is kept: the rerun downloads again and writes the record only when
the existing bytes equal the newly verified ones.

``verify`` reads no network: it checks the published pair against the source
file and the bytes on disk, and prints a ``saccade.model_verify_report/v1``.

Policy: a source without an expected sha256 is refused before any request
unless ``--allow-no-expected-hash`` is given; the record is then
``integrity: local_only``. A matched sha256 means the bytes equal the hash the
source file supplied, nothing more: every record carries
``source_authentication: not_performed``. Neither command is a model bundle
check (Gate A / ``--check-bundle``), an identity level, a rights grant or a
benchmark claim.

Only https is fetched; a redirect must stay on https, at most --max-redirects
hops, no loop. Content-Encoding other than identity is refused (the hash is
over the delivered bytes). Proxies come from the usual environment variables.

Signed URLs: the query and fragment of every URL the tool persists or
prints (the source url, redirect hops, the final url, diagnostics) are
removed, because presigned URLs carry their tokens there; the record still
pairs with the exact source file through its sha256 (``descriptor_sha256``).
A token in the URL *path* is not recognised. The record is created 0600.

Exit: 0 fetched / already present / verified; 1 verification failed (sha256
or size mismatch, truncated, more bytes than expected; for verify: anything
not verified); 2 usage (argument errors included), invalid source, policy
refusal, unsafe or conflicting target, a file system error; 3 transport
(network, TLS, HTTP status, encoding); 4 internal error (a bug; traceback on
stderr); 128+N interrupted by signal N. stdout carries one JSON object with
``exit_code`` for every exit; only ``--help`` prints help text instead.
"""

# status: stable

from __future__ import annotations

import argparse
import datetime as dt
import errno
import fcntl
import hashlib
import http.client
import json
import os
import re
import secrets
import signal
import ssl
import stat
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import urljoin, urlsplit, urlunsplit

SOURCE_SCHEMA = "saccade.model_source/v1"
RECORD_SCHEMA = "saccade.model_fetch_record/v1"
RESULT_SCHEMA = "saccade.model_fetch_result/v1"
VERIFY_SCHEMA = "saccade.model_verify_report/v1"
RECORD_SUFFIX = ".fetch.json"
PARTIAL_MARK = ".partial-"
READING = (
    "Integrity record only: the bytes at <name> are the bytes this tool "
    "downloaded, and integrity says whether they matched the expected sha256 "
    "of the source file. Not source or publisher authentication, not a rights "
    "or licence grant, not a model identity level (VL1/VL2), not a benchmark "
    "or parity claim."
)
HASH_KINDS = ("upstream_published", "user_supplied", "none")
REDIRECTS = (301, 302, 303, 307, 308)
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
CHUNK = 1 << 20
MAX_SOURCE_BYTES = 1 << 20

EXIT_OK, EXIT_VERIFY, EXIT_USAGE, EXIT_TRANSPORT, EXIT_INTERNAL = 0, 1, 2, 3, 4


class Failure(Exception):
    """A refusal or failure with its exit code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class Interrupted(BaseException):
    """SIGTERM / SIGHUP, turned into an exception so cleanup runs."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _raise_interrupted(signum: int, _frame: Any) -> NoReturn:
    raise Interrupted(signum)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_json(data: bytes, what: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in items:
            if k in out:
                raise Failure(EXIT_USAGE, f"{what}: duplicate key {k!r}")
            out[k] = v
        return out

    def constant(name: str) -> NoReturn:
        raise Failure(EXIT_USAGE, f"{what}: {name} is not JSON")

    try:
        return json.loads(
            data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise Failure(EXIT_USAGE, f"{what}: not valid JSON ({e})") from None


def _object(v: Any, keys: set[str], what: str) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise Failure(EXIT_USAGE, f"{what} is not an object")
    if set(v) != keys:
        raise Failure(
            EXIT_USAGE, f"{what} keys {sorted(v)} are not exactly {sorted(keys)}"
        )
    return v


def _text(v: Any, what: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise Failure(EXIT_USAGE, f"{what} is not a non-empty string")
    return v


def check_https_url(url: str, what: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise Failure(EXIT_USAGE, f"{what} is not an https URL: {redact(url)}")
    if parts.username is not None or parts.password is not None:
        raise Failure(EXIT_USAGE, f"{what} carries credentials; refusing")


def redact(url: str) -> str:
    """The URL without query and fragment (signed URLs carry tokens there)."""
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, p.path, "", ""))


def scrub(text: str, url: str) -> str:
    """text with url, and its query and fragment on their own, removed."""
    p = urlsplit(url)
    text = text.replace(url, redact(url))
    for part in (p.query, p.fragment):
        if part:
            text = text.replace(part, "<redacted>")
    return text


def check_name(name: Any) -> str:
    if (
        not isinstance(name, str)
        or not NAME_RE.match(name)
        or name.endswith(RECORD_SUFFIX)
        or PARTIAL_MARK in name
    ):
        raise Failure(
            EXIT_USAGE,
            f"source name {name!r} must be one path component of [A-Za-z0-9._-], "
            f"not starting with '.', not ending in {RECORD_SUFFIX}",
        )
    return name


def load_source(path: Path) -> dict[str, Any]:
    """The source file, strictly: exact keys, typed values, consistent policy."""
    try:
        with open(path, "rb") as f:
            data = f.read(MAX_SOURCE_BYTES + 1)
    except OSError as e:
        raise Failure(EXIT_USAGE, f"cannot read source {path}: {e.strerror}") from None
    if len(data) > MAX_SOURCE_BYTES:
        raise Failure(EXIT_USAGE, f"source {path} is larger than 1 MiB")
    src = _object(
        _strict_json(data, "source"),
        {"schema", "name", "url", "expected", "hash_reference", "terms"},
        "source",
    )
    if src["schema"] != SOURCE_SCHEMA:
        raise Failure(EXIT_USAGE, f"source schema is not {SOURCE_SCHEMA}")
    check_name(src["name"])
    check_https_url(_text(src["url"], "source url"), "source url")
    exp = _object(src["expected"], {"sha256", "bytes"}, "source expected")
    if exp["sha256"] is not None and (
        not isinstance(exp["sha256"], str) or not SHA_RE.match(exp["sha256"])
    ):
        raise Failure(
            EXIT_USAGE, "source expected.sha256 is not null or 64 lowercase hex"
        )
    b = exp["bytes"]
    if b is not None and (isinstance(b, bool) or not isinstance(b, int) or b <= 0):
        raise Failure(
            EXIT_USAGE, "source expected.bytes is not null or a positive integer"
        )
    ref = _object(src["hash_reference"], {"kind", "citation"}, "source hash_reference")
    if ref["kind"] not in HASH_KINDS:
        raise Failure(
            EXIT_USAGE, f"source hash_reference.kind is not one of {HASH_KINDS}"
        )
    if ref["citation"] is not None:
        _text(ref["citation"], "source hash_reference.citation")
    if (ref["kind"] == "none") != (exp["sha256"] is None):
        raise Failure(
            EXIT_USAGE,
            "source hash_reference.kind is 'none' exactly when expected.sha256 is null",
        )
    if ref["kind"] == "upstream_published" and ref["citation"] is None:
        raise Failure(
            EXIT_USAGE,
            "source hash_reference.kind upstream_published needs a citation "
            "(where upstream publishes the hash)",
        )
    terms = _object(src["terms"], {"url", "accepted", "accepted_utc"}, "source terms")
    _text(terms["url"], "source terms.url")
    if terms["accepted"] is not True:
        raise Failure(EXIT_USAGE, "source terms.accepted is not true; nothing fetched")
    _text(terms["accepted_utc"], "source terms.accepted_utc")
    src["descriptor_sha256"] = _sha256_bytes(data)
    return src


def open_dest(path: Path) -> int:
    """DIR as a directory fd; DIR itself must not be a symlink."""
    try:
        st = os.lstat(path)
    except OSError as e:
        raise Failure(EXIT_USAGE, f"--dest {path}: {e.strerror}") from None
    if not stat.S_ISDIR(st.st_mode):
        raise Failure(
            EXIT_USAGE, f"--dest {path} is not a directory (symlinks refused)"
        )
    try:
        return os.open(
            path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
    except OSError as e:
        raise Failure(EXIT_USAGE, f"--dest {path}: {e.strerror}") from None


def entry(dfd: int, name: str) -> os.stat_result | None:
    """lstat of DIR/name, refusing anything that is not a regular file."""
    try:
        st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as e:
        raise Failure(EXIT_USAGE, f"{name}: {e.strerror}") from None
    if not stat.S_ISREG(st.st_mode):
        raise Failure(
            EXIT_USAGE, f"{name} exists and is not a regular file (symlinks refused)"
        )
    return st


def read_at(dfd: int, name: str, limit: int | None = None) -> bytes:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dfd)
    with os.fdopen(fd, "rb") as f:
        return f.read() if limit is None else f.read(limit + 1)


def hash_at(dfd: int, name: str) -> tuple[int, str]:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dfd)
    h, n = hashlib.sha256(), 0
    with os.fdopen(fd, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
            n += len(chunk)
    return n, h.hexdigest()


def integrity_of(src: dict[str, Any]) -> str:
    return (
        "local_only" if src["expected"]["sha256"] is None else "expected_sha256_matched"
    )


def record_source(src: dict[str, Any]) -> dict[str, Any]:
    """The record's ``source`` object: the source file without the url's
    query and fragment, bound to the exact file by its sha256."""
    return {
        "url": redact(src["url"]),
        "expected": src["expected"],
        "hash_reference": src["hash_reference"],
        "terms": src["terms"],
        "descriptor_sha256": src["descriptor_sha256"],
    }


def published_problems(
    dfd: int, src: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Why DIR/<name> + its record are not a verified pair for this source
    (empty when they are), and what was observed."""
    name = src["name"]
    rec_name = name + RECORD_SUFFIX
    problems: list[str] = []
    observed: dict[str, Any] = {"bytes": None, "sha256": None, "integrity": None}
    try:
        file_st, rec_st = entry(dfd, name), entry(dfd, rec_name)
    except Failure as e:
        return [str(e)], observed
    if file_st is None:
        problems.append(f"{name} does not exist")
    if rec_st is None:
        problems.append(f"{rec_name} does not exist (the file is not published)")
    if problems:
        return problems, observed
    try:
        raw = read_at(dfd, rec_name, MAX_SOURCE_BYTES)
    except OSError as e:
        return [f"{rec_name}: {e.strerror}"], observed
    if len(raw) > MAX_SOURCE_BYTES:
        return [f"{rec_name} is larger than 1 MiB"], observed
    try:
        rec = _strict_json(raw, rec_name)
    except Failure as e:
        return [str(e)], observed
    if not isinstance(rec, dict) or rec.get("schema") != RECORD_SCHEMA:
        return [f"{rec_name} schema is not {RECORD_SCHEMA}"], observed
    rsrc = rec.get("source") if isinstance(rec.get("source"), dict) else {}
    want_src = record_source(src)
    if set(rsrc) != set(want_src):
        problems.append(
            f"record source keys {sorted(rsrc)} are not exactly {sorted(want_src)}"
        )
    for key, want, got in (
        ("name", name, rec.get("name")),
        *((f"source.{k}", v, rsrc.get(k)) for k, v in want_src.items()),
        ("integrity", integrity_of(src), rec.get("integrity")),
        ("source_authentication", "not_performed", rec.get("source_authentication")),
    ):
        if got != want:
            problems.append(f"record {key} {got!r} != {want!r}")
    try:
        n, sha = hash_at(dfd, name)
    except OSError as e:
        return [*problems, f"{name}: {e.strerror}"], observed
    observed.update(bytes=n, sha256=sha, integrity=rec.get("integrity"))
    if (rec.get("bytes"), rec.get("sha256")) != (n, sha):
        problems.append(
            f"{name} is {n} bytes sha256 {sha}; the record says "
            f"{rec.get('bytes')} bytes sha256 {rec.get('sha256')}"
        )
    exp = src["expected"]
    if exp["sha256"] is not None and sha != exp["sha256"]:
        problems.append(f"{name} sha256 {sha} != expected {exp['sha256']}")
    if exp["bytes"] is not None and n != exp["bytes"]:
        problems.append(f"{name} is {n} bytes, expected {exp['bytes']}")
    return problems, observed


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Redirects come back as HTTPError so every hop is checked here."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _unlink(dfd: int, name: str | None) -> None:
    if name is None:
        return
    try:
        os.unlink(name, dir_fd=dfd)
    except FileNotFoundError:
        pass


def download(
    dfd: int, src: dict[str, Any], args: argparse.Namespace, staging: str
) -> dict[str, Any]:
    """Download src into DIR/staging and check it; raise Failure otherwise."""
    try:
        ctx = ssl.create_default_context(cafile=args.ca_bundle)
    except (OSError, ssl.SSLError) as e:
        raise Failure(EXIT_USAGE, f"--ca-bundle {args.ca_bundle}: {e}") from None
    opener = urllib.request.build_opener(
        _NoRedirect(), urllib.request.HTTPSHandler(context=ctx)
    )
    url, seen, hops = src["url"], {src["url"]}, []
    for _ in range(args.max_redirects + 1):
        check_https_url(url, "redirect target" if hops else "source url")
        req = urllib.request.Request(
            url,
            headers={
                "Accept-Encoding": "identity",
                "User-Agent": "saccade-model-fetch/1",
            },
        )
        try:
            resp = opener.open(req, timeout=args.timeout)
        except urllib.error.HTTPError as e:
            location = e.headers.get("Location") if e.headers else None
            e.close()
            if e.code not in REDIRECTS:
                raise Failure(
                    EXIT_TRANSPORT, f"HTTP {e.code} from {redact(url)}"
                ) from None
            if not location:
                raise Failure(
                    EXIT_TRANSPORT, f"HTTP {e.code} without Location"
                ) from None
            nxt = urljoin(url, location)
            hops.append({"url": redact(url), "status": e.code, "location": redact(nxt)})
            if urlsplit(nxt).scheme != "https":
                raise Failure(
                    EXIT_USAGE,
                    f"redirect from {redact(url)} leaves https: {redact(nxt)}",
                ) from None
            if nxt in seen:
                raise Failure(EXIT_USAGE, f"redirect loop at {redact(nxt)}") from None
            seen.add(nxt)
            url = nxt
            continue
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            reason = scrub(str(getattr(e, "reason", e)), url)
            raise Failure(EXIT_TRANSPORT, f"{redact(url)}: {reason}") from None
        with resp:
            if resp.status != 200:
                raise Failure(EXIT_TRANSPORT, f"HTTP {resp.status} from {redact(url)}")
            return {
                **stream(dfd, src, args, staging, resp, url),
                "final_url": redact(url),
                "redirects": hops,
            }
    raise Failure(EXIT_USAGE, f"more than {args.max_redirects} redirects")


def stream(
    dfd: int,
    src: dict[str, Any],
    args: argparse.Namespace,
    staging: str,
    resp: Any,
    url: str,
) -> dict[str, Any]:
    exp = src["expected"]
    # Rejected header values may contain signed URLs or secrets; do not echo them.
    encoding = (resp.headers.get("Content-Encoding") or "identity").strip().lower()
    if encoding != "identity":
        raise Failure(EXIT_TRANSPORT, "unsupported Content-Encoding")
    length_text = resp.headers.get("Content-Length")
    try:
        length = int(length_text) if length_text is not None else None
    except ValueError:
        raise Failure(EXIT_TRANSPORT, "invalid Content-Length") from None
    cap = exp["bytes"] if exp["bytes"] is not None else args.max_bytes
    if length is not None and exp["bytes"] is not None and length != exp["bytes"]:
        raise Failure(
            EXIT_VERIFY,
            f"size mismatch: Content-Length {length} != expected {exp['bytes']}",
        )
    if length is not None and length > cap:
        raise Failure(EXIT_VERIFY, f"Content-Length {length} exceeds --max-bytes {cap}")
    fd = os.open(
        staging,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=dfd,
    )
    h, n = hashlib.sha256(), 0
    with os.fdopen(fd, "wb") as out:
        while True:
            try:
                chunk = resp.read(CHUNK)
            except http.client.IncompleteRead as e:
                raise Failure(
                    EXIT_VERIFY, f"truncated: {n + len(e.partial)} of {length} bytes"
                ) from None
            except (OSError, http.client.HTTPException) as e:
                raise Failure(
                    EXIT_TRANSPORT, f"read failed after {n} bytes: {scrub(str(e), url)}"
                ) from None
            if not chunk:
                break
            n += len(chunk)
            if n > cap:
                raise Failure(EXIT_VERIFY, f"more than {cap} bytes; stopped")
            h.update(chunk)
            out.write(chunk)
        out.flush()
        os.fchmod(out.fileno(), 0o644)
        os.fsync(out.fileno())
    sha = h.hexdigest()
    if length is not None and n != length:
        raise Failure(EXIT_VERIFY, f"truncated: {n} of {length} bytes")
    if exp["bytes"] is not None and n != exp["bytes"]:
        raise Failure(EXIT_VERIFY, f"size mismatch: {n} != expected {exp['bytes']}")
    if exp["sha256"] is not None and sha != exp["sha256"]:
        raise Failure(
            EXIT_VERIFY, f"sha256 mismatch: {sha} != expected {exp['sha256']}"
        )
    return {"bytes": n, "sha256": sha, "content_length": length}


def link_noreplace(dfd: int, src_name: str, dst_name: str) -> None:
    try:
        os.link(
            src_name, dst_name, src_dir_fd=dfd, dst_dir_fd=dfd, follow_symlinks=False
        )
    except FileExistsError:
        raise Failure(
            EXIT_USAGE, f"{dst_name} appeared during the fetch; not replaced"
        ) from None
    except OSError as e:
        if e.errno in (errno.EPERM, errno.EOPNOTSUPP, errno.EXDEV, errno.EMLINK):
            raise Failure(
                EXIT_USAGE,
                f"cannot publish {dst_name} without replacing (hard link: {e.strerror})",
            ) from None
        raise


def _tool_record() -> dict[str, Any]:
    me = Path(__file__).resolve()
    rel = me.relative_to(me.parents[2]) if len(me.parents) > 2 else me
    return {"path": str(rel), "sha256": _sha256_bytes(me.read_bytes())}


def cmd_fetch(args: argparse.Namespace) -> dict[str, Any]:
    src = load_source(args.source)
    if src["expected"]["sha256"] is None and not args.allow_no_expected_hash:
        raise Failure(
            EXIT_USAGE,
            "the source has no expected sha256; refusing before any request "
            "(--allow-no-expected-hash fetches with a local_only record)",
        )
    name, rec_name = src["name"], src["name"] + RECORD_SUFFIX
    dfd = open_dest(args.dest)
    try:
        try:
            fcntl.flock(dfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Failure(
                EXIT_USAGE, f"another model_fetch is running in {args.dest}"
            ) from None
        stale = sorted(
            n for n in os.listdir(dfd) if n.startswith("." + name) and PARTIAL_MARK in n
        )
        base = {
            "name": name,
            "path": str(Path(os.path.realpath(args.dest)) / name),
            "stale_partials": stale,
        }
        file_st, rec_st = entry(dfd, name), entry(dfd, rec_name)
        if file_st is not None and rec_st is not None:
            problems, observed = published_problems(dfd, src)
            if problems:
                raise Failure(
                    EXIT_USAGE,
                    f"{name} is already published and does not match this source "
                    f"(not replaced): {'; '.join(problems)}",
                )
            return {**base, "status": "already_present", **observed}
        if rec_st is not None:
            raise Failure(
                EXIT_USAGE, f"{rec_name} exists without {name}; remove the stale record"
            )
        orphan = None
        if file_st is not None:
            orphan = hash_at(dfd, name)
            want = src["expected"]["sha256"]
            if want is not None and orphan[1] != want:
                raise Failure(
                    EXIT_USAGE,
                    f"{name} exists unpublished with sha256 {orphan[1]} != expected; not replaced",
                )
        token = secrets.token_hex(8)
        staging = f".{name}{PARTIAL_MARK}{token}"
        rec_staging = f".{rec_name}{PARTIAL_MARK}{token}"
        try:
            got = download(dfd, src, args, staging)
            if orphan is not None and orphan != (got["bytes"], got["sha256"]):
                raise Failure(
                    EXIT_USAGE,
                    f"{name} exists unpublished and differs from the downloaded bytes; not replaced",
                )
            record = {
                "schema": RECORD_SCHEMA,
                "reading": READING,
                "name": name,
                "bytes": got["bytes"],
                "sha256": got["sha256"],
                "integrity": integrity_of(src),
                "source_authentication": "not_performed",
                "source": record_source(src),
                "transport": {
                    "final_url": got["final_url"],
                    "redirects": got["redirects"],
                    "content_length": got["content_length"],
                    "query_redacted": True,
                },
                "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "tool": _tool_record(),
            }
            fd = os.open(
                rec_staging,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=dfd,
            )
            with os.fdopen(fd, "w") as out:
                out.write(json.dumps(record, indent=2) + "\n")
                out.flush()
                os.fsync(out.fileno())
            if orphan is None:
                link_noreplace(dfd, staging, name)
            link_noreplace(dfd, rec_staging, rec_name)
            os.fsync(dfd)
        finally:
            _unlink(dfd, staging)
            _unlink(dfd, rec_staging)
        return {
            **base,
            "status": "fetched" if orphan is None else "record_recovered",
            "bytes": got["bytes"],
            "sha256": got["sha256"],
            "integrity": record["integrity"],
            "final_url": got["final_url"],
        }
    finally:
        os.close(dfd)


def cmd_verify(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    src = load_source(args.source)
    dfd = open_dest(args.dest)
    try:
        problems, observed = published_problems(dfd, src)
    finally:
        os.close(dfd)
    report = {
        "schema": VERIFY_SCHEMA,
        "reading": READING,
        "status": "failed" if problems else "verified",
        "name": src["name"],
        "path": str(Path(os.path.realpath(args.dest)) / src["name"]),
        **observed,
        "source_authentication": "not_performed",
        "problems": problems,
    }
    return (EXIT_VERIFY if problems else EXIT_OK), report


class _Parser(argparse.ArgumentParser):
    """Argument errors become a Failure, so they print the JSON result too."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        raise Failure(EXIT_USAGE, f"{self.prog}: {message}")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download, check and publish one model file")
    v = sub.add_parser("verify", help="check a published file offline")
    for p in (f, v):
        p.add_argument(
            "--source", type=Path, required=True, help="saccade.model_source/v1 JSON"
        )
        p.add_argument("--dest", type=Path, required=True, help="existing directory")
    f.add_argument(
        "--allow-no-expected-hash",
        action="store_true",
        help="fetch a source without expected sha256 (record: local_only)",
    )
    f.add_argument("--ca-bundle", help="PEM file of trusted CA certificates")
    f.add_argument("--max-redirects", type=int, default=5)
    f.add_argument(
        "--timeout", type=float, default=60.0, help="seconds per network operation"
    )
    f.add_argument(
        "--max-bytes",
        type=int,
        default=64 << 30,
        help="size cap when the source has no expected bytes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    previous = {
        s: signal.signal(s, _raise_interrupted) for s in (signal.SIGTERM, signal.SIGHUP)
    }
    command, code, result = None, EXIT_OK, {}
    try:
        args = _parser().parse_args(argv)
        command = args.command
        if command == "fetch":
            if args.max_redirects < 0 or args.max_bytes <= 0 or args.timeout <= 0:
                raise Failure(
                    EXIT_USAGE, "--max-redirects >= 0, --max-bytes > 0, --timeout > 0"
                )
            result = cmd_fetch(args)
        else:
            code, result = cmd_verify(args)
    except Failure as e:
        code, result = e.code, {"status": "failed", "error": str(e)}
    except OSError as e:
        where = f" ({e.filename})" if e.filename is not None else ""
        code, result = (
            EXIT_USAGE,
            {"status": "failed", "error": f"file system: {e.strerror or e}{where}"},
        )
    except KeyboardInterrupt:
        code, result = (
            128 + signal.SIGINT,
            {"status": "interrupted", "signal": int(signal.SIGINT)},
        )
    except Interrupted as e:
        code, result = 128 + e.signum, {"status": "interrupted", "signal": e.signum}
    except Exception as e:  # a bug: still one JSON result, traceback on stderr
        traceback.print_exc()
        code, result = (
            EXIT_INTERNAL,
            {"status": "internal_error", "error": f"{type(e).__name__}: {e}"},
        )
    finally:
        for s, h in previous.items():
            signal.signal(s, h)
    if command == "fetch":
        result = {
            "schema": RESULT_SCHEMA,
            "command": "fetch",
            **result,
            "source_authentication": "not_performed",
        }
    result["exit_code"] = code
    if code != EXIT_OK and "error" in result:
        print(f"model_fetch: {result['error']}", file=sys.stderr)
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
