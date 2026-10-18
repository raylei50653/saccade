"""Model fetch / offline verify tool (#610 slice 2, scripts/model/model_fetch.py).

docs/reference/model_fetch_610.md. Every test runs against a local https
server on 127.0.0.1 (a test CA and leaf certificate made here; the client gets
the CA through ``--ca-bundle``) with a small synthetic body; nothing leaves
the host and no real model is downloaded.

* positive: fetch publishes ``<name>`` and, last, ``<name>.fetch.json``;
  https redirects are followed and recorded with their queries redacted; a
  rerun of a published pair sends no request; ``verify`` checks offline;
* policy: no expected sha256 is refused before any request, and with
  ``--allow-no-expected-hash`` the record says ``local_only``; every record
  says ``source_authentication: not_performed``;
* negative: sha256 / size mismatch, truncation, more bytes than expected,
  a redirect to http, a redirect loop, too many redirects, HTTP errors, an
  untrusted certificate, a Content-Encoding, invalid sources, symlinked
  destinations, conflicts with existing files, a held lock -- none leaves a
  formal file, a record or a staging file, and none changes an existing file;
* interruption: SIGINT / SIGTERM mid-download remove the staging file; after
  SIGKILL the leftover staging file is reported, never published, and a rerun
  publishes; a file left without its record is recovered only when the bytes
  are identical.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import importlib.util
import ipaddress
import json
import os
import signal
import ssl
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "scripts/model/model_fetch.py"
BODY = bytes(range(256)) * 9000 + b"tail"  # ~2.2 MiB: more than one 1 MiB chunk
BODY_SHA = hashlib.sha256(BODY).hexdigest()
NAME = "model.bin"
PROXY_VARS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("model_fetch", TOOL)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mf = _load()


# ── TLS material and the local server ───────────────────────────────────────


def _certs(d: Path) -> tuple[Path, Path, Path]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = dt.datetime.now(dt.timezone.utc)

    def build(subject: str, issuer: str, key: Any, sign_key: Any, ca: bool) -> Any:
        b = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer)]))
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=1))
            .add_extension(
                x509.BasicConstraints(ca=ca, path_length=None), critical=True
            )
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
                critical=False,
            )
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(
                    sign_key.public_key()
                ),
                critical=False,
            )
        )
        if ca:
            b = b.add_extension(
                x509.KeyUsage(
                    False, False, False, False, False, True, True, False, False
                ),
                critical=True,
            )
        else:
            b = b.add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                        x509.DNSName("localhost"),
                    ]
                ),
                critical=False,
            ).add_extension(
                x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
                critical=False,
            )
        return b.sign(sign_key, hashes.SHA256())

    d.mkdir(parents=True, exist_ok=True)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    ca = build("saccade test CA", "saccade test CA", ca_key, ca_key, True)
    leaf = build("127.0.0.1", "saccade test CA", leaf_key, ca_key, False)
    pem = serialization.Encoding.PEM
    ca_pem, leaf_pem, key_pem = d / "ca.pem", d / "leaf.pem", d / "leaf.key"
    ca_pem.write_bytes(ca.public_bytes(pem))
    leaf_pem.write_bytes(leaf.public_bytes(pem))
    key_pem.write_bytes(
        leaf_key.private_bytes(
            pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    return ca_pem, leaf_pem, key_pem


class Server:
    """https on 127.0.0.1; routes map a path to a handler function."""

    def __init__(self, tmp: Path) -> None:
        self.ca, leaf, key = _certs(tmp)
        self.routes: dict[str, Callable[[BaseHTTPRequestHandler], None]] = {}
        self.requests: list[str] = []
        self.release = threading.Event()
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802 (http.server API)
                path = self.path.split("?", 1)[0]
                server.requests.append(self.path)
                route = server.routes.get(path)
                if route is None:
                    self.send_error(404)
                    return
                route(self)

            def log_message(self, *args: Any) -> None:
                pass

        class Quiet(ThreadingHTTPServer):
            def handle_error(self, request: Any, client_address: Any) -> None:
                # A client that aborts on purpose (more bytes than expected, an
                # untrusted certificate) resets the connection; anything else
                # is a bug in this server and is reported.
                if not isinstance(sys.exc_info()[1], (ConnectionError, ssl.SSLError)):
                    super().handle_error(request, client_address)

        self.httpd = Quiet(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(leaf, key)
        self.httpd.socket = ctx.wrap_socket(self.httpd.socket, server_side=True)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path: str) -> str:
        return f"https://127.0.0.1:{self.port}{path}"

    def close(self) -> None:
        self.release.set()
        self.httpd.shutdown()
        self.httpd.server_close()


def body(
    data: bytes = BODY,
    *,
    length: int | None = -1,
    headers: dict[str, str] | None = None,
    close_after: int | None = None,
) -> Callable[[BaseHTTPRequestHandler], None]:
    """200 with data; length -1 = len(data), None = no Content-Length (close)."""

    def route(h: BaseHTTPRequestHandler) -> None:
        h.send_response(200)
        for k, v in (headers or {}).items():
            h.send_header(k, v)
        if length is None:
            h.send_header("Connection", "close")
            h.close_connection = True
        else:
            h.send_header("Content-Length", str(len(data) if length == -1 else length))
        h.end_headers()
        h.wfile.write(data if close_after is None else data[:close_after])
        h.wfile.flush()
        if close_after is not None or length is None:
            h.close_connection = True

    return route


def redirect(
    location: str, code: int = 302
) -> Callable[[BaseHTTPRequestHandler], None]:
    def route(h: BaseHTTPRequestHandler) -> None:
        h.send_response(code)
        h.send_header("Location", location)
        h.send_header("Content-Length", "0")
        h.end_headers()

    return route


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for v in PROXY_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("no_proxy", "*")


@pytest.fixture
def server(tmp_path: Path) -> Iterator[Server]:
    s = Server(tmp_path / "tls")
    yield s
    s.close()


@pytest.fixture
def dest(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    return d


def make_source(
    tmp: Path,
    url: str,
    *,
    sha: str | None = BODY_SHA,
    size: int | None = len(BODY),
    kind: str | None = None,
    citation: str | None = "https://example.invalid/release",
    name: str = NAME,
    accepted: Any = True,
    extra: dict[str, Any] | None = None,
    raw: str | None = None,
) -> Path:
    p = tmp / f"source-{time.monotonic_ns()}.json"
    if raw is not None:
        p.write_text(raw)
        return p
    if kind is None:
        kind = "none" if sha is None else "upstream_published"
    src = {
        "schema": "saccade.model_source/v1",
        "name": name,
        "url": url,
        "expected": {"sha256": sha, "bytes": size},
        "hash_reference": {
            "kind": kind,
            "citation": citation if kind != "none" else None,
        },
        "terms": {
            "url": "https://example.invalid/terms",
            "accepted": accepted,
            "accepted_utc": "2026-10-18T00:00:00Z",
        },
        **(extra or {}),
    }
    p.write_text(json.dumps(src))
    return p


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = mf.main(list(argv))
    out = json.loads(capsys.readouterr().out)
    assert out["exit_code"] == code
    return code, out


def fetch(
    capsys: pytest.CaptureFixture[str],
    server: Server,
    src: Path,
    dest: Path,
    *extra: str,
) -> tuple[int, dict[str, Any]]:
    return run(
        capsys,
        "fetch",
        "--source",
        str(src),
        "--dest",
        str(dest),
        "--ca-bundle",
        str(server.ca),
        *extra,
    )


def verify(
    capsys: pytest.CaptureFixture[str], src: Path, dest: Path
) -> tuple[int, dict[str, Any]]:
    return run(capsys, "verify", "--source", str(src), "--dest", str(dest))


def listing(d: Path) -> list[str]:
    return sorted(os.listdir(d))


def record(dest: Path) -> dict[str, Any]:
    rec: dict[str, Any] = json.loads((dest / (NAME + ".fetch.json")).read_text())
    return rec


# ── positive controls ───────────────────────────────────────────────────────


def test_fetch_publishes_pair_and_verify_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], server: Server, dest: Path
) -> None:
    server.routes["/m"] = body()
    src = make_source(tmp_path, server.url("/m"))
    code, out = fetch(capsys, server, src, dest)
    assert code == 0 and out["status"] == "fetched" and out["sha256"] == BODY_SHA
    assert out["source_authentication"] == "not_performed"
    assert listing(dest) == [NAME, NAME + ".fetch.json"]
    assert (dest / NAME).read_bytes() == BODY
    rec = record(dest)
    assert rec["schema"] == "saccade.model_fetch_record/v1"
    assert rec["integrity"] == "expected_sha256_matched"
    assert rec["source_authentication"] == "not_performed"
    assert (rec["bytes"], rec["sha256"]) == (len(BODY), BODY_SHA)
    assert rec["tool"]["path"] == "scripts/model/model_fetch.py"
    code, rep = verify(capsys, src, dest)
    assert code == 0 and rep["status"] == "verified" and rep["problems"] == []


def test_https_redirects_followed_and_recorded_without_query(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], server: Server, dest: Path
) -> None:
    server.routes["/a"] = redirect("/b?sig=SECRET-TOKEN", 302)
    server.routes["/b"] = redirect(server.url("/m?X-Amz-Signature=SECRET-TOKEN"), 307)
    server.routes["/m"] = body()
    code, out = fetch(capsys, server, make_source(tmp_path, server.url("/a")), dest)
    assert code == 0
    rec = record(dest)
    assert [h["status"] for h in rec["transport"]["redirects"]] == [302, 307]
    assert rec["transport"]["final_url"] == server.url("/m")
    assert "SECRET-TOKEN" not in json.dumps(rec["transport"])


def test_rerun_of_published_pair_sends_no_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], server: Server, dest: Path
) -> None:
    server.routes["/m"] = body()
    src = make_source(tmp_path, server.url("/m"))
    assert fetch(capsys, server, src, dest)[0] == 0
    before, mtime = len(server.requests), (dest / NAME).stat().st_mtime_ns
    code, out = fetch(capsys, server, src, dest)
    assert code == 0 and out["status"] == "already_present"
    assert len(server.requests) == before
    assert (dest / NAME).stat().st_mtime_ns == mtime


def test_no_expected_hash_is_refused_before_any_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], server: Server, dest: Path
) -> None:
    server.routes["/m"] = body()
    code, out = fetch(
        capsys, server, make_source(tmp_path, server.url("/m"), sha=None), dest
    )
    assert code == 2 and "no expected sha256" in out["error"]
    assert server.requests == [] and listing(dest) == []


def test_no_expected_hash_allowed_is_local_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], server: Server, dest: Path
) -> None:
    server.routes["/m"] = body()
    src = make_source(tmp_path, server.url("/m"), sha=None, size=None)
    code, out = fetch(capsys, server, src, dest, "--allow-no-expected-hash")
    assert code == 0 and out["integrity"] == "local_only"
    assert record(dest)["integrity"] == "local_only"
    assert record(dest)["source"]["hash_reference"]["kind"] == "none"
    code, rep = verify(capsys, src, dest)
    assert code == 0 and rep["integrity"] == "local_only"
    assert rep["source_authentication"] == "not_performed"


# ── negative controls: nothing published, nothing left ──────────────────────


@pytest.mark.parametrize(
    ("route", "src_kw", "code", "needle"),
    [
        (body(), {"sha": "0" * 64}, 1, "sha256 mismatch"),
        (body(), {"size": len(BODY) - 1}, 1, "size mismatch: Content-Length"),
        (body(length=None), {"size": len(BODY) - 10, "sha": "1" * 64}, 1, "more than"),
        (body(length=len(BODY) + 100), {"size": None}, 1, "truncated"),
        (body(close_after=1000), {}, 1, "truncated"),
        (body(headers={"Content-Encoding": "gzip"}), {}, 3, "Content-Encoding gzip"),
        (lambda h: h.send_error(404), {}, 3, "HTTP 404"),
    ],
    ids=[
        "sha",
        "size-header",
        "too-many-bytes",
        "short-length",
        "connection-closed",
        "gzip",
        "404",
    ],
)
def test_failures_leave_no_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    server: Server,
    dest: Path,
    route: Callable[[BaseHTTPRequestHandler], None],
    src_kw: dict[str, Any],
    code: int,
    needle: str,
) -> None:
    server.routes["/m"] = route
    got, out = fetch(
        capsys, server, make_source(tmp_path, server.url("/m"), **src_kw), dest
    )
    assert got == code and needle in out["error"]
    assert listing(dest) == []


def test_redirect_to_http_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], server: Server, dest: Path
) -> None:
    server.routes["/a"] = redirect(f"http://127.0.0.1:{server.port}/m")
    server.routes["/m"] = body()
    code, out = fetch(capsys, server, make_source(tmp_path, server.url("/a")), dest)
    assert code == 2 and "leaves https" in out["error"]
    assert server.requests == ["/a"] and listing(dest) == []


def test_redirect_loop_and_too_many_redirects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], server: Server, dest: Path
) -> None:
    server.routes["/loop"] = redirect("/loop")
    code, out = fetch(capsys, server, make_source(tmp_path, server.url("/loop")), dest)
    assert code == 2 and "redirect loop" in out["error"]
    for i in range(3):
        server.routes[f"/c{i}"] = redirect(f"/c{i + 1}")
    server.routes["/c3"] = body()
    code, out = fetch(
        capsys,
        server,
        make_source(tmp_path, server.url("/c0")),
        dest,
        "--max-redirects",
        "2",
    )
    assert code == 2 and "more than 2 redirects" in out["error"]
    assert listing(dest) == []


def test_untrusted_certificate_is_a_transport_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], server: Server, dest: Path
) -> None:
    server.routes["/m"] = body()
    src = make_source(tmp_path, server.url("/m"))
    code, out = run(capsys, "fetch", "--source", str(src), "--dest", str(dest))
    assert code == 3 and "CERTIFICATE_VERIFY_FAILED" in out["error"]
    assert listing(dest) == []


@pytest.mark.parametrize(
    ("kw", "needle"),
    [
        ({"url": "http://127.0.0.1/m"}, "not an https URL"),
        ({"url": "https://user:pw@127.0.0.1/m"}, "credentials"),
        ({"name": "../x"}, "one path component"),
        ({"name": "a/b"}, "one path component"),
        ({"name": ".hidden"}, "one path component"),
        ({"name": "m.fetch.json"}, "one path component"),
        ({"name": "m.partial-1"}, "one path component"),
        ({"kind": "none"}, "exactly when expected.sha256 is null"),
        ({"citation": None}, "needs a citation"),
        ({"accepted": False}, "terms.accepted is not true"),
        ({"accepted": "yes"}, "terms.accepted is not true"),
        ({"size": True}, "positive integer"),
        ({"sha": BODY_SHA.upper()}, "64 lowercase hex"),
        ({"extra": {"mirror": "x"}}, "not exactly"),
        ({"raw": '{"schema": "a", "schema": "b"}'}, "duplicate key"),
    ],
)
def test_invalid_sources_are_refused_before_any_request(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    server: Server,
    dest: Path,
    kw: dict[str, Any],
    needle: str,
) -> None:
    server.routes["/m"] = body()
    url = kw.pop("url", server.url("/m"))
    code, out = fetch(capsys, server, make_source(tmp_path, url, **kw), dest)
    assert code == 2 and needle in out["error"]
    assert server.requests == [] and listing(dest) == []


def test_symlinked_destinations_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], server: Server, dest: Path
) -> None:
    server.routes["/m"] = body()
    src = make_source(tmp_path, server.url("/m"))
    link_dir = tmp_path / "link"
    link_dir.symlink_to(dest)
    code, out = fetch(capsys, server, src, link_dir)
    assert code == 2 and "not a directory" in out["error"]
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"keep")
    (dest / NAME).symlink_to(outside)
    code, out = fetch(capsys, server, src, dest)
    assert code == 2 and "not a regular file" in out["error"]
    assert outside.read_bytes() == b"keep" and server.requests == []
    (dest / NAME).unlink()
    (dest / NAME).mkdir()
    assert fetch(capsys, server, src, dest)[0] == 2


def test_published_pair_for_another_source_is_not_replaced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], server: Server, dest: Path
) -> None:
    server.routes["/m"] = body()
    server.routes["/n"] = body(b"other bytes")
    assert fetch(capsys, server, make_source(tmp_path, server.url("/m")), dest)[0] == 0
    other = make_source(
        tmp_path,
        server.url("/n"),
        sha=hashlib.sha256(b"other bytes").hexdigest(),
        size=11,
    )
    before = len(server.requests)
    code, out = fetch(capsys, server, other, dest)
    assert code == 2 and "already published and does not match" in out["error"]
    assert len(server.requests) == before and (dest / NAME).read_bytes() == BODY


def test_stale_record_without_file_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], server: Server, dest: Path
) -> None:
    server.routes["/m"] = body()
    src = make_source(tmp_path, server.url("/m"))
    assert fetch(capsys, server, src, dest)[0] == 0
    (dest / NAME).unlink()
    code, out = fetch(capsys, server, src, dest)
    assert code == 2 and "stale record" in out["error"]


def test_unpublished_file_recovered_only_when_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], server: Server, dest: Path
) -> None:
    server.routes["/m"] = body()
    src = make_source(tmp_path, server.url("/m"))
    (dest / NAME).write_bytes(BODY)
    inode = (dest / NAME).stat().st_ino
    code, out = fetch(capsys, server, src, dest)
    assert code == 0 and out["status"] == "record_recovered"
    assert (dest / NAME).stat().st_ino == inode and verify(capsys, src, dest)[0] == 0

    other = tmp_path / "other"
    other.mkdir()
    (other / NAME).write_bytes(b"x" * len(BODY))
    before = len(server.requests)
    code, out = fetch(capsys, server, src, other)
    assert code == 2 and "exists unpublished" in out["error"]
    assert len(server.requests) == before and (other / NAME).read_bytes() == b"x" * len(
        BODY
    )
    assert listing(other) == [NAME]


def test_unpublished_file_without_expected_hash_must_match_download(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], server: Server, dest: Path
) -> None:
    server.routes["/m"] = body()
    src = make_source(tmp_path, server.url("/m"), sha=None, size=None)
    (dest / NAME).write_bytes(b"different")
    code, out = fetch(capsys, server, src, dest, "--allow-no-expected-hash")
    assert code == 2 and "differs from the downloaded bytes" in out["error"]
    assert listing(dest) == [NAME] and (dest / NAME).read_bytes() == b"different"


def test_held_lock_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], server: Server, dest: Path
) -> None:
    server.routes["/m"] = body()
    fd = os.open(dest, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        code, out = fetch(capsys, server, make_source(tmp_path, server.url("/m")), dest)
    finally:
        os.close(fd)
    assert code == 2 and "another model_fetch" in out["error"] and server.requests == []


# ── offline verify ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "damage",
    [
        "flip_byte",
        "truncate",
        "no_record",
        "record_sha",
        "record_integrity",
        "source_hash",
        "source_url",
    ],
)
def test_verify_fails_closed_offline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    server: Server,
    dest: Path,
    damage: str,
) -> None:
    server.routes["/m"] = body()
    src = make_source(tmp_path, server.url("/m"))
    assert fetch(capsys, server, src, dest)[0] == 0
    server.close()  # verify must not need the network
    f, r = dest / NAME, dest / (NAME + ".fetch.json")
    if damage == "flip_byte":
        data = bytearray(f.read_bytes())
        data[5] ^= 1
        f.write_bytes(bytes(data))
    elif damage == "truncate":
        f.write_bytes(BODY[:-1])
    elif damage == "no_record":
        r.unlink()
    elif damage == "record_sha":
        rec = json.loads(r.read_text())
        rec["sha256"] = "0" * 64
        r.write_text(json.dumps(rec))
    elif damage == "record_integrity":
        rec = json.loads(r.read_text())
        rec["integrity"] = "local_only"
        r.write_text(json.dumps(rec))
    elif damage == "source_hash":
        src = make_source(tmp_path, server.url("/m"), sha="2" * 64)
    else:  # same bytes and hash, another URL: the record names where it came from
        src = make_source(tmp_path, server.url("/mirror"))
    code, rep = verify(capsys, src, dest)
    assert code == 1 and rep["status"] == "failed" and rep["problems"]


# ── interruption ────────────────────────────────────────────────────────────


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in PROXY_VARS}
    env["no_proxy"] = "*"
    return env


def _stall_route(server: Server) -> Callable[[BaseHTTPRequestHandler], None]:
    def route(h: BaseHTTPRequestHandler) -> None:
        h.send_response(200)
        h.send_header("Content-Length", str(len(BODY)))
        h.end_headers()
        h.wfile.write(BODY[: len(BODY) // 2])
        h.wfile.flush()
        server.release.wait(30)

    return route


def _start_stalled(
    tmp_path: Path, server: Server, dest: Path
) -> subprocess.Popen[bytes]:
    server.routes["/m"] = _stall_route(server)
    src = make_source(tmp_path, server.url("/m"))
    proc = subprocess.Popen(
        [
            sys.executable,
            str(TOOL),
            "fetch",
            "--source",
            str(src),
            "--dest",
            str(dest),
            "--ca-bundle",
            str(server.ca),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_env(),
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        partial = [n for n in listing(dest) if ".partial-" in n]
        if partial and (dest / partial[0]).stat().st_size > 0:
            return proc
        time.sleep(0.05)
    proc.kill()
    raise AssertionError("the download never started")


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_signal_mid_download_removes_staging(
    tmp_path: Path, server: Server, dest: Path, sig: signal.Signals
) -> None:
    proc = _start_stalled(tmp_path, server, dest)
    proc.send_signal(sig)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 128 + sig
    assert json.loads(out)["status"] == "interrupted"
    assert listing(dest) == []


def test_sigkill_leftover_is_never_published_and_rerun_recovers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], server: Server, dest: Path
) -> None:
    proc = _start_stalled(tmp_path, server, dest)
    proc.kill()
    proc.wait(timeout=30)
    leftover = [n for n in listing(dest) if ".partial-" in n]
    assert leftover and all(n.startswith(".") for n in leftover)
    assert NAME not in listing(dest)
    server.release.set()
    server.routes["/m"] = body()
    src = make_source(tmp_path, server.url("/m"))
    assert verify(capsys, src, dest)[0] == 1
    code, out = fetch(capsys, server, src, dest)
    assert (
        code == 0 and out["status"] == "fetched" and out["stale_partials"] == leftover
    )
    assert verify(capsys, src, dest)[0] == 0
